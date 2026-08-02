#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Public GAGAvatar runtime API for external integrations.

The original repository is demo-oriented: inference scripts own checkpoint
loading, tracked-avatar loading, batch construction, and rendering. This
module factors those steps into a small importable runtime so applications can
depend on GAGAvatar as a library.
"""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from pytorch3d.transforms import axis_angle_to_matrix

from gagavatar.assets import GAGAvatarAssets
from gagavatar.libs.flame_model import FLAMEModel
from gagavatar.libs.utils_renderer import render_gaussian
from gagavatar.models import GAGAvatar


GAGAVATAR_HEAD_GAUSSIAN_COUNT = 5023


@dataclass(frozen=True)
class GAGAvatarRuntimeConfig:
    asset_dir: Path | str | None = None
    assets: GAGAvatarAssets | None = None
    model_path: Path | str | None = None
    tracked_path: Path | str | None = None
    flame_model_path: Path | str | None = None
    device: str = "auto"
    point_plane_size: int = 296
    flame_scale: float = 5.0
    # Run the convolutional stages (gaussian generation, upsampler) under
    # torch.autocast with this dtype ("float16" / "bfloat16"). The Gaussian
    # rasterizer is fp32-only, so rendering decomposes around it; None keeps
    # the original single fp32 forward.
    autocast_dtype: str | None = None
    # Launch-overhead reduction for the upsampler (the largest conv block on
    # the per-frame path). "cuda-graph" captures its kernel sequence with
    # torch.cuda.CUDAGraph and replays it per call — works on any CUDA GPU
    # and any torch. torch.compile modes (e.g. "reduce-overhead") are also
    # accepted on compute capability >= 7.0, but inductor in torch 2.4
    # cannot compile the upsampler's interpolate calls (FunctionalTensor
    # bug), so "cuda-graph" is the working choice there too.
    compile_mode: str | None = None

    def resolved_assets(self) -> GAGAvatarAssets:
        if self.assets is not None:
            return self.assets
        return GAGAvatarAssets.resolve(
            root=self.asset_dir,
            model_path=self.model_path,
            tracked_path=self.tracked_path,
            flame_model_path=self.flame_model_path,
        )

    def resolved_model_path(self) -> Path:
        return self.resolved_assets().model_path

    def resolved_tracked_path(self) -> Path | None:
        return self.resolved_assets().tracked_path

    def resolved_flame_model_path(self) -> Path | None:
        return self.resolved_assets().flame_model_path


class CudaGraphReplay(torch.nn.Module):
    """Capture the wrapped module's kernel sequence once per input shape and
    replay it on later calls, collapsing per-op launch overhead into a single
    graph launch. The wrapped module must be shape-deterministic and
    side-effect-free (inference mode)."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._graphs: dict[tuple, tuple] = {}
        self._capture_failed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._capture_failed:
            return self.module(x)
        key = (tuple(x.shape), x.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(x)
            except Exception:
                # An optimization must never take the process down: fall
                # back to eager permanently for this module.
                self._capture_failed = True
                self._graphs.clear()
                import traceback

                print(
                    "[gagavatar] CUDA graph capture failed; continuing eager:\n"
                    + traceback.format_exc(limit=3)
                )
                return self.module(x)
            self._graphs[key] = entry
        graph, static_in, static_out = entry
        static_in.copy_(x)
        graph.replay()
        # The static output is overwritten by the next replay; hand the
        # caller its own copy.
        return static_out.clone()

    def _capture(self, x: torch.Tensor) -> tuple:
        # Quiesce the whole device first: capture aborts if other in-flight
        # work interleaves, and callers may run with stage syncs disabled.
        torch.cuda.synchronize()
        # Warm up on a side stream so capture sees a settled allocator,
        # then record with static input/output buffers. thread_local error
        # mode tolerates other threads touching CUDA mid-capture, which is
        # normal inside a threaded server process.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.module(x)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        static_in = x.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_out = self.module(static_in)
        return (graph, static_in, static_out)


class GAGAvatarRuntime:
    """Reusable GAGAvatar runtime for ARTalk motion outputs."""

    def __init__(self, config: GAGAvatarRuntimeConfig):
        self.config = config
        self.assets = config.resolved_assets()
        self.assets.validate()
        self.device = select_device(config.device)
        self.model = GAGAvatar().to(self.device).eval()
        checkpoint = torch.load(self.assets.model_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        state = {key: value for key, value in state.items() if "percep_loss" not in key}
        self.model.load_state_dict(state, strict=False)
        self.flame_model = FLAMEModel(
            n_shape=300,
            n_exp=100,
            scale=config.flame_scale,
            no_lmks=True,
            model_path=self.assets.flame_model_path,
        ).to(self.device)
        if config.compile_mode is not None and self.device.type == "cuda":
            if config.compile_mode == "cuda-graph":
                self.model.upsampler = CudaGraphReplay(self.model.upsampler)
            elif torch.cuda.get_device_capability(self.device) >= (7, 0):
                self.model.upsampler = torch.compile(
                    self.model.upsampler, mode=config.compile_mode
                )
        self.tracked_avatars = self._load_tracked_avatars(self.assets.tracked_path)
        self._tracked_avatar = None
        self._tracked_name = None
        self._feature_batch = None
        self._shape_code = None
        self._upper_points = None

    def available_avatar_ids(self) -> list[str]:
        return sorted(self.tracked_avatars)

    @property
    def camera_params(self) -> dict:
        return self.model.cam_params

    def load_tracked_avatar(self, avatar_id: str) -> dict:
        if avatar_id not in self.tracked_avatars:
            raise KeyError(f"Unknown GAGAvatar tracked avatar: {avatar_id}")
        return deepcopy(self.tracked_avatars[avatar_id])

    def set_avatar_id(self, avatar_id: str):
        self.set_tracked_avatar(self.load_tracked_avatar(avatar_id), avatar_id)

    def set_tracked_avatar(self, tracked_avatar: dict, avatar_id: str | None = None):
        tracked = deepcopy(tracked_avatar)
        for key, value in list(tracked.items()):
            if not isinstance(value, torch.Tensor):
                tracked[key] = torch.tensor(value).float()
        self._tracked_avatar = tracked
        self._tracked_name = avatar_id
        self._feature_batch = None
        self._shape_code = None
        self._upper_points = None
        if hasattr(self.model, "_gs_params"):
            del self.model._gs_params

    def shape_code(self, tracked_avatar: dict | None = None) -> torch.Tensor:
        tracked = tracked_avatar or self._require_tracked_avatar()
        shape = torch.as_tensor(tracked["shapecode"], dtype=torch.float32)
        if tuple(shape.shape) != (300,):
            raise ValueError(f"Invalid GAGAvatar shapecode shape: {tuple(shape.shape)}")
        return shape[None]

    @torch.no_grad()
    def build_forward_batch(self, motion_code: torch.Tensor) -> dict:
        tracked = self._require_tracked_avatar()
        if motion_code.dim() != 2:
            raise ValueError(f"motion_code must be (N, 106), got {tuple(motion_code.shape)}")
        motion_code = motion_code.to(self.device)
        if self._feature_batch is None:
            feature_batch = {}
            feature_batch["f_image"] = torchvision.transforms.functional.resize(
                tracked["image"], (518, 518), antialias=True
            )[None].to(self.device)
            feature_batch["f_planes"] = build_points_planes(
                self.config.point_plane_size,
                tracked["transform_matrix"],
            )
            feature_batch["f_planes"]["plane_points"] = feature_batch["f_planes"]["plane_points"][None].to(self.device)
            feature_batch["f_planes"]["plane_dirs"] = feature_batch["f_planes"]["plane_dirs"][None].to(self.device)
            feature_batch["t_image"] = torchvision.transforms.functional.resize(
                tracked["image"], (512, 512), antialias=True
            )[None].to(self.device)
            feature_batch["t_transform"] = tracked["transform_matrix"][None].to(self.device)
            self._feature_batch = feature_batch
            self._shape_code = tracked["shapecode"][None].to(self.device)

        feature_batch = self._expand_feature_batch(motion_code.shape[0])
        exp_code = motion_code[:, :100]
        pose_code = torch.cat([motion_code.new_zeros(motion_code.shape[0], 3), motion_code[:, 103:]], dim=-1)
        t_points = self.flame_model(
            shape_params=self._shape_code.expand(motion_code.shape[0], -1),
            pose_params=pose_code,
            expression_params=exp_code,
            eye_pose_params=pose_code.new_zeros(motion_code.shape[0], 6),
        ).float()
        t_points[:, FOREHEAD_INDICES] = self._smooth_upper_points(
            t_points[:, FOREHEAD_INDICES]
        )
        feature_batch["t_points"] = t_points
        feature_batch["t_transform"][:, :3, :3] = transform_emoca_to_p3d(motion_code[:, 100:103])[:, :3, :3]
        return feature_batch

    def _expand_feature_batch(self, batch_size: int) -> dict:
        if self._feature_batch is None:
            raise RuntimeError("Feature batch is not initialized.")
        return {
            key: self._expand_feature_value(value, batch_size)
            for key, value in self._feature_batch.items()
        }

    def _expand_feature_value(self, value, batch_size: int):
        if isinstance(value, dict):
            return {
                key: self._expand_feature_value(item, batch_size)
                for key, item in value.items()
            }
        if not torch.is_tensor(value):
            return deepcopy(value)
        if value.shape[0] == batch_size:
            return value.clone()
        if value.shape[0] != 1:
            raise ValueError(
                f"Cannot expand cached feature from batch {value.shape[0]} to {batch_size}."
            )
        return value.expand(batch_size, *value.shape[1:]).contiguous()

    def _smooth_upper_points(self, current_points: torch.Tensor) -> torch.Tensor:
        smoothed = []
        for frame_points in current_points:
            frame_points = frame_points[None]
            if self._upper_points is None:
                self._upper_points = frame_points
            else:
                self._upper_points = 0.98 * self._upper_points + 0.02 * frame_points
            smoothed.append(self._upper_points[0])
        return torch.stack(smoothed, dim=0)

    @torch.no_grad()
    def forward_gaussians(self, motion_code: torch.Tensor) -> dict:
        return self.model.forward_gaussians(self.build_forward_batch(motion_code))

    @torch.no_grad()
    def forward_gaussians_for_batch(self, batch: dict) -> dict:
        return self.model.forward_gaussians(batch)

    @torch.no_grad()
    def rasterize_gaussians(self, gs_params: dict, cam_matrix: torch.Tensor) -> torch.Tensor:
        return render_gaussian(
            gs_params=gs_params,
            cam_matrix=cam_matrix,
            cam_params=self.model.cam_params,
        )["images"]

    @torch.no_grad()
    def render_rgb_batch(self, batch: dict) -> torch.Tensor:
        if self.config.autocast_dtype is None:
            return self.model.forward_expression(batch)["sr_gen_image"].clamp(0, 1)
        # Mixed-precision path: same stages as model.forward_expression, but
        # decomposed so the fp32-only Gaussian rasterizer sits outside the
        # autocast region.
        dtype = getattr(torch, self.config.autocast_dtype)
        with torch.autocast(self.device.type, dtype=dtype):
            gs_params = self.model.forward_gaussians(batch)
        gs_params = {
            key: value.float() if torch.is_tensor(value) else value
            for key, value in gs_params.items()
        }
        gen_images = render_gaussian(
            gs_params=gs_params,
            cam_matrix=batch["t_transform"].float(),
            cam_params=self.model.cam_params,
        )["images"]
        with torch.autocast(self.device.type, dtype=dtype):
            sr_gen_images = self.model.upsampler(gen_images)
        return sr_gen_images.float().clamp(0, 1)

    @torch.no_grad()
    def render_rgb_frame(self, motion_frame: torch.Tensor) -> torch.Tensor:
        batch = self.build_forward_batch(motion_frame[None] if motion_frame.dim() == 1 else motion_frame)
        return self.render_rgb_batch(batch).cpu()[0]

    def _require_tracked_avatar(self) -> dict:
        if self._tracked_avatar is None:
            raise RuntimeError("Call set_avatar_id() or set_tracked_avatar() before rendering.")
        return self._tracked_avatar

    @staticmethod
    def _load_tracked_avatars(path: Path | None) -> dict:
        if path is None:
            return {}
        if not path.exists():
            raise FileNotFoundError(path)
        tracked = torch.load(path, map_location="cpu", weights_only=False)
        if "avatar" in tracked and len(tracked) == 1:
            return {"avatar": tracked["avatar"]}
        return tracked


def select_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def check_gaussian_rasterizer_available():
    if importlib.util.find_spec("diff_gaussian_rasterization_32d") is None:
        raise RuntimeError("GAGAvatar Gaussian rendering requires diff_gaussian_rasterization_32d.")


def build_points_planes(plane_size: int, transforms: torch.Tensor) -> dict:
    x, y = torch.meshgrid(
        torch.linspace(1, -1, plane_size, dtype=torch.float32),
        torch.linspace(1, -1, plane_size, dtype=torch.float32),
        indexing="xy",
    )
    r = transforms[:3, :3]
    t = transforms[:3, 3:]
    cam_dirs = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    ray_dirs = torch.nn.functional.pad(torch.stack([x / 12.0, y / 12.0], dim=-1), (0, 1), value=1.0)
    cam_dirs = torch.matmul(r, cam_dirs.reshape(-1, 3)[:, :, None])[..., 0]
    ray_dirs = torch.matmul(r, ray_dirs.reshape(-1, 3)[:, :, None])[..., 0]
    origins = (-torch.matmul(r, t)[..., 0]).broadcast_to(ray_dirs.shape).squeeze()
    distance = ((origins[0] * cam_dirs[0]).sum()).abs()
    plane_points = origins + distance * ray_dirs
    return {"plane_points": plane_points, "plane_dirs": cam_dirs[0]}


def transform_emoca_to_p3d(emoca_base_rotation: torch.Tensor) -> torch.Tensor:
    device = emoca_base_rotation.device
    batch_size = emoca_base_rotation.shape[0]
    initial_trans = torch.tensor([[0, 0, 5000.0 / 512]], device=device)
    rotation = emoca_base_rotation.clone()
    rotation[:, [0, 2]] *= -1
    emoca_base_matrix = axis_angle_to_matrix(rotation)
    emoca_base_matrix = torch.matmul(
        emoca_base_matrix,
        torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=device).float(),
    )
    emoca_base_matrix = emoca_base_matrix.inverse()
    return torch.cat([emoca_base_matrix, initial_trans.reshape(1, -1, 1).repeat(batch_size, 1, 1)], dim=-1)


FOREHEAD_INDICES = [
    2168, 2165, 3068, 2199, 2196, 3720, 2091, 2088, 3524, 625, 628, 3871, 705, 708, 2030, 667, 670,
    3708, 3706, 3729, 3721, 3773, 3789, 3735, 3732, 3786, 3876, 3878, 3913, 3899, 3872, 3874, 3864, 3865,
    3158, 3157, 336, 335, 3153, 3705, 2177, 2176, 3540, 671, 672, 3863, 2134, 16, 17, 2138, 2139,
    2567, 2566, 337, 338, 3154, 3712, 2178, 2179, 3495, 674, 673, 3868, 2135, 27, 18, 1429, 1430,
]
