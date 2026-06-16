#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

"""Public GAGAvatar runtime API for external renderers.

The original repository is demo-oriented: inference scripts own checkpoint
loading, tracked-avatar loading, batch construction, rendering, and file
writing. This module factors those steps into a small importable runtime so a
standalone renderer server can depend on GAGAvatar as a library.
"""

from __future__ import annotations

import gzip
import importlib.util
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import av
import torch
import torchvision
from pytorch3d.transforms import axis_angle_to_matrix

from core.libs.flame_model import FLAMEModel
from core.libs.utils_renderer import render_gaussian
from core.models import GAGAvatar


GAGAVATAR_HEAD_GAUSSIAN_COUNT = 5023
UPSAMPLER_PREVIEW_FRAME_COUNT = 32
UPSAMPLER_QUANTIZATION_LEVELS = 32
UPSAMPLER_INPUT_DTYPE = "uint8-linear"


@dataclass(frozen=True)
class GAGAvatarRuntimeConfig:
    model_path: Path | str
    tracked_path: Path | str | None = None
    flame_model_path: Path | str | None = None
    device: str = "auto"
    point_plane_size: int = 296
    flame_scale: float = 5.0

    def resolved_model_path(self) -> Path:
        return Path(self.model_path).expanduser().resolve()

    def resolved_tracked_path(self) -> Path | None:
        if self.tracked_path is None:
            return None
        return Path(self.tracked_path).expanduser().resolve()

    def resolved_flame_model_path(self) -> Path | None:
        if self.flame_model_path is None:
            return None
        return Path(self.flame_model_path).expanduser().resolve()


@dataclass
class GAGAvatarGaussianExport:
    metadata: dict
    output_dir: Path


class GAGAvatarRuntime:
    """Reusable GAGAvatar runtime for ARTalk motion outputs."""

    def __init__(self, config: GAGAvatarRuntimeConfig):
        self.config = config
        self.device = select_device(config.device)
        self.model = GAGAvatar().to(self.device).eval()
        checkpoint = torch.load(config.resolved_model_path(), map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        state = {key: value for key, value in state.items() if "percep_loss" not in key}
        self.model.load_state_dict(state, strict=False)
        self.flame_model = FLAMEModel(
            n_shape=300,
            n_exp=100,
            scale=config.flame_scale,
            no_lmks=True,
            model_path=config.resolved_flame_model_path(),
        ).to(self.device)
        self.tracked_avatars = self._load_tracked_avatars(config.resolved_tracked_path())
        self._tracked_avatar = None
        self._tracked_name = None
        self._feature_batch = None
        self._shape_code = None
        self._upper_points = None

    def available_avatar_ids(self) -> list[str]:
        return sorted(self.tracked_avatars)

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

        feature_batch = deepcopy(self._feature_batch)
        exp_code = motion_code[:, :100]
        pose_code = torch.cat([motion_code.new_zeros(motion_code.shape[0], 3), motion_code[:, 103:]], dim=-1)
        t_points = self.flame_model(
            shape_params=self._shape_code.expand(motion_code.shape[0], -1),
            pose_params=pose_code,
            expression_params=exp_code,
            eye_pose_params=pose_code.new_zeros(motion_code.shape[0], 6),
        ).float()
        if self._upper_points is None:
            self._upper_points = t_points[:, FOREHEAD_INDICES]
        else:
            current_points = t_points[:, FOREHEAD_INDICES]
            self._upper_points = 0.98 * self._upper_points + 0.02 * current_points
            t_points[:, FOREHEAD_INDICES] = self._upper_points
        feature_batch["t_points"] = t_points
        feature_batch["t_transform"][:, :3, :3] = transform_emoca_to_p3d(motion_code[:, 100:103])[:, :3, :3]
        return feature_batch

    @torch.no_grad()
    def forward_gaussians(self, motion_code: torch.Tensor) -> dict:
        return self.model.forward_gaussians(self.build_forward_batch(motion_code))

    @torch.no_grad()
    def render_rgb_frame(self, motion_frame: torch.Tensor) -> torch.Tensor:
        batch = self.build_forward_batch(motion_frame[None] if motion_frame.dim() == 1 else motion_frame)
        return self.model.forward_expression(batch)["sr_gen_image"].clamp(0, 1).cpu()[0]

    @torch.no_grad()
    def export_gaussian_artifacts(
        self,
        motions: torch.Tensor,
        output_dir: str | Path,
        *,
        write_reference_video: bool = True,
        upsampler_preview_frame_count: int | str | None = None,
        upsampler_preview_stride: int | None = None,
        quantization_levels: int = UPSAMPLER_QUANTIZATION_LEVELS,
        fps: int = 25,
    ) -> GAGAvatarGaussianExport:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        motions = motions.to(self.device)
        head_frames = []
        transform_frames = []
        reference_frames = []
        first_batch = None
        first_gs_params = None
        upsampler_input_files = []
        upsampler_input_shape = None
        preview_indices = upsampler_preview_frame_indices(
            int(motions.shape[0]),
            upsampler_preview_frame_count,
            upsampler_preview_stride,
        )
        preview_index_set = set(preview_indices)

        for frame_index, motion in enumerate(motions):
            batch = self.build_forward_batch(motion[None])
            if first_batch is None:
                first_batch = batch
            head_frames.append(batch["t_points"][0].detach().float().cpu())
            transform_frames.append(batch["t_transform"][0].detach().float().cpu())
            if write_reference_video:
                reference_frames.append(self.model.forward_expression(batch)["sr_gen_image"].cpu()[0])
            if frame_index in preview_index_set:
                gs_params_frame = self.model.forward_gaussians(batch)
                if first_gs_params is None:
                    first_gs_params = {key: value.clone() for key, value in gs_params_frame.items()}
                upsampler_input = render_gaussian(
                    gs_params=gs_params_frame,
                    cam_matrix=batch["t_transform"],
                    cam_params=self.model.cam_params,
                )["images"][0].detach().to(torch.float16).cpu()
                upsampler_input_shape = list(upsampler_input.shape)
                file_name = f"gaussians.upsampler_input_{len(upsampler_input_files):03d}.u8.gz"
                write_quantized_upsampler_input(
                    upsampler_input,
                    output_dir / file_name,
                    quantization_levels=quantization_levels,
                )
                if not upsampler_input_files:
                    write_quantized_upsampler_input(
                        upsampler_input,
                        output_dir / "gaussians.upsampler_input_first.u8.gz",
                        quantization_levels=quantization_levels,
                    )
                upsampler_input_files.append(file_name)

        if first_batch is None:
            raise ValueError("Cannot export Gaussian artifacts for an empty motion sequence.")
        gs_params = first_gs_params if first_gs_params is not None else self.model.forward_gaussians(first_batch)
        head_positions = torch.stack(head_frames)
        transforms = torch.stack(transform_frames)
        snapshot = {
            "xyz": gs_params["xyz"][0].detach().float().cpu(),
            "colors": gs_params["colors"][0].detach().float().cpu(),
            "opacities": gs_params["opacities"][0].detach().float().cpu(),
            "scales": gs_params["scales"][0].detach().float().cpu(),
            "rotations": gs_params["rotations"][0].detach().float().cpu(),
        }
        for name, tensor in snapshot.items():
            tensor.numpy().astype("float32", copy=False).tofile(output_dir / f"gaussians.{name}.f32")
        head_positions.numpy().astype("float32", copy=False).tofile(output_dir / "gaussians.head_xyz.f32")
        transforms.numpy().astype("float32", copy=False).tofile(output_dir / "gaussians.transforms.f32")
        if write_reference_video and reference_frames:
            frames = (torch.stack(reference_frames) * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
            write_rgb_video(frames, output_dir / "gaussians.reference.mp4", fps=fps)

        metadata = {
            "gaussianCount": int(snapshot["xyz"].shape[0]),
            "gaussianFormat": "gagavatar-first-frame-f32-v1",
            "gaussianColorChannels": int(snapshot["colors"].shape[1]),
            "gaussianUpsamplerInput": {
                "dtype": UPSAMPLER_INPUT_DTYPE,
                "shape": upsampler_input_shape,
                "frameCount": len(upsampler_input_files),
                "frameIndices": preview_indices,
                "quantizationLevels": quantization_levels,
            },
            "gaussianHeadCount": int(head_positions.shape[1]),
            "gaussianHeadFrameCount": int(head_positions.shape[0]),
            "gaussianTransformFrameCount": int(transforms.shape[0]),
            "gaussianUrls": {
                "xyz": "gaussians.xyz.f32",
                "headXyz": "gaussians.head_xyz.f32",
                "transforms": "gaussians.transforms.f32",
                "referenceVideo": "gaussians.reference.mp4" if write_reference_video else None,
                "upsamplerInputFirst": "gaussians.upsampler_input_first.u8.gz",
                "upsamplerInputFrames": upsampler_input_files,
                "colors": "gaussians.colors.f32",
                "opacities": "gaussians.opacities.f32",
                "scales": "gaussians.scales.f32",
                "rotations": "gaussians.rotations.f32",
            },
            "gaussianCamera": {
                "focalX": float(self.model.cam_params["focal_x"]),
                "focalY": float(self.model.cam_params["focal_y"]),
                "size": list(self.model.cam_params["size"]),
            },
        }
        return GAGAvatarGaussianExport(metadata=metadata, output_dir=output_dir)

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


def sample_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0 or max_frames <= 0:
        return []
    if frame_count <= max_frames:
        return list(range(frame_count))
    if max_frames == 1:
        return [0]
    return sorted({round(index * (frame_count - 1) / (max_frames - 1)) for index in range(max_frames)})


def upsampler_preview_frame_indices(
    frame_count: int,
    max_frames: int | str | None = None,
    stride: int | None = None,
) -> list[int]:
    if frame_count <= 0:
        return []
    if stride is None:
        raw_stride = os.environ.get("GAGAVATAR_UPSAMPLER_PREVIEW_STRIDE")
        stride = int(raw_stride) if raw_stride and raw_stride.isdigit() else None
    if stride is not None:
        return list(range(0, frame_count, max(1, stride)))
    if max_frames is None:
        max_frames = os.environ.get("GAGAVATAR_UPSAMPLER_PREVIEW_FRAMES") or UPSAMPLER_PREVIEW_FRAME_COUNT
    if isinstance(max_frames, str):
        if max_frames.lower() == "all":
            return list(range(frame_count))
        try:
            max_frames = int(max_frames)
        except ValueError:
            max_frames = UPSAMPLER_PREVIEW_FRAME_COUNT
    return sample_frame_indices(frame_count, max(1, int(max_frames)))


def write_quantized_upsampler_input(
    tensor: torch.Tensor,
    output_path: str | Path,
    *,
    quantization_levels: int = UPSAMPLER_QUANTIZATION_LEVELS,
):
    levels = min(max(int(quantization_levels), 2), 256)
    channels = int(tensor.shape[0])
    values = tensor.float().flatten(1)
    mins = values.min(dim=1).values
    maxs = values.max(dim=1).values
    scales = ((maxs - mins) / float(levels - 1)).clamp_min(1e-8)
    quantized = torch.clamp(
        torch.round((tensor.float() - mins[:, None, None]) / scales[:, None, None]),
        0,
        levels - 1,
    ).to(torch.uint8)
    params = torch.stack([mins, scales], dim=1).cpu().numpy().astype("float32", copy=False)
    with gzip.open(output_path, "wb", compresslevel=1) as f:
        f.write(quantized.cpu().numpy().tobytes(order="C"))
        f.write(params.tobytes(order="C"))
    if params.shape != (channels, 2):
        raise ValueError("Invalid upsampler quantization parameters.")


def write_rgb_video(frames: torch.Tensor, output_path: str | Path, *, fps: int):
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T, H, W, 3), got {tuple(frames.shape)}")
    frames_np = frames.detach().cpu().numpy()
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=int(fps))
    stream.width = int(frames_np.shape[2])
    stream.height = int(frames_np.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    try:
        for frame in frames_np:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


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
