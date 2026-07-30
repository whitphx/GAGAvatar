#!/usr/bin/env python

"""Asset resolution helpers for GAGAvatar runtime integrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import tomllib


class AssetConfigError(RuntimeError):
    """Raised when GAGAvatar assets cannot be resolved or validated."""


class AssetDownloadError(RuntimeError):
    """Raised when a GAGAvatar asset cannot be downloaded or verified."""


@dataclass(frozen=True)
class GAGAvatarAssets:
    root: Path
    model_path: Path
    tracked_path: Path | None = None
    flame_model_path: Path | None = None

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        flame_model_path: str | Path | None = None,
        tracked_path: str | Path | None = None,
    ) -> "GAGAvatarAssets":
        root_path = Path(root).expanduser().resolve()
        return cls(
            root=root_path,
            model_path=root_path / "GAGAvatar.pt",
            tracked_path=resolve_optional_path(tracked_path) or root_path / "tracked.pt",
            flame_model_path=resolve_optional_path(flame_model_path),
        )

    @classmethod
    def from_pyproject(cls, project_root: str | Path | None = None) -> "GAGAvatarAssets":
        pyproject = find_pyproject(Path(project_root) if project_root else Path.cwd())
        data = load_pyproject(pyproject)
        gagavatar_assets = data.get("tool", {}).get("gagavatar", {}).get("assets", {})

        root = gagavatar_assets.get("root")
        if not root:
            raise AssetConfigError(f"Missing [tool.gagavatar.assets].root in {pyproject}")
        root_path = resolve_config_path(root, pyproject.parent)
        flame_model = gagavatar_assets.get("flame_model")
        tracked = gagavatar_assets.get("tracked")
        return cls.from_root(
            root_path,
            flame_model_path=resolve_config_path(flame_model, pyproject.parent)
            if flame_model
            else None,
            tracked_path=resolve_config_path(tracked, pyproject.parent)
            if tracked
            else None,
        )

    @classmethod
    def resolve(
        cls,
        *,
        root: str | Path | None = None,
        model_path: str | Path | None = None,
        tracked_path: str | Path | None = None,
        flame_model_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> "GAGAvatarAssets":
        if root is not None:
            assets = cls.from_root(
                root,
                flame_model_path=flame_model_path,
                tracked_path=tracked_path,
            )
        elif os.environ.get("GAGAVATAR_ASSET_DIR"):
            assets = cls.from_root(
                os.environ["GAGAVATAR_ASSET_DIR"],
                flame_model_path=flame_model_path,
                tracked_path=tracked_path,
            )
        else:
            try:
                assets = cls.from_pyproject(project_root)
            except AssetConfigError:
                assets = cls.from_root("assets/GAGAvatar")

        return cls(
            root=assets.root,
            model_path=resolve_optional_path(model_path) or assets.model_path,
            tracked_path=resolve_optional_path(tracked_path) or assets.tracked_path,
            flame_model_path=resolve_optional_path(flame_model_path)
            or assets.flame_model_path,
        )

    def validate(self, require_tracked: bool = False) -> None:
        required = [self.model_path]
        if self.flame_model_path is not None:
            required.append(self.flame_model_path)
        if require_tracked and self.tracked_path is not None:
            required.append(self.tracked_path)
        missing = [path for path in required if not path.exists()]
        if missing:
            details = "\n".join(f"- {path}" for path in missing)
            raise AssetConfigError(f"Missing GAGAvatar asset files:\n{details}")


def find_pyproject(start: Path) -> Path:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for directory in [current, *current.parents]:
        pyproject = directory / "pyproject.toml"
        if pyproject.exists():
            return pyproject
    raise AssetConfigError(f"No pyproject.toml found from {start}")


def load_pyproject(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_config_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def load_asset_manifest() -> dict[str, Any]:
    return json.loads(files(__package__).joinpath("assets_manifest.json").read_text())


def iter_manifest_assets(*, include_optional: bool = False) -> list[dict[str, Any]]:
    assets = load_asset_manifest()["assets"]
    if include_optional:
        return assets
    return [asset for asset in assets if asset.get("required", True)]


def download_assets(
    root: str | Path,
    *,
    names: set[str] | None = None,
    include_optional: bool = False,
    force: bool = False,
    dry_run: bool = False,
    verify: bool = True,
) -> None:
    root_path = Path(root).expanduser().resolve()
    candidates = iter_manifest_assets(include_optional=True if names else include_optional)
    selected = [asset for asset in candidates if names is None or asset["name"] in names]
    if names:
        known = {asset["name"] for asset in iter_manifest_assets(include_optional=True)}
        unknown = names - known
        if unknown:
            raise AssetDownloadError(
                f"Unknown GAGAvatar asset names: {', '.join(sorted(unknown))}"
            )

    manual_assets = []
    for asset in selected:
        destination = root_path / asset["path"]
        if asset.get("manual"):
            manual_assets.append(asset)
            print(f"manual: {asset['name']} -> {destination}")
            continue
        url = asset_download_url(asset)
        if dry_run:
            print(f"download: {url} -> {destination}")
            continue
        if destination.exists() and not force:
            if verify and asset.get("sha256"):
                verify_asset(destination, asset)
            print(f"exists: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"download: {url} -> {destination}")
        download_file(url, temporary)
        if verify and asset.get("sha256"):
            verify_asset(temporary, asset)
        temporary.replace(destination)

    if manual_assets:
        print("\nManual assets were not downloaded:")
        for asset in manual_assets:
            print(f"- {asset['path']}: {asset.get('note', 'manual download required')}")


def asset_download_url(asset: dict[str, Any]) -> str:
    source = asset.get("source") or {}
    if source.get("type") != "huggingface":
        raise AssetDownloadError(f"Unsupported source for {asset['name']}: {source}")
    repo_id = source["repo_id"]
    revision = source.get("revision", "main")
    filename = source["filename"]
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}?download=true"


def download_file(url: str, destination: Path, timeout_s: float = 60.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def verify_asset(path: Path, asset: dict[str, Any]) -> None:
    expected_size = asset.get("size")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise AssetDownloadError(
            f"Invalid size for {path}: {path.stat().st_size}, expected {expected_size}"
        )
    expected_sha256 = asset.get("sha256")
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise AssetDownloadError(f"Invalid SHA-256 for {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gagavatar-assets")
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_parser = subcommands.add_parser("list", help="List GAGAvatar asset manifest entries.")
    list_parser.add_argument("--include-optional", action="store_true")

    download_parser = subcommands.add_parser("download", help="Download GAGAvatar assets.")
    download_parser.add_argument("--root", default="assets")
    download_parser.add_argument("--asset", action="append", dest="assets")
    download_parser.add_argument("--include-optional", action="store_true")
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.add_argument("--no-verify", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for asset in iter_manifest_assets(include_optional=args.include_optional):
                marker = "manual" if asset.get("manual") else "download"
                required = "required" if asset.get("required", True) else "optional"
                print(f"{asset['name']}: {asset['path']} ({marker}, {required})")
            return 0
        if args.command == "download":
            download_assets(
                args.root,
                names=set(args.assets) if args.assets else None,
                include_optional=args.include_optional,
                force=args.force,
                dry_run=args.dry_run,
                verify=not args.no_verify,
            )
            return 0
    except AssetDownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
