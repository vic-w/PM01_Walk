"""Helpers for resolving paths to the PM01 robot assets."""

from importlib import resources
from pathlib import Path


def get_pm01_asset_dir() -> Path:
    """Return the root directory that contains the PM01 robot assets."""

    return Path(resources.files(__name__))


def get_pm01_usd_path(filename: str = "pm01.usd") -> Path:
    """Return the absolute path to a USD asset belonging to the PM01 robot."""

    return get_pm01_asset_dir() / "usd" / filename


__all__ = ["get_pm01_asset_dir", "get_pm01_usd_path"]
