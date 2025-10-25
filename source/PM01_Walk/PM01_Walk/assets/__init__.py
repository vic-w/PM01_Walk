"""Utility helpers for packaged assets used by the PM01_Walk extension."""

from importlib import resources
from pathlib import Path


def get_assets_root() -> Path:
    """Return the root path to the packaged assets directory."""

    return Path(resources.files(__name__))


__all__ = ["get_assets_root"]
