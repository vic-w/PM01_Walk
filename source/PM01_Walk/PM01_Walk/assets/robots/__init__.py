"""Robot assets packaged with the PM01_Walk extension."""

from importlib import resources
from pathlib import Path


def get_robot_assets_root() -> Path:
    """Return the root path containing robot asset folders."""

    return Path(resources.files(__name__))


__all__ = ["get_robot_assets_root"]
