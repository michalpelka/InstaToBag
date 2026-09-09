"""Convert Insta360 ``.insv`` captures into ROS 2 MCAP bags."""

from .convert import Options, Summary, convert

__all__ = ["Options", "Summary", "convert", "__version__"]
__version__ = "1.0.0"
