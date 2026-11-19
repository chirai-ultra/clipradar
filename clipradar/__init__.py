"""ClipRadar: find the best moments in any video and export them as HD clips."""

__version__ = "0.1.0"

from .config import Options  # noqa: E402
from .pipeline import run  # noqa: E402

__all__ = ["Options", "run", "__version__"]
