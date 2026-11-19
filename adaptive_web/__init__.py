"""Adaptive Python Learning Showcase core package (`3/adaptive_web/`)."""
from pathlib import Path

__version__ = "1.0.0"

PACKAGE_DIR = Path(__file__).resolve().parent
CODE_3_ROOT = PACKAGE_DIR.parent
REPO_ROOT = CODE_3_ROOT.parent

from .config import Settings  # noqa: E402

__all__ = [
    "__version__",
    "PACKAGE_DIR",
    "CODE_3_ROOT",
    "REPO_ROOT",
    "Settings",
]
