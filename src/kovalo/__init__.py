"""Kovalo Decision Copilot."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kovalo-copilot")
except PackageNotFoundError:  # uruchomienie bez instalacji pakietu
    __version__ = "0.0.0"

__all__ = ["__version__"]
