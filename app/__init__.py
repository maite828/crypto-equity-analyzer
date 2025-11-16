"""Paquete principal para análisis configurable de señales y precios."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cryptopredicction")
except PackageNotFoundError:  # pragma: no cover - no distribución instalada
    __version__ = "0.0.0"

__all__ = ["__version__"]
