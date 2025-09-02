"""Blue noise dithering package for converting images to palettes with various color distance methods."""

__version__ = "1.0.0"
__author__ = "ccicnce113424"

from .core import BlueNoiseDitherer
from .palette import PaletteLoader
from .color_distance import ColorDistanceCalculator

__all__ = ["BlueNoiseDitherer", "PaletteLoader", "ColorDistanceCalculator"]