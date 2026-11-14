'''
File:   __init__.py
Brief:  Core processing engines for the LukutarRetro pack.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Engines land per phase; each module re-exports its public names here so
node modules import from the package root (mirrors the sibling packs).
'''

from __future__ import annotations

from .dither import (
    DIFFUSION_KERNELS,
    DIFFUSION_METHODS,
    DITHER_ATKINSON,
    DITHER_BAYER2,
    DITHER_BAYER4,
    DITHER_BAYER8,
    DITHER_FLOYDSTEINBERG,
    DITHER_METHODS,
    DITHER_NONE,
    ORDERED_METHODS,
    bayer_matrix,
    dither_frame,
)
from .palettes import (
    PALETTE_NAMES,
    get_palette,
    parse_hex_palette,
)
from .pixelize import (
    METHOD_CENTER,
    METHOD_MEAN,
    METHOD_MEDIAN,
    METHOD_MODE,
    SAMPLE_METHODS,
    Pixelizer,
)
from .quantize import (
    AMPLITUDE_GRID_LEVELS,
    LUT_BIN_SHIFT,
    MATCH_SPACE,
    PaletteQuantizer,
    make_palette_lut,
)

__all__ = [
    # dither
    "DITHER_ATKINSON",
    "DITHER_BAYER2",
    "DITHER_BAYER4",
    "DITHER_BAYER8",
    "DITHER_FLOYDSTEINBERG",
    "DITHER_METHODS",
    "DITHER_NONE",
    "DIFFUSION_KERNELS",
    "DIFFUSION_METHODS",
    "ORDERED_METHODS",
    "bayer_matrix",
    "dither_frame",
    # palettes
    "PALETTE_NAMES",
    "get_palette",
    "parse_hex_palette",
    # pixelize
    "METHOD_CENTER",
    "METHOD_MEAN",
    "METHOD_MEDIAN",
    "METHOD_MODE",
    "SAMPLE_METHODS",
    "Pixelizer",
    # quantize
    "AMPLITUDE_GRID_LEVELS",
    "LUT_BIN_SHIFT",
    "MATCH_SPACE",
    "PaletteQuantizer",
    "make_palette_lut",
]
