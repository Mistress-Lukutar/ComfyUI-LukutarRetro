'''
File:   __init__.py
Brief:  Aggregation of node classes for the LukutarRetro pack.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node classes are imported from their modules; mappings are built and
re-exported so the root package __init__ can forward them to ComfyUI.
'''

from __future__ import annotations

from .aberration import ChromaticAberrationNode
from .bloom import BloomNode
from .color_depth import ColorDepthNode
from .curvature import CurvatureNode
from .dither import DitherNode
from .lcd_dot_matrix import LcdDotMatrixNode
from .pixelize import PixelizeNode
from .quantize import PaletteQuantizeNode
from .scanlines import ScanlinesNode
from .subpixel_mask import SubpixelMaskNode
from .vhs import VhsNode
from .vignette import VignetteNode

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "RetroPixelize": PixelizeNode,
    "RetroPaletteQuantize": PaletteQuantizeNode,
    "RetroColorDepth": ColorDepthNode,
    "RetroDither": DitherNode,
    "RetroScanlines": ScanlinesNode,
    "RetroSubpixelMask": SubpixelMaskNode,
    "RetroBloom": BloomNode,
    "RetroChromaticAberration": ChromaticAberrationNode,
    "RetroCurvature": CurvatureNode,
    "RetroVignette": VignetteNode,
    "RetroVHS": VhsNode,
    "RetroLCDDotMatrix": LcdDotMatrixNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "RetroPixelize": "Pixelize",
    "RetroPaletteQuantize": "Palette Quantize",
    "RetroColorDepth": "Color Depth",
    "RetroDither": "Dither (8-bit)",
    "RetroScanlines": "Scanlines",
    "RetroSubpixelMask": "Subpixel Mask",
    "RetroBloom": "Bloom (Glow)",
    "RetroChromaticAberration": "Chromatic Aberration",
    "RetroCurvature": "Screen Curvature",
    "RetroVignette": "Vignette",
    "RetroVHS": "VHS (Analog)",
    "RetroLCDDotMatrix": "LCD Dot Matrix",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PixelizeNode",
    "PaletteQuantizeNode",
    "ColorDepthNode",
    "DitherNode",
    "ScanlinesNode",
    "SubpixelMaskNode",
    "BloomNode",
    "ChromaticAberrationNode",
    "CurvatureNode",
    "VignetteNode",
    "VhsNode",
    "LcdDotMatrixNode",
]
