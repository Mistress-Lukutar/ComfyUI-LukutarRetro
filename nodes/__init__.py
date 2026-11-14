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

from .color_depth import ColorDepthNode
from .dither import DitherNode
from .pixelize import PixelizeNode
from .quantize import PaletteQuantizeNode

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "RetroPixelize": PixelizeNode,
    "RetroPaletteQuantize": PaletteQuantizeNode,
    "RetroColorDepth": ColorDepthNode,
    "RetroDither": DitherNode,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "RetroPixelize": "Pixelize",
    "RetroPaletteQuantize": "Palette Quantize",
    "RetroColorDepth": "Color Depth",
    "RetroDither": "Dither (8-bit)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PixelizeNode",
    "PaletteQuantizeNode",
    "ColorDepthNode",
    "DitherNode",
]
