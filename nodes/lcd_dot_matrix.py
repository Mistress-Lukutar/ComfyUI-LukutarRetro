'''
File:   lcd_dot_matrix.py
Brief:  LCD dot-matrix overlay node — simulates subpixel aperture grille.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.analog.lcd_dot_matrix engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.analog import DOT_SHAPES, SHAPE_SQUARE, lcd_dot_matrix
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class LcdDotMatrixNode:
    '''Overlay a dot-matrix LCD pattern on an image.

    Each cell_size-pixel block is treated as a "dot". Pixels inside the
    active region keep full brightness; pixels in the gap are darkened.

    The overlay is computed at the current resolution — pair with Pixelize
    by setting cell_size equal to Pixelize's pixel_size for matching pitch.
    '''

    CATEGORY = "Lukutar/Retro/Analog"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_dot_matrix"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to apply dot-matrix overlay to"},
                ),
                "cell_size": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Side length of each dot cell in pixels",
                    },
                ),
                "gap_ratio": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.05,
                        "tooltip": (
                            "Fractional gap around the active region "
                            "in [0, 0.5]"
                        ),
                    },
                ),
                "grid_dark": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Darkening factor for gap pixels in [0, 1]",
                    },
                ),
                "shape": (
                    list(DOT_SHAPES),
                    {
                        "default": SHAPE_SQUARE,
                        "tooltip": "Dot shape — square or round active region",
                    },
                ),
                "show_preview": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Show the processed result as an in-node preview "
                            "(off = no preview is generated at all)"
                        ),
                    },
                ),
            },
        }

    def apply_dot_matrix(
        self,
        image: torch.Tensor,
        cell_size: int = 8,
        gap_ratio: float = 0.15,
        grid_dark: float = 0.6,
        shape: str = SHAPE_SQUARE,
        show_preview: bool = True,
    ) -> dict:
        '''Apply a dot-matrix LCD overlay to a batch of frames.

        Each cell_size-pixel block is treated as a "dot". Pixels inside the
        active region keep full brightness; pixels in the gap are darkened by
        (1 - grid_dark). The overlay is computed at the current resolution —
        pair with Pixelize by setting cell_size equal to Pixelize's pixel_size.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            cell_size: Side length of each dot cell in pixels (>= 1).
            gap_ratio: Fractional gap around the active region in [0, 0.5].
            grid_dark: Darkening factor for gap pixels in [0, 1].
                0 = no darkening; 1 = fully dark gaps.
            shape: Dot shape — one of DOT_SHAPES.
            show_preview: Whether to generate in-node preview images.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid parameters.
        '''
        frames = tensor_to_frames(image)
        out = [
            lcd_dot_matrix(f, cell_size, gap_ratio, grid_dark, shape)
            for f in frames
        ]
        logger.info(
            "LCD Dot Matrix: %d frame(s) processed (cell_size=%d, "
            "gap_ratio=%.2f, grid_dark=%.2f, shape=%s)",
            len(out),
            cell_size,
            gap_ratio,
            grid_dark,
            shape,
        )
        return build_output_payload(out, show_preview=show_preview)
