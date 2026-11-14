'''
File:   color_depth.py
Brief:  Color-depth (posterization) node — uniform quantisation per channel.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping core.dither.dither_frame with DITHER_NONE for undithered
uniform quantisation (posterization).
'''

from __future__ import annotations

import logging

import torch

from ..core.dither import DITHER_NONE, dither_frame
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class ColorDepthNode:
    '''Posterize an image by restricting per-channel colour levels.

    Two modes: "bits" quantises to 2^value levels (1..8 bits),
    "levels" sets the exact number of levels (2..64).
    The engine uses undithered uniform quantisation — identical to
    dither_frame with method "none".
    '''

    CATEGORY = "Lukutar/Retro/Pixel"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "posterize"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to posterise"},
                ),
                "mode": (
                    ("bits", "levels"),
                    {
                        "default": "bits",
                        "tooltip": (
                            "bits: quantise to 2^value levels; "
                            "levels: exact number of levels"
                        ),
                    },
                ),
                "value": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Bits (1-8) when mode=bits; "
                            "exact levels (2-64) when mode=levels"
                        ),
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

    def posterize(
        self,
        image: torch.Tensor,
        mode: str = "bits",
        value: int = 4,
        show_preview: bool = True,
    ) -> dict:
        '''Posterize a batch of frames to the given colour depth.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            mode: "bits" (2^value levels) or "levels" (exact value levels).
            value: Bit depth (1-8) when mode=bits; exact levels (2-64) when mode=levels.
            show_preview: Whether to generate in-node preview images.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid value for the selected mode.
        '''
        frames = tensor_to_frames(image)

        if mode == "bits":
            if not (1 <= value <= 8):
                raise ValueError(
                    f"Color depth bits mode: value must be 1-8, got {value}"
                )
            levels = 2 ** value
        else:  # mode == "levels"
            if not (2 <= value <= 64):
                raise ValueError(
                    f"Color depth levels mode: value must be 2-64, got {value}"
                )
            levels = value

        out = [dither_frame(f, levels, DITHER_NONE) for f in frames]
        logger.info(
            "Color depth (posterize): %d frame(s) processed "
            "(mode=%s, value=%d -> %d levels)",
            len(out),
            mode,
            value,
            levels,
        )
        return build_output_payload(out, show_preview=show_preview)
