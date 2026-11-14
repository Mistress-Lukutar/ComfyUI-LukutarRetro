'''
File:   dither.py
Brief:  Dithering node — bit-depth reduction with error diffusion or ordered dither.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping core.dither.dither_frame for retro-style colour quantisation.
'''

from __future__ import annotations

import logging

import numpy as np
import torch

from ..core.dither import (
    DITHER_ATKINSON,
    DITHER_BAYER2,
    DITHER_BAYER4,
    DITHER_BAYER8,
    DITHER_FLOYDSTEINBERG,
    dither_frame,
)
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)

#: Dither methods available in this node (excludes "none").
DITHER_NODE_METHODS: tuple[str, ...] = (
    DITHER_BAYER2,
    DITHER_BAYER4,
    DITHER_BAYER8,
    DITHER_FLOYDSTEINBERG,
    DITHER_ATKINSON,
)


class DitherNode:
    '''Apply ordered or error-diffusion dithering to reduce colour depth.

    Supports Bayer-ordered dither (bayer2/4/8) for fast vectorised output,
    and Floyd-Steinberg / Atkinson diffusion for smoother gradients.
    Mono mode converts to grayscale luma first, then dithers all channels.
    '''

    CATEGORY = "Lukutar/Retro/Pixel"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "dither"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to dither"},
                ),
                "levels": (
                    "INT",
                    {
                        "default": 4,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Quantisation levels per channel (2-32)",
                    },
                ),
                "mode": (
                    ("color", "mono"),
                    {
                        "default": "color",
                        "tooltip": (
                            "color: dither RGB; mono: grayscale luma first, "
                            "then dither"
                        ),
                    },
                ),
                "dither_method": (
                    list(DITHER_NODE_METHODS),
                    {
                        "default": DITHER_BAYER4,
                        "tooltip": (
                            "Dither algorithm: Bayer-ordered or "
                            "error-diffusion"
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Dither amplitude (0 = off, 1 = full)",
                    },
                ),
                "serpentine": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Alternate scanline direction for "
                            "error-diffusion methods"
                        ),
                    },
                ),
                "show_preview": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Show the processed result as an in-node preview "
                            "(frontend toggle; the backend accepts and ignores it)"
                        ),
                    },
                ),
            },
        }

    def dither(
        self,
        image: torch.Tensor,
        levels: int = 4,
        mode: str = "color",
        dither_method: str = DITHER_BAYER4,
        strength: float = 1.0,
        serpentine: bool = True,
        show_preview: bool = True,
    ) -> dict:
        '''Dither a batch of frames to the given colour depth.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            levels: Quantisation levels per channel (>= 2).
            mode: "color" for RGB dither; "mono" for grayscale-first.
            dither_method: One of DITHER_NODE_METHODS (never "none").
            strength: Dither amplitude in [0, 1].
            serpentine: Alternate scanline direction for diffusion methods.
            show_preview: Whether to save preview PNGs (ignored by backend).

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid parameters.
        '''
        frames = tensor_to_frames(image)

        out: list[np.ndarray] = []
        for frame in frames:
            work = frame.copy()
            if mode == "mono":
                # Convert to grayscale luma (Rec. 601), broadcast back to 3 channels
                r = frame[:, :, 0]
                g = frame[:, :, 1]
                b = frame[:, :, 2]
                luma = 0.299 * r + 0.587 * g + 0.114 * b
                work = np.stack([luma, luma, luma], axis=-1)

            result = dither_frame(
                work, levels, dither_method,
                strength=strength, serpentine=serpentine,
            )
            out.append(result)

        logger.info(
            "Dither: %d frame(s) processed "
            "(levels=%d, mode=%s, method=%s, strength=%.2f)",
            len(out),
            levels,
            mode,
            dither_method,
            strength,
        )
        return build_output_payload(out)
