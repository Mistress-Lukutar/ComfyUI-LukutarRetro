'''
File:   subpixel_mask.py
Brief:  Subpixel phosphor-mask node — fake CRT / LCD dot-grid modulation.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.crt.subpixel_mask engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.crt import (
    MASK_APERTURE_GRILLE,
    MASK_TYPES,
    subpixel_mask,
)
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class SubpixelMaskNode:
    '''Apply a fake CRT phosphor-mask (or LCD dot-grid) modulation.

    The pattern has a period of 3*pitch horizontally. Each channel only
    transmits light through its own stripe(s), and the result is multiplied
    by MASK_BRIGHTNESS_COMPENSATION to restore overall brightness.
    '''

    CATEGORY = "Lukutar/Retro/CRT"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_mask"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to apply phosphor mask to"},
                ),
                "mask": (
                    list(MASK_TYPES),
                    {
                        "default": MASK_APERTURE_GRILLE,
                        "tooltip": (
                            "Phosphor-mask type: aperture grille, "
                            "shadow mask, or slot mask"
                        ),
                    },
                ),
                "pitch": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Stripe width; the RGB triad period is 3*pitch",
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

    def apply_mask(
        self,
        image: torch.Tensor,
        mask: str = MASK_APERTURE_GRILLE,
        pitch: int = 1,
        show_preview: bool = True,
    ) -> dict:
        '''Apply a phosphor-mask pattern to a batch of frames.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            mask: Phosphor-mask type — one of MASK_TYPES.
            pitch: Stripe width in pixels (>= 1).
            show_preview: Whether to generate in-node preview images.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid parameters.
        '''
        frames = tensor_to_frames(image)
        out = [subpixel_mask(f, mask, pitch) for f in frames]
        logger.info(
            "Subpixel mask: %d frame(s) processed (mask=%s, pitch=%d)",
            len(out),
            mask,
            pitch,
        )
        return build_output_payload(out, show_preview=show_preview)
