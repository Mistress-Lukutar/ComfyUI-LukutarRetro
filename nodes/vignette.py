'''
File:   vignette.py
Brief:  Vignette node — radial corner-darkening effect.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.crt.vignette engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.crt import vignette
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class VignetteNode:
    '''Apply a radial vignette (corner-darkening) effect to an image.

    A smooth radial falloff darkens the edges of the image, with the
    transition controlled by radius and softness parameters.
    '''

    CATEGORY = "Lukutar/Retro/CRT"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_vignette"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to apply vignette to"},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Maximum darkening factor in [0, 1]",
                    },
                ),
                "radius": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Normalised distance at which darkening begins",
                    },
                ),
                "softness": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Controls the steepness of the transition",
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

    def apply_vignette(
        self,
        image: torch.Tensor,
        strength: float = 0.4,
        radius: float = 0.8,
        softness: float = 0.5,
        show_preview: bool = True,
    ) -> dict:
        '''Apply a vignette effect to a batch of frames.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            strength: Maximum darkening factor in [0, 1].
            radius: Normalised distance at which darkening begins ([0, 1]).
            softness: Controls the steepness of the transition in [0, 1].
            show_preview: Whether to generate in-node preview images.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid parameters.
        '''
        frames = tensor_to_frames(image)
        out = [
            vignette(f, strength, radius, softness)
            for f in frames
        ]
        logger.info(
            "Vignette: %d frame(s) processed (strength=%.2f, radius=%.2f, "
            "softness=%.2f)",
            len(out),
            strength,
            radius,
            softness,
        )
        return build_output_payload(out, show_preview=show_preview)
