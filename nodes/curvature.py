'''
File:   curvature.py
Brief:  Screen curvature node — barrel distortion simulating CRT convexity.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.crt.curvature engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.crt import curvature
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class CurvatureNode:
    '''Apply barrel distortion to simulate a convex CRT screen.

    Pixels near the corners are pulled inward, simulating the physical
    curvature of an old CRT display.
    '''

    CATEGORY = "Lukutar/Retro/CRT"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_curvature"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to apply screen curvature to"},
                ),
                "amount": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Distortion strength; zero is identity",
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

    def apply_curvature(
        self,
        image: torch.Tensor,
        amount: float = 0.15,
        show_preview: bool = True,
    ) -> dict:
        '''Apply barrel distortion to a batch of frames.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            amount: Distortion strength in [0, 1]. Zero is identity.
            show_preview: Whether to generate in-node preview images.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid parameters.
        '''
        frames = tensor_to_frames(image)
        out = [curvature(f, amount) for f in frames]
        logger.info(
            "Curvature: %d frame(s) processed (amount=%.2f)",
            len(out),
            amount,
        )
        return build_output_payload(out, show_preview=show_preview)
