'''
File:   aberration.py
Brief:  Chromatic aberration node — RGB channel displacement.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.crt.chromatic_aberration engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.crt import (
    ABERRATION_MODES,
    ABERRATION_RADIAL,
    chromatic_aberration,
)
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class ChromaticAberrationNode:
    '''Displace RGB channels to simulate lens chromatic aberration.

    Red and blue channels are shifted relative to green (unchanged),
    producing colour fringes on edges. Linear mode shifts horizontally;
    radial mode displaces outward/inward from the image centre.
    '''

    CATEGORY = "Lukutar/Retro/CRT"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_aberration"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to apply chromatic aberration to"},
                ),
                "amount": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Displacement magnitude in pixels",
                    },
                ),
                "mode": (
                    list(ABERRATION_MODES),
                    {
                        "default": ABERRATION_RADIAL,
                        "tooltip": (
                            "radial: outward from centre; "
                            "linear: horizontal shift"
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

    def apply_aberration(
        self,
        image: torch.Tensor,
        amount: float = 2.0,
        mode: str = ABERRATION_RADIAL,
        show_preview: bool = True,
    ) -> dict:
        '''Apply chromatic aberration to a batch of frames.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            amount: Displacement magnitude in pixels ([0, 20]).
            mode: Displacement mode — one of ABERRATION_MODES.
            show_preview: Whether to generate in-node preview images.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid parameters.
        '''
        frames = tensor_to_frames(image)
        out = [
            chromatic_aberration(f, amount, mode)
            for f in frames
        ]
        logger.info(
            "Chromatic aberration: %d frame(s) processed "
            "(amount=%.1f, mode=%s)",
            len(out),
            amount,
            mode,
        )
        return build_output_payload(out, show_preview=show_preview)
