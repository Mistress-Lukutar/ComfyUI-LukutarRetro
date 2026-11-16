'''
File:   scanlines.py
Brief:  Scanlines node — CRT scanline darkening effect.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.crt.scanlines engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.crt import (
    PROFILE_HARD,
    SCANLINE_PROFILES,
    scanlines,
)
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class ScanlinesNode:
    '''Apply a CRT-style scanline darkening effect to an image.

    Every *pitch* rows the darkest *thickness* fraction is dimmed by
    *strength*. The *profile* controls whether the transition is abrupt
    (hard) or uses a raised-cosine ramp (soft).
    '''

    CATEGORY = "Lukutar/Retro/CRT"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_scanlines"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to apply scanlines to"},
                ),
                "pitch": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Rows per scanline period (>= 1)",
                    },
                ),
                "thickness": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Fraction of period that is darkened in [0, 1]",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Maximum dimming factor in [0, 1]",
                    },
                ),
                "offset": (
                    "INT",
                    {
                        "default": 0,
                        "min": -64,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Row phase shift applied before "
                            "the modulo operation"
                        ),
                    },
                ),
                "profile": (
                    list(SCANLINE_PROFILES),
                    {
                        "default": PROFILE_HARD,
                        "tooltip": "hard: abrupt transition; soft: raised-cosine ramp",
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
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def apply_scanlines(
        self,
        image: torch.Tensor,
        pitch: int = 2,
        thickness: float = 0.5,
        strength: float = 0.35,
        offset: int = 0,
        profile: str = PROFILE_HARD,
        show_preview: bool = True,
        prompt: dict | None = None,
        extra_pnginfo: dict | None = None,
    ) -> dict:
        '''Apply CRT scanlines to a batch of frames.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            pitch: Number of rows per scanline period (>= 1).
            thickness: Fraction of the period that is darkened in [0, 1].
            strength: Maximum dimming factor in [0, 1].
            offset: Row phase shift applied before the modulo operation.
            profile: Brightness profile — one of SCANLINE_PROFILES.
            show_preview: Whether to generate in-node preview images.
            prompt: Executing prompt graph (hidden PROMPT input);
                embedded into preview PNG metadata. Optional.
            extra_pnginfo: Hidden EXTRA_PNGINFO input, the UI
                workflow; embedded into preview PNG metadata. Optional.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid parameters.
        '''
        frames = tensor_to_frames(image)
        out = [
            scanlines(f, pitch, thickness, strength, offset, profile)
            for f in frames
        ]
        logger.info(
            "Scanlines: %d frame(s) processed (pitch=%d, thickness=%.2f, "
            "strength=%.2f, offset=%d, profile=%s)",
            len(out),
            pitch,
            thickness,
            strength,
            offset,
            profile,
        )
        return build_output_payload(
            out,
            show_preview=show_preview,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
