'''
File:   bloom.py
Brief:  Bloom/glow node — bright-area glow with configurable blend mode.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.crt.bloom engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.crt import (
    BLEND_SCREEN,
    BLOOM_BLENDS,
    bloom,
)
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class BloomNode:
    '''Add a bloom/glow effect to bright areas of an image.

    Bright pixels above *threshold* are blurred and blended back into the
    original using either screen or additive blending.
    '''

    CATEGORY = "Lukutar/Retro/CRT"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_bloom"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to apply bloom to"},
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Luminance fraction above which pixels glow",
                    },
                ),
                "radius": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.5,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Gaussian blur sigma in pixels for the glow",
                    },
                ),
                "intensity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Bloom contribution multiplier",
                    },
                ),
                "blend": (
                    list(BLOOM_BLENDS),
                    {
                        "default": BLEND_SCREEN,
                        "tooltip": "screen: soft additive; add: direct addition",
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

    def apply_bloom(
        self,
        image: torch.Tensor,
        threshold: float = 0.7,
        radius: float = 8.0,
        intensity: float = 0.5,
        blend: str = BLEND_SCREEN,
        show_preview: bool = True,
        prompt: dict | None = None,
        extra_pnginfo: dict | None = None,
    ) -> dict:
        '''Apply bloom/glow to a batch of frames.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            threshold: Luminance fraction ([0, 1]) above which pixels glow.
            radius: Gaussian blur sigma in pixels (>= 0.05).
            intensity: Bloom contribution multiplier in [0, 4].
            blend: Blend mode — one of BLOOM_BLENDS.
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
            bloom(f, threshold, radius, intensity, blend)
            for f in frames
        ]
        logger.info(
            "Bloom: %d frame(s) processed (threshold=%.2f, radius=%.1f, "
            "intensity=%.2f, blend=%s)",
            len(out),
            threshold,
            radius,
            intensity,
            blend,
        )
        return build_output_payload(
            out,
            show_preview=show_preview,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
