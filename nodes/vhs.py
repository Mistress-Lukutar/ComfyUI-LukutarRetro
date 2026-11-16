'''
File:   vhs.py
Brief:  VHS node — analog degradation pipeline (chroma bleed, jitter, noise, band).
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.analog.vhs engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.analog import NOISE_COLOR, NOISE_MODES, vhs
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class VhsNode:
    '''Apply a VHS-degradation effect to an image.

    Four stages are applied in sequence (each skipped when its strength is 0):
    chroma bleed in YUV, per-row horizontal jitter, additive Gaussian noise,
    and a rolling brightening band.
    '''

    CATEGORY = "Lukutar/Retro/Analog"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_vhs"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to apply VHS degradation to"},
                ),
                "noise": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Noise amplitude in [0, 1] (sigma = noise * 40)",
                    },
                ),
                "chroma_bleed": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "U/V chroma blur amount in [0, 1]",
                    },
                ),
                "jitter": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Horizontal row-shift amplitude in [0, 1] "
                            "(max = jitter * 6 px)"
                        ),
                    },
                ),
                "band": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Rolling-band brightening strength in [0, 1]",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": (
                            "0 = fresh randomness on every run; "
                            ">0 = deterministic for that seed"
                        ),
                    },
                ),
                "noise_mode": (
                    list(NOISE_MODES),
                    {
                        "default": NOISE_COLOR,
                        "tooltip": "Noise colouring mode",
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

    def apply_vhs(
        self,
        image: torch.Tensor,
        noise: float = 0.15,
        chroma_bleed: float = 0.5,
        jitter: float = 0.3,
        band: float = 0.3,
        seed: int = 0,
        noise_mode: str = NOISE_COLOR,
        show_preview: bool = True,
        prompt: dict | None = None,
        extra_pnginfo: dict | None = None,
    ) -> dict:
        '''Apply VHS degradation to a batch of frames.

        Four stages are applied per frame (each skipped when its strength is 0):
        chroma bleed in YUV, per-row horizontal jitter, additive Gaussian noise,
        and a rolling brightening band.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            noise: Noise amplitude in [0, 1] (sigma = noise * 40).
            chroma_bleed: U/V chroma blur amount in [0, 1].
            jitter: Horizontal row-shift amplitude in [0, 1] (max = jitter * 6 px).
            band: Rolling-band brightening strength in [0, 1].
            seed: Random seed (0 = fresh randomness each run; >0 = deterministic).
            noise_mode: Noise colouring — one of NOISE_MODES.
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
            vhs(f, noise, chroma_bleed, jitter, band, seed, noise_mode)
            for f in frames
        ]
        logger.info(
            "VHS: %d frame(s) processed (noise=%.2f, chroma_bleed=%.2f, "
            "jitter=%.2f, band=%.2f, seed=%d, mode=%s)",
            len(out),
            noise,
            chroma_bleed,
            jitter,
            band,
            seed,
            noise_mode,
        )
        return build_output_payload(
            out,
            show_preview=show_preview,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
