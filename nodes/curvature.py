'''
File:   curvature.py
Brief:  Screen curvature node — barrel distortion simulating CRT convexity,
        with a mask of the blackened pixels.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.crt.curvature engine.
'''

from __future__ import annotations

import logging

import numpy as np
import torch

from ..core.crt import curvature
from ..utils.images import frames_to_mask_tensor, tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class CurvatureNode:
    '''Apply barrel distortion to simulate a convex CRT screen.

    Pixels near the corners are pulled inward, simulating the physical
    curvature of an old CRT display. The frame-edge midpoints stay on the
    original edges, so only the corners fall outside the source — the
    natural curvature artifact. The ``mask`` output marks the pixels that
    went black as a result (1 = fully black, fractional on the curved
    boundary). ``edge_softness`` feathers that boundary with a smooth
    falloff of the given width instead of a hard cut.
    '''

    CATEGORY = "Lukutar/Retro/CRT"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
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
                "edge_softness": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 32.0,
                        "step": 0.5,
                        "tooltip": (
                            "Width of the smooth falloff into the black "
                            "border, in pixels (0 = hard cut). Feathers "
                            "the curved boundary without blurring content."
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
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def apply_curvature(
        self,
        image: torch.Tensor,
        amount: float = 0.15,
        edge_softness: float = 0.0,
        show_preview: bool = True,
        prompt: dict | None = None,
        extra_pnginfo: dict | None = None,
    ) -> dict:
        '''Apply barrel distortion to a batch of frames.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            amount: Distortion strength in [0, 1]. Zero is identity.
            edge_softness: Width of the smooth falloff into the black
                border, in pixels ([0, 32]). Zero is a hard cut.
            show_preview: Whether to generate in-node preview images.
            prompt: Executing prompt graph (hidden PROMPT input);
                embedded into preview PNG metadata. Optional.
            extra_pnginfo: Hidden EXTRA_PNGINFO input, the UI
                workflow; embedded into preview PNG metadata. Optional.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE
            tensor followed by the MASK tensor (B, H, W, float in [0, 1];
            1 = pixel went black from the distortion).

        Raises:
            ValueError: On invalid parameters.
        '''
        frames = tensor_to_frames(image)
        out: list[np.ndarray] = []
        black_masks: list[np.ndarray] = []
        for frame in frames:
            warped, black = curvature(frame, amount, edge_softness)
            out.append(warped)
            black_masks.append(black)
        logger.info(
            "Curvature: %d frame(s) processed (amount=%.2f, softness=%.1f)",
            len(out),
            amount,
            edge_softness,
        )
        return build_output_payload(
            out,
            show_preview=show_preview,
            extra_result=(frames_to_mask_tensor(black_masks),),
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
