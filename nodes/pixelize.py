'''
File:   pixelize.py
Brief:  Pixelization node — reduce resolution via cell pooling and upscale.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.pixelize.Pixelizer engine.
'''

from __future__ import annotations

import logging

import torch

from ..core.pixelize import METHOD_MODE, SAMPLE_METHODS, Pixelizer
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class PixelizeNode:
    '''Pixelize an image by downsampling cells and upscaling with nearest neighbour.

    The pixel_size parameter defines the logical cell dimension in input pixels
    (1 = identity). Sampling method controls how colour is pooled inside each cell.
    Pre-blur applies a Gaussian blur before downsampling for anti-aliasing.
    '''

    CATEGORY = "Lukutar/Retro/Pixel"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pixelize"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to pixelize"},
                ),
                "pixel_size": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": (
                            "Logical pixel size in input pixels; "
                            "1 = identity"
                        ),
                    },
                ),
                "sampling": (
                    list(SAMPLE_METHODS),
                    {
                        "default": METHOD_MODE,
                        "tooltip": (
                            "center: sample centre pixel; mean: average cell; "
                            "median: median cell; mode: dominant colour"
                        ),
                    },
                ),
                "pre_blur": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "Gaussian pre-blur sigma as anti-alias"
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

    def pixelize(
        self,
        image: torch.Tensor,
        pixel_size: int,
        sampling: str = METHOD_MODE,
        pre_blur: float = 0.0,
        show_preview: bool = True,
        prompt: dict | None = None,
        extra_pnginfo: dict | None = None,
    ) -> dict:
        '''Process a batch of frames through the pixelizer.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            pixel_size: Cell dimension in pixels (>= 1).
            sampling: Pooling method from SAMPLE_METHODS.
            pre_blur: Gaussian blur sigma applied before downsampling (>= 0).
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
        engine = Pixelizer(pixel_size, method=sampling, pre_blur=pre_blur)
        out = [engine.process(f) for f in frames]
        logger.info(
            "Pixelize: %d frame(s) processed (pixel_size=%d, method=%s, pre_blur=%.1f)",
            len(out),
            pixel_size,
            sampling,
            pre_blur,
        )
        return build_output_payload(
            out,
            show_preview=show_preview,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
