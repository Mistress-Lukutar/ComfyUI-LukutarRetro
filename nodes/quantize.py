'''
File:   quantize.py
Brief:  Palette-based quantization node.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Node wrapping the core.quantize.PaletteQuantizer engine with palette selection
from built-in retro palettes or a custom hex-string input.
'''

from __future__ import annotations

import logging

import torch

from ..core.dither import DITHER_METHODS, DITHER_NONE
from ..core.palettes import PALETTE_NAMES, get_palette, parse_hex_palette
from ..core.quantize import PaletteQuantizer
from ..utils.images import tensor_to_frames
from .preview import build_output_payload

logger = logging.getLogger(__name__)


class PaletteQuantizeNode:
    '''Quantise an image through a retro palette with optional dithering.

    Select a built-in palette (e.g. PICO-8, C64, CGA) or provide custom
    hex colours. Apply ordered Bayer or Floyd-Steinberg dithering to
    simulate limited colour hardware.
    '''

    CATEGORY = "Lukutar/Retro/Pixel"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "quantize"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple]]:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Image to quantise"},
                ),
                "palette": (
                    list(PALETTE_NAMES),
                    {
                        "default": "pico8_16",
                        "tooltip": "Built-in retro palette",
                    },
                ),
                "custom_palette": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional custom palette: hex colours (#0f380f) "
                            "separated by commas/semicolons/newlines; "
                            "empty = use palette combo"
                        ),
                    },
                ),
                "dither": (
                    list(DITHER_METHODS),
                    {
                        "default": DITHER_NONE,
                        "tooltip": (
                            "Dithering method to reduce colour banding"
                        ),
                    },
                ),
                "dither_strength": (
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

    def quantize(
        self,
        image: torch.Tensor,
        palette: str = "pico8_16",
        custom_palette: str = "",
        dither: str = DITHER_NONE,
        dither_strength: float = 1.0,
        serpentine: bool = True,
        show_preview: bool = True,
        prompt: dict | None = None,
        extra_pnginfo: dict | None = None,
    ) -> dict:
        '''Process a batch of frames through palette quantisation.

        Args:
            image: ComfyUI IMAGE batch (B, H, W, 3), float in [0, 1].
            palette: Built-in palette name from PALETTE_NAMES.
            custom_palette: Hex colour string for a custom palette;
                empty uses palette combo.
            dither: Dithering method from DITHER_METHODS.
            dither_strength: Dither amplitude in [0, 1].
            serpentine: Alternate scanline direction for diffusion methods.
            show_preview: Whether to generate in-node preview images.
            prompt: Executing prompt graph (hidden PROMPT input);
                embedded into preview PNG metadata. Optional.
            extra_pnginfo: Hidden EXTRA_PNGINFO input, the UI
                workflow; embedded into preview PNG metadata. Optional.

        Returns:
            OUTPUT_NODE payload dict whose "result" carries the IMAGE tensor.

        Raises:
            ValueError: On invalid palette or parameters.
        '''
        frames = tensor_to_frames(image)

        if custom_palette.strip():
            pal = parse_hex_palette(custom_palette)
        else:
            pal = get_palette(palette)

        engine = PaletteQuantizer(
            pal,
            dither=dither,
            strength=dither_strength,
            serpentine=serpentine,
        )
        out = [engine.process(f) for f in frames]
        logger.info(
            "Palette quantize: %d frame(s) processed "
            "(palette=%s, dither=%s, strength=%.2f)",
            len(out),
            palette if not custom_palette.strip() else "custom",
            dither,
            dither_strength,
        )
        return build_output_payload(
            out,
            show_preview=show_preview,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
