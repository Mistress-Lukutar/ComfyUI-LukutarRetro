# ComfyUI-LukutarRetro

ComfyUI custom node pack by Mistress-Lukutar: **pixel-art / retro display
filters**. Pure math over images — no generative or ML content. Every
filter node is `IMAGE` in → `IMAGE` out with per-filter parameters and a
built-in node preview behind a `show_preview` toggle.

Status: **work in progress** — the node set lands in phases; this README
gets the full behavioral contract as nodes are added (see `AGENTS.md` for
the architecture).

## Planned nodes

Pixel (`Lukutar/Retro/Pixel`): Pixelize, Palette Quantize, Color Depth,
Dither (8-bit).
CRT (`Lukutar/Retro/CRT`): Scanlines, Subpixel Mask, Bloom (Glow),
Chromatic Aberration, Screen Curvature, Vignette.
Analog (`Lukutar/Retro/Analog`): VHS (Analog), LCD Dot Matrix.

## Install

Clone into `ComfyUI/custom_nodes/` (or install via the ComfyUI registry —
publisher `mistress-lukutar`, node id `lukutar-retro`, once published).
Requires `opencv-python` (see `requirements.txt`); torch and numpy come
from the ComfyUI runtime.

## License

MIT — see `LICENSE`.
