# ComfyUI-LukutarRetro

ComfyUI custom node pack by Mistress-Lukutar: **pixel-art / retro display
filters**. Pure math over images — no generative or ML content. Every filter
node is `IMAGE` in → `IMAGE` out with per-filter parameters and a built-in
node preview behind a `show_preview` toggle.

## Nodes

### Pixel (`Lukutar/Retro/Pixel`)

#### RetroPixelize «Pixelize»

Downsamples an image into square cells of user-defined size, pools colour
inside each cell, then upscales back to the original resolution via nearest-
neighbour interpolation so that downstream nodes always receive full-resolution
input. Cells at frame edges are padded by edge replication (constant-value
copy of the border pixel).

**Sampling methods.**

- `center` — subsamples at the centre pixel of each cell (`work[s//2::s, s//2::s]`).
  Fastest method; reproduces a raw grid without smoothing.
- `mean` — box-average of all pixels in the cell. Smooths fine detail inside
  each cell.
- `median` — per-channel median of cell pixels. Robust to outliers (bright or
  dark speckles) within a cell.
- `mode` — dominant colour per cell, found by binning channel values into 16
  levels (`value >> 4`) and picking the most frequent bin combination; ties
  broken by lowest combined code. Best for a true pixel-art look because it
  preserves solid colour regions without smearing them.

**Pre-blur.** A Gaussian blur with sigma `pre_blur` is applied before
downsampling. This anti-aliasing step softens high-frequency detail so that
the downsampling does not produce moiré; set to `0` for a hard pixelated look.

Default sampling is `mode`.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to pixelize. |
| pixel_size | INT | 8 | Logical cell dimension in input pixels (1–128, step 1). `1` = identity (no change). |
| sampling | COMBO | mode | Pooling method: `center`, `mean`, `median`, `mode`. |
| pre_blur | FLOAT | 0.0 | Gaussian pre-blur sigma for anti-aliasing before downsampling (0–10, step 0.1). |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Pixelized batch at original input resolution. |

#### RetroPaletteQuantize «Palette Quantize»

Maps every pixel to the nearest colour in a retro palette (CIELAB distance via
a 5-bit RGB lookup table) with optional dithering to simulate limited-colour
hardware. The LUT maps each of the 32³ = 32768 five-bit bins to the closest
palette entry, introducing a systematic bias of at most 4 levels per channel
towards the palette centre (query colour per bin is its midpoint `idx * 8 + 4`,
clamped to 255).

**Built-in palettes.**

| Palette | Colours | Notes |
|---------|---------|-------|
| dmg_4 | 4 | Game Boy DMG green palette |
| pico8_16 | 16 | PICO-8 default palette (default) |
| c64_16 | 16 | Commodore 64 Pepto palette |
| cga_16 | 16 | IBM CGA 16-colour palette |
| zx_spectrum_15 | 15 | ZX Spectrum palette |
| ega_64 | 64 | EGA full gamut (4-level-per-channel Cartesian product) |
| nes | 55 | Nintendo Entertainment System 2C02 |
| gray_4 | 4 | Four-level grayscale |
| mono_1bit | 2 | Black and white |
| sweetie_16 | 16 | Sweetie16 palette |

**Custom palette.** The `custom_palette` multiline STRING accepts hex colour
tokens (`#0f380f` or `0f380f`) separated by commas, semicolons, whitespace, or
newlines. Three-digit shorthand is not supported (exactly 6 hex digits per
token). Non-empty `custom_palette` overrides the `palette` combo entirely.

**Dithering.** Ordered Bayer methods (`bayer2`, `bayer4`, `bayer8`) are fully
vectorised and fast. Error-diffusion methods (`floyd-steinberg`, `atkinson`)
use a sequential scanline pass per pixel (not per channel — the matched palette
colour is used as a single RGB unit); they are CPU-bound in CPython (seconds
per megapixel). For ordered dithering, `dither_strength` scales the threshold
amplitude by a self-calibrated mean CIELAB distance of the palette; for
diffusion methods it scales the error propagated to neighbours.

**Serpentine mode.** When enabled, odd scanlines are processed right-to-left
with mirrored kernel offsets — standard practice for reducing directional bias
in error-diffusion dithering.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to quantise. |
| palette | COMBO | pico8_16 | Built-in retro palette (see table above). |
| custom_palette | STRING | "" | Optional custom palette: hex colours separated by commas/semicolons/newlines; empty = use the `palette` combo. |
| dither | COMBO | none | Dithering method: `none`, `bayer2`, `bayer4`, `bayer8`, `floyd-steinberg`, `atkinson`. |
| dither_strength | FLOAT | 1.0 | Dither amplitude (0 = off, 1 = full). Scales ordered-threshold bias or diffusion error. |
| serpentine | BOOLEAN | true | Alternate scanline direction for error-diffusion methods. |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Palette-quantized batch at original input resolution. |

#### RetroColorDepth «Color Depth»

Pure posterization: uniform per-channel quantisation to a fixed number of
levels. No dithering — the output is a clean step function. Two modes select
how the `value` parameter is interpreted.

**Bits mode.** `value` is the bit depth (1–8); the node computes `2^value`
quantisation levels per channel. 1-bit → 2 levels, 4-bit → 16 levels, etc.

**Levels mode.** `value` is the exact number of quantisation levels (2–64).
This lets you pick arbitrary step counts not tied to powers of two.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to posterize. |
| mode | COMBO | bits | `bits`: quantise to 2^value levels (value 1–8). `levels`: exact number of levels (value 2–64). |
| value | INT | 4 | Bit depth when `mode=bits` (1–8); exact level count when `mode=levels` (2–64). |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Posterized batch at original input resolution. |

#### RetroDither «Dither (8-bit)»

Bit-depth reduction with ordered or error-diffusion dithering on a raw RGB
image (no palette mapping). The engine is the same as Palette Quantize's
dither core but operates on continuous colour values rather than through a
palette LUT.

**Modes.** `color` dithers each RGB channel independently. `mono` converts
the frame to grayscale luma first (Rec. 601 weights: 0.299 R + 0.587 G +
0.114 B), broadcasts the single-channel result back to three channels, then
dithers — producing a greyscale output.

**Dither methods.** `bayer2`, `bayer4`, `bayer8` are fast vectorised ordered
dithering with threshold matrices of increasing granularity. `floyd-steinberg`
and `atkinson` are error-diffusion methods (CPU-bound sequential scanline).
The node excludes `none` from its combo — use **Color Depth** for undithered
posterization.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to dither. |
| levels | INT | 4 | Quantisation levels per channel (2–32, step 1). |
| mode | COMBO | color | `color`: RGB dither. `mono`: grayscale luma (Rec. 601) first, then dither all channels. |
| dither_method | COMBO | bayer4 | Dither algorithm: `bayer2`, `bayer4`, `bayer8`, `floyd-steinberg`, `atkinson`. |
| strength | FLOAT | 1.0 | Dither amplitude (0 = off, 1 = full). |
| serpentine | BOOLEAN | true | Alternate scanline direction for error-diffusion methods. |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Dithered batch at original input resolution. |

### CRT (`Lukutar/Retro/CRT`)

Scanlines, Subpixel Mask, Bloom (Glow), Chromatic Aberration, Screen Curvature,
Vignette — landing in a later phase.

### Analog (`Lukutar/Retro/Analog`)

VHS, LCD Dot Matrix — landing in a later phase.

## Preview

Every filter node carries a `show_preview` BOOLEAN widget (default **on**).
When on, the backend saves per-frame preview PNGs to the ComfyUI temp
directory with the long side capped at 1024 px; the `IMAGE` output always
carries the full-resolution processed result. The web extension
(`web/js/retro_preview.js`) hides the v2 output section (`node.hideOutputImages = true`)
and feeds `/view` URLs of the temp files into `node.imgs` on execution, so
right-click **Open Image / Copy Image / Save Image** works against the real
temp files.

When `show_preview` is toggled off, the next execution clears `node.imgs`
(the v2 output section stays hidden; no preview renders inside the node body).
The pack remains fully functional without the web extension: in that case
the preview images simply appear in the standard ComfyUI output section.

Nodes are `OUTPUT_NODE = True`, meaning they re-execute whenever their inputs
change — same behaviour as the built-in Preview Image node. The `show_preview`
widget value is declared in `INPUT_TYPES` so it serializes into workflows
without frontend-only hacks; the backend accepts and ignores it (preview
saving is always performed, the toggle only controls frontend rendering).

## Development

```bash
# Engine unit tests — any python with numpy, opencv-python, pytest (no torch)
python -m pytest

# Lint (ruff: line-length 88, rules E,F,W,I,UP,B,SIM)
python -m ruff check .

# Type check — plain `mypy core` FAILS (see gotchas); use:
python -m mypy --explicit-package-bases core

# Node load + e2e smoke test — MUST use ComfyUI's embedded python
"C:/Ai/ComfyUI_windows_portable/python_embeded/python.exe" tests/smoke_test_comfyui_load.py
```

See `AGENTS.md` for architecture details (code layout, import conventions,
release flow).

## Install

Clone into `ComfyUI/custom_nodes/` (or install via the ComfyUI registry —
publisher `mistress-lukutar`, node id `lukutar-retro`, once published).
Requires `opencv-python` (see `requirements.txt`); torch and numpy come
from the ComfyUI runtime.

## License

MIT — see `LICENSE`.
