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
colour is used as a single RGB unit); they are CPU-bound in CPython (the scan
runs over plain Python lists with a sliding window of pending rows, roughly
3-5x faster than the earlier numpy-scalar loop, but still seconds per
megapixel). For ordered dithering, `dither_strength` scales the threshold
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
and `atkinson` are error-diffusion methods (CPU-bound sequential scanline,
same optimized list-based scan as Palette Quantize).
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

CRT filter nodes — Scanlines, Subpixel Mask, Bloom (Glow), Chromatic Aberration,
Screen Curvature, and Vignette. Each node applies one atomic screen effect to an
IMAGE batch at its original resolution, with per-filter parameters and a built-in
preview behind a `show_preview` toggle.

#### RetroScanlines «Scanlines»

Simulates the dark horizontal bands of a CRT display by dimming every *pitch* rows.
Within each pitch-period the darkest *thickness* fraction is reduced by *strength*;
the remaining rows pass through unchanged. The *profile* parameter selects whether
the transition within the dark band is abrupt (**hard**) or uses a raised-cosine
ramp (**soft**). An integer *offset* shifts the pattern phase, useful for aligning
scanlines with an underlying pixel grid.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to apply scanlines to. |
| pitch | INT | 2 | Rows per scanline period (1–32, step 1). `1` = every row darkened. |
| thickness | FLOAT | 0.5 | Fraction of the period that is darkened in [0, 1]. |
| strength | FLOAT | 0.35 | Maximum dimming factor in [0, 1]. |
| offset | INT | 0 | Row phase shift applied before the modulo operation (−64 to 64). |
| profile | COMBO | hard | Brightness profile: `hard` (abrupt transition) or `soft` (raised-cosine ramp). |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Scanline-darkened batch at original input resolution. |

#### RetroSubpixelMask «Subpixel Mask»

Fakes a CRT phosphor-mask (or LCD dot-grid) structure: horizontal RGB triad
period is `3 * pitch`, each channel only transmits light through its own vertical
stripe(s). Overall brightness is multiplied by 3 (`MASK_BRIGHTNESS_COMPENSATION`)
because each channel lights only one-third of the area — saturated pixels may clip.

Three mask types are available: **aperture grille** (straight vertical RGB stripes),
**shadow mask** (odd rows staggered by half a triad period, mimicking real CRT masks),
and **slot mask** (aperture grille with every second horizontal band of height *pitch*
dimmed to 50%, simulating slot-mask aperture plates). For clean alignment with the
pixel grid, pair this node with **Pixelize** using a `pixel_size` that is a multiple
of `3 * pitch`.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to apply phosphor mask to. |
| mask | COMBO | aperture_grille | Phosphor-mask type: `aperture_grille`, `shadow_mask`, or `slot_mask`. |
| pitch | INT | 1 | Stripe width in pixels (1–16, step 1); RGB triad period is `3 * pitch`. |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Phosphor-masked batch at original input resolution. Saturated areas may clip. |

#### RetroBloom «Bloom (Glow)»

Extracts bright pixels above a luminance *threshold* (Rec. 601 luma, linear ramp
to white), applies a Gaussian blur with sigma *radius*, and blends the glow back
into the original image. The blend can be **screen** (`255 − (255 − a)(255 − b) / 255`)
for soft additive glow or **add** for direct addition. An *intensity* multiplier
scales the bloom contribution before blending.

When `threshold >= 1.0` the bright mask is entirely zero and the node returns the
input unchanged (early-out). The Gaussian kernel size is computed as `2 * ceil(3σ) + 1`,
minimum 3.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to apply bloom to. |
| threshold | FLOAT | 0.7 | Luminance fraction ([0, 1]) above which pixels glow. `1.0` = no glow. |
| radius | FLOAT | 8.0 | Gaussian blur sigma in pixels for the glow (0.5–100, step 0.5). |
| intensity | FLOAT | 0.5 | Bloom contribution multiplier ([0, 4]). Scales the blurred glow before blending. |
| blend | COMBO | screen | Blend mode: `screen` (soft additive) or `add` (direct addition). |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Bloom-glow batch at original input resolution. |

#### RetroChromaticAberration «Chromatic Aberration»

Displaces the red and blue channels relative to green (which stays fixed) using
`cv2.remap` with linear interpolation and `BORDER_REFLECT`. Green channel is
copied through unchanged. Two displacement modes: **linear** shifts red right by
*amount* pixels and blue left (red fringe on the right edge of bright features,
blue on the left); **radial** samples red outward from center (contracted toward
center in the output) and blue inward (expanded outward), producing fringes on the
inner/outer sides of edges respectively. `amount = 0` is identity.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to apply chromatic aberration to. |
| amount | FLOAT | 2.0 | Displacement magnitude in pixels ([0, 20], step 0.5). `0` = identity. |
| mode | COMBO | radial | Displacement mode: `radial` (outward from centre) or `linear` (horizontal shift). |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Chromatic-aberration batch at original input resolution. |

#### RetroCurvature «Screen Curvature»

Simulates the convex barrel distortion of a CRT screen. Normalized coordinates
are remapped from `center + (pos − center)` to `center + (pos − center) * (1 +
amount * r²)` where r² is the squared normalized distance from centre. The
sampling coordinates are compensated by `1 / (1 + amount)` (zoom
compensation): the midpoints of all four frame edges sample exactly the
original edge pixels, so no straight black bands appear along the borders —
only the corners fall outside the source and fill with black (the natural
curvature artifact, constant-border remap). The centre pixel is unchanged;
`amount = 0` is identity.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to apply screen curvature to. |
| amount | FLOAT | 0.15 | Distortion strength ([0, 1], step 0.05). `0` = identity. |
| edge_softness | FLOAT | 0.0 | Width of the smooth falloff into the black border, in pixels ([0, 32], step 0.5). `0` = hard cut. |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Barrel-distorted batch at original input resolution. Corners are black. |
| mask | MASK | The pixels that went black from the distortion: `1` = fully black, `0` = untouched content, fractional on the curved boundary. Shape (B, H, W), float32. |

**Edge feathering.** `edge_softness` widens the darkening ramp at the curved
black boundary to a smoothstep of the given width (in source pixels). The
image itself is never blurred: content is remapped with clamped border
extension and multiplied by the feathered alpha, so the interior stays
sharp and the black region keeps its shape — the fade just becomes gradual,
like the glass-edge falloff of a real tube. At `edge_softness = 0` the node
is bit-identical to the hard constant-border cut. The ramp is defined in
source coordinates, so under the distortion it compresses where the remap
derivative is large (near the corners) and stays ~`edge_softness` px at the
edge midpoints. The `mask` output follows the feather (`1 − alpha`).

#### RetroVignette «Vignette»

Radar corner-darkening effect: computes a normalized distance *d* from the image
centre (0 at centre, 1 at corners). Darkening begins at normalised radius *radius*,
follows a power curve with exponent `1 + softness * 4`, and reaches maximum
darkening of factor *strength*. Setting `radius = 1` or `strength = 0` returns the
input unchanged.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to apply vignette to. |
| strength | FLOAT | 0.4 | Maximum darkening factor ([0, 1], step 0.05). `0` = no effect. |
| radius | FLOAT | 0.8 | Normalised distance at which darkening begins ([0, 1], step 0.05). |
| softness | FLOAT | 0.5 | Controls the steepness of the transition curve ([0, 1], step 0.05). |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Vignetted batch at original input resolution. |

### Analog (`Lukutar/Retro/Analog`)

VHS degradation and LCD dot-matrix overlay nodes. Each node applies one
analog-style effect to an `IMAGE` batch at its original resolution, with
per-filter parameters and a built-in preview behind a `show_preview` toggle.

#### RetroVHS «VHS (Analog)»

Applies four degradation stages in sequence (each skipped when its strength is
0): chroma bleed, horizontal jitter, additive noise, and rolling-band brightening.

**Chroma bleed.** Converts to YUV, box-blurs the U/V planes horizontally with a
kernel width proportional to *chroma_bleed* (range 3–15, odd), then blends back to
RGB. Luma is preserved through this stage.

**Jitter.** Each row is shifted by a random integer horizontal offset using edge
replication (no wrap-around). Maximum amplitude = *jitter* × 6 px.

**Noise.** Additive Gaussian noise with sigma = *noise* × 40. `mono` mode uses a
single shared noise plane across all channels; `color` generates independent noise
per channel.

**Band.** Multiplicative vertical brightening band: a Gaussian profile centered at a
random row position (drawn from the same RNG as jitter and noise). Only brightens,
never darkens.

The *seed* parameter controls randomness: `0` = fresh random seed every run; > 0 =
deterministic per frame for that seed value.

Because the input is named `seed`, the ComfyUI frontend auto-attaches a
`control_after_generate` widget to it. On nodes created with this pack the
control defaults to **fixed** — with the default `seed = 0` every run gets
fresh randomness, and the workflow state does not change between runs. If
you switch the control to *randomize*, the seed (and the serialized
workflow) changes on every execution — combined with the queue button's
"Run (On change)" mode that re-queues the prompt forever. Workflows saved
with an explicit control mode keep the saved value.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to apply VHS degradation to. |
| noise | FLOAT | 0.15 | Noise amplitude in [0, 1] (sigma = noise * 40). |
| chroma_bleed | FLOAT | 0.5 | U/V chroma blur amount in [0, 1]. Kernel width = round(chroma_bleed * 15), clamped to [3, 15], forced odd. |
| jitter | FLOAT | 0.3 | Horizontal row-shift amplitude in [0, 1] (max = jitter * 6 px). |
| band | FLOAT | 0.3 | Rolling-band brightening strength in [0, 1]. Gaussian profile, only brightens. |
| seed | INT | 0 | Random seed (0 = fresh randomness each run; > 0 = deterministic for that seed). Range: 0–2147483647. |
| noise_mode | COMBO | color | Noise colouring: `mono` (shared plane) or `color` (independent per channel). |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Degraded batch at original input resolution. |

#### RetroLCDDotMatrix «LCD Dot Matrix»

Overlays a dot-matrix LCD pattern on the image at its current resolution. Each
`cell_size`-pixel block is treated as a "dot": pixels inside the active region keep
full brightness, pixels in the gap are darkened by factor `(1 - grid_dark)`.

**Square shape.** The active region is a rectangle defined by *gap_ratio*: within each
cell the normalized local coordinates must satisfy `gap_ratio/2 <= coord <= 1 -
gap_ratio/2` on both axes.

**Round shape.** The active region is a circle with normalised radius `<= 1 - gap_ratio`,
computed from fractional coordinates centred at 0 in range [-1, 1].

When *gap_ratio* = 0 or *grid_dark* = 0 the overlay is identity (no visible change).
Pair with **Pixelize** by setting `cell_size` equal to Pixelize's `pixel_size` for
matching pitch.

##### Inputs / Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| image | IMAGE | — | Image to apply dot-matrix overlay to. |
| cell_size | INT | 8 | Side length of each dot cell in pixels (1–64, step 1). |
| gap_ratio | FLOAT | 0.15 | Fractional gap around the active region in [0, 0.5]. `0` = identity. |
| grid_dark | FLOAT | 0.6 | Darkening factor for gap pixels in [0, 1]. `0` = no darkening; `1` = fully dark gaps. |
| shape | COMBO | square | Dot shape: `square` or `round`. |
| show_preview | BOOLEAN | true | Render an in-node preview of the result. |

| Name | Type | Description |
|------|------|-------------|
| image | IMAGE | Dot-matrix-overlay batch at original input resolution. |

## Preview

Every filter node carries a `show_preview` BOOLEAN widget (default **on**).
When on, the backend saves per-frame preview PNGs to the ComfyUI temp
directory at the frame's FULL resolution — the preview shows exactly the
pixels the `IMAGE` output carries, with no downscaling. The web extension
(`web/js/retro_preview.js`) feeds `/view` URLs of the temp files into
`node.imgs` on execution, so right-click **Open Image / Copy Image /
Save Image** works against the real temp files. The v2 (Vue node renderer)
output section is left visible — under that renderer it is the only surface
that can show the executed result, because the litegraph draw-background
preview path does not run there.

When `show_preview` is toggled off, the backend emits no preview images at
all (no temp PNGs are written), and the next execution clears the previous
in-node preview: `node.imgs` is emptied and the frontend's in-canvas image
preview widget (`$$canvas-image-preview`) is removed, so no stale preview
remains inside the node body. The pack remains fully functional without the
web extension: in that case the preview images simply appear in the standard
ComfyUI output section.

The preview PNGs embed the same workflow metadata as ComfyUI's own Save
Image node — the `prompt` (API graph) and `workflow` chunks, received via
hidden `PROMPT` / `EXTRA_PNGINFO` inputs. A preview PNG can therefore be
drag-and-dropped onto the canvas to restore the workflow that produced it.

Nodes are `OUTPUT_NODE = True`, meaning they re-execute whenever their inputs
change — same behaviour as the built-in Preview Image node. The `show_preview`
widget value is declared in `INPUT_TYPES` so it serializes into workflows
without frontend-only hacks; the backend honours it (off = no preview is
generated, on = temp PNGs are emitted for the frontend to render).

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
