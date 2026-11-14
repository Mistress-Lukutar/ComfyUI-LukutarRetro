# AGENTS.md — ComfyUI-LukutarRetro

ComfyUI custom node pack by Mistress-Lukutar. **Pixel-art / retro display
filters** — pure math over images, no generative/ML content: pixelization,
color-palette & bit-depth restriction with dithering, and CRT/analog-video
artifacts. Every filter node has the same shape: `IMAGE` in → `IMAGE` out
(batch preserved, resolution preserved unless the filter is geometric),
per-filter parameters, and a `show_preview` BOOLEAN widget driving an
in-node preview. Filters are non-destructive: the output IMAGE always
carries the processed result, the preview is extra.

Current state: ALL 12 nodes are implemented — Pixel (RetroPixelize,
RetroPaletteQuantize, RetroColorDepth, RetroDither), CRT (RetroScanlines,
RetroSubpixelMask, RetroBloom, RetroChromaticAberration, RetroCurvature,
RetroVignette) and Analog (RetroVHS, RetroLCDDotMatrix). The smoke test
runs every node end-to-end plus a 4-node chain. Per-node behavior
contracts live in `README.md` — keep it in sync.

## Layout & architecture boundaries

- `core/` — pure numpy + OpenCV processing engines. **No ComfyUI imports,
  no torch** — must stay importable and unit-testable on any plain
  python that has numpy + opencv. Modules: `pixelize.py` (cell pooling:
  center/mean/median/mode), `dither.py` (Bayer 2/4/8 + FS/Atkinson error
  diffusion), `palettes.py` (10 hardware palettes + hex-string parser),
  `quantize.py` (CIELAB nearest match via 5-bit RGB LUT + palette
  dithering).
- `nodes/` — ComfyUI node classes (`INPUT_TYPES`, tensor glue). `comfy`
  imports must be lazy and guarded with
  `try: from comfy... except ImportError: return None` (pattern:
  `_make_progress_bar` in the sibling pack's `nodes/color_match.py` —
  these filters are single-pass, so most nodes need no progress bar).
  `utils.images` provides the tensor⇄frame glue and needs torch, so it is
  imported only from node modules.
- `web/js/` — frontend extension served via `WEB_DIRECTORY` in the root
  `__init__.py`: `retro_preview.js` covers ALL filter nodes (explicit
  node-name list). Contract: set `node.hideOutputImages = true` on
  creation; on `onExecuted` feed `node.imgs` with `/view` URLs of the
  emitted temp images UNLESS the node's `show_preview` widget is off —
  then set `node.imgs = []` (right-click Open/Save Image still works from
  temp files when preview is on). Vanilla ES module, no build step; the
  pack must stay fully functional without it (preview simply shows in the
  v2 output section).
- `utils/` — torch IMAGE-tensor ⇄ numpy frame conversion helpers
  (B,H,W,3 float [0,1] RGB ⇄ HWC float32 [0,255] RGB).
- `tests/unit/` — engine tests, torch-free.
  `tests/smoke_test_comfyui_load.py` — node loading + end-to-end run,
  ComfyUI's embedded python only.

### Preview contract (nodes/preview.py)

Every filter node: `OUTPUT_NODE = True`, `RETURN_TYPES = ("IMAGE",)`.
`build_preview(frames)` in `nodes/preview.py` saves each frame as a temp
PNG via guarded PIL (Pillow from the ComfyUI runtime), caps the long side
at 1024 px (preview only — the IMAGE output keeps full resolution),
returns the `{"ui": {"images": [...]}, "result": (tensor,)}` payload; on
any failure it logs and returns a payload without ui images (preview must
never kill the filter). The `show_preview` widget IS declared in
INPUT_TYPES (frontend stores it, backend accepts and ignores it) so it
serializes without frontend-only widget hacks.

Adding a node: engine in `core/`, node class in `nodes/`, register in
`nodes/__init__.py` (`NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS`)
and add the mapping key to the JS name list in `web/js/retro_preview.js`.
ComfyUI menu categories: `Lukutar/Retro/Pixel` (pixelization/palette/
dither nodes), `Lukutar/Retro/CRT` (atomic screen effects), 
`Lukutar/Retro/Analog` (VHS, LCD dot matrix). Mapping keys carry the
`Retro` prefix for cross-pack uniqueness (sibling packs share the menu).

## Known implementation traps (do not regress)

- Error-diffusion scan loops MUST copy the pixel before overwriting it
  (`pixel = plane[row, col].copy()`): basic indexing returns a view, so
  writing the matched value through it silently zeroes the diffused
  error. Regression: `TestErrorDiffusionSpreads` in
  `tests/unit/test_quantize.py`.
- Clamp the quantized value to `[0, 255]` inside diffusion loops —
  accumulated error can overshoot the outer levels and would otherwise
  leak oversized residues into downstream pixels.
- `folder_paths` is a TOP-LEVEL module of the ComfyUI root
  (`import folder_paths`), never `from comfy import folder_paths`.
  Temp files go through `folder_paths.get_save_image_path(...)` plus
  `os.path.join(full_output_dir, filename)` (see `nodes/preview.py`).
- The preview helper never raises — any failure logs a warning and
  returns `[]`. Do not turn that into raising: previews must never kill
  the filter.
- `utils/images.py` needs torch — import it only from node modules;
  the pytest bootstrap (`tests/conftest.py`) must never execute it.

## Commands

```bash
# Unit tests — any python with numpy, opencv-python, pytest (no torch)
python -m pytest

# Lint (ruff: line-length 88, rules E,F,W,I,UP,B,SIM)
python -m ruff check .

# Type check — plain `mypy core` FAILS (see gotchas); use:
python -m mypy --explicit-package-bases core

# Node load + e2e smoke test — MUST use ComfyUI's embedded python
"C:/Ai/ComfyUI_windows_portable/python_embeded/python.exe" tests/smoke_test_comfyui_load.py
```

The pack is **not pip-installed**; it is cloned into `ComfyUI/custom_nodes`.
`requirements.txt` carries `opencv-python` — torch and numpy come from the
ComfyUI runtime. The `[project]`/`[tool.comfy]` tables in `pyproject.toml`
are registry + package metadata (no pip semantics); pytest/ruff/mypy
config lives in its tool sections.

## Releases (ComfyUI registry)

Published by CI, never by hand. Pushing a `v*` tag (or pressing "Run
workflow" in the Actions tab) makes `.github/workflows/publish_action.yml`
run Comfy-Org/publish-node-action with the `REGISTRY_ACCESS_TOKEN` repo
secret and publish the version written in `pyproject.toml` to the
registry — publisher `mistress-lukutar`, node id `lukutar-retro`
(registry.comfy.org/publishers/mistress-lukutar).

**Note (as of v0.1.0): the workflow file is NOT committed yet** — no
`.github/` exists in the repo. Tag pushes publish nothing until it
lands; treat the flow below as the intended process, and verify
`.github/workflows/publish_action.yml` is tracked before trusting a
tag to publish.

```bash
# checks green first: pytest, ruff, mypy, smoke test (see Commands)
# bump the version in BOTH __init__.py.__version__ and pyproject.toml
git commit -m "chore(release): bump version to X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

- Verify: `gh run list --workflow=publish_action.yml` (expect success),
  then `curl -s https://api.comfy.org/nodes/lukutar-retro/versions`
  must list the new version.
- A published version can never be republished — the registry answers
  400 "The node version already exists". A forgotten bump = red run.
- The registry node name (`lukutar-retro` in `pyproject.toml`) is
  immutable after the first publish; `DisplayName` is changeable.
- The workflow deliberately does NOT fire on `pyproject.toml` edits:
  the file changes for non-release reasons too. The tag is the release.
- The package is every git-tracked file minus `.comfyignore` — when
  adding dev-only files, extend `.comfyignore` so they don't ship.

## Import / pytest gotchas

- The repo folder name contains a hyphen and is **not a valid python
  package name**. Consequences:
  - mypy errors with "not a valid Python package name" unless run with
    `--explicit-package-bases`.
  - Unit tests import `core` through a stub parent package aliased
    `comfyui_lukutar_retro` (`tests/conftest.py`), which deliberately
    does **not** execute the root `__init__.py` (it would pull in torch).
- pytest runs with `--import-mode=importlib` and root `conftest.py` sets
  `collect_ignore = ["__init__.py"]` — don't remove either; pytest 9
  would otherwise try to import the root `__init__` and fail.
- Root `__init__.py` is dual-mode: imported as a package it exports the
  real node mappings; imported as a bare module (no package context) it
  exposes empty mappings instead of failing. ComfyUI never takes the
  bare-module branch.

## Conventions

- Python >= 3.10. Every module starts with the header docstring
  (`File / Brief / Author / Date / Version`) followed by
  `from __future__ import annotations`.
- Google-style docstrings (`Args:` / `Returns:`); `#:` doc comments on
  module constants; combo choices exposed as module-level string tuples
  (e.g. `SAMPLE_METHODS` in `core/pixelize.py`, `PALETTES` in
  `core/palettes.py`, `DITHER_METHODS` in `core/dither.py`) and
  re-exported via the package `__init__` `__all__`.
- Use module loggers (`logging.getLogger(__name__)`), not print.
- Version is tracked in **both** `__init__.py.__version__` and
  `pyproject.toml` — bump them together (release flow: see Releases).
- Behavior of every node (inputs, parameters, edge cases, preview
  semantics) is specified in `README.md`; keep the README in sync with
  any behavior change.
