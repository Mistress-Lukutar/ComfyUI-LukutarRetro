'''
File:   smoke_test_comfyui_load.py
Brief:  Simulates ComfyUI custom node loading without launching the server.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.3.0

Run with ComfyUI's own python (no server launch required):

    C:/Ai/ComfyUI_windows_portable/python_embeded/python.exe \
        tests/smoke_test_comfyui_load.py
'''

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

PACK_ROOT = Path(__file__).resolve().parent.parent
#: Portable-install ComfyUI root, used by the preview file-check part.
COMFYUI_ROOT = Path("C:/Ai/ComfyUI_windows_portable/ComfyUI")


def load_pack() -> object:
    '''Import the pack exactly the way ComfyUI's load_custom_node does.'''
    module_name = "custom_nodes.ComfyUI-LukutarRetro"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PACK_ROOT / "__init__.py",
        submodule_search_locations=[str(PACK_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for the pack")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_batch(
    height: int = 72, width: int = 96, batch_size: int = 2
) -> torch.Tensor:
    '''Build a synthetic IMAGE batch tensor (B, H, W, 3), float32 [0, 1].

    A colourful gradient plus deterministic noise so that mode/median
    pooling and dithering have texture to operate on.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        batch_size: Number of frames in the batch.

    Returns:
        torch.Tensor of shape (batch_size, H, W, 3), dtype float32.
    '''
    rng = np.random.default_rng(42)
    xs = np.linspace(0, 255, width, dtype=np.float32)
    ys = np.linspace(0, 255, height, dtype=np.float32)
    grid = xs[np.newaxis, :] + ys[:, np.newaxis]

    frame = np.stack([
        np.clip(grid, 0, 255),
        np.clip(grid * 0.5 + 60.0, 0, 255),
        np.clip(255.0 - grid * 0.3, 0, 255),
    ], axis=-1).astype(np.float32)

    # Add deterministic noise for texture (each frame gets a different slice)
    noise = rng.integers(-30, 30, size=(height, width, 3)).astype(np.float32)
    frame = np.clip(frame + noise, 0, 255)

    frames: list[np.ndarray] = []
    for _ in range(batch_size):
        noise_frame = rng.integers(-30, 30, size=(height, width, 3)).astype(np.float32)
        f = np.clip(frame + noise_frame, 0, 255).astype(np.float32)
        frames.append(f)

    # Convert to tensor: HWC [0, 255] -> BHWC [0, 1]
    tensors = []
    for f in frames:
        t = torch.from_numpy((f / 255.0).astype(np.float32))[None]
        tensors.append(t)
    return torch.cat(tensors, dim=0)


def _check_image_tensor(tensor: torch.Tensor, expected_shape: tuple) -> None:
    '''Assert that an IMAGE tensor has the expected shape, dtype, and range.'''
    assert tensor.dtype == torch.float32, f"bad dtype: {tensor.dtype}"
    assert tensor.shape == expected_shape, (
        f"bad shape: {tensor.shape} != {expected_shape}"
    )
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0, (
        f"values out of [0, 1]: min={float(tensor.min()):.4f}, "
        f"max={float(tensor.max()):.4f}"
    )


def main() -> None:
    # ------------------------------------------------------------------ #
    # a) Pack loads; version; mappings                                   #
    # ------------------------------------------------------------------ #
    print("--- Check a: pack load, version, mappings ---")

    sys.path.insert(0, str(COMFYUI_ROOT))
    try:
        pack = load_pack()
    except Exception as exc:
        raise SystemExit(f"FAIL: pack load failed: {exc}") from None
    finally:
        sys.path.remove(str(COMFYUI_ROOT))

    assert getattr(pack, "__version__", None) == "0.2.0", (
        f"__version__ mismatch: {getattr(pack, '__version__', None)!r}"
    )

    mappings = getattr(pack, "NODE_CLASS_MAPPINGS", None)
    assert mappings is not None, "NODE_CLASS_MAPPINGS missing"

    expected_class_keys = {
        "RetroPixelize",
        "RetroPaletteQuantize",
        "RetroColorDepth",
        "RetroDither",
        "RetroScanlines",
        "RetroSubpixelMask",
        "RetroBloom",
        "RetroChromaticAberration",
        "RetroCurvature",
        "RetroVignette",
        "RetroVHS",
        "RetroLCDDotMatrix",
    }
    assert set(mappings.keys()) == expected_class_keys, (
        f"NODE_CLASS_MAPPINGS keys mismatch: {set(mappings.keys())}"
    )

    display = getattr(pack, "NODE_DISPLAY_NAME_MAPPINGS", None)
    assert display is not None, "NODE_DISPLAY_NAME_MAPPINGS missing"
    expected_display = {
        "RetroPixelize": "Pixelize",
        "RetroPaletteQuantize": "Palette Quantize",
        "RetroColorDepth": "Color Depth",
        "RetroDither": "Dither (8-bit)",
        "RetroScanlines": "Scanlines",
        "RetroSubpixelMask": "Subpixel Mask",
        "RetroBloom": "Bloom (Glow)",
        "RetroChromaticAberration": "Chromatic Aberration",
        "RetroCurvature": "Screen Curvature",
        "RetroVignette": "Vignette",
        "RetroVHS": "VHS (Analog)",
        "RetroLCDDotMatrix": "LCD Dot Matrix",
    }
    assert display == expected_display, (
        f"NODE_DISPLAY_NAME_MAPPINGS mismatch: {display}"
    )

    print("  PASS")

    # ------------------------------------------------------------------ #
    # b) INPUT_TYPES contract per node                                    #
    # ------------------------------------------------------------------ #
    print("--- Check b: INPUT_TYPES has 'image' and 'show_preview' ---")

    required_keys = {"image", "show_preview"}
    for key, cls in mappings.items():
        instance = cls()
        it = instance.INPUT_TYPES()
        assert isinstance(it, dict) and "required" in it, (
            f"{key}.INPUT_TYPES must return dict with 'required'"
        )
        req = it["required"]
        assert required_keys.issubset(req.keys()), (
            f"{key} missing required inputs: {required_keys - set(req.keys())}"
        )

    print("  PASS")

    # ------------------------------------------------------------------ #
    # c) Synthetic tensor                                                 #
    # ------------------------------------------------------------------ #
    print("--- Check c: synthetic batch tensor ---")

    batch = _synthetic_batch(height=72, width=96, batch_size=2)
    assert batch.shape == (2, 72, 96, 3), f"bad batch shape: {batch.shape}"
    assert batch.dtype == torch.float32
    assert float(batch.min()) >= 0.0 and float(batch.max()) <= 1.0
    print("  PASS")

    # ------------------------------------------------------------------ #
    # d) End-to-end run of each node with default-ish params              #
    # ------------------------------------------------------------------ #
    pixelize_node = mappings["RetroPixelize"]()
    quantize_node = mappings["RetroPaletteQuantize"]()
    colordepth_node = mappings["RetroColorDepth"]()
    dither_node = mappings["RetroDither"]()
    scanlines_node = mappings["RetroScanlines"]()
    subpixel_node = mappings["RetroSubpixelMask"]()
    bloom_node = mappings["RetroBloom"]()
    aberration_node = mappings["RetroChromaticAberration"]()
    curvature_node = mappings["RetroCurvature"]()
    vignette_node = mappings["RetroVignette"]()
    vhs_node = mappings["RetroVHS"]()
    lcd_node = mappings["RetroLCDDotMatrix"]()

    expected_shape = (2, 72, 96, 3)

    # --- RetroPixelize: pixel_size=4, sampling="center" (fast) ---
    print("--- Check d1: RetroPixelize end-to-end ---")
    payload_px = pixelize_node.pixelize(
        image=batch,
        pixel_size=4,
        sampling="center",
        pre_blur=0.0,
        show_preview=True,
    )
    assert isinstance(payload_px, dict) and "result" in payload_px
    assert isinstance(payload_px["result"], tuple) and len(payload_px["result"]) == 1
    _check_image_tensor(payload_px["result"][0], expected_shape)
    assert isinstance(payload_px.get("ui"), dict)
    assert "images" in payload_px["ui"]
    print("  PASS")

    # --- RetroPaletteQuantize: default palette, no dither ---
    print("--- Check d2: RetroPaletteQuantize end-to-end ---")
    payload_qz = quantize_node.quantize(
        image=batch,
        palette="pico8_16",
        custom_palette="",
        dither="none",
        dither_strength=1.0,
        serpentine=True,
        show_preview=True,
    )
    assert isinstance(payload_qz, dict) and "result" in payload_qz
    assert isinstance(payload_qz["result"], tuple) and len(payload_qz["result"]) == 1
    _check_image_tensor(payload_qz["result"][0], expected_shape)
    assert isinstance(payload_qz.get("ui"), dict)
    assert "images" in payload_qz["ui"]
    print("  PASS")

    # --- RetroColorDepth: bits mode, value=4 (16 levels) ---
    print("--- Check d3: RetroColorDepth end-to-end ---")
    payload_cd = colordepth_node.posterize(
        image=batch,
        mode="bits",
        value=4,
        show_preview=True,
    )
    assert isinstance(payload_cd, dict) and "result" in payload_cd
    assert isinstance(payload_cd["result"], tuple) and len(payload_cd["result"]) == 1
    _check_image_tensor(payload_cd["result"][0], expected_shape)
    assert isinstance(payload_cd.get("ui"), dict)
    assert "images" in payload_cd["ui"]
    print("  PASS")

    # --- RetroDither: levels=4, bayer4 (fast ordered dither) ---
    print("--- Check d4: RetroDither end-to-end ---")
    payload_dt = dither_node.dither(
        image=batch,
        levels=4,
        mode="color",
        dither_method="bayer4",
        strength=1.0,
        serpentine=True,
        show_preview=True,
    )
    assert isinstance(payload_dt, dict) and "result" in payload_dt
    assert isinstance(payload_dt["result"], tuple) and len(payload_dt["result"]) == 1
    _check_image_tensor(payload_dt["result"][0], expected_shape)
    assert isinstance(payload_dt.get("ui"), dict)
    assert "images" in payload_dt["ui"]
    print("  PASS")

    # --- RetroScanlines: pitch=2, thickness=0.5, strength=0.35 ---
    print("--- Check d5: RetroScanlines end-to-end ---")
    payload_sl = scanlines_node.apply_scanlines(
        image=batch,
        pitch=2,
        thickness=0.5,
        strength=0.35,
        offset=0,
        profile="hard",
        show_preview=True,
    )
    assert isinstance(payload_sl, dict) and "result" in payload_sl
    assert isinstance(payload_sl["result"], tuple) and len(payload_sl["result"]) == 1
    _check_image_tensor(payload_sl["result"][0], expected_shape)
    assert isinstance(payload_sl.get("ui"), dict)
    assert "images" in payload_sl["ui"]
    print("  PASS")

    # --- RetroSubpixelMask: aperture_grille, pitch=1 ---
    print("--- Check d6: RetroSubpixelMask end-to-end ---")
    payload_sp = subpixel_node.apply_mask(
        image=batch,
        mask="aperture_grille",
        pitch=1,
        show_preview=True,
    )
    assert isinstance(payload_sp, dict) and "result" in payload_sp
    assert isinstance(payload_sp["result"], tuple) and len(payload_sp["result"]) == 1
    _check_image_tensor(payload_sp["result"][0], expected_shape)
    assert isinstance(payload_sp.get("ui"), dict)
    assert "images" in payload_sp["ui"]
    print("  PASS")

    # --- RetroBloom: threshold=0.7, radius=3.0, intensity=0.5, blend=screen ---
    print("--- Check d7: RetroBloom end-to-end ---")
    payload_bl = bloom_node.apply_bloom(
        image=batch,
        threshold=0.7,
        radius=3.0,
        intensity=0.5,
        blend="screen",
        show_preview=True,
    )
    assert isinstance(payload_bl, dict) and "result" in payload_bl
    assert isinstance(payload_bl["result"], tuple) and len(payload_bl["result"]) == 1
    _check_image_tensor(payload_bl["result"][0], expected_shape)
    assert isinstance(payload_bl.get("ui"), dict)
    assert "images" in payload_bl["ui"]
    print("  PASS")

    # --- RetroChromaticAberration: amount=2.0, mode=linear ---
    print("--- Check d8: RetroChromaticAberration end-to-end ---")
    payload_ab = aberration_node.apply_aberration(
        image=batch,
        amount=2.0,
        mode="linear",
        show_preview=True,
    )
    assert isinstance(payload_ab, dict) and "result" in payload_ab
    assert isinstance(payload_ab["result"], tuple) and len(payload_ab["result"]) == 1
    _check_image_tensor(payload_ab["result"][0], expected_shape)
    assert isinstance(payload_ab.get("ui"), dict)
    assert "images" in payload_ab["ui"]
    print("  PASS")

    # --- RetroCurvature: amount=0.15, edge_softness=2.0 ---
    print("--- Check d9: RetroCurvature end-to-end ---")
    payload_cv = curvature_node.apply_curvature(
        image=batch,
        amount=0.15,
        edge_softness=2.0,
        show_preview=True,
    )
    assert isinstance(payload_cv, dict) and "result" in payload_cv
    assert isinstance(payload_cv["result"], tuple) and len(payload_cv["result"]) == 2
    _check_image_tensor(payload_cv["result"][0], expected_shape)
    mask_t = payload_cv["result"][1]
    assert mask_t.shape == (batch.shape[0], batch.shape[1], batch.shape[2]), (
        f"MASK shape mismatch: {mask_t.shape}"
    )
    assert mask_t.dtype == torch.float32
    assert bool((mask_t >= 0.0).all()) and bool((mask_t <= 1.0).all())
    assert isinstance(payload_cv.get("ui"), dict)
    assert "images" in payload_cv["ui"]
    print("  PASS")

    # --- RetroVignette: strength=0.4, radius=0.8, softness=0.5 ---
    print("--- Check d10: RetroVignette end-to-end ---")
    payload_vg = vignette_node.apply_vignette(
        image=batch,
        strength=0.4,
        radius=0.8,
        softness=0.5,
        show_preview=True,
    )
    assert isinstance(payload_vg, dict) and "result" in payload_vg
    assert isinstance(payload_vg["result"], tuple) and len(payload_vg["result"]) == 1
    _check_image_tensor(payload_vg["result"][0], expected_shape)
    assert isinstance(payload_vg.get("ui"), dict)
    assert "images" in payload_vg["ui"]
    print("  PASS")

    # --- RetroVHS: noise, chroma_bleed, jitter, band, seed=42 (deterministic) ---
    print("--- Check d11: RetroVHS end-to-end ---")
    payload_vhs = vhs_node.apply_vhs(
        image=batch,
        noise=0.15,
        chroma_bleed=0.5,
        jitter=0.3,
        band=0.3,
        seed=42,
        noise_mode="color",
        show_preview=True,
    )
    assert isinstance(payload_vhs, dict) and "result" in payload_vhs
    assert isinstance(payload_vhs["result"], tuple) and len(payload_vhs["result"]) == 1
    _check_image_tensor(payload_vhs["result"][0], expected_shape)
    assert isinstance(payload_vhs.get("ui"), dict)
    assert "images" in payload_vhs["ui"]
    print("  PASS")

    # --- RetroLCDDotMatrix: cell_size=8, gap_ratio=0.15, grid_dark=0.6 ---
    print("--- Check d12: RetroLCDDotMatrix end-to-end ---")
    payload_lcd = lcd_node.apply_dot_matrix(
        image=batch,
        cell_size=8,
        gap_ratio=0.15,
        grid_dark=0.6,
        shape="square",
        show_preview=True,
    )
    assert isinstance(payload_lcd, dict) and "result" in payload_lcd
    assert isinstance(payload_lcd["result"], tuple) and len(payload_lcd["result"]) == 1
    _check_image_tensor(payload_lcd["result"][0], expected_shape)
    assert isinstance(payload_lcd.get("ui"), dict)
    assert "images" in payload_lcd["ui"]
    print("  PASS")

    # ------------------------------------------------------------------ #
    # e) Preview file check (best effort)                                 #
    # ------------------------------------------------------------------ #
    print("--- Check e: preview files ---")

    try:
        sys.path.insert(0, str(COMFYUI_ROOT))
        try:
            import folder_paths  # type: ignore[import-not-found]
        except ImportError:
            print("  NOTE: folder_paths unavailable, skipping preview file check")
        else:
            # Run RetroPixelize once more and verify temp files exist.
            px_payload = pixelize_node.pixelize(
                image=batch,
                pixel_size=8,
                sampling="mode",
                pre_blur=0.0,
                show_preview=True,
            )
            assert px_payload["ui"]["images"] is not None and len(
                px_payload["ui"]["images"]
            ) > 0, (
                "preview images list must be non-empty when folder_paths available"
            )

            temp_dir = folder_paths.get_temp_directory()
            for img_info in px_payload["ui"]["images"]:
                fname = img_info.get("filename", "")
                if not fname:
                    continue
                fpath = Path(temp_dir) / fname
                assert fpath.exists(), (
                    f"preview file not found: {fpath}"
                )

            print("  PASS")
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"FAIL: preview check raised: {exc}") from None
    finally:
        sys.path.remove(str(COMFYUI_ROOT))

    # ------------------------------------------------------------------ #
    # f) Chained sanity: Pixelize -> Scanlines -> VHS -> Quantize         #
    # ------------------------------------------------------------------ #
    print("--- Check f: chained Pixelize -> Scanlines -> VHS -> Quantize ---")

    chained_px = pixelize_node.pixelize(
        image=batch,
        pixel_size=4,
        sampling="center",
        pre_blur=0.0,
        show_preview=True,
    )
    chained_sl = scanlines_node.apply_scanlines(
        image=chained_px["result"][0],
        pitch=2,
        thickness=0.5,
        strength=0.35,
        offset=0,
        profile="hard",
        show_preview=True,
    )
    chained_vhs = vhs_node.apply_vhs(
        image=chained_sl["result"][0],
        noise=0.15,
        chroma_bleed=0.5,
        jitter=0.3,
        band=0.3,
        seed=42,
        noise_mode="color",
        show_preview=True,
    )
    chained_qz = quantize_node.quantize(
        image=chained_vhs["result"][0],
        palette="dmg_4",
        custom_palette="",
        dither="none",
        dither_strength=1.0,
        serpentine=True,
        show_preview=True,
    )
    _check_image_tensor(chained_qz["result"][0], expected_shape)
    print("  PASS")

    # ------------------------------------------------------------------ #
    # g) Summary                                                          #
    # ------------------------------------------------------------------ #
    print()
    print("SMOKE PASS: all checks passed — pack loads, 12 nodes execute, "
          "preview files written, chain works")


if __name__ == "__main__":
    main()
