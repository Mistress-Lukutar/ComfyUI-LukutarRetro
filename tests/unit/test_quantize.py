'''
File:   test_quantize.py
Brief:  Unit tests for core/quantize — make_palette_lut and PaletteQuantizer.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import numpy as np
import pytest
from comfyui_lukutar_retro.core.dither import DIFFUSION_KERNELS
from comfyui_lukutar_retro.core.palettes import get_palette
from comfyui_lukutar_retro.core.quantize import (
    LUT_BIN_SHIFT,
    PaletteQuantizer,
    make_palette_lut,
)

# Preload dmg_4 palette for reuse across tests.
_DMG4 = get_palette("dmg_4")


class TestMakePaletteLut:
    """Tests for the make_palette_lut function."""

    def test_shape_and_dtype(self) -> None:
        """Output is (32, 32, 32, 3) uint8."""
        lut = make_palette_lut(_DMG4)
        assert lut.shape == (32, 32, 32, 3)
        assert lut.dtype == np.uint8

    def test_entries_are_palette_colors(self) -> None:
        """Every LUT entry is one of the 4 dmg_4 colours."""
        lut = make_palette_lut(_DMG4)
        flat = lut.reshape(-1, 3)
        for row in flat:
            assert np.any(np.all(row == _DMG4, axis=1)), (
                f"LUT entry {row} not in palette"
            )

    def test_invalid_palette_shape(self) -> None:
        """Non-(K,3) palette → ValueError."""
        with pytest.raises(ValueError):
            make_palette_lut(np.zeros((3,), dtype=np.uint8))


class TestPaletteQuantizerNone:
    """Tests for the 'none' dither method."""

    def test_exact_palette_color(self) -> None:
        """Constant frame of an exact palette color maps to itself everywhere."""
        q = PaletteQuantizer(_DMG4, dither="none")
        frame = np.full((16, 16, 3), [15, 56, 15], dtype=np.float32)
        result = q.process(frame)
        assert np.allclose(result, [15.0, 56.0, 15.0])

    def test_random_frame_pixels_in_palette(self) -> None:
        """Every output pixel in a random frame is a palette row."""
        q = PaletteQuantizer(_DMG4, dither="none")
        frame = np.random.rand(16, 16, 3).astype(np.float32) * 255
        result = q.process(frame)

        flat_result = result.reshape(-1, 3)
        for pixel in flat_result:
            found = False
            for pal_row in _DMG4:
                if np.allclose(pixel, pal_row.astype(np.float32)):
                    found = True
                    break
            assert found, f"Pixel {pixel} not in palette"

    def test_shape_dtype_preserved(self) -> None:
        """Output shape and dtype match input."""
        q = PaletteQuantizer(_DMG4, dither="none")
        frame = np.random.rand(8, 8, 3).astype(np.float32) * 255
        result = q.process(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.float32


class TestPaletteQuantizerOrdered:
    """Tests for ordered (Bayer) dithering."""

    def test_bayer4_constant_splits(self) -> None:
        """Bayer4 on constant 128 with dmg palette → ≥ 2 distinct output colours."""
        q = PaletteQuantizer(_DMG4, dither="bayer4", strength=1.0)
        frame = np.full((16, 16, 3), [128.0, 128.0, 128.0], dtype=np.float32)
        result = q.process(frame)

        # Every pixel must be a palette colour
        flat_result = result.reshape(-1, 3)
        for pixel in flat_result:
            found = False
            for pal_row in _DMG4:
                if np.allclose(pixel, pal_row.astype(np.float32)):
                    found = True
                    break
            assert found

        # Must contain ≥ 2 distinct palette colours
        unique_cols = set()
        for pixel in flat_result:
            for i, pal_row in enumerate(_DMG4):
                if np.allclose(pixel, pal_row.astype(np.float32)):
                    unique_cols.add(i)
                    break
        assert len(unique_cols) >= 2

    def test_bayer4_deterministic(self) -> None:
        """Two identical calls produce identical results."""
        q = PaletteQuantizer(_DMG4, dither="bayer4", strength=1.0)
        frame = np.random.rand(8, 8, 3).astype(np.float32) * 255
        r1 = q.process(frame)
        r2 = q.process(frame)
        assert np.array_equal(r1, r2)


class TestPaletteQuantizerDiffusion:
    """Tests for error-diffusion dithering."""

    def _gradient_frame(self, size: int = 16) -> np.ndarray:
        """Create a horizontal gradient frame (H, W, 3)."""
        grad = (
            np.linspace(0, 255, size).astype(np.float32)
            [:, np.newaxis, np.newaxis]
        )  # (size, 1, 1)
        return np.broadcast_to(grad, (size, size, 3)).copy()

    def test_floyd_steinberg_palette_subset(self) -> None:
        """FS on gradient → every pixel in palette."""
        q = PaletteQuantizer(_DMG4, dither="floyd-steinberg", strength=1.0)
        frame = self._gradient_frame(16)
        result = q.process(frame)

        flat_result = result.reshape(-1, 3)
        for pixel in flat_result:
            found = False
            for pal_row in _DMG4:
                if np.allclose(pixel, pal_row.astype(np.float32)):
                    found = True
                    break
            assert found, f"Pixel {pixel} not in palette"

    def test_floyd_steinberg_deterministic(self) -> None:
        """FS is deterministic."""
        q1 = PaletteQuantizer(_DMG4, dither="floyd-steinberg", strength=1.0)
        q2 = PaletteQuantizer(_DMG4, dither="floyd-steinberg", strength=1.0)
        frame = self._gradient_frame(8)
        assert np.array_equal(q1.process(frame), q2.process(frame))

    def test_floyd_steinberg_serpentine_false(self) -> None:
        """serpentine=False runs and is deterministic."""
        q = PaletteQuantizer(
            _DMG4, dither="floyd-steinberg", strength=1.0, serpentine=False,
        )
        frame = self._gradient_frame(8)
        r1 = q.process(frame)
        r2 = q.process(frame)
        assert np.array_equal(r1, r2)

    def test_strength_zero_equals_none(self) -> None:
        """strength=0.0 diffusion equals 'none' output."""
        frame = self._gradient_frame(8)
        q_none = PaletteQuantizer(_DMG4, dither="none")
        q_diff = PaletteQuantizer(
            _DMG4, dither="floyd-steinberg", strength=0.0,
        )
        assert np.allclose(q_none.process(frame), q_diff.process(frame))

    def test_atkinson_palette_subset(self) -> None:
        """Atkinson on 8x8 random frame → every pixel in palette."""
        q = PaletteQuantizer(_DMG4, dither="atkinson", strength=1.0)
        frame = np.random.rand(8, 8, 3).astype(np.float32) * 255
        result = q.process(frame)

        flat_result = result.reshape(-1, 3)
        for pixel in flat_result:
            found = False
            for pal_row in _DMG4:
                if np.allclose(pixel, pal_row.astype(np.float32)):
                    found = True
                    break
            assert found, f"Pixel {pixel} not in palette"


class TestQuantizerShapeDtypeRange:
    """Tests for output shape, dtype, and value range across all methods."""

    def test_all_methods_preserve_shape_dtype_range(self) -> None:
        """All DITHER_METHODS return float32 in [0, 255] with correct shape."""
        from comfyui_lukutar_retro.core.dither import DITHER_METHODS
        frame = np.random.rand(16, 16, 3).astype(np.float32) * 255

        for method in DITHER_METHODS:
            q = PaletteQuantizer(_DMG4, dither=method)
            result = q.process(frame)
            assert result.shape == frame.shape, f"{method}: shape mismatch"
            assert result.dtype == np.float32, f"{method}: dtype mismatch"
            assert np.all(result >= 0), f"{method}: negative values"
            assert np.all(result <= 255), f"{method}: values > 255"


class TestQuantizerInvalidParams:
    """Tests for invalid parameter handling."""

    def test_empty_palette(self) -> None:
        """Palette with 0 rows → ValueError."""
        with pytest.raises(ValueError):
            PaletteQuantizer(np.zeros((0, 3), dtype=np.uint8))

    def test_wrong_ndim_palette(self) -> None:
        """1-D palette → ValueError."""
        with pytest.raises(ValueError):
            PaletteQuantizer(np.array([1, 2, 3], dtype=np.uint8))

    def test_unknown_dither_method(self) -> None:
        """Unknown dither method → ValueError."""
        with pytest.raises(ValueError):
            PaletteQuantizer(_DMG4, dither="invalid")

    def test_strength_too_large(self) -> None:
        """strength=1.5 → ValueError."""
        with pytest.raises(ValueError):
            PaletteQuantizer(_DMG4, dither="none", strength=1.5)

    def test_strength_negative(self) -> None:
        """strength=-0.1 → ValueError."""
        with pytest.raises(ValueError):
            PaletteQuantizer(_DMG4, dither="none", strength=-0.1)


class TestAmplitudeCalibration:
    """Tests for the amplitude property."""

    def test_none_method_amplitude_is_none(self) -> None:
        """'none' method → amplitude is None."""
        q = PaletteQuantizer(_DMG4, dither="none")
        assert q.amplitude is None

    def test_ordered_method_amplitude_positive(self) -> None:
        """bayer4 method → amplitude is a positive float."""
        q = PaletteQuantizer(_DMG4, dither="bayer4", strength=1.0)
        amp = q.amplitude
        assert amp is not None
        assert amp > 0


class TestErrorDiffusionSpreads:
    """Regression: error diffusion must actually diffuse (non-zero error).

    A mid-gray constant through the 2-color mono palette with a healthy
    Floyd-Steinberg pass produces the classic two-color checkerboard; a
    zero-error implementation (e.g. diffusing against a stale view into
    the output array) collapses to a single palette color.
    """

    def test_fs_constant_gray_yields_both_palette_colors(self) -> None:
        """Constant gray 128 + mono_1bit + FS → both black and white."""
        palette = get_palette("mono_1bit")
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)
        out = PaletteQuantizer(palette, dither="floyd-steinberg").process(
            frame
        )
        colors = np.unique(out.reshape(-1, 3), axis=0)
        assert colors.shape[0] == 2

    def test_fs_serpentine_false_constant_gray_yields_both(self) -> None:
        """Same spread holds without serpentine scanning."""
        palette = get_palette("mono_1bit")
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)
        out = PaletteQuantizer(
            palette, dither="floyd-steinberg", serpentine=False
        ).process(frame)
        colors = np.unique(out.reshape(-1, 3), axis=0)
        assert colors.shape[0] == 2


class TestPaletteDiffusionStability:
    """Regression: palette diffusion must clamp the running value.

    A palette can be unable to absorb error on a channel — all dmg_4 greens
    are blue≈15, so blue error can never be matched away.  Without clamping
    the running value to ``[0, 255]`` before the LUT match (and diffusing
    the error of the clamped value) the residual grows without bound, the
    LUT bin saturates and bright detail collapses into the surrounding dark
    regions (giant flat blobs instead of a fine dither texture).
    """

    @staticmethod
    def _textbook_diffusion(
        frame: np.ndarray,
        palette: np.ndarray,
        method: str,
        strength: float = 1.0,
        serpentine: bool = True,
    ) -> np.ndarray:
        """Naive per-pixel error-diffusion reference (clamp → match → diffuse)."""
        kernel = DIFFUSION_KERNELS[method]
        lut = make_palette_lut(palette)
        mirrored = tuple((-dx, dy, wt) for dx, dy, wt in kernel)
        h, w, _ = frame.shape
        work = frame.astype(np.float64).copy()
        for y in range(h):
            k = mirrored if (serpentine and y % 2 == 1) else kernel
            xs = range(w - 1, -1, -1) if (serpentine and y % 2 == 1) else range(w)
            for x in xs:
                r, g, b = work[y, x]
                r = 0.0 if r < 0.0 else (255.0 if r > 255.0 else r)
                g = 0.0 if g < 0.0 else (255.0 if g > 255.0 else g)
                b = 0.0 if b < 0.0 else (255.0 if b > 255.0 else b)
                m = lut[
                    int(r) >> LUT_BIN_SHIFT,
                    int(g) >> LUT_BIN_SHIFT,
                    int(b) >> LUT_BIN_SHIFT,
                ].astype(np.float64)
                work[y, x] = m
                er = (r - m[0]) * strength
                eg = (g - m[1]) * strength
                eb = (b - m[2]) * strength
                for dx, dy, wt in k:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        work[ny, nx, 0] += er * wt
                        work[ny, nx, 1] += eg * wt
                        work[ny, nx, 2] += eb * wt
        return np.clip(work, 0, 255).astype(np.float32)

    @pytest.mark.parametrize("method", ["floyd-steinberg", "atkinson"])
    @pytest.mark.parametrize("serpentine", [True, False])
    def test_matches_textbook_reference(
        self, method: str, serpentine: bool,
    ) -> None:
        """Engine diffusion equals the clamp-then-match textbook pass."""
        rng = np.random.default_rng(1234)
        frame = rng.random((24, 18, 3)).astype(np.float32) * 255
        q = PaletteQuantizer(
            _DMG4, dither=method, strength=1.0, serpentine=serpentine,
        )
        expected = self._textbook_diffusion(
            frame, _DMG4, method, serpentine=serpentine,
        )
        assert np.array_equal(q.process(frame), expected)

    def test_preserves_bright_detail_on_degenerate_palette(self) -> None:
        """Bright detail survives FS even though no palette colour can
        absorb blue error."""
        frame = np.full((48, 48, 3), 30.0, dtype=np.float32)
        frame[16:32, 12:36] = (200.0, 150.0, 60.0)
        out = PaletteQuantizer(
            _DMG4, dither="floyd-steinberg", strength=1.0,
        ).process(frame)
        bright = [c.astype(np.float32) for c in _DMG4 if int(c.sum()) > 200]
        region = out[16:32, 12:36].reshape(-1, 3)
        hits = sum(
            1 for px in region if any(np.allclose(px, c) for c in bright)
        )
        assert hits >= 0.5 * region.shape[0]
