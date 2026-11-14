'''
File:   test_analog.py
Brief:  Unit tests for core/analog — vhs degradation and lcd_dot_matrix.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import numpy as np
import pytest
from comfyui_lukutar_retro.core.analog import (
    NOISE_MONO,
    SHAPE_ROUND,
    SHAPE_SQUARE,
    lcd_dot_matrix,
    vhs,
)
from comfyui_lukutar_retro.core.crt import LUMA_WEIGHTS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _constant_frame(h: int = 16, w: int = 16, val: float = 128.0) -> np.ndarray:
    """Build a constant HWC float32 [0, 255] RGB frame.

    Args:
        h: Height in rows.
        w: Width in columns.
        val: Constant value for all channels.

    Returns:
        HWC float32 array.
    """
    return np.full((h, w, 3), val, dtype=np.float32)


# ---------------------------------------------------------------------------
# VHS — identity / zero-strength
# ---------------------------------------------------------------------------


class TestVhsZeroStrength:
    """All strengths at 0 must yield the input frame."""

    def test_all_strengths_zero_identity(self) -> None:
        """noise=chroma_bleed=jitter=band=0 → output == input for any seed."""
        frame = _constant_frame(16, 16, 128.0)
        for s in (0, 42):
            result = vhs(frame, noise=0, chroma_bleed=0, jitter=0, band=0, seed=s)
            assert np.allclose(result, frame)

    def test_seed_zero_fresh_randomness(self) -> None:
        """seed=0 must produce different outputs on repeated calls."""
        frame = _constant_frame(16, 16, 128.0)
        r1 = vhs(frame, noise=0.1, seed=0)
        r2 = vhs(frame, noise=0.1, seed=0)
        assert not np.allclose(r1, r2)


# ---------------------------------------------------------------------------
# VHS — determinism
# ---------------------------------------------------------------------------


class TestVhsDeterminism:
    """Same seed → identical; different seed → different (when randomness matters)."""

    def test_same_seed_identical(self) -> None:
        """seed=7 twice with noise>0 → identical outputs."""
        frame = _constant_frame(16, 16, 128.0)
        r1 = vhs(frame, noise=0.5, seed=7)
        r2 = vhs(frame, noise=0.5, seed=7)
        assert np.array_equal(r1, r2)

    def test_different_seed_different(self) -> None:
        """seed=7 vs seed=8 with noise>0 → different outputs."""
        frame = _constant_frame(16, 16, 128.0)
        r7 = vhs(frame, noise=0.5, seed=7)
        r8 = vhs(frame, noise=0.5, seed=8)
        assert not np.array_equal(r7, r8)


# ---------------------------------------------------------------------------
# VHS — noise only
# ---------------------------------------------------------------------------


class TestVhsNoiseOnly:
    """Noise-only pipeline tests."""

    def test_noise_changes_frame(self) -> None:
        """noise>0 with others=0 → output != input."""
        frame = _constant_frame(16, 16, 128.0)
        result = vhs(frame, noise=0.3, chroma_bleed=0, jitter=0, band=0, seed=42)
        assert not np.allclose(result, frame)

    def test_noise_max_diff_bound(self) -> None:
        """max abs diff <= noise * 40 * 3 + 1."""
        frame = _constant_frame(16, 16, 128.0)
        n = 0.5
        result = vhs(frame, noise=n, chroma_bleed=0, jitter=0, band=0, seed=42)
        max_diff = np.max(np.abs(result.astype(np.float64) - frame.astype(np.float64)))
        assert max_diff <= n * 40 * 3 + 1

    def test_noise_shape_preserved(self) -> None:
        """Output shape/dtype matches input."""
        frame = _constant_frame(16, 16, 128.0)
        result = vhs(frame, noise=0.5, seed=42)
        assert result.shape == frame.shape
        assert result.dtype == np.float32

    def test_noise_mono_mode_equal_diffs(self) -> None:
        """Mono noise: diff planes for R and G are equal on a constant frame."""
        frame = _constant_frame(16, 16, 128.0)
        result = vhs(
            frame, noise=0.3, chroma_bleed=0, jitter=0,
            band=0, seed=42, noise_mode=NOISE_MONO,
        )
        diff_r = result[:, :, 0].astype(np.float64) - frame[:, :, 0].astype(np.float64)
        diff_g = result[:, :, 1].astype(np.float64) - frame[:, :, 1].astype(np.float64)
        assert np.allclose(diff_r, diff_g)


# ---------------------------------------------------------------------------
# VHS — chroma bleed only
# ---------------------------------------------------------------------------


class TestVhsChromaBleed:
    """Chroma-bleed pipeline tests."""

    def _red_blue_frame(self, h: int = 16, w: int = 32) -> np.ndarray:
        """Left half pure red (255,0,0), right half pure blue (0,0,255)."""
        frame = np.zeros((h, w, 3), dtype=np.float32)
        frame[:, :w // 2, 0] = 255.0  # red channel for left half
        frame[:, w // 2:, 2] = 255.0  # blue channel for right half
        return frame

    def test_chroma_bleed_luma_preserved(self) -> None:
        """With bleed=1.0, Y (luma) of every pixel stays within 15.0 of original."""
        frame = self._red_blue_frame(16, 32)
        result = vhs(frame, noise=0, chroma_bleed=1.0, jitter=0, band=0, seed=42)
        luma_in = np.dot(frame.astype(np.float64), LUMA_WEIGHTS)
        luma_out = np.dot(result.astype(np.float64), LUMA_WEIGHTS)
        assert np.all(np.abs(luma_out - luma_in) < 15.0)

    def test_chroma_bleed_changes_output(self) -> None:
        """With bleed=1.0, output != input."""
        frame = self._red_blue_frame(16, 32)
        result = vhs(frame, noise=0, chroma_bleed=1.0, jitter=0, band=0, seed=42)
        assert not np.allclose(result, frame)


# ---------------------------------------------------------------------------
# VHS — jitter only
# ---------------------------------------------------------------------------


class TestVhsJitterOnly:
    """Jitter pipeline tests."""

    def test_jitter_changes_frame(self) -> None:
        """jitter=1 with others=0 → output != input."""
        h, w = 16, 16
        frame = np.zeros((h, w, 3), dtype=np.float32)
        # Vertical stripe pattern at column 0 (non-constant frame)
        frame[:, 0, :] = 255.0
        result = vhs(frame, noise=0, chroma_bleed=0, jitter=1, band=0, seed=42)
        assert not np.array_equal(result, frame)

    def test_jitter_displaces_rows(self) -> None:
        """Vertical stripe pattern: at least one row differs from input."""
        h, w = 16, 16
        frame = np.zeros((h, w, 3), dtype=np.float32)
        # Vertical stripe at column 0
        frame[:, 0, :] = 255.0
        result = vhs(frame, noise=0, chroma_bleed=0, jitter=1, band=0, seed=42)
        # At least one row must differ (shifted stripe)
        any_row_diff = False
        for r in range(h):
            if not np.allclose(result[r], frame[r]):
                any_row_diff = True
                break
        assert any_row_diff


# ---------------------------------------------------------------------------
# VHS — band only
# ---------------------------------------------------------------------------


class TestVhsBandOnly:
    """Band pipeline tests."""

    def test_band_only_brightens(self) -> None:
        """band>0, others=0 → output >= input - 1e-3 everywhere on mid-gray frame."""
        frame = _constant_frame(48, 32, 128.0)
        result = vhs(frame, noise=0, chroma_bleed=0, jitter=0, band=1, seed=42)
        assert np.all(result >= frame - 1e-3)

    def test_band_max_increases(self) -> None:
        """band>0 on mid-gray → output.max() > input.max()."""
        frame = _constant_frame(48, 32, 128.0)
        result = vhs(frame, noise=0, chroma_bleed=0, jitter=0, band=1, seed=42)
        assert result.max() > frame.max()


# ---------------------------------------------------------------------------
# VHS — invalid params
# ---------------------------------------------------------------------------


class TestVhsInvalidParams:
    """Must raise ValueError."""

    def test_noise_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            vhs(_constant_frame(16, 16), noise=1.5)

    def test_jitter_negative(self) -> None:
        with pytest.raises(ValueError):
            vhs(_constant_frame(16, 16), jitter=-0.1)

    def test_seed_negative(self) -> None:
        with pytest.raises(ValueError):
            vhs(_constant_frame(16, 16), seed=-1)

    def test_noise_mode_invalid(self) -> None:
        with pytest.raises(ValueError):
            vhs(_constant_frame(16, 16), noise_mode="weird")


# ---------------------------------------------------------------------------
# LCD dot matrix — identity cases
# ---------------------------------------------------------------------------


class TestLcdDotMatrixIdentity:
    """gap_ratio=0 or grid_dark=0 must be identity on constant frames."""

    def test_gap_ratio_zero_identity(self) -> None:
        """gap_ratio=0 → identity (constant gray frame)."""
        frame = _constant_frame(48, 32, 200.0)
        result = lcd_dot_matrix(frame, cell_size=8, gap_ratio=0, grid_dark=0.6)
        assert np.allclose(result, frame)

    def test_grid_dark_zero_identity(self) -> None:
        """grid_dark=0 → identity (constant gray frame)."""
        frame = _constant_frame(48, 32, 200.0)
        result = lcd_dot_matrix(frame, cell_size=8, gap_ratio=0.15, grid_dark=0)
        assert np.allclose(result, frame)


# ---------------------------------------------------------------------------
# LCD dot matrix — square shape
# ---------------------------------------------------------------------------


class TestLcdDotMatrixSquare:
    """Square-dot pattern tests."""

    def test_square_pattern_columns(self) -> None:
        """cell_size=4, gap_ratio=0.5, grid_dark=1.0, gray 200:
        cols follow dark,bright,bright,dark per cell — col 0==0, col 1==200,
        col 2==200, col 3==0 (repeats at col 4==0)."""
        h, w = 16, 12
        frame = _constant_frame(h, w, 200.0)
        result = lcd_dot_matrix(
            frame, cell_size=4, gap_ratio=0.5,
            grid_dark=1.0, shape=SHAPE_SQUARE,
        )
        # row 1: fy=(1%4+0.5)/4=0.375, inside active region [0.25, 0.75]
        assert result[1, 0, 0] == pytest.approx(0.0, abs=1e-3)  # dark
        assert result[1, 1, 0] == pytest.approx(200.0, abs=1e-3)  # bright
        assert result[1, 2, 0] == pytest.approx(200.0, abs=1e-3)  # bright
        assert result[1, 3, 0] == pytest.approx(0.0, abs=1e-3)  # dark
        assert result[1, 4, 0] == pytest.approx(0.0, abs=1e-3)  # next cell col 0


# ---------------------------------------------------------------------------
# LCD dot matrix — round shape
# ---------------------------------------------------------------------------


class TestLcdDotMatrixRound:
    """Round-dot pattern tests."""

    def test_round_center_and_corner(self) -> None:
        """cell_size=8, gap_ratio=0.5, grid_dark=1.0, gray 200:
        pixel nearest cell center is 200, cell corner pixel is 0."""
        h, w = 16, 16
        frame = _constant_frame(h, w, 200.0)
        result = lcd_dot_matrix(
            frame, cell_size=8, gap_ratio=0.5,
            grid_dark=1.0, shape=SHAPE_ROUND,
        )

        # Cell (0,0) center: col 4 row 4 → fx=(4+0.5)/8=0.5625, fy=0.5625
        # |2*(0.5625-0.5)| = 0.125 → r^2 = 0.03125 <= 0.25 → inside
        assert result[4, 4, 0] == pytest.approx(200.0, abs=1e-3)

        # Cell (0,0) corner: col 0 row 0 → fx=(0+0.5)/8=0.0625, fy=0.0625
        # |2*(0.0625-0.5)| = 0.875 → r^2 = 0.765625 > 0.25 → gap
        assert result[0, 0, 0] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# LCD dot matrix — invalid params
# ---------------------------------------------------------------------------


class TestLcdDotMatrixInvalidParams:
    """Must raise ValueError."""

    def test_cell_size_zero(self) -> None:
        with pytest.raises(ValueError):
            lcd_dot_matrix(_constant_frame(16, 16), cell_size=0)

    def test_gap_ratio_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            lcd_dot_matrix(_constant_frame(16, 16), gap_ratio=0.6)

    def test_grid_dark_negative(self) -> None:
        with pytest.raises(ValueError):
            lcd_dot_matrix(_constant_frame(16, 16), grid_dark=-0.1)

    def test_shape_invalid(self) -> None:
        with pytest.raises(ValueError):
            lcd_dot_matrix(_constant_frame(16, 16), shape="tri")


# ---------------------------------------------------------------------------
# Shape / dtype / range preservation
# ---------------------------------------------------------------------------


class TestShapeDtypeRangePreservation:
    """Both functions preserve shape, dtype, and value range."""

    def _random_frame(self, h: int = 48, w: int = 32) -> np.ndarray:
        return (np.random.RandomState(99).rand(h, w, 3) * 255).astype(np.float32)

    def test_vhs_preserves_shape_dtype_range(self) -> None:
        frame = self._random_frame(48, 32)
        result = vhs(frame, noise=0.3, chroma_bleed=0.2, jitter=0.1, band=0.1, seed=42)
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert result.min() >= 0 - 1e-6
        assert result.max() <= 255 + 1e-6

    def test_lcd_dot_matrix_preserves_shape_dtype_range(self) -> None:
        frame = self._random_frame(48, 32)
        result = lcd_dot_matrix(frame, cell_size=8, gap_ratio=0.15, grid_dark=0.6)
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert result.min() >= 0 - 1e-6
        assert result.max() <= 255 + 1e-6
