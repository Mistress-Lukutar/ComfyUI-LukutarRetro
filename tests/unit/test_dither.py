'''
File:   test_dither.py
Brief:  Unit tests for core/dither — bayer_matrix and dither_frame.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import numpy as np
import pytest
from comfyui_lukutar_retro.core.dither import (
    DITHER_ATKINSON,
    DITHER_BAYER4,
    DITHER_FLOYDSTEINBERG,
    DITHER_METHODS,
    DITHER_NONE,
    ORDERED_METHODS,
    bayer_matrix,
    dither_frame,
)


class TestBayerMatrix:
    """Tests for the bayer_matrix function."""

    def test_bayer_matrix_size_2(self) -> None:
        """Size 2 returns [[0, 2], [3, 1]]."""
        m = bayer_matrix(2)
        expected = np.array([[0, 2], [3, 1]], dtype=np.uint8)
        assert np.array_equal(m, expected)

    def test_bayer_matrix_size_4_shape_and_values(self) -> None:
        """Size 4 has shape (4, 4) with all values 0..15 present exactly once."""
        m = bayer_matrix(4)
        assert m.shape == (4, 4)
        assert m.dtype == np.uint8
        values = sorted(m.ravel().tolist())
        assert values == list(range(16))

    def test_bayer_matrix_size_8_values(self) -> None:
        """Size 8 has shape (8, 8) with all values 0..63 present."""
        m = bayer_matrix(8)
        assert m.shape == (8, 8)
        values = sorted(m.ravel().tolist())
        assert values == list(range(64))

    def test_bayer_matrix_invalid_sizes(self) -> None:
        """Sizes 3 and 0 raise ValueError."""
        for size in [3, 0]:
            with pytest.raises(ValueError):
                bayer_matrix(size)


class TestDitherFrameNone:
    """Tests for the 'none' (plain quantisation) method."""

    def test_none_constant_gray(self) -> None:
        """Constant gray 100, levels=4 → uniform value = round(100/85)*85."""
        frame = np.full((16, 16, 3), 100.0, dtype=np.float32)
        result = dither_frame(frame, levels=4, method=DITHER_NONE)

        step = 255.0 / (4 - 1)  # 85.0
        expected = round(100.0 / step) * step  # round(1.176...) * 85 = 85.0
        assert np.allclose(result, expected)


class TestDitherFrameOrdered:
    """Tests for ordered (Bayer matrix) dithering methods."""

    def test_ordered_strength_zero_equals_none(self) -> None:
        """strength=0.0 produces the same output as 'none' for every ordered method."""
        frame = np.random.rand(16, 16, 3).astype(np.float32) * 255

        for method in ORDERED_METHODS:
            result_zero = dither_frame(frame, levels=4, method=method, strength=0.0)
            expected_none = dither_frame(frame, levels=4, method=DITHER_NONE)
            msg = f"{method} strength=0 mismatch"
            assert np.allclose(result_zero, expected_none), msg

    def test_bayer4_constant_128_levels_2(self) -> None:
        """Bayer4 on constant 128.0 with levels=2 → values in {0, 255}, both present."""
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)
        result = dither_frame(frame, levels=2, method=DITHER_BAYER4, strength=1.0)

        unique_vals = set(np.unique(result[:, :, 0]).tolist())
        assert unique_vals.issubset({0.0, 255.0})
        assert 0.0 in unique_vals and 255.0 in unique_vals

    def test_bayer4_deterministic(self) -> None:
        """Two identical calls produce identical results."""
        frame = np.random.rand(16, 16, 3).astype(np.float32) * 255

        result1 = dither_frame(frame, levels=4, method=DITHER_BAYER4, strength=1.0)
        result2 = dither_frame(frame, levels=4, method=DITHER_BAYER4, strength=1.0)
        assert np.array_equal(result1, result2)


class TestDitherFrameDiffusion:
    """Tests for error-diffusion dithering methods."""

    def test_floyd_steinberg_constant(self) -> None:
        """FS on constant 128.0, levels=2 → values in {0, 255}, both present."""
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)

        result_fs = dither_frame(
            frame, levels=2, method=DITHER_FLOYDSTEINBERG, strength=1.0,
        )
        unique_vals = set(np.unique(result_fs[:, :, 0]).tolist())
        assert unique_vals.issubset({0.0, 255.0})
        assert 0.0 in unique_vals and 255.0 in unique_vals

    def test_floyd_steinberg_serpentine_variants(self) -> None:
        """Both serpentine=True and False run without error."""
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)

        result_true = dither_frame(
            frame, levels=2, method=DITHER_FLOYDSTEINBERG, serpentine=True,
        )
        result_false = dither_frame(
            frame, levels=2, method=DITHER_FLOYDSTEINBERG, serpentine=False,
        )

        assert result_true.shape == frame.shape
        assert result_false.shape == frame.shape

    def test_floyd_steinberg_deterministic(self) -> None:
        """Two identical calls produce identical results."""
        frame = np.random.rand(16, 16, 3).astype(np.float32) * 255

        result1 = dither_frame(frame, levels=4, method=DITHER_FLOYDSTEINBERG)
        result2 = dither_frame(frame, levels=4, method=DITHER_FLOYDSTEINBERG)
        assert np.array_equal(result1, result2)

    def test_atkinson_constant(self) -> None:
        """Atkinson on constant 128.0, levels=2 → runs, values in {0, 255}."""
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)

        result = dither_frame(frame, levels=2, method=DITHER_ATKINSON, strength=1.0)
        unique_vals = set(np.unique(result[:, :, 0]).tolist())
        assert unique_vals.issubset({0.0, 255.0})

    def test_diffusion_strength_half_deterministic(self) -> None:
        """strength=0.5 diffusion runs and is deterministic."""
        frame = np.random.rand(16, 16, 3).astype(np.float32) * 255

        result1 = dither_frame(
            frame, levels=4, method=DITHER_FLOYDSTEINBERG, strength=0.5,
        )
        result2 = dither_frame(
            frame, levels=4, method=DITHER_FLOYDSTEINBERG, strength=0.5,
        )
        assert np.array_equal(result1, result2)


class TestDitherFrameInvalidParams:
    """Tests for invalid parameter handling in dither_frame."""

    def test_levels_too_small(self) -> None:
        """levels=1 → ValueError."""
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)
        with pytest.raises(ValueError):
            dither_frame(frame, levels=1, method=DITHER_NONE)

    def test_unknown_method(self) -> None:
        """Unknown method → ValueError."""
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)
        with pytest.raises(ValueError):
            dither_frame(frame, levels=4, method="invalid")

    def test_strength_too_large(self) -> None:
        """strength=1.5 → ValueError."""
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)
        with pytest.raises(ValueError):
            dither_frame(frame, levels=4, method=DITHER_FLOYDSTEINBERG, strength=1.5)

    def test_strength_negative(self) -> None:
        """strength=-0.1 → ValueError."""
        frame = np.full((16, 16, 3), 128.0, dtype=np.float32)
        with pytest.raises(ValueError):
            dither_frame(frame, levels=4, method=DITHER_FLOYDSTEINBERG, strength=-0.1)


class TestDitherFrameShapeDtypePreserved:
    """Tests that shape and dtype are preserved for every method."""

    def test_shape_dtype_range_all_methods(self) -> None:
        """All DITHER_METHODS preserve shape, return float32 in [0, 255]."""
        frame = np.random.rand(16, 16, 3).astype(np.float32) * 255

        for method in DITHER_METHODS:
            result = dither_frame(frame, levels=4, method=method)
            assert result.shape == frame.shape, f"{method}: shape mismatch"
            assert result.dtype == np.float32, f"{method}: dtype mismatch"
            assert np.all(result >= 0), f"{method}: negative values found"
            assert np.all(result <= 255), f"{method}: values > 255 found"
