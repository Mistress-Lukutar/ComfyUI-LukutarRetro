'''
File:   test_pixelize.py
Brief:  Unit tests for core/pixelize — Pixelizer class with all pooling methods.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import numpy as np
import pytest
from comfyui_lukutar_retro.core.pixelize import (
    METHOD_CENTER,
    METHOD_MEAN,
    METHOD_MEDIAN,
    METHOD_MODE,
    SAMPLE_METHODS,
    Pixelizer,
)


class TestPixelizerCenter:
    """Tests for the 'center' subsampling method."""

    def test_center_4x4_gradient(self) -> None:
        """4×4 gradient with pixel_size=2 → exact subsample at offset 1 tiled back."""
        # Create a 4×4 grayscale gradient (values spread across [0, 255])
        values = np.linspace(0, 255, 16).reshape(4, 4)
        frame = np.stack([values] * 3, axis=2).astype(np.float32)

        pix = Pixelizer(pixel_size=2, method=METHOD_CENTER)
        result = pix.process(frame)

        # Subsample at offset pixel_size//2 = 1
        subsampled = frame[1::2, 1::2]  # shape (2, 2, 3)

        # Upscale with nearest neighbour: each cell becomes a 2×2 block
        expected = np.repeat(np.repeat(subsampled, 2, axis=0), 2, axis=1)

        assert result.shape == frame.shape
        assert np.allclose(result, expected)


class TestPixelizerMean:
    """Tests for the 'mean' pooling method."""

    def test_mean_constant_cells(self) -> None:
        """4×4 frame where each 2×2 cell has a constant colour → mean preserves it."""
        h, w = 4, 4
        s = 2
        cells_h, cells_w = 2, 2

        # Each cell gets a unique constant colour
        colors = np.array([
            [[100, 50, 200]],
            [[200, 100, 50]],
            [[50, 200, 100]],
            [[250, 250, 0]],
        ], dtype=np.float32)

        frame = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(cells_h):
            for j in range(cells_w):
                frame[i * s:(i + 1) * s, j * s:(j + 1) * s] = colors[i * cells_w + j]

        result = Pixelizer(pixel_size=2, method=METHOD_MEAN).process(frame)

        for i in range(cells_h):
            for j in range(cells_w):
                expected_color = colors[i * cells_w + j]
                assert np.allclose(
                    result[i * s:(i + 1) * s, j * s:(j + 1) * s],
                    expected_color,
                ), f"Cell ({i},{j}) mismatch"

    def test_mean_non_uniform_cell(self) -> None:
        """Cell with values 0, 10, 20, 30 → mean = 15."""
        frame = np.zeros((4, 4, 3), dtype=np.float32)
        frame[0, 0] = [0.0, 0.0, 0.0]
        frame[0, 1] = [10.0, 10.0, 10.0]
        frame[1, 0] = [20.0, 20.0, 20.0]
        frame[1, 1] = [30.0, 30.0, 30.0]

        result = Pixelizer(pixel_size=2, method=METHOD_MEAN).process(frame)
        # Mean of top-left cell = (0 + 10 + 20 + 30) / 4 = 15
        assert np.allclose(result[0:2, 0:2], 15.0)


class TestPixelizerMedian:
    """Tests for the 'median' pooling method."""

    def test_median_outlier(self) -> None:
        """Cell with values [0, 0, 0, 240] → median = 0."""
        frame = np.zeros((4, 4, 3), dtype=np.float32)
        frame[0, 0] = [0, 0, 0]
        frame[0, 1] = [0, 0, 0]
        frame[1, 0] = [0, 0, 0]
        frame[1, 1] = [240, 240, 240]

        result = Pixelizer(pixel_size=2, method=METHOD_MEDIAN).process(frame)
        assert np.allclose(result[0:2, 0:2], [0, 0, 0])


class TestPixelizerMode:
    """Tests for the 'mode' (dominant colour) pooling method."""

    def test_mode_dominant(self) -> None:
        """Cell of 8 pixels at (10,10,10) + 1 pixel at (250,250,250), pixel_size=3."""
        frame = np.zeros((6, 6, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                if i == 2 and j == 2:
                    frame[0 + i, 0 + j] = [250, 250, 250]
                else:
                    frame[0 + i, 0 + j] = [10, 10, 10]

        result = Pixelizer(pixel_size=3, method=METHOD_MODE).process(frame)
        # Dominant bin centroid = mean of 8 binned pixels at (10,10,10)
        assert np.allclose(
            result[0:3, 0:3],
            [10, 10, 10],
            atol=2.0,
        )


class TestPixelizerOddSize:
    """Tests for odd-dimension frames."""

    def test_odd_size_every_method(self) -> None:
        """5x5, pixel_size=2 → output shape (5,5,3) for every method."""
        frame = np.random.rand(5, 5, 3).astype(np.float32) * 255

        for method in SAMPLE_METHODS:
            result = Pixelizer(pixel_size=2, method=method).process(frame)
            assert result.shape == (5, 5, 3), f"Method {method}: shape mismatch"


class TestPixelizerPreBlur:
    """Tests for the pre_blur parameter."""

    def test_pre_blur_runs(self) -> None:
        """pre_blur=2.0 on gradient → runs, shape preserved, values in [0, 255]."""
        frame = np.random.rand(64, 64, 3).astype(np.float32) * 255

        result = Pixelizer(
            pixel_size=4, method=METHOD_MEAN, pre_blur=2.0,
        ).process(frame)

        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 255)


class TestPixelizerIdentity:
    """Tests for identity behaviour."""

    def test_pixel_size_one_pre_blur_zero(self) -> None:
        """pixel_size=1, pre_blur=0 → output == input (allclose)."""
        frame = np.random.rand(64, 64, 3).astype(np.float32) * 255

        result = Pixelizer(
            pixel_size=1, method=METHOD_MEAN, pre_blur=0.0,
        ).process(frame)
        assert np.array_equal(result, frame)


class TestPixelizerInvalidParams:
    """Tests for invalid parameter handling."""

    def test_invalid_pixel_size(self) -> None:
        """pixel_size=0 → ValueError."""
        with pytest.raises(ValueError):
            Pixelizer(pixel_size=0)

    def test_invalid_method(self) -> None:
        """Unknown method → ValueError."""
        with pytest.raises(ValueError):
            Pixelizer(pixel_size=2, method="invalid")

    def test_negative_pre_blur(self) -> None:
        """pre_blur < 0 → ValueError."""
        with pytest.raises(ValueError):
            Pixelizer(pixel_size=2, pre_blur=-1.0)


class TestPixelizerDtypeRange:
    """Tests for output dtype and value range across all methods."""

    def test_dtype_and_range_all_methods(self) -> None:
        """Every method returns float32 with values in [0, 255] for a random frame."""
        frame = np.random.rand(32, 32, 3).astype(np.float32) * 255

        for method in SAMPLE_METHODS:
            result = Pixelizer(pixel_size=4, method=method).process(frame)
            assert result.dtype == np.float32, f"Method {method}: dtype mismatch"
            assert np.all(result >= 0), f"Method {method}: negative values found"
            assert np.all(result <= 255), f"Method {method}: values > 255 found"
