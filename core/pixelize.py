'''
File:   pixelize.py
Brief:  Pixelization engine — reduce resolution via cell pooling and upscale.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Sampling methods for pixel cell pooling.
SAMPLE_METHODS: tuple[str, ...] = ("center", "mean", "median", "mode")
METHOD_CENTER: str = "center"
METHOD_MEAN: str = "mean"
METHOD_MEDIAN: str = "median"
METHOD_MODE: str = "mode"

#: Per-channel bin levels used by mode pooling to merge near-duplicates.
MODE_BIN_LEVELS: int = 16


class Pixelizer:
    """Pixelize an image by downsampling cells and upscaling with nearest neighbour."""

    def __init__(
        self,
        pixel_size: int,
        method: str = METHOD_MODE,
        pre_blur: float = 0.0,
    ) -> None:
        """Initialise the pixelizer.

        Args:
            pixel_size: Cell dimension in pixels (>= 1).
            method: Pooling method from SAMPLE_METHODS.
            pre_blur: Gaussian blur sigma applied before downsampling (>= 0).

        Raises:
            ValueError: On invalid parameters.
        """
        if pixel_size < 1:
            raise ValueError("pixel_size must be >= 1")
        if method not in SAMPLE_METHODS:
            raise ValueError(f"method must be one of {SAMPLE_METHODS}")
        if pre_blur < 0:
            raise ValueError("pre_blur must be >= 0")

        self.pixel_size = pixel_size
        self.method = method
        self.pre_blur = pre_blur

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pixelizer.

        Args:
            frame: HWC float32 [0, 255] RGB image.

        Returns:
            Processed frame with same shape/dtype/range as input.
        """
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HWC with 3 channels")

        h, w, _ = frame.shape
        s = self.pixel_size

        # pixel_size == 1 and pre_blur <= 0 → identity
        if s == 1 and self.pre_blur <= 0:
            return frame.copy()

        work = frame.copy()

        # Pre-blur
        if self.pre_blur > 0:
            sigma = self.pre_blur
            ksize = max(3, 2 * int(np.ceil(3 * sigma)) + 1)
            work = cv2.GaussianBlur(work, (ksize, ksize), sigma)

        # Pad up to a multiple of pixel_size with edge replication
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h > 0 or pad_w > 0:
            work = np.pad(work, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        ph, pw, _ = work.shape
        cells_h = ph // s
        cells_w = pw // s

        # Downsample based on method
        if self.method == METHOD_CENTER:
            reduced = work[s // 2::s, s // 2::s]
        elif self.method == METHOD_MEAN:
            cells = work.reshape(cells_h, s, cells_w, s, 3)
            reduced = cells.mean(axis=(1, 3))
        elif self.method == METHOD_MEDIAN:
            cells = work.reshape(cells_h, s, cells_w, s, 3)
            reduced = np.median(cells, axis=(1, 3))
        else:  # METHOD_MODE
            reduced = self._mode_pool(work, cells_h, cells_w, s)

        # Upscale to original (w, h) with nearest-neighbour interpolation
        result = cv2.resize(reduced, (w, h), interpolation=cv2.INTER_NEAREST)
        result = np.clip(result.astype(np.float32), 0, 255)
        return result

    def _mode_pool(
        self,
        padded: np.ndarray,
        cells_h: int,
        cells_w: int,
        s: int,
    ) -> np.ndarray:
        """Vectorised mode pooling for pixel cells.

        For each cell the dominant colour is found by binning channel values
        into 16 levels per channel (value >> 4), forming a combined code, and
        picking the most frequent code group.  The winning centroid is the
        mean of the actual pixel values in that group.

        Args:
            padded: Frame already padded to a multiple of *s* in both dims.
            cells_h: Number of cells vertically.
            cells_w: Number of cells horizontally.
            s: Pixel size (cell dimension).

        Returns:
            Reduced grid of shape ``(cells_h, cells_w, 3)`` with dominant colours.
        """
        n_cells = cells_h * cells_w

        # Reshape to cell-pixel layout; swap axes so reshape groups pixels per cell
        small_grid = padded.reshape(cells_h, s, cells_w, s, 3)
        flat = (
            np.swapaxes(small_grid, 1, 2)
            .reshape(n_cells, s * s, 3)
            .astype(np.float64)
        )

        # Per-channel bin values (16 levels: value >> 4)
        bb = np.floor(flat[:, :, 0]).astype(np.int32) >> 4
        gb = np.floor(flat[:, :, 1]).astype(np.int32) >> 4
        rb = np.floor(flat[:, :, 2]).astype(np.int32) >> 4

        # Combined code for lexsort: rb*256 + gb*16 + bb
        codes = (rb << 8) | (gb << 4) | bb  # int32, max value 4095

        ids = np.repeat(np.arange(n_cells), s * s)
        codes_flat = codes.ravel()

        # Flatten to (n_pixels, 3) so order indexes into the first axis directly
        flat_raveled = flat.reshape(n_cells * s * s, 3)

        # Sort by (cell_id, code) — lexsort: secondary key first
        order = np.lexsort((codes_flat, ids))
        ids_sorted = ids[order]
        codes_sorted = codes_flat[order]
        pixels_sorted = flat_raveled[order]  # float64, shape (n_pixels, 3)

        # Cumulative sums for segmented-sum computation
        cumsum = np.cumsum(pixels_sorted, axis=0)

        # Find run boundaries (where cell_id or code changes)
        diff_ids = np.diff(ids_sorted)
        diff_codes = np.diff(codes_sorted)
        boundary = (diff_ids != 0) | (diff_codes != 0)
        run_end = np.append(np.where(boundary)[0], len(ids_sorted) - 1)
        run_start = np.concatenate([[0], run_end[:-1] + 1])
        run_count = run_end - run_start + 1

        # Segment sums via cumsum (handle index-0 prefix carefully)
        n_runs = len(run_end)
        segment_sums = np.zeros((n_runs, 3), dtype=np.float64)
        for ch in range(3):
            mask_pos = run_start > 0
            segment_sums[mask_pos, ch] = (
                cumsum[run_end[mask_pos], ch] - cumsum[run_start[mask_pos] - 1, ch]
            )
            segment_sums[~mask_pos, ch] = cumsum[run_end[~mask_pos], ch]

        # Find best run per cell: max count, tie → lowest code
        best_count = np.zeros(n_cells, dtype=np.int32)
        best_sum = np.zeros((n_cells, 3), dtype=np.float64)
        best_code = np.full(n_cells, np.iinfo(np.int32).max, dtype=np.int32)

        for r in range(n_runs):
            pos = run_start[r]
            cell = ids_sorted[pos]
            code_v = codes_sorted[pos]
            cnt = run_count[r]

            if cnt > best_count[cell]:
                best_count[cell] = cnt
                best_sum[cell] = segment_sums[r]
                best_code[cell] = code_v
            elif cnt == best_count[cell] and code_v < best_code[cell]:
                best_sum[cell] = segment_sums[r]
                best_code[cell] = code_v

        # Centroids = sum / count (only for cells with runs)
        centroids = np.zeros((n_cells, 3), dtype=np.float64)
        mask = best_count > 0
        centroids[mask] = best_sum[mask] / best_count[mask][:, None]

        return centroids.reshape(cells_h, cells_w, 3)
