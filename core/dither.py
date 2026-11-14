'''
File:   dither.py
Brief:  Dithering engine — bit-depth reduction with ordered or error-diffusion methods.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Ordered methods (bayer2/4/8) are fully vectorised via ``np.tile``.
Error-diffusion methods (floyd-steinberg, atkinson) use a sequential scanline
pass per channel plane — CPU-bound in CPython (seconds per megapixel), which
is documented behaviour, not a bug.
'''

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Dithering methods.
DITHER_METHODS: tuple[str, ...] = (
    "none", "bayer2", "bayer4", "bayer8",
    "floyd-steinberg", "atkinson",
)
DITHER_NONE: str = "none"
DITHER_BAYER2: str = "bayer2"
DITHER_BAYER4: str = "bayer4"
DITHER_BAYER8: str = "bayer8"
DITHER_FLOYDSTEINBERG: str = "floyd-steinberg"
DITHER_ATKINSON: str = "atkinson"

#: Methods applied as a tileable ordered threshold matrix.
ORDERED_METHODS: tuple[str, ...] = (DITHER_BAYER2, DITHER_BAYER4, DITHER_BAYER8)
#: Methods applied as sequential error diffusion.
DIFFUSION_METHODS: tuple[str, ...] = (DITHER_FLOYDSTEINBERG, DITHER_ATKINSON)

#: Error-diffusion kernels: ``(dx, dy, weight)`` offsets and weights.
DIFFUSION_KERNELS: dict[str, tuple[tuple[int, int, float], ...]] = {
    DITHER_FLOYDSTEINBERG: (
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16),
    ),
    DITHER_ATKINSON: (
        (1, 0, 1 / 8),
        (2, 0, 1 / 8),
        (-1, 1, 1 / 8),
        (0, 1, 1 / 8),
        (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ),
}


def bayer_matrix(size: int) -> np.ndarray:
    """Construct a Bayer threshold matrix of the given size.

    Uses recursive construction from ``M2 = [[0, 2], [3, 1]]``:
    ``M(2n) = [[4M, 4M+2], [4M+3, 4M+1]]``.

    Args:
        size: Matrix dimension — must be 2, 4, or 8.

    Returns:
        uint8 array of shape ``(size, size)`` with values ``0 .. (size*size - 1)``,
        each appearing exactly once.

    Raises:
        ValueError: If *size* is not 2, 4, or 8.
    """
    if size not in (2, 4, 8):
        raise ValueError("bayer_matrix: size must be 2, 4, or 8")

    M = np.array([[0, 2], [3, 1]], dtype=np.uint8)
    if size == 2:
        return M

    n_steps = int(np.log2(size)) - 1
    for _ in range(n_steps):
        M = np.block([
            [4 * M, 4 * M + 2],
            [4 * M + 3, 4 * M + 1],
        ])
    return M.astype(np.uint8)


def dither_frame(
    frame: np.ndarray,
    levels: int,
    method: str,
    strength: float = 1.0,
    serpentine: bool = True,
) -> np.ndarray:
    """Dither a frame to the given number of quantisation levels.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        levels: Number of quantisation levels (>= 2).
        method: One of :data:`DITHER_METHODS`.
        strength: Error diffusion / ordered-dither strength in ``[0, 1]``.
        serpentine: Alternate scanline direction for error-diffusion methods.

    Returns:
        float32 HWC array with values clipped to ``[0, 255]``.

    Raises:
        ValueError: On invalid parameters.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    if method not in DITHER_METHODS:
        raise ValueError(f"method must be one of {DITHER_METHODS}")
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")

    step = 255.0 / (levels - 1)

    def _quantize(x: np.ndarray) -> np.ndarray:
        """Vectorised quantisation to *levels* levels."""
        return np.clip(np.round(x / step) * step, 0, 255).astype(np.float32)

    if method == DITHER_NONE:
        return _quantize(frame)

    if method in ORDERED_METHODS:
        return _dither_ordered(frame, levels, method, strength)

    # Error diffusion
    return _dither_diffusion(frame, levels, method, strength, serpentine)


def _dither_ordered(
    frame: np.ndarray,
    levels: int,
    method: str,
    strength: float,
) -> np.ndarray:
    """Vectorised ordered dithering (Bayer threshold matrix)."""
    step = 255.0 / (levels - 1)

    size_map = {DITHER_BAYER2: 2, DITHER_BAYER4: 4, DITHER_BAYER8: 8}
    size = size_map[method]

    m = bayer_matrix(size).astype(np.float64)
    m_norm = (m + 0.5) / (size * size)
    bias = (m_norm - 0.5) * step * strength

    h, w, _ = frame.shape
    bias_3d = bias[:, :, np.newaxis]  # (size, size, 1) for correct tile broadcasting
    tiled_bias = np.tile(bias_3d, (h // size + 1, w // size + 1, 1))[:h, :w, :]

    quantized = np.clip(frame.astype(np.float64) + tiled_bias, 0, 255)
    return _quantize_ordered(quantized, step)


def _quantize_ordered(x: np.ndarray, step: float) -> np.ndarray:
    """Quantise a float64 array to *levels* levels."""
    return np.clip(np.round(x / step) * step, 0, 255).astype(np.float32)


def _dither_diffusion(
    frame: np.ndarray,
    levels: int,
    method: str,
    strength: float,
    serpentine: bool,
) -> np.ndarray:
    """Sequential error-diffusion dithering (Floyd-Steinberg or Atkinson).

    Scans each channel plane row by row, propagating quantisation error to
    unprocessed neighbours according to the kernel.  For *serpentine=True*
    odd rows are scanned right-to-left and kernel x-offsets are mirrored.
    """
    step = 255.0 / (levels - 1)
    kernel = DIFFUSION_KERNELS[method]

    h, w, _ = frame.shape
    result = frame.astype(np.float64).copy()

    for ch in range(3):
        plane = result[:, :, ch]

        for row in range(h):
            if serpentine and row % 2 == 1:
                # Right-to-left scan, mirror col offsets
                scan_kernel: tuple[tuple[int, int, float], ...] = tuple(
                    (-dc, dr, wt) for dc, dr, wt in kernel
                )
                col_start = w - 1
                col_step = -1
            else:
                scan_kernel = kernel
                col_start = 0
                col_step = 1

            end_col = -1 if col_step == -1 else w
            for col in range(col_start, end_col, col_step):
                old_val = plane[row, col]
                # Clamp to [0, 255]: accumulated error can push the running
                # value past the outer levels; without the clamp the
                # quantized residue would leak oversized errors downstream.
                new_val = min(255.0, max(0.0, round(old_val / step) * step))
                plane[row, col] = new_val
                err = (old_val - new_val) * strength

                for dc, dr, wt in scan_kernel:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        plane[nr, nc] += err * wt

    return np.clip(result, 0, 255).astype(np.float32)
