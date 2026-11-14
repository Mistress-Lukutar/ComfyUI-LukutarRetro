'''
File:   test_crt.py
Brief:  Unit tests for core/crt — scanlines, subpixel mask, bloom, chromatic
        aberration, curvature, and vignette.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import cv2
import numpy as np
import pytest
from comfyui_lukutar_retro.core.crt import (
    BLEND_ADD,
    BLEND_SCREEN,
    MASK_APERTURE_GRILLE,
    MASK_SHADOW_MASK,
    MASK_SLOT_MASK,
    PROFILE_HARD,
    PROFILE_SOFT,
    bloom,
    chromatic_aberration,
    curvature,
    scanlines,
    subpixel_mask,
    vignette,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gradient_frame(h: int = 16, w: int = 8) -> np.ndarray:
    """Build a deterministic HWC gradient frame for testing.

    Channel 0 (R): row index ramp scaled to [0, 255].
    Channel 1 (G): col index ramp scaled to [0, 255].
    Channel 2 (B): sum of both ramps scaled to [0, 255].

    Args:
        h: Height in rows.
        w: Width in columns.

    Returns:
        HWC float32 array with values in ``[0, 255]``.
    """
    row_vals = np.linspace(0, 255, h, dtype=np.float64)[:, np.newaxis]
    col_vals = np.linspace(0, 255, w, dtype=np.float64)[np.newaxis, :]

    # Broadcast to (h, w) for stacking
    r = np.broadcast_to(row_vals / row_vals.max() * 255.0, (h, w))
    g = np.broadcast_to(col_vals / col_vals.max() * 255.0, (h, w))
    b = (row_vals + col_vals) / (row_vals.max() + col_vals.max()) * 255.0

    frame = np.stack([r, g, b], axis=2).astype(np.float32)
    return frame


# ---------------------------------------------------------------------------
# Scanlines — hard profile
# ---------------------------------------------------------------------------


class TestScanlinesHard:
    """Tests for scanlines with PROFILE_HARD."""

    def test_hard_pitch2_strength05_offset0(self) -> None:
        """pitch=2, thickness=0.5, strength=0.5, offset=0
        → row 0 dimmed, row 1 bright."""
        frame = _gradient_frame()
        result = scanlines(
            frame, pitch=2, thickness=0.5,
            strength=0.5, offset=0, profile=PROFILE_HARD,
        )

        # Row 0: p = (0 % 2)/2 = 0 < 0.5 → factor = 1 - 0.5 = 0.5
        assert np.allclose(result[0], frame[0] * 0.5)
        # Row 1: p = (1 % 2)/2 = 0.5, not < 0.5 → factor = 1.0
        assert np.allclose(result[1], frame[1])

    def test_hard_pitch2_offset1(self) -> None:
        """pitch=2, offset=1 → row 1 is dimmed (phase shifted)."""
        frame = _gradient_frame()
        result = scanlines(
            frame, pitch=2, thickness=0.5,
            strength=0.5, offset=1, profile=PROFILE_HARD,
        )

        # Row 1: p = ((1 + 1) % 2)/2 = 0 < 0.5 → factor = 0.5
        assert np.allclose(result[1], frame[1] * 0.5)

    def test_hard_thickness_zero_identity(self) -> None:
        """thickness=0 → no dark band → identity."""
        frame = _gradient_frame()
        result = scanlines(frame, thickness=0.0)
        assert np.allclose(result, frame)

    def test_hard_strength_zero_identity(self) -> None:
        """strength=0 → no dimming → identity."""
        frame = _gradient_frame()
        result = scanlines(frame, strength=0.0)
        assert np.allclose(result, frame)

    def test_hard_thickness1_all_rows_dimmed(self) -> None:
        """thickness=1, pitch=2 → ALL rows scaled by (1 - strength)."""
        frame = _gradient_frame()
        strength = 0.4
        result = scanlines(
            frame, pitch=2, thickness=1.0,
            strength=strength, profile=PROFILE_HARD,
        )

        expected_factor = 1.0 - strength
        assert np.allclose(result, frame * expected_factor)


# ---------------------------------------------------------------------------
# Scanlines — soft profile
# ---------------------------------------------------------------------------


class TestScanlinesSoft:
    """Tests for scanlines with PROFILE_SOFT."""

    def test_soft_pitch2_thickness1_strength05(self) -> None:
        """pitch=2, thickness=1 → every row in dark band, raised-cosine ramp."""
        # Use a constant frame so result / frame = exact per-pixel factor
        frame = np.full((8, 16, 3), 200.0, dtype=np.float32)
        strength = 0.5
        result = scanlines(
            frame, pitch=2, thickness=1.0,
            strength=strength, profile=PROFILE_SOFT,
        )

        # With thickness=1: every p < 1 → cos window over full period
        # min factor = 0.5 at p=0 (cos(0)=1 → 1 - 0.5*0.5*(1+1) = 0.5)
        expected_min = 1.0 - strength
        actual_min = result.min() / frame.max() if frame.max() > 0 else 0
        assert actual_min >= expected_min - 1e-6

        # All factors should be in [0.5, 1.0]
        row_factors = result / (frame + 1e-6)
        assert np.all(row_factors <= 1.0 + 1e-6)
        assert np.all(row_factors >= expected_min - 1e-6)

    def test_soft_factor_broadcast(self) -> None:
        """Soft profile output == orig * per-row factors broadcast over channels."""
        # Use a constant frame so all channels share the same factor
        frame = np.full((8, 16, 3), 200.0, dtype=np.float32)
        result = scanlines(
            frame, pitch=2, thickness=0.5,
            strength=0.3, profile=PROFILE_SOFT,
        )

        # For each row, check that all columns share the same factor
        h, w, _ = frame.shape
        for ch in range(3):
            factors_ch = result[:, :, ch] / (frame[:, :, ch] + 1e-6)
            for row in range(h):
                diffs_row = np.max(np.abs(factors_ch[row, :] - factors_ch[row, 0]))
                assert diffs_row < 1e-5, (
                    f"Row {row} has varying factors: "
                    f"{factors_ch[row, :]}"
                )


# ---------------------------------------------------------------------------
# Scanlines — invalid parameters
# ---------------------------------------------------------------------------


class TestScanlinesInvalid:
    """Tests for invalid scanline parameters."""

    def test_invalid_pitch_zero(self) -> None:
        """pitch=0 → ValueError."""
        with pytest.raises(ValueError):
            scanlines(_gradient_frame(), pitch=0)

    def test_invalid_thickness_over_one(self) -> None:
        """thickness=1.5 → ValueError."""
        with pytest.raises(ValueError):
            scanlines(_gradient_frame(), thickness=1.5)

    def test_invalid_strength_negative(self) -> None:
        """strength=-0.1 → ValueError."""
        with pytest.raises(ValueError):
            scanlines(_gradient_frame(), strength=-0.1)

    def test_invalid_profile_string(self) -> None:
        """profile="weird" → ValueError."""
        with pytest.raises(ValueError):
            scanlines(_gradient_frame(), profile="weird")


# ---------------------------------------------------------------------------
# Subpixel mask — aperture grille
# ---------------------------------------------------------------------------


class TestSubpixelMaskApertureGrille:
    """Tests for subpixel_mask with MASK_APERTURE_GRILLE."""

    def test_grille_pitch1_constant_gray(self) -> None:
        """Constant gray 60, pitch=1:
        R at cols 0==180, col1==0; G at col1==180; B at col2==180."""
        frame = np.full((8, 16, 3), 60.0, dtype=np.float32)
        result = subpixel_mask(frame, mask=MASK_APERTURE_GRILLE, pitch=1)

        # Check row 0 pattern repeats every 3 columns (within bounds)
        for row in range(8):
            for col in range(0, frame.shape[1] - 1, 3):
                # R stripe at col % 3 == 0
                assert result[row, col, 0] == pytest.approx(180.0, abs=0.01), \
                    f"R at ({row},{col}) expected ~180, got {result[row, col, 0]}"
                # G stripe at col % 3 == 1
                assert result[row, col + 1, 1] == pytest.approx(180.0, abs=0.01), \
                    f"G at ({row},{col+1}) expected ~180"
                # B stripe at col % 3 == 2
                if col + 2 < frame.shape[1]:
                    assert result[row, col + 2, 2] == pytest.approx(180.0, abs=0.01), \
                        f"B at ({row},{col+2}) expected ~180"

    def test_grille_pitch1_zero_channels(self) -> None:
        """In aperture grille, non-matching channels are zero."""
        frame = np.full((4, 9, 3), 60.0, dtype=np.float32)
        result = subpixel_mask(frame, mask=MASK_APERTURE_GRILLE, pitch=1)

        # Row 0: col 0 is R stripe → G and B should be 0
        assert result[0, 0, 1] == pytest.approx(0.0, abs=0.01)
        assert result[0, 0, 2] == pytest.approx(0.0, abs=0.01)

    def test_grille_pitch2(self) -> None:
        """pitch=2: stripe width 2 → R at cols 0-1==180, 0 at 2-5."""
        frame = np.full((4, 12, 3), 60.0, dtype=np.float32)
        result = subpixel_mask(frame, mask=MASK_APERTURE_GRILLE, pitch=2)

        # R stripe: cols 0-1 should be 180
        assert result[0, 0, 0] == pytest.approx(180.0, abs=0.01)
        assert result[0, 1, 0] == pytest.approx(180.0, abs=0.01)
        # G stripe: cols 2-3 should be 180 for G channel
        assert result[0, 2, 1] == pytest.approx(180.0, abs=0.01)
        assert result[0, 3, 1] == pytest.approx(180.0, abs=0.01)


# ---------------------------------------------------------------------------
# Subpixel mask — shadow mask
# ---------------------------------------------------------------------------


class TestSubpixelMaskShadowMask:
    """Tests for subpixel_mask with MASK_SHADOW_MASK."""

    def test_shadow_mask_stagger(self) -> None:
        """Row 1 pattern is row 0 pattern shifted by (3*pitch)//2 columns."""
        frame = np.full((4, 9, 3), 60.0, dtype=np.float32)
        pitch = 1
        result = subpixel_mask(frame, mask=MASK_SHADOW_MASK, pitch=pitch)

        shift = (3 * pitch) // 2  # = 1 for pitch=1

        # Compare: row 1 channel factors should equal rolled row 0
        for ch in range(3):
            row0_factors = result[0, :, ch] > 0.5
            row1_expected = np.roll(row0_factors, shift)
            row1_actual = result[1, :, ch] > 0.5
            assert np.array_equal(row1_actual, row1_expected), \
                f"Shadow mask stagger mismatch for channel {ch}"


# ---------------------------------------------------------------------------
# Subpixel mask — slot mask
# ---------------------------------------------------------------------------


class TestSubpixelMaskSlotMask:
    """Tests for subpixel_mask with MASK_SLOT_MASK."""

    def test_slot_mask_bands(self) -> None:
        """pitch=2, rows 0-1 (band 0) full, rows 2-3 (band 1) == band 0 * 0.5."""
        frame = np.full((8, 6, 3), 60.0, dtype=np.float32)
        pitch = 2
        result = subpixel_mask(frame, mask=MASK_SLOT_MASK, pitch=pitch)

        # Band 0 (rows 0-1): full pattern
        band0 = result[0:2]
        # Band 1 (rows 2-3): half pattern
        band1 = result[2:4]

        assert np.allclose(band1, band0 * 0.5)

    def test_slot_mask_band2(self) -> None:
        """Band 2 (rows 4-5) should be full again."""
        frame = np.full((12, 6, 3), 60.0, dtype=np.float32)
        pitch = 2
        result = subpixel_mask(frame, mask=MASK_SLOT_MASK, pitch=pitch)

        band0 = result[0:2]
        band2 = result[4:6]
        assert np.allclose(band2, band0)


# ---------------------------------------------------------------------------
# Subpixel mask — invalid parameters
# ---------------------------------------------------------------------------


class TestSubpixelMaskInvalid:
    """Tests for invalid subpixel_mask parameters."""

    def test_invalid_mask_string(self) -> None:
        """mask="dots" → ValueError."""
        with pytest.raises(ValueError):
            subpixel_mask(_gradient_frame(), mask="dots")

    def test_invalid_pitch_zero(self) -> None:
        """pitch=0 → ValueError."""
        with pytest.raises(ValueError):
            subpixel_mask(_gradient_frame(), pitch=0)


# ---------------------------------------------------------------------------
# Bloom
# ---------------------------------------------------------------------------


class TestBloom:
    """Tests for the bloom function."""

    def test_bloom_screen_constant_250(self) -> None:
        """Constant gray 250, threshold=0.7, radius=3,
        intensity=1.0, blend=screen → [250, 255]."""
        frame = np.full((8, 16, 3), 250.0, dtype=np.float32)
        result = bloom(
            frame, threshold=0.7, radius=3.0,
            intensity=1.0, blend=BLEND_SCREEN,
        )

        assert np.all(result >= 250.0 - 1e-6)
        assert np.all(result <= 255.0 + 1e-6)

    def test_bloom_add_intensity05(self) -> None:
        """blend=add, intensity=0.5 → output >= frame, <= 255."""
        frame = np.full((8, 16, 3), 200.0, dtype=np.float32)
        result = bloom(frame, threshold=0.7, radius=3.0, intensity=0.5, blend=BLEND_ADD)

        assert np.all(result >= frame - 1e-6)
        assert np.all(result <= 255.0 + 1e-6)

    def test_bloom_black_frame(self) -> None:
        """Constant black → mask is 0 → output == 0."""
        frame = np.zeros((8, 16, 3), dtype=np.float32)
        result = bloom(frame, threshold=0.7, radius=3.0, intensity=1.0)

        assert np.allclose(result, 0.0)

    def test_bloom_threshold_one_identity(self) -> None:
        """threshold=1.0 → mask all zeros → output == input."""
        frame = _gradient_frame()
        result = bloom(frame, threshold=1.0)
        assert np.allclose(result, frame)

    def test_bloom_invalid_threshold(self) -> None:
        """threshold=1.5 → ValueError."""
        with pytest.raises(ValueError):
            bloom(_gradient_frame(), threshold=1.5)

    def test_bloom_invalid_radius(self) -> None:
        """radius=0.01 (< 0.05) → ValueError."""
        with pytest.raises(ValueError):
            bloom(_gradient_frame(), radius=0.01)

    def test_bloom_invalid_intensity(self) -> None:
        """intensity=5 (> 4) → ValueError."""
        with pytest.raises(ValueError):
            bloom(_gradient_frame(), intensity=5.0)

    def test_bloom_invalid_blend(self) -> None:
        """blend="mul" → ValueError."""
        with pytest.raises(ValueError):
            bloom(_gradient_frame(), blend="mul")


# ---------------------------------------------------------------------------
# Chromatic aberration — linear
# ---------------------------------------------------------------------------


class TestChromaticAberrationLinear:
    """Tests for chromatic_aberration with linear mode."""

    def test_linear_amount2_gradient(self) -> None:
        """amount=2 on gradient: R col x == input R col (x-2),
        B col x == input B col (x+2)."""
        frame = _gradient_frame()
        result = chromatic_aberration(frame, amount=2.0, mode="linear")

        h, w, _ = frame.shape
        # Interior columns: R[x] ≈ orig_R[x - 2], B[x] ≈ orig_B[x + 2]
        for x in range(2, w - 2):
            assert np.allclose(result[:, x, 0], frame[:, x - 2, 0])
            assert np.allclose(result[:, x, 2], frame[:, x + 2, 2])

    def test_linear_green_identity(self) -> None:
        """Green channel is identity in linear mode."""
        frame = _gradient_frame()
        result = chromatic_aberration(frame, amount=2.0, mode="linear")
        assert np.allclose(result[:, :, 1], frame[:, :, 1])

    def test_linear_amount_zero_identity(self) -> None:
        """amount=0 → identity."""
        frame = _gradient_frame()
        result = chromatic_aberration(frame, amount=0.0)
        assert np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# Chromatic aberration — radial
# ---------------------------------------------------------------------------


class TestChromaticAberrationRadial:
    """Tests for chromatic_aberration with radial mode."""

    def test_radial_amount_zero_identity(self) -> None:
        """amount=0 → identity."""
        frame = _gradient_frame()
        result = chromatic_aberration(frame, amount=0.0, mode="radial")
        assert np.array_equal(result, frame)

    def test_radial_center_pixel_odd_dims(self) -> None:
        """Center pixel of odd-dims frame unchanged for any amount."""
        # 15x9 frame (odd height and width)
        h, w = 9, 15
        cx, cy = w // 2, h // 2
        frame = _gradient_frame(h=h, w=w)

        for amount in [0.5, 2.0, 5.0]:
            result = chromatic_aberration(frame, amount=amount, mode="radial")
            assert np.allclose(result[cy, cx], frame[cy, cx]), \
                f"Center pixel changed at amount={amount}"

    def test_radial_invalid_mode(self) -> None:
        """mode="spin" → ValueError."""
        with pytest.raises(ValueError):
            chromatic_aberration(_gradient_frame(), mode="spin")


# ---------------------------------------------------------------------------
# Chromatic aberration — invalid parameters
# ---------------------------------------------------------------------------


class TestChromaticAberrationInvalid:
    """Tests for invalid chromatic_aberration parameters."""

    def test_invalid_amount_over_20(self) -> None:
        """amount=25 → ValueError."""
        with pytest.raises(ValueError):
            chromatic_aberration(_gradient_frame(), amount=25.0)


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------


class TestCurvature:
    """Tests for the curvature function."""

    def test_curvature_amount_zero_identity(self) -> None:
        """amount=0 → identity, empty mask."""
        frame = _gradient_frame()
        result, mask = curvature(frame, amount=0.0)
        assert np.array_equal(result, frame)
        assert np.all(mask == 0.0)

    def test_curvature_center_unchanged(self) -> None:
        """Exact center pixel unchanged for any non-zero amount."""
        h, w = 9, 15
        cx, cy = w // 2, h // 2
        frame = _gradient_frame(h=h, w=w)

        result, mask = curvature(frame, amount=0.4)
        assert np.allclose(result[cy, cx], frame[cy, cx])
        assert mask[cy, cx] == 0.0

    def test_curvature_edge_midpoints_preserved(self) -> None:
        """Regression: no straight black bands along the frame edges.

        The 1/(1 + amount) zoom compensation must make the midpoints of
        all four edges sample exactly the original edge pixels.
        """
        h, w = 9, 15  # odd dims → exact edge midpoints exist
        frame = _gradient_frame(h=h, w=w)
        cy = h // 2
        cx = w // 2

        result, mask = curvature(frame, amount=0.4)
        assert np.allclose(result[cy, 0], frame[cy, 0])  # left
        assert np.allclose(result[cy, w - 1], frame[cy, w - 1])  # right
        assert np.allclose(result[0, cx], frame[0, cx])  # top
        assert np.allclose(result[h - 1, cx], frame[h - 1, cx])  # bottom
        assert mask[cy, 0] == 0.0
        assert mask[0, cx] == 0.0

    def test_curvature_corner_black(self) -> None:
        """Corner pixel (0, 0) is black (source outside frame)."""
        frame = _gradient_frame()
        result, mask = curvature(frame, amount=0.4)
        assert np.allclose(result[0, 0], 0.0)
        assert mask[0, 0] == 1.0

    def test_curvature_mask_range_and_center(self) -> None:
        """Mask is float32, [0, 1], zero at the centre, positive near corner."""
        frame = _gradient_frame(h=32, w=32)
        result, mask = curvature(frame, amount=0.4)

        assert mask.dtype == np.float32
        assert mask.shape == frame.shape[:2]
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
        assert mask[16, 16] == 0.0
        assert mask[1, 1] > 0.0
        # Blackened pixels really are black
        assert np.allclose(result[mask == 1.0], 0.0)

    def test_curvature_shape_preserved(self) -> None:
        """Output shape matches input."""
        frame = _gradient_frame()
        result, mask = curvature(frame, amount=0.4)
        assert result.shape == frame.shape
        assert mask.shape == frame.shape[:2]

    def test_curvature_invalid_amount_over_one(self) -> None:
        """amount=1.5 → ValueError."""
        with pytest.raises(ValueError):
            curvature(_gradient_frame(), amount=1.5)

    def test_curvature_invalid_softness_negative(self) -> None:
        """edge_softness=-1 → ValueError."""
        with pytest.raises(ValueError):
            curvature(_gradient_frame(), amount=0.4, edge_softness=-1.0)

    def test_curvature_softness_zero_matches_constant_border(self) -> None:
        """edge_softness=0 is identical to the hard constant-border cut.

        The clamped-border remap multiplied by the hard bilinear
        coverage must reproduce the BORDER_CONSTANT remap exactly.
        """
        frame = _gradient_frame(h=32, w=48)
        amount = 0.4
        result, _ = curvature(frame, amount=amount, edge_softness=0.0)

        # Independent reference: straight BORDER_CONSTANT remap.
        h, w = frame.shape[:2]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        xx, yy = np.meshgrid(
            np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)
        )
        u = (xx - cx) / cx
        v = (yy - cy) / cy
        scale = (1.0 + amount * (u * u + v * v)) / (1.0 + amount)
        src_x = cx + (xx - cx) * scale
        src_y = cy + (yy - cy) * scale
        ref = np.zeros_like(frame, dtype=np.float64)
        for ch in range(3):
            ref[:, :, ch] = cv2.remap(
                frame[:, :, ch],
                src_x.astype(np.float32),
                src_y.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        ref = np.clip(ref, 0, 255).astype(np.float32)

        assert np.allclose(result, ref, atol=1e-3)

    def test_curvature_softness_feathers_boundary(self) -> None:
        """edge_softness>0 widens the fade band and shrinks the black area."""
        frame = _gradient_frame(h=64, w=64)

        hard, hard_mask = curvature(frame, amount=0.4, edge_softness=0.0)
        soft, soft_mask = curvature(frame, amount=0.4, edge_softness=6.0)

        hard_black = (hard_mask == 1.0).mean()
        soft_black = (soft_mask == 1.0).mean()
        assert soft_black < hard_black
        # The fractional (feather) band is wider than the hard 1 px ramp
        assert ((soft_mask > 0.0) & (soft_mask < 1.0)).sum() > (
            (hard_mask > 0.0) & (hard_mask < 1.0)
        ).sum()
        # Fully blackened pixels really are black
        assert np.allclose(soft[soft_mask == 1.0], 0.0)

    def test_curvature_softness_keeps_interior(self) -> None:
        """edge_softness>0 leaves pixels away from the border untouched."""
        frame = _gradient_frame(h=64, w=64)
        hard, _ = curvature(frame, amount=0.4, edge_softness=0.0)
        soft, _ = curvature(frame, amount=0.4, edge_softness=6.0)

        # Interior crop: every pixel ≥ 6 px inside the frame edge is
        # untouched by the feather.
        interior = (slice(8, 56), slice(8, 56))
        assert np.allclose(soft[interior], hard[interior])
        assert np.allclose(soft[32, 32], hard[32, 32])


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------


class TestVignette:
    """Tests for the vignette function."""

    def test_vignette_strength_zero_identity(self) -> None:
        """strength=0 → identity."""
        frame = _gradient_frame()
        result = vignette(frame, strength=0.0)
        assert np.allclose(result, frame)

    def test_vignette_radius_one_identity(self) -> None:
        """radius=1.0 → t all zero → identity."""
        frame = _gradient_frame()
        result = vignette(frame, radius=1.0)
        assert np.allclose(result, frame)

    def test_vignette_corner_darkens(self) -> None:
        """strength=0.6, radius=0.5, softness=0 → corner factor < center factor."""
        frame = np.full((16, 32, 3), 128.0, dtype=np.float32)
        result = vignette(frame, strength=0.6, radius=0.5, softness=0.0)

        # Center pixel should be brighter than corner
        h, w = frame.shape[:2]
        center_val = result[h // 2, w // 2, 0]
        corner_val = result[0, 0, 0]

        assert center_val > corner_val + 1e-6, \
            f"Center ({center_val}) should be brighter than corner ({corner_val})"

    def test_vignette_invalid_strength(self) -> None:
        """strength=1.5 → ValueError."""
        with pytest.raises(ValueError):
            vignette(_gradient_frame(), strength=1.5)

    def test_vignette_invalid_radius(self) -> None:
        """radius=1.5 → ValueError."""
        with pytest.raises(ValueError):
            vignette(_gradient_frame(), radius=1.5)

    def test_vignette_invalid_softness(self) -> None:
        """softness=-0.2 → ValueError."""
        with pytest.raises(ValueError):
            vignette(_gradient_frame(), softness=-0.2)


# ---------------------------------------------------------------------------
# Shape / dtype / range for all functions
# ---------------------------------------------------------------------------


class TestAllFunctionsShapeDtypeRange:
    """Every function preserves shape, returns float32, values in [0, 255]."""

    def _random_frame(self, h: int = 16, w: int = 16) -> np.ndarray:
        """Create a random HWC float32 frame in [0, 255]."""
        return (np.random.rand(h, w, 3) * 255.0).astype(np.float32)

    def test_scanlines_shape_dtype_range(self) -> None:
        """scanlines on random frame → shape/dtype/range correct."""
        frame = self._random_frame()
        result = scanlines(frame, pitch=2, thickness=0.5, strength=0.3)
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_subpixel_mask_shape_dtype_range(self) -> None:
        """subpixel_mask on random frame → shape/dtype/range correct."""
        frame = self._random_frame()
        result = subpixel_mask(frame, mask=MASK_APERTURE_GRILLE, pitch=1)
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_bloom_shape_dtype_range(self) -> None:
        """bloom on random frame → shape/dtype/range correct."""
        frame = self._random_frame()
        result = bloom(frame, threshold=0.7, radius=3.0, intensity=0.5)
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_chromatic_aberration_shape_dtype_range(self) -> None:
        """chromatic_aberration on random frame → shape/dtype/range correct."""
        frame = self._random_frame()
        result = chromatic_aberration(frame, amount=2.0)
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_curvature_shape_dtype_range(self) -> None:
        """curvature on random frame → shape/dtype/range correct."""
        frame = self._random_frame()
        result, mask = curvature(frame, amount=0.4)
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 255)
        assert mask.shape == frame.shape[:2]
        assert mask.dtype == np.float32
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

    def test_vignette_shape_dtype_range(self) -> None:
        """vignette on random frame → shape/dtype/range correct."""
        frame = self._random_frame()
        result = vignette(frame, strength=0.4, radius=0.8, softness=0.5)
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 255)
