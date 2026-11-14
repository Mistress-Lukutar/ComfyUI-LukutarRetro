'''
File:   crt.py
Brief:  CRT/analog-video artifact engine — scanlines, phosphor mask, bloom,
        chromatic aberration, barrel curvature, and vignette.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Frames are ``HWC float32 [0, 255] RGB``; output has the same shape/dtype.
'''

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Scanline brightness profiles.
SCANLINE_PROFILES: tuple[str, ...] = ("hard", "soft")
PROFILE_HARD: str = "hard"
PROFILE_SOFT: str = "soft"

#: Phosphor-mask types (fake CRT / LCD dot structure).
MASK_TYPES: tuple[str, ...] = ("aperture_grille", "shadow_mask", "slot_mask")
MASK_APERTURE_GRILLE: str = "aperture_grille"
MASK_SHADOW_MASK: str = "shadow_mask"
MASK_SLOT_MASK: str = "slot_mask"

#: Bloom blend modes.
BLOOM_BLENDS: tuple[str, ...] = ("screen", "add")
BLEND_SCREEN: str = "screen"
BLEND_ADD: str = "add"

#: Chromatic-aberration displacement modes.
ABERRATION_MODES: tuple[str, ...] = ("radial", "linear")
ABERRATION_RADIAL: str = "radial"
ABERRATION_LINEAR: str = "linear"

#: Brightness compensation for phosphor masks (each channel lights 1/3 of the area).
MASK_BRIGHTNESS_COMPENSATION: float = 3.0

#: Rec. 601 luma weights (also used by the sibling nodes).
LUMA_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)


# ---------------------------------------------------------------------------
# Scanlines
# ---------------------------------------------------------------------------


def scanlines(
    frame: np.ndarray,
    pitch: int = 2,
    thickness: float = 0.5,
    strength: float = 0.35,
    offset: int = 0,
    profile: str = PROFILE_HARD,
) -> np.ndarray:
    """Apply a CRT scanline darkening effect to *frame*.

    Every ``pitch`` rows the darkest ``thickness`` fraction is dimmed by
    *strength*.  The ``profile`` controls whether the transition is abrupt
    (hard) or uses a raised-cosine ramp (soft).

    Args:
        frame: HWC float32 [0, 255] RGB image.
        pitch: Number of rows per scanline period (>= 1).
        thickness: Fraction of the period that is darkened in ``[0, 1]``.
        strength: Maximum dimming factor in ``[0, 1]``.
        offset: Row phase shift applied before the modulo operation.
        profile: Brightness profile — one of :data:`SCANLINE_PROFILES`.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if pitch < 1:
        raise ValueError("pitch must be >= 1")
    if thickness < 0 or thickness > 1:
        raise ValueError("thickness must be in [0, 1]")
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")
    if profile not in SCANLINE_PROFILES:
        raise ValueError(f"profile must be one of {SCANLINE_PROFILES}")

    h = frame.shape[0]
    row_indices = np.arange(h, dtype=np.float64)

    # Row-phase p = ((row + offset) mod pitch) / pitch  →  p in [0, 1)
    p = (row_indices + offset) % pitch / pitch

    if profile == PROFILE_HARD:
        # Hard: factor = 1 - strength when p < thickness, else 1.
        mask = np.where(p < thickness, 1.0 - strength, 1.0)
    else:  # PROFILE_SOFT
        # Soft: raised-cosine dip inside the dark band.
        # factor = 1 - strength * 0.5 * (1 + cos(pi * p / thickness))
        #          when p < thickness
        # Guard thickness == 0 → no darkening; thickness == 1 → full window.
        mask = np.ones(h, dtype=np.float64)
        dark_mask = p < thickness
        if np.any(dark_mask):
            # When thickness == 1, p < 1 so cos covers the full range.
            # Guard: avoid division by zero when thickness == 0
            # (already handled by dark_mask).
            cos_arg = np.pi * p[dark_mask] / max(thickness, 1e-12)
            soft_factor = 1.0 - strength * 0.5 * (1.0 + np.cos(cos_arg))
            mask[dark_mask] = soft_factor

    result = frame.astype(np.float64) * mask[:, np.newaxis, np.newaxis]
    return np.clip(result, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Subpixel mask
# ---------------------------------------------------------------------------


def subpixel_mask(
    frame: np.ndarray,
    mask: str = MASK_APERTURE_GRILLE,
    pitch: int = 1,
) -> np.ndarray:
    """Apply a fake CRT phosphor-mask (or LCD dot-grid) modulation to *frame*.

    The pattern has a period of ``3 * pitch`` horizontally.  Each channel only
    transmits light through its own stripe(s), and the result is multiplied by
    :data:`MASK_BRIGHTNESS_COMPENSATION` to restore overall brightness.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        mask: Phosphor-mask type — one of :data:`MASK_TYPES`.
        pitch: Stripe width in pixels (>= 1).

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if mask not in MASK_TYPES:
        raise ValueError(f"mask must be one of {MASK_TYPES}")
    if pitch < 1:
        raise ValueError("pitch must be >= 1")

    h, w, _ = frame.shape
    period = 3 * pitch

    # Column index array for broadcasting
    col_indices = np.arange(w, dtype=np.float64)
    row_indices = np.arange(h, dtype=np.float64)

    if mask == MASK_APERTURE_GRILLE:
        pattern = _aperture_grille_pattern(h, w, pitch, period, col_indices)

    elif mask == MASK_SHADOW_MASK:
        pattern = _shadow_mask_pattern(h, w, pitch, period, col_indices, row_indices)

    else:  # MASK_SLOT_MASK
        pattern = _slot_mask_pattern(h, w, pitch, period, col_indices, row_indices)

    # Apply pattern and compensate brightness
    result = frame.astype(np.float64) * pattern
    result *= MASK_BRIGHTNESS_COMPENSATION
    return np.clip(result, 0, 255).astype(np.float32)


def _aperture_grille_pattern(
    h: int, w: int, pitch: int, period: int, col_indices: np.ndarray,
) -> np.ndarray:
    """Build an aperture-grille modulation pattern (H, W, 3)."""
    stripe_idx = (col_indices // pitch) % 3  # (W,)
    # Broadcast to (H, W, 3): channel c is 1 where stripe_idx == c
    pattern = np.zeros((h, w, 3), dtype=np.float64)
    for ch in range(3):
        pattern[:, :, ch] = (stripe_idx[np.newaxis, :] == ch).astype(np.float64)
    return pattern


def _shadow_mask_pattern(
    h: int, w: int, pitch: int, period: int,
    col_indices: np.ndarray, row_indices: np.ndarray,
) -> np.ndarray:
    """Build a shadow-mask (staggered triads) modulation pattern (H, W, 3)."""
    shift = (3 * pitch) // 2  # horizontal offset for odd rows
    stripe_idx = (col_indices // pitch) % 3  # (W,)

    pattern = np.zeros((h, w, 3), dtype=np.float64)
    for ch in range(3):
        even_rows = row_indices[:, np.newaxis] % 2 == 0  # (H, 1)
        shifted = (col_indices - shift) // pitch % 3  # (W,)
        pattern[:, :, ch] = np.where(
            even_rows,
            (stripe_idx[np.newaxis, :] == ch).astype(np.float64),
            (shifted[np.newaxis, :] == ch).astype(np.float64),
        )
    return pattern


def _slot_mask_pattern(
    h: int, w: int, pitch: int, period: int,
    col_indices: np.ndarray, row_indices: np.ndarray,
) -> np.ndarray:
    """Build a slot-mask (horizontal-slot gaps) modulation pattern (H, W, 3)."""
    # Base aperture-grille pattern
    base = _aperture_grille_pattern(h, w, pitch, period, col_indices)

    # Every second band of height `pitch` is multiplied by 0.5
    band_idx = row_indices // pitch  # (H,)
    band_mask = np.where(band_idx % 2 == 1, 0.5, 1.0)[:, np.newaxis, np.newaxis]

    return base * band_mask


# ---------------------------------------------------------------------------
# Bloom
# ---------------------------------------------------------------------------


def bloom(
    frame: np.ndarray,
    threshold: float = 0.7,
    radius: float = 8.0,
    intensity: float = 0.5,
    blend: str = BLEND_SCREEN,
) -> np.ndarray:
    """Add a bloom/glow effect to bright areas of *frame*.

    Bright pixels above *threshold* are blurred and blended back into the
    original using either screen or additive blending.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        threshold: Luminance fraction (``[0, 1]``) above which pixels glow.
        radius: Gaussian blur sigma in pixels (>= 0.05).
        intensity: Bloom contribution multiplier in ``[0, 4]``.
        blend: Blend mode — one of :data:`BLOOM_BLENDS`.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be in [0, 1]")
    if radius < 0.05:
        raise ValueError("radius must be >= 0.05")
    if radius > 200:
        raise ValueError("radius must be <= 200")
    if intensity < 0 or intensity > 4:
        raise ValueError("intensity must be in [0, 4]")
    if blend not in BLOOM_BLENDS:
        raise ValueError(f"blend must be one of {BLOOM_BLENDS}")

    h, w, _ = frame.shape

    # Soft bright mask from luma
    luma = np.dot(frame[:, :, :3], LUMA_WEIGHTS)  # (H, W)
    thresh_px = threshold * 255.0
    denom = max(255.0 - thresh_px, 1.0)
    m = np.clip((luma - thresh_px) / denom, 0, 1)

    if threshold >= 1.0:
        # All mask values are zero → output == input
        return frame.copy()

    # Gaussian blur kernel size (odd, minimum 3)
    ksize = max(3, 2 * int(np.ceil(3 * radius)) + 1)
    glow = cv2.GaussianBlur(frame * m[..., np.newaxis], (ksize, ksize), sigmaX=radius)
    glow *= intensity

    if blend == BLEND_SCREEN:
        # out = 255 - (255 - frame) * (255 - glow) / 255
        result = 255.0 - (255.0 - frame.astype(np.float64)) * (255.0 - glow) / 255.0
    else:  # BLEND_ADD
        result = frame.astype(np.float64) + glow

    return np.clip(result, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Chromatic aberration
# ---------------------------------------------------------------------------


def chromatic_aberration(
    frame: np.ndarray,
    amount: float = 2.0,
    mode: str = ABERRATION_RADIAL,
) -> np.ndarray:
    """Apply a chromatic-aberration displacement to *frame*.

    The red channel is displaced rightward (linear mode) / toward the
    center (radial mode) and the blue channel the opposite way, so red
    fringes appear on the right / inner side of edges and blue on the
    left / outer side.  Green remains at its original position.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        amount: Displacement magnitude in pixels (``[0, 20]``).
        mode: Displacement mode — one of :data:`ABERRATION_MODES`.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if amount < 0 or amount > 20:
        raise ValueError("amount must be in [0, 20]")
    if mode not in ABERRATION_MODES:
        raise ValueError(f"mode must be one of {ABERRATION_MODES}")

    if amount == 0.0:
        return frame.copy()

    h, w, _ = frame.shape
    result = np.zeros((h, w, 3), dtype=np.float64)

    if mode == ABERRATION_LINEAR:
        # Linear: R at (x - amount, y), B at (x + amount, y), G identity.
        # Red fringe on the left edge, blue on the right.
        maps_linear = _build_linear_maps(h, w, amount)
        for ch in range(3):
            result[:, :, ch] = cv2.remap(
                frame[:, :, ch],
                maps_linear[ch][0].astype(np.float32),
                maps_linear[ch][1].astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

    else:  # ABERRATION_RADIAL
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        maps_radial = _build_radial_maps(h, w, amount, cx, cy)
        for ch in range(3):
            result[:, :, ch] = cv2.remap(
                frame[:, :, ch],
                maps_radial[ch][0].astype(np.float32),
                maps_radial[ch][1].astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

    return np.clip(result, 0, 255).astype(np.float32)


def _build_linear_maps(
    h: int, w: int, amount: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build remap maps for linear chromatic aberration.

    Returns a list of (map_x, map_y) tuples for R, G, B channels.
    """
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    maps = []
    # Red: sampled at (x - amount, y) → red content shifts right,
    # red fringe on the right edge (blue on the left)
    map_x_r = xx - amount
    map_y_r = yy.copy()
    maps.append((map_x_r, map_y_r))

    # Green: identity
    maps.append((xx.copy(), yy.copy()))

    # Blue: sampled at (x + amount, y) → blue content shifts left
    map_x_b = xx + amount
    map_y_b = yy.copy()
    maps.append((map_x_b, map_y_b))

    return maps


def _build_radial_maps(
    h: int, w: int, amount: float, cx: float, cy: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build remap maps for radial chromatic aberration.

    Returns a list of (map_x, map_y) tuples for R, G, B channels.
    Sampling red outward contracts it toward the center (red displaced
    inward, blue expanded outward).
    """
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Direction vectors from center
    u_x = xx - cx
    u_y = yy - cy
    dist = np.sqrt(u_x * u_x + u_y * u_y)
    dist = np.maximum(dist, 1e-6)

    maps = []
    # Red: sampled outward → contracted toward center (red displaced inward)
    map_x_r = xx + (u_x / dist) * amount
    map_y_r = yy + (u_y / dist) * amount
    maps.append((map_x_r, map_y_r))

    # Green: identity
    maps.append((xx.copy(), yy.copy()))

    # Blue: sampled inward → expanded outward (blue displaced outward)
    map_x_b = xx - (u_x / dist) * amount
    map_y_b = yy - (u_y / dist) * amount
    maps.append((map_x_b, map_y_b))

    return maps


# ---------------------------------------------------------------------------
# Curvature (barrel screen curve)
# ---------------------------------------------------------------------------


def curvature(
    frame: np.ndarray,
    amount: float = 0.15,
) -> np.ndarray:
    """Apply a barrel-distortion (CRT screen curve) to *frame*.

    Pixels near the corners are pulled inward, simulating the convex
    curvature of an old CRT display.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        amount: Distortion strength in ``[0, 1]``. Zero is identity.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if amount < 0 or amount > 1:
        raise ValueError("amount must be in [0, 1]")

    if amount == 0.0:
        return frame.copy()

    h, w, _ = frame.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Normalized coordinates
    u = (xx - cx) / max(cx, 1e-6)
    v = (yy - cy) / max(cy, 1e-6)
    r2 = u * u + v * v

    # Barrel: source coords = center + (pos - center) * (1 + amount * r^2)
    src_x = cx + (xx - cx) * (1.0 + amount * r2)
    src_y = cy + (yy - cy) * (1.0 + amount * r2)

    result = np.zeros((h, w, 3), dtype=np.float64)
    for ch in range(3):
        result[:, :, ch] = cv2.remap(
            frame[:, :, ch],
            src_x.astype(np.float32),
            src_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    return np.clip(result, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------


def vignette(
    frame: np.ndarray,
    strength: float = 0.4,
    radius: float = 0.8,
    softness: float = 0.5,
) -> np.ndarray:
    """Apply a vignette (corner-darkening) effect to *frame*.

    A smooth radial falloff darkens the edges of the image, with the
    transition controlled by *radius* and *softness*.

    Args:
        frame: HWC float32 [0, 255] RGB image.
        strength: Maximum darkening factor in ``[0, 1]``.
        radius: Normalised distance at which darkening begins (``[0, 1]``).
        softness: Controls the steepness of the transition in ``[0, 1]``.

    Returns:
        Processed frame with same shape/dtype as input.

    Raises:
        ValueError: On invalid parameters.
    """
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in [0, 1]")
    if radius < 0 or radius > 1:
        raise ValueError("radius must be in [0, 1]")
    if softness < 0 or softness > 1:
        raise ValueError("softness must be in [0, 1]")

    h, w, _ = frame.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    half_diag = np.sqrt(cx * cx + cy * cy)

    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Normalised distance: 0 at center, 1 at corners
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(half_diag, 1e-6)

    # Transition factor t
    t = np.clip((d - radius) / max(1.0 - radius, 1e-6), 0, 1)

    # Factor: darkening exponent increases with softness
    exponent = 1.0 + softness * 4.0
    factor = 1.0 - strength * (t ** exponent)

    result = frame.astype(np.float64) * factor[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.float32)
