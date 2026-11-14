'''
File:   test_palettes.py
Brief:  Unit tests for core/palettes — palette constants, get_palette,
        parse_hex_palette.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import numpy as np
import pytest
from comfyui_lukutar_retro.core.palettes import (
    PALETTE_NAMES,
    PALETTES,
    get_palette,
    parse_hex_palette,
)


class TestPaletteCounts:
    """Tests for palette count and coverage."""

    def test_counts_match(self) -> None:
        """len(PALETTES) == len(PALETTE_NAMES) == 10."""
        assert len(PALETTES) == 10
        assert len(PALETTE_NAMES) == 10

    def test_all_names_are_keys(self) -> None:
        """Every name in PALETTE_NAMES is a key of PALETTES."""
        for name in PALETTE_NAMES:
            assert name in PALETTES


class TestPaletteShapeAndRange:
    """Tests for palette shape, dtype, and value range."""

    def test_all_palettes_valid(self) -> None:
        """Every named palette: uint8, (K,3), unique rows, values [0,255]."""
        for name in PALETTE_NAMES:
            arr = get_palette(name)
            assert arr.dtype == np.uint8
            assert arr.ndim == 2
            assert arr.shape[1] == 3
            # All rows unique
            _, counts = np.unique(arr, axis=0, return_counts=True)
            assert np.all(counts == 1), f"{name}: duplicate rows found"
            # Values within [0, 255]
            assert np.all(arr >= 0) and np.all(arr <= 255)


class TestPaletteExactK:
    """Tests for exact palette sizes."""

    EXPECTED_K: dict[str, int] = {
        "dmg_4": 4,
        "pico8_16": 16,
        "c64_16": 16,
        "cga_16": 16,
        "zx_spectrum_15": 15,
        "ega_64": 64,
        "nes": 55,
        "gray_4": 4,
        "mono_1bit": 2,
        "sweetie_16": 16,
    }

    @pytest.mark.parametrize("name,expected_k", EXPECTED_K.items())
    def test_palette_size(self, name: str, expected_k: int) -> None:
        """Exact K per palette matches spec."""
        arr = get_palette(name)
        assert arr.shape[0] == expected_k


class TestDmg4Exact:
    """Tests for exact dmg_4 palette values."""

    def test_dmg4_rows(self) -> None:
        """dmg_4 rows match the expected RGB tuples."""
        arr = get_palette("dmg_4")
        expected = np.array([
            [15, 56, 15],
            [48, 98, 48],
            [139, 172, 15],
            [155, 188, 15],
        ], dtype=np.uint8)
        assert np.array_equal(arr, expected)


class TestGetPaletteCopy:
    """Tests that get_palette returns a copy."""

    def test_mutation_does_not_affect_palletes(self) -> None:
        """Mutating the result does not change PALETTES internals."""
        arr1 = get_palette("dmg_4")
        arr2 = get_palette("dmg_4")
        arr1[0, 0] = 99
        assert arr2[0, 0] != 99


class TestGetPaletteUnknown:
    """Tests for unknown palette name handling."""

    def test_unknown_name_raises(self) -> None:
        """Unknown palette name → ValueError."""
        with pytest.raises(ValueError):
            get_palette("nonexistent")


class TestParseHexPalette:
    """Tests for parse_hex_palette function."""

    def test_multi_token_mixed_separators(self) -> None:
        """Mixed separators produce exact 4 rows, case-insensitive."""
        result = parse_hex_palette("#0f380f, #306230\n#8BAC0F; 9BBC0F")
        assert result.shape == (4, 3)
        expected = np.array([
            [15, 56, 15],
            [48, 98, 48],
            [139, 172, 15],
            [155, 188, 15],
        ], dtype=np.uint8)
        assert np.array_equal(result, expected)

    def test_single_token(self) -> None:
        """Single '#ff0000' → (1, 3)."""
        result = parse_hex_palette("#ff0000")
        assert result.shape == (1, 3)
        assert np.array_equal(result, [[255, 0, 0]])

    @pytest.mark.parametrize("text", ["", "   ", ";;,,"])
    def test_empty_input_raises(self, text: str) -> None:
        """Empty / whitespace-only / separator-only input → ValueError."""
        with pytest.raises(ValueError):
            parse_hex_palette(text)

    def test_invalid_hex_digits(self) -> None:
        """'zzzzzz' → ValueError."""
        with pytest.raises(ValueError):
            parse_hex_palette("zzzzzz")

    def test_wrong_digit_count_5(self) -> None:
        """'12345' (5 digits) → ValueError."""
        with pytest.raises(ValueError):
            parse_hex_palette("12345")

    def test_three_digit_shorthand_raises(self) -> None:
        """'f0a' (3-digit) → ValueError."""
        with pytest.raises(ValueError):
            parse_hex_palette("f0a")
