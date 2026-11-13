'''
File:   test_skeleton.py
Brief:  Sanity checks for the pack skeleton: alias bootstrap works.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

import sys


def test_pack_alias_registered() -> None:
    '''The conftest bootstrap exposes the pack under a valid package name.'''
    assert "comfyui_lukutar_retro" in sys.modules


def test_core_importable_torch_free() -> None:
    '''core imports on any plain python with numpy + opencv.'''
    from comfyui_lukutar_retro import core

    assert core is not None
