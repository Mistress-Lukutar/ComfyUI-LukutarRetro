'''
File:   __init__.py
Brief:  Helpers shared by the LukutarRetro node modules.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

``images`` needs torch (ComfyUI IMAGE tensor conversions) and is therefore
imported only from node modules, never from the torch-free test bootstrap.
'''

from __future__ import annotations
