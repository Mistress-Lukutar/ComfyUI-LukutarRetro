'''
File:   __init__.py
Brief:  Aggregation of node classes for the LukutarRetro pack.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0
'''

from __future__ import annotations

#: Phase 0 skeleton — node classes land with the engine phases. ComfyUI
#: tolerates an empty mapping; the root __init__ re-exports these dicts.
NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
