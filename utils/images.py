'''
File:   images.py
Brief:  Conversion helpers between ComfyUI IMAGE tensors and numpy frames.
Author: Mistress-Lukutar
Date:   2026-08-28
Version: v0.1.1
'''

from __future__ import annotations

import numpy as np
import torch


def tensor_to_frames(image: torch.Tensor) -> list[np.ndarray]:
    '''Convert a ComfyUI IMAGE tensor into per-frame numpy arrays.

    Args:
        image: Tensor of shape (B, H, W, 3), float in [0, 1], RGB. May
            live on any device; moved to CPU automatically. May be a
            non-contiguous view (e.g. a VAE-decoded batch after a
            ``permute``).

    Returns:
        List of B frames, HWC float32 in [0, 255], RGB, each
        C-contiguous so downstream cv2 drawing accepts it.
    '''
    frames_np = image.detach().cpu().numpy()
    return [
        np.ascontiguousarray(
            np.clip(frame, 0.0, 1.0) * 255.0, dtype=np.float32
        )
        for frame in frames_np
    ]


def frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    '''Convert numpy frames back into a ComfyUI IMAGE tensor.

    Args:
        frames: List of B frames, HWC, uint8 or float32 in [0, 255], RGB.

    Returns:
        Tensor of shape (B, H, W, 3), float32 in [0, 1], RGB, on CPU.
    '''
    stacked = (
        np.stack([np.clip(frame, 0.0, 255.0) for frame in frames]).astype(
            np.float32
        )
        / 255.0
    )
    return torch.from_numpy(stacked)


def frames_to_mask_tensor(masks: list[np.ndarray]) -> torch.Tensor:
    '''Convert numpy masks into a ComfyUI MASK tensor.

    Args:
        masks: List of B masks, HW float32, values in ``[0, 1]``
            (``1`` = masked).

    Returns:
        Tensor of shape (B, H, W), float32 in [0, 1], on CPU.
    '''
    stacked = np.stack(
        [np.clip(mask, 0.0, 1.0).astype(np.float32) for mask in masks]
    ).astype(np.float32)
    return torch.from_numpy(stacked)
