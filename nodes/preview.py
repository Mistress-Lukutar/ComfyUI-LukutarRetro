'''
File:   preview.py
Brief:  Shared OUTPUT_NODE payload helper — save in-node preview PNGs.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Every filter node returns IMAGE from build_output_payload(). The preview
images are saved to the ComfyUI temp directory and fed to the frontend
via node.imgs by web/js/retro_preview.js.
'''

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from ..utils.images import frames_to_tensor

logger = logging.getLogger(__name__)

#: Long-side cap for in-node preview PNGs (the IMAGE output keeps full res).
MAX_PREVIEW_SIZE: int = 1024


def _load_pil() -> object | None:
    """Load the PIL Image module, or None when Pillow is unavailable.

    Returns:
        The ``PIL.Image`` module, or ``None`` on import failure.
    """
    try:
        from PIL import Image  # noqa: F811
        return Image
    except ImportError:
        return None


#: Cached PIL Image module (module-level guarded import).
_Image = _load_pil()


def _folder_paths() -> object | None:
    """Load the ``folder_paths`` module, or None outside the ComfyUI runtime.

    ``folder_paths`` is a TOP-LEVEL module in the ComfyUI root, not a
    ``comfy`` subpackage member — import it accordingly.

    Returns:
        The ``folder_paths`` module, or ``None`` on import failure.
    """
    try:
        import folder_paths  # type: ignore[import-not-found]
        return folder_paths
    except ImportError:
        return None


#: Cached folder_paths module (module-level guarded import).
_folder_props = _folder_paths()


def save_preview_files(frames: list[np.ndarray]) -> list[dict[str, str]]:
    """Save per-frame preview PNGs to the ComfyUI temp directory.

    NEVER raises — catches all exceptions, logs a warning, and returns [].
    When called outside the ComfyUI runtime (PIL or folder_paths missing),
    returns [] immediately.

    Args:
        frames: List of HWC float32 [0, 255] RGB numpy arrays.

    Returns:
        List of ``{"filename", "subfolder", "type": "temp"}`` dicts.
    """
    if _folder_props is None or _Image is None:
        logger.debug("Preview skipped: ComfyUI runtime not available")
        return []

    try:
        full_output_dir, base_name, counter, subfolder, _ = (
            _folder_props.get_save_image_path(
                "retro_preview",
                _folder_props.get_temp_directory(),
                1024,
                1024,
            )
        )

        results: list[dict[str, str]] = []
        for frame in frames:
            h, w, _ = frame.shape
            long_side = max(h, w)

            work = frame.copy()
            if long_side > MAX_PREVIEW_SIZE:
                scale = MAX_PREVIEW_SIZE / long_side
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                work = cv2.resize(work, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to uint8 for PIL
            arr = np.clip(work, 0, 255).astype(np.uint8)
            pil_img = _Image.fromarray(arr)

            filename = f"{base_name}_{counter:05}.png"
            pil_img.save(
                os.path.join(full_output_dir, filename),
                compress_level=1,
            )
            results.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "temp",
            })
            counter += 1

        return results

    except Exception:  # noqa: BLE001 — preview must not kill the filter
        logger.warning("Preview save failed", exc_info=True)
        return []


def build_output_payload(frames: list[np.ndarray]) -> dict:
    """Build the uniform OUTPUT_NODE return payload.

    Args:
        frames: List of HWC float32 [0, 255] RGB numpy arrays.

    Returns:
        Dict with ``"ui": {"images": [...]}`` and ``"result": (IMAGE_tensor,)``.
    """
    ui_images = save_preview_files(frames)
    return {
        "ui": {"images": ui_images},
        "result": (frames_to_tensor(frames),),
    }
