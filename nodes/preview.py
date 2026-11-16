'''
File:   preview.py
Brief:  Shared OUTPUT_NODE payload helper — save in-node preview PNGs.
Author: Mistress-Lukutar
Date:   2026-11-14
Version: v0.1.0

Every filter node returns IMAGE from build_output_payload(). The preview
images are saved to the ComfyUI temp directory at FULL resolution (same
pixels as the IMAGE output) and fed to the frontend via node.imgs by
web/js/retro_preview.js. When show_preview is off no preview is
generated at all.
'''

from __future__ import annotations

import json
import logging
import os

import numpy as np

from ..utils.images import frames_to_tensor

logger = logging.getLogger(__name__)


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


def save_preview_files(
    frames: list[np.ndarray],
    prompt: dict | None = None,
    extra_pnginfo: dict | None = None,
) -> list[dict[str, str]]:
    """Save per-frame preview PNGs to the ComfyUI temp directory.

    Previews are saved at the frame's FULL resolution — the preview shows
    exactly the pixels the IMAGE output carries, with no downscaling.
    When *prompt* / *extra_pnginfo* are provided (ComfyUI passes them as
    hidden inputs), they are embedded into the PNG chunks the same way
    ComfyUI's own SaveImage node does, so the saved preview can be
    drag-and-dropped back onto the canvas to restore the workflow.

    NEVER raises — catches all exceptions, logs a warning, and returns [].
    When called outside the ComfyUI runtime (PIL or folder_paths missing),
    returns [] immediately.

    Args:
        frames: List of HWC float32 [0, 255] RGB numpy arrays.
        prompt: Executing prompt graph (hidden ``PROMPT`` input), or None.
        extra_pnginfo: Extra workflow info (hidden ``EXTRA_PNGINFO``
            input — carries the UI workflow), or None.

    Returns:
        List of ``{"filename", "subfolder", "type": "temp"}`` dicts.
    """
    if _folder_props is None or _Image is None:
        logger.debug("Preview skipped: ComfyUI runtime not available")
        return []

    try:
        first = frames[0] if frames else None
        hint_h, hint_w = (
            (first.shape[0], first.shape[1]) if first is not None else (1024, 1024)
        )
        full_output_dir, base_name, counter, subfolder, _ = (
            _folder_props.get_save_image_path(
                "retro_preview",
                _folder_props.get_temp_directory(),
                hint_w,
                hint_h,
            )
        )

        # Workflow metadata — same chunks as ComfyUI's SaveImage node.
        metadata = None
        if prompt is not None or extra_pnginfo is not None:
            from PIL.PngImagePlugin import PngInfo

            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value))

        results: list[dict[str, str]] = []
        for frame in frames:
            # Convert to uint8 for PIL — full resolution, no downscale.
            arr = np.clip(frame, 0, 255).astype(np.uint8)
            pil_img = _Image.fromarray(arr)

            filename = f"{base_name}_{counter:05}.png"
            pil_img.save(
                os.path.join(full_output_dir, filename),
                compress_level=1,
                pnginfo=metadata,
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


def build_output_payload(
    frames: list[np.ndarray],
    show_preview: bool = True,
    extra_result: tuple = (),
    prompt: dict | None = None,
    extra_pnginfo: dict | None = None,
) -> dict:
    """Build the uniform OUTPUT_NODE return payload.

    Args:
        frames: List of HWC float32 [0, 255] RGB numpy arrays.
        show_preview: When False, no preview PNGs are generated and the
            ui payload carries an empty image list — the frontend clears
            any previously shown preview on the next execution.
        extra_result: Additional output tensors appended to the result
            tuple after the IMAGE (e.g. a MASK output).
        prompt: Hidden ``PROMPT`` input from ComfyUI — embedded into the
            preview PNGs as workflow metadata (optional).
        extra_pnginfo: Hidden ``EXTRA_PNGINFO`` input from ComfyUI —
            embedded into the preview PNGs as workflow metadata
            (optional).

    Returns:
        Dict with ``"ui": {"images": [...]}`` and
        ``"result": (IMAGE_tensor, *extra_result)``.
    """
    ui_images = (
        save_preview_files(frames, prompt, extra_pnginfo) if show_preview else []
    )
    return {
        "ui": {"images": ui_images},
        "result": (frames_to_tensor(frames), *extra_result),
    }
