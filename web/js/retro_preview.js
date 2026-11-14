/**
 * File:   retro_preview.js
 * Brief:  Retro filter nodes — hide the v2 output section and feed
 *         in-node previews via /view URLs on execution.
 * Author: Mistress-Lukutar
 * Date:   2026-11-14
 * Version: v0.1.0
 *
 * Contract:
 * - `node.hideOutputImages = true` keeps the v2 output section hidden.
 *   The v2 output display prefers live preview frames over executed
 *   outputs, so rendering the section would show the same picture a
 *   second time below the preview.
 * - `node.imgs` is fed with /view URLs of the temp files on
 *   `onExecuted`, which is what the right-click menu (Open Image /
 *   Copy Image / Save Image) is built from. Open and Save then work
 *   against the real full-resolution temp file instead of the JPEG
 *   preview blob.
 * - The `show_preview` BOOLEAN widget on each node controls whether
 *   previews exist at all: the backend emits NO ui images when it is
 *   off, so nothing re-appears from the output store. This extension
 *   additionally clears `node.imgs` and removes the core's
 *   `$$canvas-image-preview` DOM widget on the next execution — the
 *   core's async preview loader (`useNodeImage().showPreview()`, run
 *   from the draw-background handler) would otherwise repopulate
 *   `node.imgs` from the previous execution's output store after our
 *   synchronous clear. When on (default), the backend emits temp image
 *   refs and this extension maps them to /view URLs for rendering.
 *
 * Served at /extensions/<pack>/js/retro_preview.js, hence the THREE
 * parent hops to reach /scripts/app.js.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.info("[LukutarRetro.Preview] extension loaded");

/** Build the /view URL for an executed image reference. */
function viewUrl(image) {
  const params = new URLSearchParams();
  params.set("filename", image.filename ?? "");
  params.set("subfolder", image.subfolder ?? "");
  params.set("type", image.type ?? "temp");
  return api.apiURL(`/view?${params.toString()}`);
}

/**
 * Remove the core's in-canvas image preview DOM widget, if present.
 * Mirrors the frontend's removeCanvasImagePreview(): call onRemove and
 * splice the widget out so the stale preview image disappears too.
 */
function removeCanvasPreviewWidget(node) {
  if (!node.widgets) return;
  const idx = node.widgets.findIndex((w) => w.name === "$$canvas-image-preview");
  if (idx > -1) {
    node.widgets[idx].onRemove?.();
    node.widgets.splice(idx, 1);
  }
}

/** All retro filter node names — covers current and future slices. */
const PREVIEW_NODES = new Set([
  "RetroPixelize",
  "RetroPaletteQuantize",
  "RetroColorDepth",
  "RetroDither",
  "RetroScanlines",
  "RetroSubpixelMask",
  "RetroBloom",
  "RetroChromaticAberration",
  "RetroCurvature",
  "RetroVignette",
  "RetroVHS",
  "RetroLCDDotMatrix",
]);

app.registerExtension({
  name: "LukutarRetro.Preview",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!PREVIEW_NODES.has(nodeData.name)) return;

    // Hide the v2 output section for these nodes.
    const originalNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalNodeCreated?.apply(this, arguments);
      this.hideOutputImages = true;
      return result;
    };

    // Feed /view URLs into node.imgs on execution.
    const originalOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      originalOnExecuted?.apply(this, arguments);

      // Read the show_preview widget FIRST: when off we must clear the
      // preview even if the payload carries no images at all.
      const widget = this.widgets?.find((w) => w.name === "show_preview");
      const show = widget ? widget.value !== false : true;

      if (!show) {
        // show_preview is off — the backend emitted no preview images;
        // drop anything left over from a previous execution. The core's
        // async loader will not repopulate node.imgs because the output
        // store no longer contains image refs for this node.
        this.imgs = [];
        removeCanvasPreviewWidget(this);
        this.graph?.setDirtyCanvas?.(true, false);
        return;
      }

      const images = output?.images;
      if (!Array.isArray(images) || images.length === 0) return;

      this.previewMediaType = "image";

      this.imgs = images
        .filter((image) => image && image.filename)
        .map((image) => {
          const img = new Image();
          img.src = viewUrl(image);
          return img;
        });
    };
  },
});
