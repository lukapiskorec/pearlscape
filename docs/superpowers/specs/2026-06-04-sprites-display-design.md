# Sprite Display Mode — Design Spec

**Date:** 2026-06-04
**Status:** Approved (brainstorming phase)
**Target environment:** Rhino 8 SR23, CPython 3 runtime, numpy available

## 1. Overview

Add a third `display_mode`, `"sprites"`: screen-aligned colored circles drawn by
a `DisplayConduit` via the batched `DisplayPipeline.DrawSprites`. Goal is speed —
instanced mesh spheres are too heavy at the bead counts we want (target 600k–1.2M).
Sprites render all beads in a single GPU draw call per frame.

## 2. Goals and non-goals

**Goals**
- Colored circles, `bead_diameter` mm (6 mm default), world-sized so they scale
  with camera distance.
- A single batched draw call for all beads, list rebuilt only when geometry changes.
- Available in `pipeline_mode` `"cave"` and `"curtains"` (screen-only).
- Conduit persists across F5 (stashed in `scriptcontext.sticky`); replaced each run;
  disabled when switching to a non-sprite mode.

**Non-goals (deferred)**
- Sprites in `"export"` — conduits don't respect per-curtain layer isolation and
  aren't real geometry; PDFs still require pointcloud/instances. `validate()` rejects
  `sprites` + `export`.
- Per-curtain layer toggling while in sprites mode (single combined conduit is
  all-beads-or-nothing; use pointcloud/instances for per-layer control).

## 3. Components

### `SpriteConduit(Rhino.Display.DisplayConduit)` (display.py)
- `__init__(points, colors, diameter)`:
  - Build one shaded glass-bead `DisplayBitmap` — grayscale + alpha, so per-point
    colors multiply it: an opaque disc (transparent only outside the circle, with
    a ~2px antialiased edge), a darker off-centre core, and a soft colour glint at
    the bottom-left. A second white-on-transparent bitmap holds a crisp specular
    blob at the top-right. Both built once (per-pixel cost irrelevant at draw time).
  - Build two `DisplayBitmapDrawList`s over the same points: one with the bead
    colours (body), one all-white (specular pass stays pure white).
  - Cache a `BoundingBox` of the points, inflated by the bead radius.
- `CalculateBoundingBox(e)`: `e.IncludeBoundingBox(self._bbox)` so ZoomExtents
  frames the beads and nothing is near-plane clipped.
- `PostDrawObjects(e)`: two `DrawSprites` passes — body tinted per-point, then the
  pure-white specular on top — both `float(diameter)`, `sizeInWorldSpace=True`.

### `render_sprites(points, colors, diameter)` (display.py)
Owns lifecycle: disable+drop any prior conduit in `sc.sticky`, create+enable the
new one, store it in `sc.sticky["pearlscape_sprite_conduit"]`, redraw.

### `clear_sprite_conduit()` (display.py)
Disable + drop any conduit in `sc.sticky`. Called at the start of every render
path so switching modes leaves no stale sprite layer.

## 4. Parameter / wiring changes

- `params.py`: `display_mode` accepts `"sprites"`; `validate()` asserts membership
  and `not (display_mode == "sprites" and pipeline_mode == "export")`.
- `build_scene.py`:
  - Call `display.clear_sprite_conduit()` before rendering in every mode.
  - Cave path: `display_mode == "sprites"` → `render_sprites(pts, colors, bead_diameter)`,
    else `render_cave_reference`.
  - Curtains path: add a `"sprites"` branch that flattens every curtain's projected
    `(plane_x, y, z)` points and colors into one combined array, then one
    `render_sprites` call.

## 5. Color tinting

Per-point colors in `DisplayBitmapDrawList` *multiply* the grayscale bead bitmap
(confirmed against McNeel's sprite-drawing sample) → colored beads. Because
multiply can't brighten past the bead's own colour, the in-body glint is the
brightest version of that colour; a literally-white specular needs the separate
all-white second pass (one extra `DrawSprites` call).

## 6. Verification

Manual (requires Rhino — conduits are viewport-only, untestable headless):
- `pipeline_mode="cave"`, `display_mode="sprites"`: colored circles, scale with
  zoom, persist after F5 finishes.
- `pipeline_mode="curtains"`, `display_mode="sprites"`: all curtains' beads as sprites.
- Switch back to `pointcloud`: sprites disappear (conduit cleared).
- `display_mode="sprites"` + `pipeline_mode="export"`: `validate()` raises.
- Eyeball speed vs `instances` at a high `total_surface_samples`.
