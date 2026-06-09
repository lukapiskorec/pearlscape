# Thin-shell curtain mode — design

**Date:** 2026-06-09
**Status:** Approved, pending implementation plan

## Goal

Add a new curtain generation mode, **"thin shell"**, alongside the existing
"band" mode. The current band logic is left untouched.

The thin-shell result should look as close as possible to the default cave model
(the raw Poisson surface cloud) while resolving every bead onto a discrete,
densely-spaced cutting plane so the model can be fabricated as thin sheets. It
must NOT read as a stack of continuous bead rings — each curtain should be a
gappy, scattered line of beads.

## Background: the two existing tracks

- **Cave track** (`pipeline_mode` `cave` / `export_cave_ply`):
  `cave.sample_surface_points()` returns a Poisson-disk cloud (minimum spacing
  `cave_bead_spacing`, default 12mm) of beads sitting directly on the craggy
  cave surface. This is the "default cave" look to preserve.
- **Curtain track** (`pipeline_mode` `curtains` / `export` / `export_ply`):
  `build_curtains()` ignores the surface cloud and samples a thick band
  (`curtain_band_thickness`, default 120mm) outward from each cross-section
  boundary at `curtain_spacing`-spaced planes (default 50mm). Each curtain is a
  fat volumetric slab.

## Core mechanic (shell mode)

A new parameter `curtain_mode: "band" | "shell"` selects the track within the
curtain pipeline. Default `"band"` preserves current behavior. In `"shell"`
mode:

1. **Sample once.** Call `cave.sample_surface_points()` — the existing Poisson
   cave cloud. This is the reference point set.
2. **Snap in X.** For each bead, set its `X` to the nearest curtain plane's `X`
   (`Y`/`Z` unchanged). Planes tile the cave's X extent at
   `shell_curtain_spacing` (≈ `2 * bead_diameter` = 12mm). Each bead snaps to
   exactly one plane (a partition of X), so no beads are lost at this step. Max
   displacement is half the plane spacing (~6mm).
3. **Drop overlaps per plane.** Within each plane, run a deterministic greedy
   keep-order filter: a bead is dropped if it lands within `bead_diameter` (in
   the Y/Z plane) of a bead already kept on that same plane. This thins the
   collapsed beads into gappy, non-continuous lines and keeps spacing on each
   sheet.
4. **Emit** the same per-curtain dict structure as band mode:
   `{plane_x, points_2d (M,2) = (Y,Z), points_3d (M,3) = (plane_x,Y,Z)}`, so the
   existing color, display, PDF, and PLY-export paths work unchanged.

### Why this produces the intended look

- The source is a sparse Poisson cloud, never a continuous ring, so each plane
  already catches only a scattered subset of the boundary.
- The per-plane overlap drop further thins beads that collapse together when
  their X is zeroed onto the plane, opening gaps.
- Every bead moves ≤6mm, so the cloud as a whole still reads as the original
  craggy surface — now resolved onto thin cuttable planes.

## Changes by file

### `params.py`
- Add `curtain_mode: str = "band"`.
- Add `shell_curtain_spacing: float = 0.0` (0 = auto, resolved to
  `2 * bead_diameter` at use). Documented in mm.
- In `validate()`: assert `curtain_mode in ("band", "shell")` and
  `shell_curtain_spacing >= 0.0`.

### `curtains.py`
- Add `build_shell_curtains(cave, plane_xs, params) -> (curtains, bead_spacing)`:
  - Source points from `cave.sample_surface_points()`.
  - Snap: `idx = round((x - x0) / spacing)`, clamped to `[0, len(plane_xs)-1]`;
    assign each bead to `plane_xs[idx]`.
  - Per plane: greedy overlap filter at radius `bead_diameter` over `(Y, Z)`,
    using a `bead_diameter`-cell spatial hash for O(n) neighbor checks.
    Deterministic bead order (e.g. stable sort by original index).
  - Returns curtain dicts (same shape as `build_curtains`) and a representative
    `bead_spacing` (the `bead_diameter` drop radius) for the run summary.
- `curtain_planes(cave, params)`: in shell mode derive plane positions from the
  X extent using `shell_curtain_spacing` (resolving 0 → `2 * bead_diameter`),
  reusing the same extent-fitting logic as the nurbs band path. Band mode is
  unchanged.

### `build_scene.py`
- In the curtain pipeline (`curtains` / `export` / `export_ply` modes), dispatch
  to `build_shell_curtains` when `params.curtain_mode == "shell"`, else
  `build_curtains`. Everything downstream (color, display, PDF, PLY, summary) is
  shared.

## Testing (headless, in `curtains.py` `__main__`)

On `CylinderFBMCave` (pure numpy, no Rhino):
- Every kept bead's `X` equals its assigned plane's `X`.
- No two kept beads on the same plane are closer than `bead_diameter` in Y/Z.
- Result is deterministic (same params → identical beads).
- Kept count is `> 0` and `<=` the source cloud size.
- Plane spacing auto-resolves: `shell_curtain_spacing = 0` → `2 * bead_diameter`.

## Concerns

- **PDF export volume.** ~167 planes over a 2000mm cave at 12mm spacing (vs ~40
  at 50mm in band mode) → ~167 layouts/PDFs in `export` mode. Inherent to thin
  curtains; not capped unless requested.
- **Spacing interaction.** `shell_curtain_spacing` and `cave_bead_spacing` both
  ~12mm means each plane catches a thin slab of the cloud. Widening
  `cave_bead_spacing` yields sparser planes; behavior stays sensible either way.

## Out of scope

- No changes to band mode, the cave samplers, color, display, or export
  internals beyond the dispatch.
- No PDF-count cap or new export format.
