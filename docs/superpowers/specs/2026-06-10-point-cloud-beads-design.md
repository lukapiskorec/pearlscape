# Point-cloud-sourced beads (`cave_type = "points"`)

**Date:** 2026-06-10
**Status:** Approved, ready for implementation

## Goal

Build bead geometry in Rhino from a point cloud on a `CavePoints` layer, populating
beads directly from those points. For now the points are used as-is (no geometric
modification); fbm displacement is a future addition. The workflow must reuse the
existing thin-shell / dense-curtains path and the PLY export for the web explorer.

## Resolved decisions

- **Pipeline routing:** CavePoints feed `build_shell_curtains` — X snapped to thin
  planes (`2 × bead_diameter` spacing) + per-plane overlap drop, identical to the
  current shell model. "Use as-is" means *no fbm displacement yet*; the shell snap
  still applies. (User confirmed over "direct beads, no snap".)
- **Axis convention:** The CavePoints cloud is already oriented with the cave's long
  axis along world X and Z up, matching cylinder/nurbs. Coordinates used directly,
  no transform. (User confirmed.)

## Architecture — reuse the `CaveSurface` seam

A third `cave_type = "points"` is backed by a new `PointCloudCave` that reads the
`CavePoints` layer. Everything downstream (curtain planes, shell snapping, color,
PLY export) is cave-type-agnostic and works unchanged.

```
make_default_cave(params)
  cave_type == "points"  ->  display.find_cave_points()  ->  PointCloudCave(points)
                                                                    |
build_scene main()  ->  curtain_planes(cave)  ->  build_shell_curtains(cave)  ->  color  ->  PLY
```

## Components

1. **`display.py`** — new constant `CAVE_POINTS_LAYER = "CavePoints"` and
   `find_cave_points() -> Optional[np.ndarray]`. Aggregates **all** point geometry
   on `Pearlscape::CavePoints` — both `PointCloud` objects (expanded to their member
   points) and individual `Point` objects — into one `(N, 3)` world-coordinate array.
   Returns `None` if the layer or its points are absent.

2. **`cave.py`** — new `PointCloudCave` class (pure-numpy, no Rhino import; the array
   is supplied to it):
   - `sample_surface_points()` → returns the stored points (a copy). The seam the
     shell path consumes.
   - `x_extent()` → `(X.min(), X.max())` from the cloud.
   - `inner_boundary(...)` → raises `NotImplementedError` (band mode needs an
     analytic cross-section; never called in the shell path — this is a guard).

3. **`make_default_cave`** — branch for `"points"`: lazy-import `display`, call
   `find_cave_points()`, raise a clear `RuntimeError` (guidance to populate the
   layer) if none found, else return `PointCloudCave`.

4. **`params.py` `validate()`** — allow `cave_type == "points"`; assert `points` is
   not combined with `curtain_mode == "band"` (unsupported). Other modes are fine,
   including `pipeline_mode = "cave"` / `"export_cave_ply"` (render/export the raw
   imported cloud).

5. **`build_scene.py print_run_summary`** — gate the cylinder/nurbs-only "cave
   geometry" detail and "wall noise" sections off `cave_type == "points"`, and add a
   short "point source" line (layer + point count).

## What does NOT change

- `build_scene.main()` flow — already cave-type-agnostic.
- `build_shell_curtains`, `curtain_planes`, color, `ply_export`, `_export_ply_bundle`.

## Data flow notes

- Shell planes tile the cloud's X extent at `2 × bead_diameter`; each point snaps to
  its nearest plane, then per-plane overlaps within `bead_diameter` drop — same data
  shape the web explorer already consumes.
- Future fbm displacement plugs into `PointCloudCave.sample_surface_points()` (needs
  per-point normals — a later concern, noted in code).

## Testing

Headless numpy test (no Rhino): build `PointCloudCave` from a synthetic array, run
`curtain_planes` + `build_shell_curtains`, assert planes tile the extent, every kept
bead lies on its plane, no two beads on a plane are closer than `bead_diameter`, and
determinism. Rhino-only `find_cave_points` is exercised manually in Rhino.

## Scope guardrails

One function, one class, one branch + two gated edits. No new dependencies. No
changes to tuned param values. Assistant does not commit (user handles git).
