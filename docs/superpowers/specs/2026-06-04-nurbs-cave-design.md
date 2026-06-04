# Irregular NURBS Cave — Design Spec

**Date:** 2026-06-04
**Status:** Approved (brainstorming phase)
**Target environment:** Rhino 8 SR23, CPython 3 runtime, numpy available

## 1. Overview

Add a second cave geometry: an irregular lofted-NURBS tube, as an alternative to
the existing `CylinderFBMCave`. The current cylinder + radial-noise cave is too
regular. The NURBS cave builds a free-form tube from jittered cross-section rings,
then displaces sampled surface points with the existing ridged/FBM noise. It must
scale to the project's new target of 600k–1.2M beads (10–20× today's ~60k) and
generate in under ~10 seconds at that count.

The downstream pipeline (`curtains.slice_and_project`) only needs an `(N, 3)`
array spanning the cave along X — it slices by X-bin and projects `(Y, Z)`. So
this feature is purely a new `CaveSurface` implementation; nothing downstream
changes.

## 2. Goals and non-goals

**Goals**
- A `NurbsLoftCave` implementing the `CaveSurface` protocol.
- Free-form but coherent single-tube shape: meandering centerline (Y/Z only),
  non-circular cross-sections varying along length.
- Reuse `ridged3_01` / `fbm3_01` (via `noise_type`) for surface displacement.
- Vectorized sampling that hits 600k–1.2M points in < ~10s.
- A `cave_type` toggle (`"cylinder" | "nurbs"`); cylinder path stays Rhino-free.
- The base surface kept as editable document geometry, with a rebuild/reuse toggle.

**Non-goals (deferred)**
- Wild topology (branches, holes, overhangs) — that was the SDF path, not chosen.
- True Poisson/blue-noise at 1.2M — jittered-grid stratified sampling is used for
  speed; "slightly less organic, but even and instant" is an accepted trade.
- Area-weighted (exactly 3D-even) sampling — param-space-even is accepted for v1.
- Sweeping/curved cave *centerline along X* — meander is Y/Z only so X stays a
  clean slicing axis.

## 3. Coordinate conventions

Unchanged from the project: X = cave axis (slicing axis), Y = lateral, Z = up,
cave centerline at Z = `cave_center_z`. Cross-section rings sit at x ∈ [0,
`cave_length`]. Centerline meander offsets Y and Z only, never X.

## 4. Geometric construction (Rhino, one-time)

1. **Rings.** `K` closed cross-section rings along X. Ring `k` at `x_k` (evenly
   spaced 0 → `cave_length`) has:
   - center `(x_k, cy_k, cz_k)` where `cy_k, cz_k` come from low-frequency noise
     (`nurbs_centerline_freq`) scaled by `nurbs_centerline_amp` (mm),
   - `P` control points around θ at radius
     `cave_radius · (1 + nurbs_radius_jitter · n)` where `n ∈ [-1,1]` is
     low-frequency noise (`nurbs_radius_jitter_freq`) over `(θ, x_k)`, evaluated
     continuously around the θ seam so the ring closes smoothly.
   Each ring is a closed (periodic) `NurbsCurve` through its `P` control points.
2. **Loft.** `Brep.CreateFromLoft(rings, closed=false)` → an open tube (no end
   caps). Take the single wall face's underlying surface as a `NurbsSurface`.
3. Determinism seeded by `nurbs_shape_seed` (independent of `noise_seed`).

## 5. Surface lifecycle — rebuild vs reuse

`nurbs_surface_source`:
- `"rebuild"` (default): build the loft from `nurbs_*` params; render it on layer
  `Pearlscape::CaveSurface`, clearing any prior surface objects there first; then
  sample.
- `"reuse"`: read the existing surface from `Pearlscape::CaveSurface` (the user
  may have dragged its control points) and sample from it without modifying it.
  If no surface is found there, print a warning and fall back to `"rebuild"`.

Because the surface is real document geometry it survives F5 runs, enabling the
edit-then-resample loop. (Point clouds still regenerate per run as today.)

## 6. Mass sampling + displacement (pure numpy, scales to 1.2M)

1. **Grid eval (Rhino, one-time).** Evaluate the `NurbsSurface` on a regular
   `nurbs_grid_u × nurbs_grid_v` grid via `PointAt` → `grid_xyz` of shape
   `(nu, nv, 3)`. (~180×180 ≈ 32k calls, a few seconds.)
2. **Normals (numpy).** Derive per-node normals from cross-products of grid
   differences along u and v — no `NormalAt` calls. The v axis is periodic.
3. **Jittered-grid sampling (numpy).** Sample the `(u, v)` param domain, scaled to
   approximate arc-length (u → `cave_length`, v → mean circumference) so spacing is
   ≈ uniform in mm, v periodic. One jittered point per cell; target count from
   `total_surface_samples`.
4. **Map to 3D (numpy).** Bilinearly interpolate `grid_xyz` and the normal field at
   each `(u, v)` sample.
5. **Displace (numpy).** Move each point inward along its interpolated normal by
   `fbm_amplitude · noise01`, where `noise01` is `ridged3_01`/`fbm3_01` (per
   `noise_type`) evaluated at the base 3D position (spatially coherent), seeded by
   `noise_seed`. Returns `(N, 3)`.

## 7. Module structure

- **`nurbs_cave.py` (new, Rhino-dependent):**
  - `build_loft_surface(params) -> NurbsSurface` — rings + loft.
  - `eval_surface_grid(surface, nu, nv) -> np.ndarray (nu,nv,3)` — `PointAt` loop.
  - `make_nurbs_cave(params) -> NurbsLoftCave` — resolves rebuild/reuse (using the
    `display` helpers below), returns the cave with its surface.
  - `class NurbsLoftCave` — holds the surface; `sample_surface_points()` runs grid
    eval then delegates the numpy pipeline to `surface_sampling`.
- **`surface_sampling.py` (new, pure numpy, headless-testable):**
  - `grid_normals(grid_xyz) -> (nu,nv,3)`.
  - `sample_and_displace(grid_xyz, grid_normals, *, target, fbm_amplitude,
    noise params, seeds) -> (N,3)` — uses `jittered_grid` + bilinear + noise.
- **`sampling.py` (extend, pure numpy):**
  - `jittered_grid(width, wrap, target_n, seed) -> (N,2)` — stratified jittered
    sampler, columns `(wrap_axis, width_axis)`, matching `bridson_torus`'s shape.
- **`display.py` (extend, Rhino):**
  - `render_cave_surface(surface)` — add to `Pearlscape::CaveSurface`, clearing old.
  - `find_cave_surface() -> surface | None` — read first surface from that layer.
- **`cave.py` (extend):** `make_default_cave` dispatches on `cave_type`, importing
  `nurbs_cave` lazily so the cylinder path stays Rhino-free. `CaveSurface` protocol
  and `CylinderFBMCave` unchanged.
- **`params.py` (extend):** new params + `validate()` membership checks.

## 8. Parameters

Reuses `cave_radius` (base ring radius), `cave_length`, `fbm_*`, `noise_type`,
`noise_seed`, `total_surface_samples`, `cave_center_z`. New:

| Param | Default | Meaning |
|-------|---------|---------|
| `cave_type` | `"cylinder"` | `"cylinder"` or `"nurbs"`. |
| `nurbs_surface_source` | `"rebuild"` | `"rebuild"` or `"reuse"`. |
| `nurbs_sections` | `8` | `K`, cross-section rings along X. |
| `nurbs_section_points` | `12` | `P`, control points per ring. |
| `nurbs_radius_jitter` | `0.35` | Per-control-point radius variation, fraction of `cave_radius`. |
| `nurbs_radius_jitter_freq` | `0.0006` | Noise freq for radius jitter (cycles/mm). |
| `nurbs_centerline_amp` | `400.0` | Centerline meander amplitude (mm, Y/Z). |
| `nurbs_centerline_freq` | `0.0003` | Noise freq for centerline meander (cycles/mm). |
| `nurbs_shape_seed` | `7` | Seed for shape noise (independent of `noise_seed`). |
| `nurbs_grid_u` | `180` | Grid divisions along X for surface eval. |
| `nurbs_grid_v` | `180` | Grid divisions around θ for surface eval. |

`validate()`: `cave_type in {"cylinder","nurbs"}`,
`nurbs_surface_source in {"rebuild","reuse"}`, positive counts/resolutions,
`0 ≤ nurbs_radius_jitter < 1`.

## 9. Performance

- Grid eval: `nu·nv` `PointAt` calls (~32k at 180²), a few seconds, one-time and
  independent of bead count.
- Normals, sampling, interpolation, displacement: vectorized numpy, sub-second to
  ~1–2s at 1.2M.
- Target: < ~10s end-to-end at 1.2M. Grid resolution is the main tunable if eval
  is too slow; bead count never drives the Rhino-side cost.

## 10. Testing

- **Headless (numpy, runnable outside Rhino):** feed `surface_sampling` a
  synthetic analytic grid (e.g. a plain cylinder grid) and assert: point count
  ≈ target, points lie within `fbm_amplitude` of the base surface, normals unit
  length, determinism, no NaNs. `jittered_grid` count ≈ target and within domain.
- **In-Rhino (manual, user F5):** `cave_type="nurbs"`, `pipeline_mode="cave"` →
  irregular tube point cloud + base surface on `CaveSurface` layer; switch
  `noise_type` ridged/fbm; bump `total_surface_samples` toward 1.2M and confirm
  < ~10s; edit the surface, set `nurbs_surface_source="reuse"`, re-run, confirm
  beads resample from the edited surface; `cave_type="cylinder"` unchanged.

## 11. Known limitations

- Param-space-even ≈ 3D-even only up to surface stretch; bulges may show mild
  density variation. Refinable with area-weighting later.
- Jittered-grid sampling has faint residual regularity vs true Poisson; accepted
  for speed.
