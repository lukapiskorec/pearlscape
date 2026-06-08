# Pearlscape — Curtain Cross-Section Sampling Design Spec

**Date:** 2026-06-05
**Status:** Approved (brainstorming phase)
**Target environment:** Rhino 8 SR23, CPython 3 runtime, numpy available
**Supersedes:** the curtain-generation portion of `2026-06-02-pearlscape-design.md` (slice-and-project)

## 1. Problem

Curtain sections currently read too much like discrete **bands**. The cause is structural: `curtains.slice_and_project` partitions the cave cloud into contiguous box-sections along X, then flattens each onto a single plane by dropping the X coordinate. This produces N zero-thickness sheets separated by `curtain_spacing` (100 mm) of air, and it piles ~6k beads (Poisson-spaced *in 3D*, so stacked along X) onto each plane — where they overlap heavily because their X-separation collapses to zero.

Trying to fix this by widening the box-sections and de-colliding the projected pile (overlapping slabs + fuzzy edges + in-plane relaxation) is a workaround layered on a flawed foundation: it adds thickness, then has to repair the collisions that thickness creates.

## 2. Approach

Stop arranging beads on the original 3D geometry and projecting. Instead, for each curtain plane, sample beads **directly in the plane**, bounded by the cave's cross-section curve at that X:

- The cross-section curve is a **hard inner border** — no bead is placed on the cave-interior side of it.
- Beads are placed from that border **outward**, with their frequency **fading stochastically** with distance.

This gives, in one move:
- a **sharp inner edge** (the cave silhouette stays crisp),
- a **soft outer volume** whose thickness we control directly, bridging the gaps between curtains,
- and **no projection stacking** — beads are sampled once in the plane (stratified blue-noise via `jittered_grid`, the same sampler already used for the cave surface), so the ~16-deep piles that projection produced are gone. In-plane spacing matches the surface beads the artist already accepts; strict Poisson-disk (`bridson_torus`) is a one-line swap if tighter spacing is ever wanted.

It also decouples band thickness from `curtain_spacing`: a single `curtain_band_thickness` knob controls gap-bridging.

## 3. Key insight: the craggy detail is analytic, not in the surface

The lofted NURBS cave surface (and the cylinder base) is **smooth**. All craggy character — the ridged crevices, up to `fbm_amplitude` (currently 900 mm) of inward bite — is applied as **per-point noise displacement at sample time** (`surface_sampling.sample_and_displace` line 139; `cave.CylinderFBMCave.sample_surface_points`), never baked into the surface.

Consequence: a naive `Surface ∩ Plane` intersection in Rhino would return the **smooth** cross-section, losing exactly the character that makes the cave interesting.

Fix: the inner boundary is **analytic**. For any angle θ around the cross-section, the wall position is

```
r_inner(θ) = r_base(θ) − fbm_amplitude · noise(base_point(θ)) · (radial component of normal)
```

and `noise()` is a numpy function we already have (`fbm3_01` / `ridged3_01`). So the craggy inner curve is computed directly from the base ring + the noise field, at **any angular resolution we want, for free** — no PointAt loop, no mesh, no Rhino intersection. The base NURBS grid only carries the smooth shape; the crags are added in numpy.

## 4. Assumptions (confirmed)

- **Outward = radially from the cross-section centroid.** Simplest; reuses the polar frame. (Alternative — offset along the curve normal, keeping deep-crevice walls thicker — is deferred. Switchable later.)
- **Bead count is spacing-driven, not target-driven.** Count emerges from `Σ band_area / spacing²` across curtains. `total_surface_samples` no longer governs the curtain count; it still drives the `"cave"` reference view. The artist shapes the count by tuning `curtain_band_thickness` and `bead_min_spacing`.
- **Inner boundary is single-valued in angle about the centerline (star-convex).** Holds because the noise displaces radially and cannot create overhangs; deep ridged crevices stay single-valued. This restricts the method to tube topology, which is the current geometry. A future non-tube geometry (e.g. Lidar) would need the deferred mesh-slice path.

## 5. Coordinate conventions (unchanged)

- **X** = viewing axis = cave axis. Curtains are YZ planes spaced along X.
- **Y** = lateral, **Z** = vertical (up). Cave centered at Y = 0, Z = `cave_center_z`.
- Per-curtain 2D space is (Y, Z), matching the existing `points_2d` contract.

## 6. Module structure

```
RhinoPython/pearlscape/
  cross_section.py    # NEW — pure numpy: inner-boundary + band sampling (headlessly testable)
  cave.py             # +inner_boundary(...) on CylinderFBMCave; CaveSurface protocol gains the method
  nurbs_cave.py       # +inner_boundary(...) on NurbsLoftCave (interpolates base grid ring + noise)
  curtains.py         # REWRITTEN — orchestrates per-plane sampling; same output dict contract
  params.py           # +curtain_band_thickness, curtain_band_fade, bead_min_spacing
```

Downstream modules **unchanged**: `display.py`, `color.py`, `export.py`. The rewritten `curtains.py` returns the same per-curtain dict (`plane_x`, `points_2d`, `points_3d`, `colors`).

### 6.1 `CaveSurface` protocol addition

```python
def inner_boundary(self, plane_x: float, n_angular: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid_yz (2,), theta (n_angular,), r_inner (n_angular,)) for the
    cave's cross-section at X = plane_x. r_inner is measured from centroid_yz.
    Angles are evenly spaced in [0, 2*pi). Returns the craggy (displaced) boundary."""
```

- **CylinderFBMCave**: centroid = `(0, center_z)`; `r_inner(θ) = radius − fbm_amplitude · noise(base_ring_point(θ, plane_x))`. Pure analytic, reuses the same noise call as `sample_surface_points`.
- **NurbsLoftCave**: evaluate/interpolate the base surface grid to the ring at `plane_x` (axis 0 ≈ X, axis 1 = θ); centroid = ring centroid; apply the same noise displacement analytically at `n_angular` resolution (interpolating the smooth base ring to full resolution first, so inner-edge crispness is not limited by `nurbs_grid_v`).

## 7. Per-curtain algorithm (`cross_section.py`)

For each curtain plane at `plane_x`:

1. **Boundary** — `centroid, theta, r_inner = cave.inner_boundary(plane_x, n_angular)`. `n_angular` is derived from the inner perimeter and `bead_min_spacing` (internal; not a user knob).
2. **Band sampling** — sample the band in **(arc-length × radial-offset)** space with the existing `sampling.jittered_grid`:
   - arc-length domain = inner perimeter (periodic), radial-offset domain = `[0, curtain_band_thickness]`.
   - `target_n = perimeter · band_thickness / bead_min_spacing²` so cells are ~`bead_min_spacing` square → stratified spacing, no piles.
3. **Map to world** — for each candidate at `(arc, offset)`: recover `θ` from arc position, `r = r_inner(θ) + offset`, then
   `y = cy + r·cos(θ)`, `z = cz + r·sin(θ)`, `x = plane_x`.
4. **Stochastic outward fade** — keep each candidate with probability
   `p(offset) = (1 − offset / curtain_band_thickness) ** curtain_band_fade`
   (full density at the inner curve, zero at the outer extent). Thinning a blue-noise set never creates collisions, so spacing is preserved.
5. **Clip** — drop beads falling outside the curtain rectangle (`|y| > curtain_width/2` or `z < 0` or `z > curtain_height`).
6. Return `points_2d = (y, z)` and `points_3d = (plane_x, y, z)`.

Deterministic per run: all randomness flows from seeded RNGs (sampling seed + a per-curtain offset), matching the existing pattern.

## 8. Parameters (new)

Defaults are as-designed; live-tuned values live in `params.py` per project convention (README documents as-designed defaults; do not "fix" the drift).

| Param | Default | Role |
|---|---|---|
| `curtain_band_thickness` | `120.0` mm | Maximum outward reach of beads; the gap-bridging knob. |
| `curtain_band_fade` | `1.5` | Fade exponent `k` in `p = (1 − offset/T)^k`. Higher = beads hug the inner edge tighter. |
| `bead_min_spacing` | `bead_diameter` (6.0 mm) | In-plane blue-noise spacing. Must be ≥ `bead_diameter` to avoid physical overlap. |

`validate()` additions: `curtain_band_thickness > 0`, `curtain_band_fade >= 0`, `bead_min_spacing >= bead_diameter`.

## 9. Colour

Colour is sampled at each bead's **plane position** `(plane_x, y, z)` (its `points_3d`), via the existing `color.apply_to_curtains`. Adjacent planes are close in X, so colour blobs stay coherent across curtains. No change to `color.py` is required — it already reads `points_3d`.

## 10. Data flow change in `build_scene.py`

- `"cave"` mode: unchanged — `cave.sample_surface_points()` → reference cloud.
- `"curtains"` / `"export"` modes: instead of `slice_and_project(pts, ...)`, call the rewritten `curtains.build_curtains(cave, plane_xs, params)`, which calls `cave.inner_boundary(...)` per plane. **The dense surface point cloud (`cave.sample_surface_points()`) is skipped entirely in these modes** — it is only produced in `"cave"` mode for the reference view. `print_run_summary` still measures the actual beads, now gathered from the built curtains rather than the surface cloud.

## 11. Edge cases

- **Plane outside cave length**: `inner_boundary` returns empty / the curtain yields no beads.
- **Band exceeds curtain rectangle**: at the equator, `cave_radius + band_thickness` can exceed `curtain_width/2`; step 5 clips. Acceptable.
- **Centroid for `nurbs_surface_source="reuse"`**: ring centroid from the (hand-edited) grid still works.
- **Seam continuity**: noise is evaluated on continuous base-ring positions (not on a wrapped angle), so the inner curve is continuous across the θ seam, matching `CylinderFBMCave`'s existing approach.

## 12. What gets deleted / kept

- **Deleted**: `slice_and_project` and `array_x_center`'s box-sectioning role in the curtain path; the entire overlap/collision concern.
- **Kept**: `curtain_x_positions` (still needed for plane positions and `render_curtain_planes`); cave reference sampling; colour; display; export.

## 13. Testing

`cross_section.py` is pure numpy → headlessly testable like `surface_sampling.py`, with a `__main__` smoke test on a synthetic analytic cylinder:
- inner edge respected: every bead has radial offset ≥ 0, i.e. `r ≥ r_inner(θ)` at its angle (guaranteed by construction);
- no piles: binning beads into `bead_min_spacing`-sized cells, no cell holds more than a small constant (e.g. ≤ 4) — the stacking the projection caused is gone;
- count sane: total ≈ `Σ kept-fraction · perimeter · band_thickness / bead_min_spacing²`;
- fade monotonic: bead density decreases with radial offset;
- determinism: two runs identical;
- empty/degenerate inputs return `(0, ...)` cleanly.

Final visual verification is manual by the user in Rhino, per project convention.

## 14. Non-goals (deferred)

- Curve-normal outward offset (vs radial) for thicker crevice walls.
- Mesh-slice / point-in-polygon path for non-tube topologies (Lidar, subdivision).
- Per-curtain variation of band thickness or fade.
- Re-tuning the colour field for the new bead distribution.
