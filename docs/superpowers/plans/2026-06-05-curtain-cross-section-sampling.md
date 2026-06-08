# Curtain Cross-Section Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the curtain pipeline's slice-and-project box-sectioning with direct in-plane bead sampling, bounded by the cave's craggy cross-section curve and fading outward — giving a sharp inner edge, a soft gap-bridging outer volume, and no projection stacking.

**Architecture:** For each curtain plane, the cave reports a craggy inner-boundary curve `(centroid, theta, r_inner)` computed analytically (base ring + the same per-point noise the surface already uses). A new pure-numpy `cross_section.py` blue-noise-samples the band outside that curve and thins it outward. `curtains.py` orchestrates per plane; downstream (`color`, `display`, `export`) is untouched because the per-curtain dict contract is preserved.

**Tech Stack:** Python 3 (Rhino 8 CPython runtime; headless dev runtime has Python 3.14 + numpy 2.4), numpy. No new dependencies. Rhino (`Rhino.Geometry`) only for the NURBS surface eval, which is cached.

---

## Conventions for this plan

- **Testing:** This codebase has no pytest. Modules carry `__main__` smoke tests with `assert`s (see `surface_sampling.py`, `sampling.py`, `color.py`). Pure-numpy modules run headlessly with the system `python`; Rhino-dependent paths are verified manually in Rhino. We follow that pattern.
- **Run tests from the repo root**, e.g. `python RhinoPython/pearlscape/cross_section.py`. Each module inserts its package parent on `sys.path`, so direct execution resolves `pearlscape.*` imports.
- **Commits are the user's job.** Never run `git commit`. At each commit checkpoint, stop and ask the user to commit, offering a plain-English message (no `feat:`/`fix:` prefixes — the user prefers plain titles).
- **Units:** millimetres throughout.

## File structure

- **Create** `RhinoPython/pearlscape/cross_section.py` — pure numpy: `displaced_ring_boundary()` + `sample_band()` + `__main__` smoke tests.
- **Modify** `RhinoPython/pearlscape/params.py` — add `curtain_band_thickness`, `curtain_band_fade`, `bead_min_spacing` + `validate()` assertions.
- **Modify** `RhinoPython/pearlscape/cave.py` — add `inner_boundary()` to the `CaveSurface` protocol and to `CylinderFBMCave`.
- **Modify** `RhinoPython/pearlscape/nurbs_cave.py` — add `inner_boundary()` + a cached grid/normals to `NurbsLoftCave`.
- **Rewrite** `RhinoPython/pearlscape/curtains.py` — replace `slice_and_project` with `build_curtains`; keep `curtain_x_positions` and `array_x_center`.
- **Modify** `RhinoPython/build_scene.py` — call `build_curtains`; skip surface sampling in curtain/export modes.
- **Modify** `README.md` — document the new params and the new curtain method.

---

## Task 1: Add curtain-band parameters

**Files:**
- Modify: `RhinoPython/pearlscape/params.py`

- [ ] **Step 1: Add the three parameters**

In `params.py`, immediately after the `bead_diameter` field (line 71, inside the `# --- Beads ---` group), add:

```python
    # --- Curtain band (in-plane bead placement) ---
    # Beads are sampled directly in each curtain plane, outward from the cave's
    # cross-section curve. Thickness controls how far the soft outer volume
    # reaches (and thus how much it bridges the gaps between curtains); fade is
    # the exponent of the outward density falloff p = (1 - d/T)^fade.
    curtain_band_thickness: float = 120.0   # max outward reach of beads (mm)
    curtain_band_fade: float = 1.5          # outward density falloff exponent
    bead_min_spacing: float = 6.0           # in-plane blue-noise spacing (mm)
```

- [ ] **Step 2: Add validation**

In `params.py`, inside `validate()`, after the existing `assert self.color_dither >= 0.0` line, add:

```python
        assert self.curtain_band_thickness > 0.0
        assert self.curtain_band_fade >= 0.0
        assert self.bead_min_spacing >= self.bead_diameter, (
            f"bead_min_spacing ({self.bead_min_spacing}) must be >= bead_diameter "
            f"({self.bead_diameter}) so placed beads cannot physically overlap."
        )
```

- [ ] **Step 3: Verify defaults validate**

Run: `python -c "import sys; sys.path.insert(0, 'RhinoPython'); from pearlscape.params import PearlscapeParams; p = PearlscapeParams(); p.validate(); print('defaults OK', p.curtain_band_thickness, p.curtain_band_fade, p.bead_min_spacing)"`
Expected: `defaults OK 120.0 1.5 6.0`

- [ ] **Step 4: Verify the spacing assertion fires**

Run: `python -c "import sys; sys.path.insert(0, 'RhinoPython'); from pearlscape.params import PearlscapeParams; p = PearlscapeParams(); p.bead_min_spacing = 1.0; p.validate()"`
Expected: `AssertionError` mentioning `bead_min_spacing` must be `>= bead_diameter`.

- [ ] **Step 5: Commit (user runs git)**

Stop and ask the user to commit. Suggested message: `Add curtain band parameters`

---

## Task 2: cross_section.py — boundary geometry + band sampling

**Files:**
- Create: `RhinoPython/pearlscape/cross_section.py`

- [ ] **Step 1: Write the module with the failing smoke test first**

Create `RhinoPython/pearlscape/cross_section.py` with the full content below. (It defines both functions *and* the `__main__` test; we run it next to watch it pass — for true test-first, you may temporarily stub the two function bodies with `raise NotImplementedError`, run to see the asserts fail, then paste the real bodies.)

```python
#! python 3
# r: numpy
"""Place curtain beads directly in the curtain plane, bounded by the cave's
craggy cross-section curve and fading outward. Pure numpy, no Rhino — testable
headlessly (see __main__).

Public functions:
  displaced_ring_boundary(base_xyz, normals, n01, *, fbm_amplitude)
      -> (centroid_yz (2,), theta (n,), r_inner (n,))
      Apply the cave's per-point inward noise displacement to a ring of base-
      surface points and return the resulting craggy inner boundary in polar
      form about the ring's Y/Z centroid. The caller supplies n01 (the noise
      value per point) using its own noise convention, so this stays pure
      geometry and matches whichever cave produced the ring.

  sample_band(centroid_yz, theta, r_inner, *, band_thickness, fade, spacing, seed)
      -> (M, 2) bead (Y, Z) positions
      Stratified blue-noise sampling of the band outside the boundary, with
      density full at the boundary and fading to zero at band_thickness.
"""

import os
import sys
from typing import Tuple

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape.sampling import jittered_grid


def displaced_ring_boundary(
    base_xyz: np.ndarray,
    normals: np.ndarray,
    n01: np.ndarray,
    *,
    fbm_amplitude: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Displace a base ring inward by fbm_amplitude * n01 along -normals and
    return (centroid_yz, theta, r_inner). theta is each displaced point's angle
    about the ring's Y/Z centroid, sorted ascending; r_inner its distance from
    that centroid. base_xyz, normals: (n, 3) world points / unit outward normals."""
    displaced = base_xyz - fbm_amplitude * n01[:, None] * normals
    yz = displaced[:, 1:3]
    centroid = yz.mean(axis=0)
    rel = yz - centroid
    theta = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)   # [0, 2pi)
    r_inner = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
    order = np.argsort(theta)
    return centroid, theta[order], r_inner[order]


def sample_band(
    centroid_yz: np.ndarray,
    theta: np.ndarray,
    r_inner: np.ndarray,
    *,
    band_thickness: float,
    fade: float,
    spacing: float,
    seed: int,
) -> np.ndarray:
    """Blue-noise-sample the band outside the inner boundary, fading outward.
    Returns (M, 2) bead positions in the curtain's (Y, Z) plane."""
    n = theta.shape[0]
    if n == 0 or band_thickness <= 0.0:
        return np.empty((0, 2), dtype=np.float64)

    cy, cz = float(centroid_yz[0]), float(centroid_yz[1])
    by = cy + r_inner * np.cos(theta)
    bz = cz + r_inner * np.sin(theta)
    # Closed-loop chord lengths -> cumulative arc-length knots (n + 1 entries).
    dby = np.diff(np.append(by, by[0]))
    dbz = np.diff(np.append(bz, bz[0]))
    seg = np.sqrt(dby ** 2 + dbz ** 2)
    arc_knots = np.concatenate([[0.0], np.cumsum(seg)])
    perimeter = float(arc_knots[-1])
    if perimeter <= 0.0:
        return np.empty((0, 2), dtype=np.float64)

    # Periodic knot arrays aligned with arc_knots: close theta to 2pi, r to r[0].
    theta_knots = np.concatenate([theta, [2.0 * np.pi]])
    r_knots = np.concatenate([r_inner, [r_inner[0]]])

    # Stratified blue-noise in (arc, radial offset): columns (wrap=arc, width=offset).
    target_n = max(1, int(round(perimeter * band_thickness / (spacing * spacing))))
    samp = jittered_grid(width=band_thickness, wrap=perimeter, target_n=target_n, seed=seed)
    if samp.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    arc = samp[:, 0]
    offset = samp[:, 1]

    th = np.interp(arc, arc_knots, theta_knots)
    ri = np.interp(arc, arc_knots, r_knots)

    # Outward fade: keep with p = (1 - offset/T)^fade.
    rng = np.random.default_rng(seed)
    p_keep = np.clip(1.0 - offset / band_thickness, 0.0, 1.0) ** fade
    keep = rng.random(arc.shape[0]) < p_keep
    th, ri, offset = th[keep], ri[keep], offset[keep]

    r = ri + offset
    y = cy + r * np.cos(th)
    z = cz + r * np.sin(th)
    return np.column_stack([y, z])


if __name__ == "__main__":
    # --- displaced_ring_boundary: zero noise -> exact circle ---
    n = 720
    th0 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R, cz = 1000.0, 1500.0
    base = np.column_stack([np.zeros(n), R * np.cos(th0), cz + R * np.sin(th0)])
    nrm = np.column_stack([np.zeros(n), np.cos(th0), np.sin(th0)])
    centroid, theta, r_inner = displaced_ring_boundary(
        base, nrm, np.zeros(n), fbm_amplitude=300.0)
    assert np.allclose(centroid, [0.0, cz], atol=1e-6), centroid
    assert np.allclose(r_inner, R, atol=1e-6), (r_inner.min(), r_inner.max())
    assert np.all(np.diff(theta) >= 0), "theta not sorted"

    # constant inward noise shrinks the radius by amplitude * n01
    _, _, r_in2 = displaced_ring_boundary(base, nrm, np.full(n, 0.5), fbm_amplitude=300.0)
    assert np.allclose(r_in2, R - 0.5 * 300.0, atol=1e-6), r_in2.mean()

    # --- sample_band: beads outside boundary, density fades outward ---
    T, spacing = 120.0, 6.0
    pts = sample_band(centroid, theta, r_inner, band_thickness=T, fade=1.5,
                      spacing=spacing, seed=7)
    assert pts.shape[0] > 0, "no beads"
    r = np.sqrt((pts[:, 0] - centroid[0]) ** 2 + (pts[:, 1] - centroid[1]) ** 2)
    assert r.min() >= R - 1e-6, f"bead inside inner edge: {r.min()} < {R}"
    assert r.max() <= R + T + spacing, f"bead beyond band: {r.max()}"
    inner_half = int(((r - R) < T / 2).sum())
    outer_half = int(((r - R) >= T / 2).sum())
    assert inner_half > outer_half, f"fade not outward: {inner_half} vs {outer_half}"

    # determinism
    pts2 = sample_band(centroid, theta, r_inner, band_thickness=T, fade=1.5,
                       spacing=spacing, seed=7)
    assert np.array_equal(pts, pts2), "sample_band not deterministic"

    # no piles: no spacing-cell holds many beads
    keys = np.floor(pts / spacing).astype(np.int64)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    assert counts.max() <= 4, f"pile detected: max {counts.max()} per cell"

    print(f"sample_band: {pts.shape[0]} beads, "
          f"r in [{r.min():.1f}, {r.max():.1f}], max/cell {counts.max()}")
    print("OK")
```

- [ ] **Step 2: Run the smoke test**

Run: `python RhinoPython/pearlscape/cross_section.py`
Expected: a `sample_band: ... beads ...` line, then `OK`. (If you stubbed the bodies first, you'll instead see an `AssertionError`/`NotImplementedError` — then paste the real bodies and re-run to get `OK`.)

- [ ] **Step 3: Commit (user runs git)**

Stop and ask the user to commit. Suggested message: `Add in-plane curtain band sampling`

---

## Task 3: CylinderFBMCave.inner_boundary + protocol

**Files:**
- Modify: `RhinoPython/pearlscape/cave.py`
- Test: extend `RhinoPython/pearlscape/cross_section.py` `__main__`

- [ ] **Step 1: Extend the smoke test with a cylinder boundary check**

In `cross_section.py`, just before the final `print("OK")`, insert:

```python
    # --- craggy inner edge via CylinderFBMCave.inner_boundary ---
    from pearlscape.cave import CylinderFBMCave
    cave = CylinderFBMCave(
        radius=1200.0, length=2000.0, fbm_amplitude=900.0, fbm_base_freq=0.00035,
        fbm_octaves=6, fbm_lacunarity=2.0, fbm_gain=0.55, noise_seed=3,
        target_samples=1000, center_z=1500.0, noise_type="ridged",
    )
    c2, th2, ri2 = cave.inner_boundary(1000.0, 1440)
    assert ri2.shape == (1440,), ri2.shape
    assert ri2.max() <= 1200.0 + 1e-6, ri2.max()
    assert ri2.min() >= 1200.0 - 900.0 - 1e-6, ri2.min()
    band = sample_band(c2, th2, ri2, band_thickness=120.0, fade=1.5, spacing=6.0, seed=11)
    rel = band - c2
    bth = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
    br = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
    ri_at = np.interp(bth, np.concatenate([th2, [2.0 * np.pi]]),
                      np.concatenate([ri2, [ri2[0]]]))
    assert np.all(br >= ri_at - 1e-6), f"bead inside craggy edge: {(br - ri_at).min()}"
    print(f"cylinder inner_boundary: r_inner in [{ri2.min():.1f}, {ri2.max():.1f}], "
          f"{band.shape[0]} band beads")
```

- [ ] **Step 2: Run it to confirm it fails (method missing)**

Run: `python RhinoPython/pearlscape/cross_section.py`
Expected: FAIL — `AttributeError: 'CylinderFBMCave' object has no attribute 'inner_boundary'`.

- [ ] **Step 3: Add `inner_boundary` to the protocol**

In `cave.py`, inside `class CaveSurface(Protocol)`, after the existing `sample_surface_points` method, add:

```python
    def inner_boundary(self, plane_x: float, n_angular: int):
        """Return (centroid_yz (2,), theta (n_angular,), r_inner (n_angular,)) for
        the cave's craggy cross-section at X = plane_x, in the curtain's Y/Z plane.
        theta is sorted ascending in [0, 2*pi); r_inner is measured from centroid_yz."""
        ...
```

- [ ] **Step 4: Implement `inner_boundary` on `CylinderFBMCave`**

In `cave.py`, add this method to `class CylinderFBMCave`, right after `sample_surface_points`:

```python
    def inner_boundary(self, plane_x: float, n_angular: int):
        """Craggy cross-section boundary at X = plane_x. Reuses the exact noise
        convention of sample_surface_points so the boundary matches the wall."""
        from pearlscape.cross_section import displaced_ring_boundary

        th = np.linspace(0.0, 2.0 * math.pi, n_angular, endpoint=False)
        cos, sin = np.cos(th), np.sin(th)
        xcol = np.full(n_angular, float(plane_x))
        base_xyz = np.column_stack([
            xcol,
            self.radius * cos,
            self.center_z + self.radius * sin,
        ])
        # Outward normals: radial in Y/Z, zero along X.
        normals = np.column_stack([np.zeros(n_angular), cos, sin])
        # Noise sampled exactly as sample_surface_points: (R cos, R sin, x) * freq.
        base_noise = np.column_stack([
            self.radius * cos, self.radius * sin, xcol,
        ]) * self.fbm_base_freq
        perm = make_perm(self.noise_seed)
        noise_fn = ridged3_01 if self.noise_type == "ridged" else fbm3_01
        n01 = noise_fn(
            base_noise, perm,
            octaves=self.fbm_octaves, lacunarity=self.fbm_lacunarity, gain=self.fbm_gain,
        )
        return displaced_ring_boundary(
            base_xyz, normals, n01, fbm_amplitude=self.fbm_amplitude,
        )
```

(`math`, `numpy as np`, `fbm3_01`, `ridged3_01`, `make_perm` are already imported at the top of `cave.py`.)

- [ ] **Step 5: Run the smoke test to confirm it passes**

Run: `python RhinoPython/pearlscape/cross_section.py`
Expected: both new lines print (`cylinder inner_boundary: ...`) and `OK`.

- [ ] **Step 6: Commit (user runs git)**

Stop and ask the user to commit. Suggested message: `Add cross-section boundary to cylinder cave`

---

## Task 4: NurbsLoftCave.inner_boundary (Rhino-verified)

**Files:**
- Modify: `RhinoPython/pearlscape/nurbs_cave.py`

This path needs Rhino (surface evaluation), so it is verified in Rhino, not headlessly. The numpy boundary math is already covered by Task 2/3 via the shared `displaced_ring_boundary`.

- [ ] **Step 1: Add a cached grid/normals to `NurbsLoftCave`**

In `nurbs_cave.py`, replace the `NurbsLoftCave.__init__` body so it caches lazily:

```python
    def __init__(self, surface, params) -> None:
        self.surface = surface
        self.params = params
        self._grid = None       # (nu, nv, 3) base-surface eval, cached
        self._normals = None    # (nu, nv, 3) per-node unit normals, cached

    def _ensure_grid(self):
        """Evaluate (and cache) the base surface grid + normals once."""
        if self._grid is None:
            p = self.params
            self._grid = eval_surface_grid(self.surface, p.nurbs_grid_u, p.nurbs_grid_v)
            self._normals = surface_sampling.grid_normals(self._grid)
        return self._grid, self._normals
```

- [ ] **Step 2: Add `inner_boundary` to `NurbsLoftCave`**

In `nurbs_cave.py`, add this method to `class NurbsLoftCave` (after `sample_surface_points`):

```python
    def inner_boundary(self, plane_x: float, n_angular: int):
        """Craggy cross-section boundary at X = plane_x, derived from the cached
        base grid + the same noise convention as sample_and_displace."""
        from pearlscape.cross_section import displaced_ring_boundary

        grid, normals = self._ensure_grid()
        nu, nv = grid.shape[0], grid.shape[1]

        # Interpolate the base ring at the u where X == plane_x. X is ~monotonic
        # along u (centerline meanders in Y/Z only), so per-u mean X is a safe key.
        xu = grid[:, :, 0].mean(axis=1)                      # (nu,)
        u_frac = float(np.interp(plane_x, xu, np.arange(nu)))
        u0 = int(np.clip(np.floor(u_frac), 0, nu - 2))
        tu = u_frac - u0
        ring = (1.0 - tu) * grid[u0] + tu * grid[u0 + 1]     # (nv, 3)
        ring_n = (1.0 - tu) * normals[u0] + tu * normals[u0 + 1]

        # Resample the periodic ring to n_angular points.
        v_idx = np.linspace(0.0, nv, n_angular, endpoint=False)
        v0 = np.floor(v_idx).astype(np.int64) % nv
        v1 = (v0 + 1) % nv
        tv = (v_idx - np.floor(v_idx))[:, None]
        base_xyz = (1.0 - tv) * ring[v0] + tv * ring[v1]     # (n_angular, 3)
        nrm = (1.0 - tv) * ring_n[v0] + tv * ring_n[v1]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

        # Orient normals outward from the ring's center (match sample_and_displace).
        center3 = base_xyz.mean(axis=0)
        outward = base_xyz - center3
        sign = np.sign(np.sum(nrm * outward, axis=1, keepdims=True))
        sign = np.where(sign == 0, 1.0, sign)
        nrm = nrm * sign

        # Noise at world coordinates (matches sample_and_displace: base_pts * freq).
        p = self.params
        perm = make_perm(p.noise_seed)
        noise_fn = ridged3_01 if p.noise_type == "ridged" else fbm3_01
        n01 = noise_fn(
            base_xyz * p.fbm_base_freq, perm,
            octaves=p.fbm_octaves, lacunarity=p.fbm_lacunarity, gain=p.fbm_gain,
        )
        return displaced_ring_boundary(
            base_xyz, nrm, n01, fbm_amplitude=p.fbm_amplitude,
        )
```

(`np`, `fbm3_01`, `ridged3_01`, `make_perm`, `surface_sampling`, and `eval_surface_grid` are already imported/defined at module top in `nurbs_cave.py`.)

- [ ] **Step 3: Verify in Rhino (manual)**

This step requires Task 5 + 6 to be wired, so it is exercised by the Task 6 Rhino run. For now just confirm the module imports cleanly under the headless runtime (it will fail to import `Rhino` — that is expected and fine; we only check there is no Python syntax error):

Run: `python -c "import ast; ast.parse(open('RhinoPython/pearlscape/nurbs_cave.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 4: Commit (user runs git)**

Stop and ask the user to commit. Suggested message: `Add cross-section boundary to nurbs cave`

---

## Task 5: Rewrite curtains.py — build_curtains

**Files:**
- Rewrite: `RhinoPython/pearlscape/curtains.py`

- [ ] **Step 1: Replace the file contents**

Replace the entire contents of `curtains.py` with:

```python
#! python 3
# r: numpy
"""Build per-curtain bead point sets by sampling each curtain plane outward from
the cave's craggy cross-section boundary. Replaces the old slice-and-project
approach: beads are placed directly in the plane, so there is no projection
stacking and the inner cave edge stays sharp."""

from typing import List, Sequence

import numpy as np

from pearlscape import cross_section


def curtain_x_positions(curtain_count: int, spacing: float, x_center: float) -> np.ndarray:
    """Return the X coordinate of each curtain plane, centered on x_center."""
    offsets = (np.arange(curtain_count) - (curtain_count - 1) / 2.0) * spacing
    return offsets + x_center


def array_x_center(cave_length: float) -> float:
    """The curtain array centers on the cave's midpoint along X."""
    return cave_length / 2.0


def build_curtains(cave, plane_xs: Sequence[float], params) -> List[dict]:
    """For each curtain plane, sample beads in-plane outside the cave's cross-
    section boundary, fading outward. Returns one dict per curtain:
        {'plane_x': float,
         'points_2d': np.ndarray (M, 2)  # columns (Y, Z), the in-plane beads
         'points_3d': np.ndarray (M, 3)} # (plane_x, Y, Z), for colour sampling
    """
    # Angular resolution for the boundary curve: ~2 samples per bead spacing
    # around the nominal circumference (finer than bead size is wasted detail).
    n_angular = max(720, int(2.0 * np.pi * params.cave_radius / params.bead_min_spacing))

    curtains: List[dict] = []
    for i, plane_x in enumerate(plane_xs):
        plane_x = float(plane_x)
        centroid, theta, r_inner = cave.inner_boundary(plane_x, n_angular)
        pts_2d = cross_section.sample_band(
            centroid, theta, r_inner,
            band_thickness=params.curtain_band_thickness,
            fade=params.curtain_band_fade,
            spacing=params.bead_min_spacing,
            seed=params.noise_seed + 1000 + i,
        )
        if pts_2d.shape[0]:
            pts_3d = np.column_stack([
                np.full(pts_2d.shape[0], plane_x),
                pts_2d[:, 0],
                pts_2d[:, 1],
            ])
        else:
            pts_3d = np.zeros((0, 3), dtype=np.float64)
        curtains.append({
            "plane_x": plane_x,
            "points_2d": pts_2d,
            "points_3d": pts_3d,
        })
    return curtains


if __name__ == "__main__":
    # Headless smoke test on the pure-numpy cylinder cave.
    from pearlscape.cave import CylinderFBMCave
    from pearlscape.params import PearlscapeParams

    params = PearlscapeParams()   # uses default band params (120 / 1.5 / 6)
    cave = CylinderFBMCave(
        radius=params.cave_radius, length=2000.0, fbm_amplitude=900.0,
        fbm_base_freq=0.00035, fbm_octaves=6, fbm_lacunarity=2.0, fbm_gain=0.55,
        noise_seed=params.noise_seed, target_samples=1000, center_z=1500.0,
        noise_type="ridged",
    )
    xs = curtain_x_positions(5, 100.0, array_x_center(2000.0))
    curtains = build_curtains(cave, xs, params)

    assert len(curtains) == 5, len(curtains)
    for c in curtains:
        assert c["points_2d"].ndim == 2 and c["points_2d"].shape[1] == 2
        assert c["points_3d"].shape == (c["points_2d"].shape[0], 3)
        if c["points_3d"].shape[0]:
            assert np.allclose(c["points_3d"][:, 0], c["plane_x"])
            assert np.allclose(c["points_3d"][:, 1], c["points_2d"][:, 0])
            assert np.allclose(c["points_3d"][:, 2], c["points_2d"][:, 1])
    total = sum(len(c["points_2d"]) for c in curtains)
    assert total > 0, "no beads generated"

    # Determinism: same params -> identical beads.
    curtains2 = build_curtains(cave, xs, params)
    same = all(np.array_equal(a["points_2d"], b["points_2d"])
               for a, b in zip(curtains, curtains2))
    assert same, "build_curtains not deterministic"

    print(f"5 curtains, {total} beads, deterministic")
    print("OK")
```

- [ ] **Step 2: Run the curtains smoke test**

Run: `python RhinoPython/pearlscape/curtains.py`
Expected: `5 curtains, <N> beads, deterministic` then `OK`, with N in the low tens of thousands.

- [ ] **Step 3: Commit (user runs git)**

Stop and ask the user to commit. Suggested message: `Rewrite curtains to sample in-plane from cross-section`

---

## Task 6: Wire build_scene.py + skip surface sampling in curtain modes

**Files:**
- Modify: `RhinoPython/build_scene.py`

- [ ] **Step 1: Update the curtains import**

In `build_scene.py` line 24, change:

```python
from pearlscape.curtains import array_x_center, curtain_x_positions, slice_and_project
```

to:

```python
from pearlscape.curtains import array_x_center, curtain_x_positions, build_curtains
```

- [ ] **Step 2: Move surface sampling into the cave-only branch and call build_curtains**

In `build_scene.py`, replace the block from line 87 (`t0 = time.time()`) through line 125 (the `print(f"Curtains: ...")` line) — i.e. the current surface-sampling call, the `if params.pipeline_mode == "cave":` branch, and the slice-and-project call — with:

```python
    cave = make_default_cave(params)

    if params.pipeline_mode == "cave":
        # Mode 1 — raw cave only: single coloured PointCloud on CaveReference.
        t0 = time.time()
        pts = cave.sample_surface_points()
        print(f"Cave: {len(pts)} surface points in {time.time()-t0:.2f}s")
        t0 = time.time()
        colors = color_mod.assign_colors(
            pts,
            palette=params.palette,
            base_freq=params.color_base_freq,
            octaves=params.color_fbm_octaves,
            lacunarity=params.color_fbm_lacunarity,
            gain=params.color_fbm_gain,
            seed=params.color_noise_seed,
            dither=params.color_dither,
        )
        print(f"Colors assigned in {time.time()-t0:.2f}s")
        if params.display_mode == "sprites":
            display.render_sprites(pts, colors, params.bead_diameter)
        else:
            display.render_cave_reference(pts, colors=colors)
        print(f"Rendered cave only ({params.display_mode}, "
              f"pipeline_mode={params.pipeline_mode!r}).")
        print_run_summary(pts, params)
        return

    # Modes "curtains" and "export": sample each curtain plane directly outward
    # from the cave's cross-section boundary. The dense surface cloud is NOT
    # produced in these modes — only the cross-section boundary is needed.
    t0 = time.time()
    x_center = array_x_center(params.cave_length)
    plane_xs = curtain_x_positions(
        params.curtain_count, params.curtain_spacing, x_center
    )
    curtains = build_curtains(cave, plane_xs, params)
    total = sum(len(c["points_2d"]) for c in curtains)
    print(f"Curtains: {len(curtains)} planes, {total} beads in {time.time()-t0:.2f}s")
```

Note: the original code had the `cave`/`sample_surface_points` call and the first `print(f"Cave: ...")` *above* the mode check (old lines 88–90); those are now folded into the `"cave"` branch above, so delete the old standalone lines 88–90. The existing `"cave"` branch body (old lines 93–113) is replaced by the version inside the new block.

- [ ] **Step 3: Confirm the rest of `main()` still lines up**

After your edit, the code immediately following must be the existing colour-assignment-for-curtains call (old line 127 onward):

```python
    t0 = time.time()
    color_mod.apply_to_curtains(curtains, params)
    print(f"Colors assigned in {time.time()-t0:.2f}s")

    plane_xs = curtain_x_positions(
        params.curtain_count, params.curtain_spacing, x_center
    )
```

The second `plane_xs = curtain_x_positions(...)` is now redundant (we computed `plane_xs` above). Delete that redundant re-assignment block (the three lines), leaving the subsequent `display.render_curtain_planes(plane_xs, ...)` call intact and using the `plane_xs` computed earlier.

- [ ] **Step 4: Syntax-check headlessly**

Run: `python -c "import ast; ast.parse(open('RhinoPython/build_scene.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 5: Verify in Rhino (manual — the real test)**

In Rhino 8, open `RhinoPython/build_scene.py` in the new Script Editor and press F5 for each of these `params.py` settings, confirming the console shows no errors and a sane bead count, and the viewport looks right:

1. `pipeline_mode="curtains"`, `cave_type="nurbs"`, `display_mode="sprites"` — curtains show a sharp inner cave edge with a soft outward fade; **no banding gaps** at the default `curtain_band_thickness=120`. Console: `Curtains: 50 planes, <N> beads`.
2. `pipeline_mode="cave"` — unchanged single reference cloud still renders.
3. `cave_type="cylinder"`, `pipeline_mode="curtains"` — also renders correctly.

Tuning to expect: raise `curtain_band_thickness` if gaps remain; raise `curtain_band_fade` to pull beads tighter to the inner edge; `bead_min_spacing` controls density.

- [ ] **Step 6: Commit (user runs git)**

Stop and ask the user to commit. Suggested message: `Sample curtains from cross-section in build pipeline`

---

## Task 7: Update README

**Files:**
- Modify: `README.md`

Per project convention, the README documents *as-designed* defaults; live-tuned values live in `params.py` and the drift is intentional — do not "fix" it.

- [ ] **Step 1: Revise the curtain-method description**

In `README.md`, update the project intro paragraph (line 3) and the `"curtains"` pipeline-mode row (line 28) so they describe direct in-plane sampling rather than slicing/projecting. Replace line 3's clause "the beads on each curtain trace a thin slice of a virtual cave" with:

```
The beads on each curtain fill the plane outward from the cave's cross-section curve at that depth — a sharp inner cave edge with a soft outer volume — so that, viewed perpendicular to the stack, the layered slices reconstruct the cave's interior depth.
```

And change the `"curtains"` row's "What it produces" cell to:

```
Cave is sliced into 50 curtain planes; each plane's beads are sampled in-plane, outward from the cave's cross-section curve.
```

- [ ] **Step 2: Add the new parameters to the Curtain array table**

In `README.md`, in the "Curtain array" parameter table (around line 89), add these rows after `cave_center_z`:

```
| `curtain_band_thickness` | `120.0`   | Max outward reach of beads from the inner cross-section curve (mm). The gap-bridging knob — raise to close inter-curtain gaps. |
| `curtain_band_fade`      | `1.5`     | Outward density falloff exponent: `p = (1 − d/T)^fade`. Higher = beads hug the inner edge tighter. |
| `bead_min_spacing`       | `6.0`     | In-plane blue-noise spacing (mm). Must be ≥ `bead_diameter`. |
```

- [ ] **Step 3: Note the bead-count behavior change**

In `README.md`, in the Beads section (around line 106), after the Bridson under-fill note, add:

```
Note: in `"curtains"`/`"export"` modes the bead count is **not** governed by `total_surface_samples`. Beads are sampled directly in each curtain plane, so the count emerges from the band geometry — tune it via `curtain_band_thickness` and `bead_min_spacing`. `total_surface_samples` governs only the `"cave"` reference cloud.
```

- [ ] **Step 4: Commit (user runs git)**

Stop and ask the user to commit. Suggested message: `Document in-plane curtain sampling`

---

## Self-review notes (already checked)

- **Spec coverage:** boundary analytic extraction (Tasks 3, 4), in-plane band sampling + outward fade (Task 2), params (Task 1), curtains orchestration + dict contract (Task 5), skip surface sampling in curtain modes + wiring (Task 6), docs (Task 7). Colour needs no code change (reads `points_3d`, provided by `build_curtains`). All spec sections map to a task.
- **Type/name consistency:** `displaced_ring_boundary(base_xyz, normals, n01, *, fbm_amplitude) -> (centroid_yz, theta, r_inner)` and `sample_band(centroid_yz, theta, r_inner, *, band_thickness, fade, spacing, seed) -> (M,2)` are used identically by both caves and `build_curtains`. The per-curtain dict keys `plane_x` / `points_2d` / `points_3d` (+ `colors` added later by `color.apply_to_curtains`) match what `display.py` and `export.py` already consume.
- **No placeholders:** every code step shows complete code; every test step gives an exact command and expected output. The only manual steps are the Rhino verifications (Task 4 Step 3, Task 6 Step 5), unavoidable because they require the Rhino surface eval / viewport.
```
