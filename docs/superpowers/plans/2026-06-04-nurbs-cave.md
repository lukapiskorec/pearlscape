# Irregular NURBS Cave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `cave_type="nurbs"` cave — an irregular lofted-NURBS tube whose surface is sampled (600k–1.2M beads, <~10s) and displaced with the existing ridged/FBM noise — alongside the existing cylinder cave, with the base surface kept as editable document geometry (rebuild/reuse toggle).

**Architecture:** Build the lofted NURBS tube in Rhino, evaluate it once on a regular `(u,v)` grid, then do all mass work in numpy: jittered-grid blue-noise sampling in arc-length-scaled param space, bilinear map to 3D, inward displacement along grid-derived normals. The Rhino-dependent geometry lives in `nurbs_cave.py`; the headless-testable numpy pipeline lives in `surface_sampling.py` and an extension to `sampling.py`.

**Tech Stack:** Rhino 8 CPython 3, numpy, RhinoCommon (`Rhino.Geometry` loft/surface eval). Spec: `docs/superpowers/specs/2026-06-04-nurbs-cave-design.md`.

---

## Conventions for this plan

- **Commits are the user's.** Where a task ends in "review and commit," pause for the user; do not run `git commit`. Suggested plain-English titles are given (no Conventional-Commits prefixes).
- **Headless tests** (pure-numpy modules) run via the project's `__main__`-smoke-test convention:
  `python -c "import runpy; runpy.run_path('pearlscape/<mod>.py', run_name='__main__')"` from `RhinoPython/`. (Plain `python pearlscape/<mod>.py` fails — the `#! python 3` shebang confuses the Windows `py` launcher; `runpy` sidesteps it.)
- **Rhino-dependent modules** (`nurbs_cave.py`, `display.py` additions) can't run headless. Verify by byte-compile (`python -c "import py_compile; py_compile.compile('pearlscape/<mod>.py', doraise=True)"`) plus the manual F5 checklist in Task 7. Delete any `__pycache__` created by byte-compiling.
- All file paths are relative to the repo root `C:\Users\lukap\Documents\GitHub\pearlscape`.

## File structure

- **Create** `RhinoPython/pearlscape/surface_sampling.py` — pure numpy: `grid_normals`, `_bilinear`, `sample_and_displace`. Headless-testable.
- **Modify** `RhinoPython/pearlscape/sampling.py` — add `jittered_grid`; extend `__main__` smoke test.
- **Create** `RhinoPython/pearlscape/nurbs_cave.py` — Rhino: `build_loft_surface`, `eval_surface_grid`, `NurbsLoftCave`, `make_nurbs_cave`.
- **Modify** `RhinoPython/pearlscape/display.py` — add `render_cave_surface`, `find_cave_surface`, `CAVE_SURFACE_LAYER`.
- **Modify** `RhinoPython/pearlscape/cave.py` — `make_default_cave` dispatches on `cave_type`.
- **Modify** `RhinoPython/pearlscape/params.py` — `cave_type`, `nurbs_*` params, `validate()` checks.
- **Modify** `README.md` — document the NURBS cave.

---

## Task 1: Parameters

**Files:**
- Modify: `RhinoPython/pearlscape/params.py`

- [ ] **Step 1: Add the new parameters**

In the `PearlscapeParams` dataclass, after the `noise_seed` field (keep existing fields), add a NURBS block. Insert immediately before the `# --- Curtain array (mm) ---` comment:

```python
    # --- Cave geometry (NURBS, used when cave_type == "nurbs") ---
    # "cylinder" -> CylinderFBMCave (radial noise on a cylinder).
    # "nurbs"    -> NurbsLoftCave (lofted irregular tube, sampled + displaced).
    cave_type: str = "cylinder"
    # "rebuild" -> loft from the nurbs_* params each run (replaces the surface
    #              on the Pearlscape::CaveSurface layer).
    # "reuse"   -> sample the existing (possibly hand-edited) surface on that
    #              layer; falls back to "rebuild" with a warning if none found.
    nurbs_surface_source: str = "rebuild"
    nurbs_sections: int = 8            # K: cross-section rings along X
    nurbs_section_points: int = 12     # P: control points per ring
    nurbs_radius_jitter: float = 0.35  # per-point radius variation, fraction of cave_radius
    nurbs_radius_jitter_freq: float = 0.0006   # cycles/mm
    nurbs_centerline_amp: float = 400.0        # meander amplitude (mm, Y/Z only)
    nurbs_centerline_freq: float = 0.0003      # cycles/mm
    nurbs_shape_seed: int = 7          # seed for shape noise (independent of noise_seed)
    nurbs_grid_u: int = 180            # surface-eval grid divisions along X
    nurbs_grid_v: int = 180            # surface-eval grid divisions around theta
```

- [ ] **Step 2: Add validation**

In `validate()`, after the existing `assert self.display_mode ...` lines and before `assert len(self.palette) >= 2`, add:

```python
        assert self.cave_type in ("cylinder", "nurbs")
        assert self.nurbs_surface_source in ("rebuild", "reuse")
        assert self.nurbs_sections >= 2
        assert self.nurbs_section_points >= 3
        assert 0.0 <= self.nurbs_radius_jitter < 1.0
        assert self.nurbs_grid_u >= 2 and self.nurbs_grid_v >= 3
```

- [ ] **Step 3: Verify**

Run from `RhinoPython/`:

```
python -c "import importlib.util as u; s=u.spec_from_file_location('params','pearlscape/params.py'); m=u.module_from_spec(s); s.loader.exec_module(m); p=m.PearlscapeParams(); p.validate(); print('default cave_type', p.cave_type); p.cave_type='nurbs'; p.validate(); p.nurbs_surface_source='reuse'; p.validate(); print('nurbs+reuse OK');
import traceback
try:
    p.nurbs_radius_jitter=1.5; p.validate(); print('ERROR: bad jitter accepted')
except AssertionError: print('bad jitter rejected')"
```

Expected: prints `default cave_type cylinder`, `nurbs+reuse OK`, `bad jitter rejected`.

- [ ] **Step 4: Review and commit** (user) — suggested title: `Add NURBS cave parameters`

---

## Task 2: Jittered-grid sampler

**Files:**
- Modify: `RhinoPython/pearlscape/sampling.py`

- [ ] **Step 1: Add the `jittered_grid` function**

After `bridson_torus` (before the `if __name__ == "__main__":` block), add:

```python
def jittered_grid(
    width: float,
    wrap: float,
    target_n: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Stratified jittered sampling of a 2D domain [0, wrap) x [0, width).

    One point per grid cell, uniformly jittered within the cell. Vectorized —
    sub-second for ~1M points. Columns are (wrap_axis, width_axis), matching
    bridson_torus, so callers can swap samplers freely. The wrap axis is treated
    as periodic only implicitly (cells tile it exactly); no wrap arithmetic is
    needed because points never need neighbour queries.

    The cell count is chosen to make cells ~square and the total ~target_n.
    """
    if target_n <= 0:
        return np.empty((0, 2), dtype=np.float64)
    area = max(width, 1e-9) * max(wrap, 1e-9)
    cell = math.sqrt(area / target_n)
    ncols = max(1, int(round(wrap / cell)))    # along the wrap axis
    nrows = max(1, int(round(width / cell)))   # along the width axis
    cw = wrap / ncols
    ch = width / nrows

    rng = np.random.default_rng(seed)
    cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows), indexing="xy")
    cols = cols.ravel()
    rows = rows.ravel()
    n = cols.size
    wrap_coord = cols * cw + rng.uniform(0.0, cw, size=n)
    width_coord = rows * ch + rng.uniform(0.0, ch, size=n)
    return np.column_stack([wrap_coord, width_coord])
```

`math` and `numpy as np` are already imported at the top of `sampling.py`.

- [ ] **Step 2: Extend the `__main__` smoke test**

At the end of the `if __name__ == "__main__":` block in `sampling.py`, append:

```python
    print("\n--- jittered_grid ---")
    jpts = jittered_grid(width=length, wrap=wrap, target_n=target, seed=0)
    print(f"Requested {target}, got {len(jpts)} "
          f"(within 15%: {abs(len(jpts) - target) < 0.15 * target})")
    assert jpts[:, 0].min() >= 0.0 and jpts[:, 0].max() < wrap + 1e-6
    assert jpts[:, 1].min() >= 0.0 and jpts[:, 1].max() < length + 1e-6
    jpts2 = jittered_grid(width=length, wrap=wrap, target_n=target, seed=0)
    print("Deterministic:", bool(np.array_equal(jpts, jpts2)))
    print("In-domain + count OK")
```

- [ ] **Step 3: Verify**

Run from `RhinoPython/`:

```
python -c "import runpy; runpy.run_path('pearlscape/sampling.py', run_name='__main__')"
```

Expected: existing Bridson output, then a `--- jittered_grid ---` section reporting count within 15%, `Deterministic: True`, `In-domain + count OK`. No `AssertionError`.

- [ ] **Step 4: Review and commit** (user) — suggested title: `Add vectorized jittered-grid sampler`

---

## Task 3: Surface sampling + displacement (numpy)

**Files:**
- Create: `RhinoPython/pearlscape/surface_sampling.py`

- [ ] **Step 1: Create the module with the numpy pipeline**

Create `RhinoPython/pearlscape/surface_sampling.py` with exactly:

```python
#! python 3
# r: numpy
"""Sample a NURBS surface (given as an evaluated (nu, nv, 3) grid) into a dense
displaced point cloud — pure numpy, no Rhino. Kept separate from nurbs_cave.py
so this pipeline can be tested headlessly.

Grid convention: axis 0 is u (along the cave's X, open); axis 1 is v (around the
cross-section, periodic).
"""

import os
import sys

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape.noise import fbm3_01, ridged3_01, make_perm
from pearlscape.sampling import jittered_grid


def grid_normals(grid_xyz: np.ndarray) -> np.ndarray:
    """Per-node unit normals from cross-products of grid tangents.

    u (axis 0) is open (one-sided differences at the ends); v (axis 1) is
    periodic (np.roll). Orientation is arbitrary here; sample_and_displace
    re-orients per sample relative to the centerline.
    """
    g = grid_xyz
    du = np.empty_like(g)
    du[1:-1] = g[2:] - g[:-2]
    du[0] = g[1] - g[0]
    du[-1] = g[-1] - g[-2]
    dv = np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return n / norm


def _bilinear(grid: np.ndarray, u_frac: np.ndarray, v_frac: np.ndarray) -> np.ndarray:
    """Bilinearly sample `grid` (nu, nv, C) at fractional indices.

    u clamps to [0, nu-1] (open); v wraps modulo nv (periodic).
    Returns (N, C).
    """
    nu, nv = grid.shape[0], grid.shape[1]
    uf = np.clip(u_frac, 0.0, nu - 1.0)
    u0 = np.floor(uf).astype(np.int64)
    u0 = np.clip(u0, 0, nu - 2)
    u1 = u0 + 1
    tu = (uf - u0)[:, None]

    v0 = np.floor(v_frac).astype(np.int64) % nv
    v1 = (v0 + 1) % nv
    tv = (v_frac - np.floor(v_frac))[:, None]

    g00 = grid[u0, v0]
    g01 = grid[u0, v1]
    g10 = grid[u1, v0]
    g11 = grid[u1, v1]
    return (
        (1 - tu) * (1 - tv) * g00
        + (1 - tu) * tv * g01
        + tu * (1 - tv) * g10
        + tu * tv * g11
    )


def sample_and_displace(
    grid_xyz: np.ndarray,
    base_normals: np.ndarray,
    *,
    target: int,
    fbm_amplitude: float,
    fbm_base_freq: float,
    octaves: int,
    lacunarity: float,
    gain: float,
    noise_type: str,
    noise_seed: int,
    sample_seed: int = 0,
) -> np.ndarray:
    """Blue-noise-sample the surface grid and displace inward by ridged/FBM noise.

    Returns an (N, 3) array of displaced surface points in world coordinates.
    """
    nu, nv = grid_xyz.shape[0], grid_xyz.shape[1]

    # Approximate arc-lengths so jittered spacing is ~uniform in mm.
    centerline = grid_xyz.mean(axis=1)                       # (nu, 3)
    seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    width_len = float(seg.sum()) or 1.0                      # along u (~cave length)
    ring_edges = np.linalg.norm(
        np.roll(grid_xyz, -1, axis=1) - grid_xyz, axis=2
    ).sum(axis=1)                                            # perimeter per u
    wrap_len = float(ring_edges.mean()) or 1.0               # around v (~circumference)

    samp = jittered_grid(width=width_len, wrap=wrap_len, target_n=target, seed=sample_seed)
    if samp.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    wrap_coord = samp[:, 0]
    width_coord = samp[:, 1]

    u_frac = width_coord / width_len * (nu - 1)
    v_frac = wrap_coord / wrap_len * nv                      # periodic; _bilinear mods nv

    base_pts = _bilinear(grid_xyz, u_frac, v_frac)
    nrm = _bilinear(base_normals, u_frac, v_frac)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

    # Orient outward relative to the centerline at each sample's u.
    c_at_u = _bilinear(centerline[:, None, :].repeat(2, axis=1), u_frac, v_frac * 0.0)
    outward = base_pts - c_at_u
    sign = np.sign(np.sum(nrm * outward, axis=1, keepdims=True))
    sign = np.where(sign == 0, 1.0, sign)
    nrm = nrm * sign

    perm = make_perm(noise_seed)
    noise_fn = ridged3_01 if noise_type == "ridged" else fbm3_01
    n01 = noise_fn(
        base_pts * fbm_base_freq, perm,
        octaves=octaves, lacunarity=lacunarity, gain=gain,
    )

    return base_pts - fbm_amplitude * n01[:, None] * nrm
```

Note on `c_at_u`: centerline is `(nu, 3)`; we lift it to a `(nu, 2, 3)` grid (constant across v) and bilinearly sample at `v_frac*0` so only u interpolates — giving the centerline point at each sample's u.

- [ ] **Step 2: Add a headless smoke test**

Append to `surface_sampling.py`:

```python
if __name__ == "__main__":
    import time

    # Synthetic analytic cylinder grid (radius R along X), to exercise the
    # numpy pipeline without Rhino.
    nu, nv = 120, 120
    R, length = 1000.0, 5000.0
    u = np.linspace(0.0, length, nu)
    v = np.linspace(0.0, 2 * np.pi, nv, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    grid = np.stack([uu, R * np.cos(vv), R * np.sin(vv)], axis=2)

    normals = grid_normals(grid)
    assert np.allclose(np.linalg.norm(normals, axis=2), 1.0, atol=1e-6), "normals not unit"

    target = 200_000
    t0 = time.time()
    pts = sample_and_displace(
        grid, normals,
        target=target, fbm_amplitude=300.0, fbm_base_freq=0.0008,
        octaves=4, lacunarity=2.0, gain=0.5, noise_type="ridged", noise_seed=1,
    )
    dt = time.time() - t0
    print(f"Sampled {len(pts)} pts (target {target}) in {dt:.2f}s")
    assert abs(len(pts) - target) < 0.2 * target, "count off"
    assert not np.isnan(pts).any(), "NaNs in output"

    # Every displaced point must lie within [R - amplitude, R] of the axis
    # (inward displacement only), with a small tolerance for grid discretization.
    radial = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
    assert radial.max() <= R + 1.0, f"point outside base surface: {radial.max()}"
    assert radial.min() >= R - 300.0 - 1.0, f"over-displaced: {radial.min()}"

    pts2 = sample_and_displace(
        grid, normals,
        target=target, fbm_amplitude=300.0, fbm_base_freq=0.0008,
        octaves=4, lacunarity=2.0, gain=0.5, noise_type="ridged", noise_seed=1,
    )
    print("Deterministic:", bool(np.array_equal(pts, pts2)))
    print("X spans cave:", float(pts[:, 0].min()), "->", float(pts[:, 0].max()))
    print("OK")
```

- [ ] **Step 3: Verify**

Run from `RhinoPython/`:

```
python -c "import runpy; runpy.run_path('pearlscape/surface_sampling.py', run_name='__main__')"
```

Expected: `Sampled ~200000 pts ... in <Xs`, `Deterministic: True`, an X-span ≈ `0.0 -> 5000.0`, `OK`. No `AssertionError`. (Confirms count, inward-only displacement, unit normals, no NaNs, determinism.)

- [ ] **Step 4: Review and commit** (user) — suggested title: `Add numpy surface sampling and displacement`

---

## Task 4: Document-surface helpers (display.py)

**Files:**
- Modify: `RhinoPython/pearlscape/display.py`

- [ ] **Step 1: Add the layer constant**

Next to the other layer constants near the top of `display.py` (after `CURTAIN_PLANES_LAYER = "CurtainPlanes"`), add:

```python
CAVE_SURFACE_LAYER = "CaveSurface"
```

- [ ] **Step 2: Add render/find helpers**

Add these two functions (e.g. after `render_curtain_planes`):

```python
def render_cave_surface(surface) -> None:
    """Add the base NURBS cave surface to Pearlscape::CaveSurface, replacing any
    surface already on that layer. Real document geometry, so it persists across
    F5 runs and can be hand-edited (see nurbs_surface_source="reuse")."""
    doc = sc.doc
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CAVE_SURFACE_LAYER}"
    layer_idx = _ensure_layer(layer_path)
    # Clear any existing objects on the layer first.
    layer = doc.Layers[layer_idx]
    existing = doc.Objects.FindByLayer(layer)
    if existing:
        for obj in existing:
            doc.Objects.Delete(obj, True)
    attrs = Rhino.DocObjects.ObjectAttributes()
    attrs.LayerIndex = layer_idx
    doc.Objects.AddSurface(surface, attrs)
    doc.Views.Redraw()


def find_cave_surface():
    """Return the first surface on Pearlscape::CaveSurface, or None.

    Accepts a raw Surface or a single-face Brep (what AddSurface round-trips to);
    returns a NurbsSurface for uniform downstream evaluation.
    """
    doc = sc.doc
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CAVE_SURFACE_LAYER}"
    idx = doc.Layers.FindByFullPath(layer_path, -1)
    if idx < 0:
        return None
    layer = doc.Layers[idx]
    objs = doc.Objects.FindByLayer(layer)
    if not objs:
        return None
    geo = objs[0].Geometry
    if isinstance(geo, rg.Brep):
        if geo.Faces.Count == 0:
            return None
        return geo.Faces[0].UnderlyingSurface().ToNurbsSurface()
    if isinstance(geo, rg.Surface):
        return geo.ToNurbsSurface()
    return None
```

- [ ] **Step 3: Byte-compile check**

Run from `RhinoPython/`:

```
python -c "import py_compile; py_compile.compile('pearlscape/display.py', doraise=True); print('display.py OK')"
```

Expected: `display.py OK`. Then delete any created cache: `python -c "import shutil; shutil.rmtree('pearlscape/__pycache__', ignore_errors=True)"`.

(Full behavior is verified in Task 7 under Rhino.)

- [ ] **Step 4: Review and commit** (user) — suggested title: `Add cave-surface document helpers`

---

## Task 5: NURBS cave (nurbs_cave.py)

**Files:**
- Create: `RhinoPython/pearlscape/nurbs_cave.py`

- [ ] **Step 1: Create the module**

Create `RhinoPython/pearlscape/nurbs_cave.py` with exactly:

```python
#! python 3
# r: numpy
"""Irregular lofted-NURBS cave: build a tube from jittered cross-section rings,
evaluate it on a grid, then hand off to surface_sampling for the dense displaced
point cloud. Rhino-dependent (loft + surface eval); the numpy mass-work lives in
surface_sampling.py.
"""

import math
import os
import sys

import numpy as np

import Rhino.Geometry as rg

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape.noise import fbm3_01, make_perm
from pearlscape import surface_sampling


def build_loft_surface(params):
    """Loft K jittered closed rings into an irregular NURBS tube; return its
    NurbsSurface. Centerline meanders in Y/Z only (X stays the slicing axis)."""
    K = params.nurbs_sections
    P = params.nurbs_section_points
    R = params.cave_radius
    perm = make_perm(params.nurbs_shape_seed)

    def noise01_at(pt_xyz, freq, tag):
        # One scalar of smooth FBM noise at a tagged location, in [0, 1].
        q = np.array([[pt_xyz[0] * freq + tag * 13.7,
                       pt_xyz[1] * freq + tag * 7.1,
                       pt_xyz[2] * freq + tag * 3.3]])
        return float(fbm3_01(q, perm, octaves=3, lacunarity=2.0, gain=0.5)[0])

    rings = []
    for k in range(K):
        x = params.cave_length * k / (K - 1)
        cy = (noise01_at((x, 0.0, 0.0), params.nurbs_centerline_freq, 1) * 2 - 1) \
            * params.nurbs_centerline_amp
        cz = (noise01_at((x, 0.0, 0.0), params.nurbs_centerline_freq, 2) * 2 - 1) \
            * params.nurbs_centerline_amp
        pts = []
        for p in range(P):
            theta = 2.0 * math.pi * p / P
            base = (R * math.cos(theta), R * math.sin(theta), x)
            jit = noise01_at(base, params.nurbs_radius_jitter_freq, 3) * 2 - 1
            r = R * (1.0 + params.nurbs_radius_jitter * jit)
            pts.append(rg.Point3d(
                x,
                cy + r * math.cos(theta),
                params.cave_center_z + cz + r * math.sin(theta),
            ))
        # Closed periodic degree-3 curve through the ring control points.
        ring = rg.NurbsCurve.Create(True, 3, pts)
        rings.append(ring)

    breps = rg.Brep.CreateFromLoft(
        rings, rg.Point3d.Unset, rg.Point3d.Unset,
        rg.LoftType.Normal, False,
    )
    if not breps:
        raise RuntimeError("Brep.CreateFromLoft returned no result")
    brep = breps[0]
    return brep.Faces[0].UnderlyingSurface().ToNurbsSurface()


def eval_surface_grid(surface, nu: int, nv: int) -> np.ndarray:
    """Evaluate `surface` on a regular (nu x nv) grid -> (nu, nv, 3) array.

    The loft may assign U or V to the ring direction, so detect which surface
    direction is CLOSED (the periodic ring) and map it to axis 1 (v); the OPEN
    direction (along X) maps to axis 0 (u). u spans its domain inclusively; v
    spans [v0, v1) with the endpoint excluded so there's no duplicate seam."""
    closed_dir = 1 if surface.IsClosed(1) else (0 if surface.IsClosed(0) else 1)
    open_dir = 1 - closed_dir
    od = surface.Domain(open_dir)
    cd = surface.Domain(closed_dir)
    u_open = np.linspace(od.T0, od.T1, nu)
    v_closed = np.linspace(cd.T0, cd.T1, nv, endpoint=False)
    grid = np.empty((nu, nv, 3), dtype=np.float64)
    for i in range(nu):
        for j in range(nv):
            params = [0.0, 0.0]
            params[open_dir] = float(u_open[i])
            params[closed_dir] = float(v_closed[j])
            pt = surface.PointAt(params[0], params[1])
            grid[i, j, 0] = pt.X
            grid[i, j, 1] = pt.Y
            grid[i, j, 2] = pt.Z
    return grid


class NurbsLoftCave:
    """CaveSurface backed by a lofted NURBS tube `surface`."""

    def __init__(self, surface, params) -> None:
        self.surface = surface
        self.params = params

    def sample_surface_points(self) -> np.ndarray:
        p = self.params
        grid = eval_surface_grid(self.surface, p.nurbs_grid_u, p.nurbs_grid_v)
        normals = surface_sampling.grid_normals(grid)
        return surface_sampling.sample_and_displace(
            grid, normals,
            target=p.total_surface_samples,
            fbm_amplitude=p.fbm_amplitude,
            fbm_base_freq=p.fbm_base_freq,
            octaves=p.fbm_octaves,
            lacunarity=p.fbm_lacunarity,
            gain=p.fbm_gain,
            noise_type=p.noise_type,
            noise_seed=p.noise_seed,
        )


def make_nurbs_cave(params) -> NurbsLoftCave:
    """Resolve the base surface (rebuild or reuse) and return the cave.

    rebuild: loft from params, render onto Pearlscape::CaveSurface (replacing).
    reuse:   read the surface from that layer; fall back to rebuild if absent.
    """
    from pearlscape import display

    source = params.nurbs_surface_source
    if source == "reuse":
        srf = display.find_cave_surface()
        if srf is None:
            print("nurbs_surface_source='reuse' but no surface on "
                  "Pearlscape::CaveSurface; rebuilding.")
            source = "rebuild"
    if source == "rebuild":
        srf = build_loft_surface(params)
        display.render_cave_surface(srf)
    return NurbsLoftCave(srf, params)
```

- [ ] **Step 2: Byte-compile check**

Run from `RhinoPython/`:

```
python -c "import py_compile; py_compile.compile('pearlscape/nurbs_cave.py', doraise=True); print('nurbs_cave.py OK')"
```

Expected: `nurbs_cave.py OK`. Then `python -c "import shutil; shutil.rmtree('pearlscape/__pycache__', ignore_errors=True)"`.

(Loft + eval are exercised under Rhino in Task 7.)

- [ ] **Step 3: Review and commit** (user) — suggested title: `Add lofted NURBS cave geometry`

---

## Task 6: Dispatch in make_default_cave

**Files:**
- Modify: `RhinoPython/pearlscape/cave.py`

- [ ] **Step 1: Dispatch on cave_type**

Replace the body of `make_default_cave` so it branches. The function currently starts with its docstring then `return CylinderFBMCave(...)`. Change it to:

```python
def make_default_cave(params):
    """Build the cave selected by params.cave_type.

    "cylinder" -> CylinderFBMCave (pure numpy, Rhino-free).
    "nurbs"    -> NurbsLoftCave (imported lazily; needs Rhino for the loft).
    """
    if params.cave_type == "nurbs":
        # Lazy import so the cylinder path stays importable outside Rhino.
        from pearlscape.nurbs_cave import make_nurbs_cave
        return make_nurbs_cave(params)

    return CylinderFBMCave(
        radius=params.cave_radius,
        length=params.cave_length,
        fbm_amplitude=params.fbm_amplitude,
        fbm_base_freq=params.fbm_base_freq,
        fbm_octaves=params.fbm_octaves,
        fbm_lacunarity=params.fbm_lacunarity,
        fbm_gain=params.fbm_gain,
        noise_seed=params.noise_seed,
        target_samples=params.total_surface_samples,
        center_z=params.cave_center_z,
        noise_type=params.noise_type,
    )
```

(Keep the existing `CylinderFBMCave` keyword args exactly as they are in the file today — the block above must match the current call.)

- [ ] **Step 2: Verify the cylinder path is unchanged headless**

Run from `RhinoPython/`:

```
python -c "import runpy; runpy.run_path('pearlscape/cave.py', run_name='__main__')"
```

Expected: the existing cave smoke test still prints a generated point count and X/Y/Z ranges (cave_type defaults to "cylinder", so no Rhino import is triggered). No errors.

- [ ] **Step 3: Review and commit** (user) — suggested title: `Dispatch cave generation on cave_type`

---

## Task 7: In-Rhino integration test + README

**Files:**
- Modify: `README.md`

This task is verified manually in Rhino (the loft, surface eval, and document helpers can't run headless). No code beyond README changes.

- [ ] **Step 1: Manual smoke — rebuild + cave mode**

In `params.py` set `cave_type="nurbs"`, `pipeline_mode="cave"`, `nurbs_surface_source="rebuild"`, `total_surface_samples=60_000`. In Rhino: fresh document, `_ScriptEditor`, open `build_scene.py`, F5.

Expected: an irregular tube point cloud on `Pearlscape::CaveReference`, and a lofted surface on `Pearlscape::CaveSurface`. Console prints the cave point count and timing. Confirm the cloud is an irregular bending tube, not a plain cylinder.

- [ ] **Step 2: Manual smoke — noise + scale + timing**

Toggle `noise_type` between `"ridged"` and `"fbm"`; confirm the wall detail changes. Then set `total_surface_samples=1_200_000` and F5.

Expected: generation completes in under ~10s (watch the printed timings; the `PointAt` grid eval is the fixed cost, sampling is sub-second). Bead density looks even.

- [ ] **Step 3: Manual smoke — reuse loop**

With the surface present on `Pearlscape::CaveSurface`, drag some of its control points (`_PointsOn`, move, `_PointsOff`). Set `nurbs_surface_source="reuse"` and F5.

Expected: the surface is NOT rebuilt (your edit is preserved), and the new bead cloud follows the edited surface. Then temporarily delete the surface and F5 in `"reuse"` mode → console prints the "no surface … rebuilding" warning and rebuilds.

- [ ] **Step 4: Manual smoke — curtains + cylinder regression**

Set `pipeline_mode="curtains"`, `display_mode="sprites"`, F5 → confirm curtains slice the NURBS cave and sprites render. Then set `cave_type="cylinder"`, F5 → confirm the original cylinder cave is unchanged.

- [ ] **Step 5: Document in README**

In `README.md`, under "### Cave geometry", add after the `noise_seed` row a pointer line and a new subsection. Add this row to the cave-geometry table:

```
| `cave_type`      | `"cylinder"` | `"cylinder"` (radial-noise cylinder) or `"nurbs"` (lofted irregular tube). See below. |
```

Then add a new subsection after the cave-geometry table:

```markdown
### NURBS cave (`cave_type = "nurbs"`)

An irregular tube lofted from jittered cross-section rings, sampled and displaced
with the same `fbm_*` / `noise_type` noise. The base surface is kept as editable
document geometry on the `Pearlscape::CaveSurface` layer.

| Parameter                  | Default     | Meaning                                                            |
|----------------------------|-------------|--------------------------------------------------------------------|
| `nurbs_surface_source`     | `"rebuild"` | `"rebuild"` lofts from the params below; `"reuse"` samples the (hand-edited) surface on the `CaveSurface` layer (falls back to rebuild if none). |
| `nurbs_sections`           | `8`         | Cross-section rings along X.                                       |
| `nurbs_section_points`     | `12`        | Control points per ring.                                           |
| `nurbs_radius_jitter`      | `0.35`      | Per-point radius variation, fraction of `cave_radius`.             |
| `nurbs_radius_jitter_freq` | `0.0006`    | Noise frequency for radius jitter (cycles/mm).                     |
| `nurbs_centerline_amp`     | `400.0`     | Centerline meander amplitude (mm, Y/Z only).                       |
| `nurbs_centerline_freq`    | `0.0003`    | Noise frequency for centerline meander (cycles/mm).               |
| `nurbs_shape_seed`         | `7`         | Seed for the shape noise (independent of `noise_seed`).            |
| `nurbs_grid_u` / `nurbs_grid_v` | `180` / `180` | Surface-eval grid resolution (X / θ). Raise for finer base form. |

To hand-edit: run once in `"rebuild"`, then `_PointsOn` the surface, move points,
`_PointsOff`, switch to `"reuse"`, and re-run to resample from your edited form.
```

- [ ] **Step 6: Review and commit** (user) — suggested title: `Document the NURBS cave`

---

## Self-review notes (author)

- **Spec coverage:** §4 construction → Task 5 `build_loft_surface`; §5 rebuild/reuse → Task 5 `make_nurbs_cave` + Task 4 helpers; §6 sampling pipeline → Tasks 2–3; §7 module structure → Tasks 2–6; §8 params → Task 1; §9 perf → Task 7 step 2; §10 testing → headless smoke tests in Tasks 1–3, 6 + manual in Task 7; §11 limitations are accepted, no task needed.
- **Type/name consistency:** `jittered_grid(width, wrap, target_n, *, seed)`, `grid_normals(grid_xyz)`, `sample_and_displace(grid_xyz, base_normals, *, target, fbm_amplitude, fbm_base_freq, octaves, lacunarity, gain, noise_type, noise_seed, sample_seed)`, `build_loft_surface(params)`, `eval_surface_grid(surface, nu, nv)`, `NurbsLoftCave(surface, params).sample_surface_points()`, `make_nurbs_cave(params)`, `render_cave_surface(surface)`, `find_cave_surface()` — used consistently across tasks.
- **No placeholders:** every code step is complete.
```
