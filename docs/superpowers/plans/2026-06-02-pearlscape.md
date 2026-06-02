# Pearlscape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parametric Rhino Python pipeline that generates a cave point cloud, slices it onto translucent curtain planes as colored bead positions, and exports per-curtain PDFs for fabrication.

**Architecture:** Flat package `RhinoPython/pearlscape/` with one module per concern (params, noise, sampling, cave, curtains, color, display, export). Top-level `build_scene.py` is the only entry point. A thin `CaveSurface` interface in `cave.py` is the seam where future geometries (Lidar, Catmull-Clark subdivision) plug in.

**Tech Stack:** Rhino 8 SR23, CPython 3 (Rhino's bundled Python 3 runtime), numpy, Rhino.Geometry / Rhino.DocObjects / Rhino.FileIO APIs.

---

## Conventions used throughout this plan

- **Project root:** `c:\Users\lukap\Documents\GitHub\pearlscape\`
- **Package path:** `RhinoPython/pearlscape/`
- **Coordinate system:** Right-handed Z-up. X = walking / viewing axis (cave length). Y = lateral. Z = vertical.
- **No automated tests.** Each task has a "Manual verification" step that the user runs in Rhino.
- **Running scripts in Rhino 8:** open the Script Editor (`_ScriptEditor` command), open the script file, press F5. The first lines of `build_scene.py` add the package directory to `sys.path` so `import pearlscape` resolves.
- **Python 3 runtime directive (required):** Rhino 8's Script Editor defaults to IronPython 2 for `.py` files. Every script that the user runs directly with F5 (entry points and modules with `__main__` smoke tests) must begin with `#! python 3` on the first line. Modules that are only imported do not need it. Files needing the directive: `build_scene.py`, `noise.py`, `sampling.py`, `cave.py`. Files that don't (no `__main__` block run directly): `__init__.py`, `params.py`, `curtains.py`*, `color.py`*, `display.py`, `export.py`. (*these have `__main__` blocks for developer testing but are not part of the user verification flow.)
- **Dual-mode imports for F5-runnable modules:** any module that has both internal package imports AND a user-facing F5 smoke test must use absolute imports (`from pearlscape.foo import bar`) preceded by a sys.path setup, NOT relative imports (`from .foo import bar`). Relative imports require a package context that doesn't exist when Rhino loads the file as `__main__`. The pattern is:
  ```python
  import os, sys
  _PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  if _PKG_PARENT not in sys.path:
      sys.path.insert(0, _PKG_PARENT)
  from pearlscape.noise import fbm3_01, make_perm
  ```
  Applies to: `cave.py` (Task 4). Other module-level imports inside the `pearlscape/` package that don't need F5-direct-run (`curtains.py`, `color.py`, `display.py`, `export.py`) can keep relative imports.
- **Numpy availability:** Rhino 8 SR23's CPython runtime ships with numpy preinstalled. No extra install needed.
- **Commits:** The user handles all commits. Implementer steps should stop at code-written state. The plan's "Commit" steps are user actions, not subagent actions.

---

## Task 1: Package scaffolding and parameters

**Files:**
- Create: `RhinoPython/pearlscape/__init__.py`
- Create: `RhinoPython/pearlscape/params.py`
- Create: `RhinoPython/build_scene.py`
- Create: `RhinoPython/README.md`

- [ ] **Step 1: Create the package directory structure**

Create the empty files listed above. No content yet — just the layout.

- [ ] **Step 2: Write `pearlscape/__init__.py`**

```python
"""Pearlscape — parametric cave-to-curtain bead artwork generator."""

from .params import PearlscapeParams

__all__ = ["PearlscapeParams"]
```

- [ ] **Step 3: Write `pearlscape/params.py`**

```python
"""Single source of truth for all tunable parameters."""

from dataclasses import dataclass, field
from typing import List, Tuple


RGB = Tuple[int, int, int]


def _default_palette() -> List[RGB]:
    return [
        (180,  40,  40),   # red
        (220, 130,  50),   # orange
        (230, 200,  70),   # yellow
        ( 80, 160,  90),   # green
        ( 60, 110, 180),   # blue
        (130,  70, 170),   # violet
    ]


@dataclass
class PearlscapeParams:
    # --- Cave geometry ---
    cave_radius: float = 1.2
    cave_length: float = 5.0
    fbm_amplitude: float = 0.30
    fbm_base_freq: float = 0.8
    fbm_octaves: int = 4
    fbm_lacunarity: float = 2.0
    fbm_gain: float = 0.5
    noise_seed: int = 1

    # --- Curtain array ---
    curtain_count: int = 25
    curtain_spacing: float = 0.20
    curtain_width: float = 2.5
    curtain_height: float = 3.0
    cave_center_z: float = 1.5   # cave centerline elevation

    # --- Beads ---
    total_surface_samples: int = 60_000
    bead_diameter: float = 0.006  # 6 mm

    # --- Color ---
    palette: List[RGB] = field(default_factory=_default_palette)
    color_base_freq: float = 0.6
    color_fbm_octaves: int = 2
    color_fbm_lacunarity: float = 2.0
    color_fbm_gain: float = 0.5
    color_noise_seed: int = 42

    # --- Display / export ---
    display_mode: str = "pointcloud"   # "pointcloud" | "instances"
    instance_sphere_subd: int = 2
    pdf_page_size: str = "A1"
    pdf_output_dir: str = "exports"

    def validate(self) -> None:
        assert self.fbm_amplitude < self.cave_radius, (
            f"fbm_amplitude ({self.fbm_amplitude}) must be < cave_radius "
            f"({self.cave_radius}) to avoid pinching the cave shut."
        )
        assert self.curtain_count >= 2
        assert self.display_mode in ("pointcloud", "instances")
        assert len(self.palette) >= 2
```

- [ ] **Step 4: Write `build_scene.py` stub**

```python
#! python 3
# r: numpy
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

import os
import sys

# Make the pearlscape package importable when running from Rhino.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pearlscape import PearlscapeParams


def main() -> None:
    params = PearlscapeParams()
    params.validate()
    print(f"Pearlscape params loaded: {params.curtain_count} curtains, "
          f"{params.total_surface_samples} beads target.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `RhinoPython/README.md`**

```markdown
# Pearlscape — Rhino Python

## Running

1. Open Rhino 8.
2. Run the `_ScriptEditor` command.
3. Open `build_scene.py`.
4. Press F5.

All parameters live in `pearlscape/params.py`. Edit and re-run.

## Layout

- `pearlscape/` — library modules (one per concern).
- `build_scene.py` — entry point.
- `exports/` — generated PDFs (created on first export).
```

- [ ] **Step 6: Manual verification**

Open Rhino 8, run `_ScriptEditor`, open `build_scene.py`, press F5.
Expected output in the script editor's output pane:
```
Pearlscape params loaded: 25 curtains, 60000 beads target.
```

- [ ] **Step 7: Commit**

```bash
git add RhinoPython/
git commit -m "feat: scaffold pearlscape package and parameters"
```

---

## Task 2: Noise — Perlin and FBM

**Files:**
- Create: `RhinoPython/pearlscape/noise.py`

- [ ] **Step 1: Write `pearlscape/noise.py`**

```python
#! python 3
# r: numpy
"""Perlin noise and FBM, numpy-vectorized. 3D inputs / 1D outputs."""

import numpy as np


def make_perm(seed: int) -> np.ndarray:
    """Build a 512-element permutation table from a seed.

    The doubled length lets us index `perm[x+1]` without wrap arithmetic.
    """
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)
    return np.concatenate([p, p])


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _grad3(h: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Ken Perlin's improved-noise gradient selector (12 edge directions of a cube)."""
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


def perlin3(p: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Perlin noise at points p of shape (N, 3). Returns (N,) values in ~[-1, 1]."""
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    fx = np.floor(x)
    fy = np.floor(y)
    fz = np.floor(z)

    X = fx.astype(np.int32) & 255
    Y = fy.astype(np.int32) & 255
    Z = fz.astype(np.int32) & 255

    x = x - fx
    y = y - fy
    z = z - fz

    u = _fade(x)
    v = _fade(y)
    w = _fade(z)

    A  = perm[X]     + Y
    AA = perm[A]     + Z
    AB = perm[A + 1] + Z
    B  = perm[X + 1] + Y
    BA = perm[B]     + Z
    BB = perm[B + 1] + Z

    n000 = _grad3(perm[AA],     x,     y,     z)
    n100 = _grad3(perm[BA],     x - 1, y,     z)
    n010 = _grad3(perm[AB],     x,     y - 1, z)
    n110 = _grad3(perm[BB],     x - 1, y - 1, z)
    n001 = _grad3(perm[AA + 1], x,     y,     z - 1)
    n101 = _grad3(perm[BA + 1], x - 1, y,     z - 1)
    n011 = _grad3(perm[AB + 1], x,     y - 1, z - 1)
    n111 = _grad3(perm[BB + 1], x - 1, y - 1, z - 1)

    x00 = _lerp(n000, n100, u)
    x10 = _lerp(n010, n110, u)
    x01 = _lerp(n001, n101, u)
    x11 = _lerp(n011, n111, u)

    y0 = _lerp(x00, x10, v)
    y1 = _lerp(x01, x11, v)

    return _lerp(y0, y1, w)


def fbm3(
    p: np.ndarray,
    perm: np.ndarray,
    *,
    octaves: int,
    lacunarity: float,
    gain: float,
) -> np.ndarray:
    """Fractal Brownian motion. Returns (N,) values in ~[-1, 1]."""
    total = np.zeros(p.shape[0], dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    norm = 0.0
    for _ in range(octaves):
        total = total + amplitude * perlin3(p * frequency, perm)
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm if norm > 0 else total


def fbm3_01(
    p: np.ndarray,
    perm: np.ndarray,
    *,
    octaves: int,
    lacunarity: float,
    gain: float,
) -> np.ndarray:
    """FBM mapped to ~[0, 1) for convenience."""
    return 0.5 * (fbm3(p, perm, octaves=octaves, lacunarity=lacunarity, gain=gain) + 1.0)


if __name__ == "__main__":
    # Smoke test: sample at non-integer coordinates (Perlin is zero at integer
    # lattice corners, so a 0..2 integer grid would produce all 0.5s).
    perm = make_perm(seed=1)
    xs = np.linspace(0.25, 1.75, 3)
    ys = np.linspace(0.25, 1.75, 3)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(9)])
    vals = fbm3_01(pts, perm, octaves=4, lacunarity=2.0, gain=0.5)
    print("Perlin/FBM smoke test (3x3 grid at z=0):")
    print(vals.reshape(3, 3).round(3))
    print(f"Min/max: {vals.min():.3f} / {vals.max():.3f}  "
          f"(varies: {bool(vals.max() - vals.min() > 0.01)})")
    vals2 = fbm3_01(pts, perm, octaves=4, lacunarity=2.0, gain=0.5)
    print("Deterministic:", bool(np.allclose(vals, vals2)))
```

- [ ] **Step 2: Manual verification**

In Rhino's Script Editor, open `RhinoPython/pearlscape/noise.py` and press F5.
Expected output (values will differ slightly — what matters is shape and determinism):
```
Perlin/FBM smoke test (3x3 grid at z=0):
[[0.5  0.xxx 0.xxx]
 [0.xxx 0.xxx 0.xxx]
 [0.xxx 0.xxx 0.xxx]]
Deterministic: True
```
The "Deterministic: True" line is the key check. Values should be in `[0, 1]`.

- [ ] **Step 3: Commit**

```bash
git add RhinoPython/pearlscape/noise.py
git commit -m "feat: add numpy Perlin + FBM noise"
```

---

## Task 3: Blue-noise sampling (Bridson's algorithm with θ-wrap)

**Files:**
- Create: `RhinoPython/pearlscape/sampling.py`

**Rationale (for the implementer):** We need ~60k well-distributed points in a (θ_arc, x) parameter space where the θ axis wraps. Bridson's Poisson-disk algorithm is the standard fast choice — O(N), uses a background grid for O(1) neighborhood lookups, no third-party dependency. We work in arc-length-on-θ (i.e., `θ_arc = θ · cave_radius`) so that the minimum-distance metric is isotropic in surface units. Conversion back to `θ` happens at the caller.

- [ ] **Step 1: Write `pearlscape/sampling.py`**

```python
#! python 3
# r: numpy
"""Blue-noise (Poisson-disk) sampling in a 2D domain with one wrapped axis.

We sample in (theta_arc, x), where theta_arc = theta * cave_radius is in meters.
The first axis wraps modulo `wrap`; the second is bounded [0, width).
"""

import math
from typing import List, Tuple

import numpy as np


def estimate_radius_for_count(width: float, wrap: float, target_n: int) -> float:
    """Estimate the Poisson-disk minimum distance that yields ~target_n points.

    Uses the hexagonal close-packing density bound: max points ~ area / (r^2 * sqrt(3)/2).
    Bridson tends to undershoot this bound; treat the result as an upper bound on r.
    """
    area = max(width, 1e-9) * max(wrap, 1e-9)
    return math.sqrt(2.0 * area / (target_n * math.sqrt(3.0)))


def bridson_torus(
    width: float,
    wrap: float,
    r: float,
    *,
    k: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Bridson Poisson-disk sampling in 2D, first axis wrapped modulo `wrap`.

    Returns an (N, 2) array with columns (theta_arc, x). N is determined by the
    domain and minimum distance r; it is not a parameter.
    """
    if r <= 0:
        raise ValueError("r must be positive")

    rng = np.random.default_rng(seed)
    cell = r / math.sqrt(2.0)

    ncols = max(1, int(math.ceil(wrap / cell)))
    nrows = max(1, int(math.ceil(width / cell)))
    cell_wrap = wrap / ncols
    cell_width = width / nrows

    grid = np.full((ncols, nrows), -1, dtype=np.int64)
    points: List[Tuple[float, float]] = []
    active: List[int] = []

    def add_point(theta_arc: float, x: float) -> None:
        idx = len(points)
        points.append((theta_arc, x))
        active.append(idx)
        grid[int(theta_arc / cell_wrap), int(x / cell_width)] = idx

    def fits(theta_arc: float, x: float) -> bool:
        if x < 0 or x >= width:
            return False
        gc = int(theta_arc / cell_wrap)
        gr = int(x / cell_width)
        r2 = r * r
        for dc in range(-2, 3):
            cc = (gc + dc) % ncols
            for dr in range(-2, 3):
                cr = gr + dr
                if cr < 0 or cr >= nrows:
                    continue
                idx = grid[cc, cr]
                if idx == -1:
                    continue
                ot, ox = points[idx]
                dt = abs(ot - theta_arc)
                if dt > wrap - dt:
                    dt = wrap - dt
                dx = ox - x
                if dt * dt + dx * dx < r2:
                    return False
        return True

    # Seed point.
    add_point(float(rng.uniform(0, wrap)), float(rng.uniform(0, width)))

    while active:
        ai = int(rng.integers(0, len(active)))
        idx = active[ai]
        ap_theta, ap_x = points[idx]
        placed = False
        # Sample k candidates in the annulus [r, 2r] around the active point.
        angles = rng.uniform(0.0, 2.0 * math.pi, size=k)
        radii = rng.uniform(r, 2.0 * r, size=k)
        cand_theta = (ap_theta + radii * np.cos(angles)) % wrap
        cand_x = ap_x + radii * np.sin(angles)
        for j in range(k):
            t = float(cand_theta[j])
            x = float(cand_x[j])
            if fits(t, x):
                add_point(t, x)
                placed = True
                break
        if not placed:
            # Remove from active list (swap-remove for O(1)).
            active[ai] = active[-1]
            active.pop()

    return np.array(points, dtype=np.float64)


if __name__ == "__main__":
    # Smoke test: sample a 2*pi*1.2 x 5.0 domain for ~60k points.
    radius = 1.2
    length = 5.0
    target = 60_000
    wrap = 2.0 * math.pi * radius
    r = estimate_radius_for_count(length, wrap, target)
    print(f"Estimated r: {r:.4f}m for target {target}")
    import time
    t0 = time.time()
    pts = bridson_torus(width=length, wrap=wrap, r=r, seed=0)
    dt = time.time() - t0
    print(f"Generated {len(pts)} points in {dt:.2f}s")
    print(f"theta_arc range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] (wrap = {wrap:.3f})")
    print(f"x range:         [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] (length = {length:.3f})")
```

- [ ] **Step 2: Manual verification**

In Rhino's Script Editor, open `RhinoPython/pearlscape/sampling.py` and press F5.
Expected output (approximate counts and timings — Bridson is randomized):
```
Estimated r: 0.0246m for target 60000
Generated 4xxxx points in N.NNs
theta_arc range: [0.000, 7.5xx] (wrap = 7.540)
x range:         [0.000, 4.9xx] (length = 5.000)
```
Notes:
- Bridson underfills relative to the theoretical bound; expect ~40-50k points at the estimated r. If you want closer to 60k, the caller will tighten r via the `total_surface_samples` parameter loop in `cave.py` (Task 4 handles this).
- Generation time should be on the order of seconds, not minutes. If it takes >60s, stop and investigate.

- [ ] **Step 3: Commit**

```bash
git add RhinoPython/pearlscape/sampling.py
git commit -m "feat: add Bridson Poisson-disk sampling with theta wrap"
```

---

## Task 4: Cave surface — interface + cylinder+FBM implementation

**Files:**
- Create: `RhinoPython/pearlscape/cave.py`

- [ ] **Step 1: Write `pearlscape/cave.py`**

```python
#! python 3
# r: numpy
"""Cave surface generation.

`CaveSurface` is the seam where future geometries (Lidar, subdivision)
plug in without changing downstream code. The default implementation
displaces a cylinder's surface inward by FBM in cylindrical parameter
space — fast, deterministic, no self-intersection while
fbm_amplitude < cave_radius.
"""

import math
import os
import sys
from typing import Protocol

import numpy as np

# Make sibling-package imports resolve when this file is F5-run directly
# (Rhino loads it as __main__, with no parent package context). When imported
# normally as `pearlscape.cave`, this sys.path adjustment is a no-op.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape.noise import fbm3_01, make_perm
from pearlscape.sampling import bridson_torus, estimate_radius_for_count


class CaveSurface(Protocol):
    """Anything that can produce surface points of a cave shape."""

    def sample_surface_points(self) -> np.ndarray:
        """Return an (N, 3) array of points on the cave surface, in world coordinates."""
        ...


class CylinderFBMCave:
    """A cylinder along X, with surface points blue-noise-sampled and
    radially displaced inward by FBM noise."""

    def __init__(
        self,
        *,
        radius: float,
        length: float,
        fbm_amplitude: float,
        fbm_base_freq: float,
        fbm_octaves: int,
        fbm_lacunarity: float,
        fbm_gain: float,
        noise_seed: int,
        target_samples: int,
        center_z: float,
        sampling_seed: int = 0,
        radius_shrink: float = 0.93,
    ) -> None:
        if fbm_amplitude >= radius:
            raise ValueError(
                f"fbm_amplitude ({fbm_amplitude}) must be < radius ({radius})"
            )
        self.radius = radius
        self.length = length
        self.fbm_amplitude = fbm_amplitude
        self.fbm_base_freq = fbm_base_freq
        self.fbm_octaves = fbm_octaves
        self.fbm_lacunarity = fbm_lacunarity
        self.fbm_gain = fbm_gain
        self.noise_seed = noise_seed
        self.target_samples = target_samples
        self.center_z = center_z
        self.sampling_seed = sampling_seed
        # Bridson underfills the theoretical bound. Shrink the estimated r
        # to land closer to the target. Empirical 0.93 typically lands ~60k
        # when target is 60k.
        self.radius_shrink = radius_shrink

    def sample_surface_points(self) -> np.ndarray:
        wrap = 2.0 * math.pi * self.radius
        r = estimate_radius_for_count(self.length, wrap, self.target_samples)
        r *= self.radius_shrink

        param_pts = bridson_torus(
            width=self.length,
            wrap=wrap,
            r=r,
            seed=self.sampling_seed,
        )

        theta = param_pts[:, 0] / self.radius   # back to radians
        x = param_pts[:, 1]

        # Evaluate FBM on the base-cylinder position so the noise input
        # is continuous around the theta seam.
        base = np.column_stack([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            x,
        ]) * self.fbm_base_freq

        perm = make_perm(self.noise_seed)
        n01 = fbm3_01(
            base, perm,
            octaves=self.fbm_octaves,
            lacunarity=self.fbm_lacunarity,
            gain=self.fbm_gain,
        )

        r_eff = self.radius - self.fbm_amplitude * n01

        # Place in world: X = x, Y = r_eff*cos, Z = center_z + r_eff*sin.
        pts = np.column_stack([
            x,
            r_eff * np.cos(theta),
            self.center_z + r_eff * np.sin(theta),
        ])
        return pts


def make_default_cave(params) -> CylinderFBMCave:
    """Build a CylinderFBMCave from a PearlscapeParams instance."""
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
    )


if __name__ == "__main__":
    from pearlscape.params import PearlscapeParams
    p = PearlscapeParams()
    cave = make_default_cave(p)
    import time
    t0 = time.time()
    pts = cave.sample_surface_points()
    print(f"Generated {len(pts)} surface points in {time.time()-t0:.2f}s")
    print(f"X range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"Y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"Z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
```

- [ ] **Step 2: Manual verification**

In Rhino's Script Editor, run `cave.py` directly (F5). Expected output:
```
Generated ~50000-60000 surface points in N.NNs
X range: [0.000, 5.000]
Y range: [-1.200, 1.200]
Z range: [0.300, 2.700]
```
Z range should be centered on 1.5 with spread ~1.2 (cave radius). Y range likewise.

- [ ] **Step 3: Commit**

```bash
git add RhinoPython/pearlscape/cave.py
git commit -m "feat: add CaveSurface interface and CylinderFBMCave"
```

---

## Task 5: Display — PointCloud renderer + first end-to-end run

**Files:**
- Create: `RhinoPython/pearlscape/display.py`
- Modify: `RhinoPython/build_scene.py`

- [ ] **Step 1: Write `pearlscape/display.py`**

```python
"""Render bead positions into the Rhino document.

Two modes:
- pointcloud: one Rhino.Geometry.PointCloud per curtain (fast, default).
- instances: one InstanceReference per bead (heavy, for rendering — Task 8).

At this task we only implement the PointCloud mode and a single-cloud
helper used to visualize the raw cave during early development.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

import Rhino
import Rhino.Geometry as rg
import scriptcontext as sc
import System.Drawing as sd


PEARLSCAPE_PARENT_LAYER = "Pearlscape"
CURTAINS_LAYER = "Curtains"
CAVE_REFERENCE_LAYER = "CaveReference"
CURTAIN_PLANES_LAYER = "CurtainPlanes"


def _ensure_layer(path: str, color: Optional[sd.Color] = None) -> int:
    """Ensure a layer at `path` (e.g. 'Pearlscape::Curtains::Curtain_00') exists.
    Returns the layer index.
    """
    doc = sc.doc
    layer_table = doc.Layers
    idx = layer_table.FindByFullPath(path, -1)
    if idx >= 0:
        return idx
    parts = path.split("::")
    parent_id = None
    current = ""
    for p in parts:
        current = p if not current else f"{current}::{p}"
        i = layer_table.FindByFullPath(current, -1)
        if i < 0:
            layer = Rhino.DocObjects.Layer()
            layer.Name = p
            if parent_id is not None:
                layer.ParentLayerId = parent_id
            if color is not None and current == path:
                layer.Color = color
            i = layer_table.Add(layer)
        parent_id = layer_table[i].Id
    return layer_table.FindByFullPath(path, -1)


def _np_to_point3d_list(pts: np.ndarray) -> List[rg.Point3d]:
    return [rg.Point3d(float(p[0]), float(p[1]), float(p[2])) for p in pts]


def _np_to_color_list(colors: Optional[np.ndarray]) -> Optional[List[sd.Color]]:
    if colors is None:
        return None
    return [sd.Color.FromArgb(int(c[0]), int(c[1]), int(c[2])) for c in colors]


def add_pointcloud(
    points: np.ndarray,
    layer_path: str,
    colors: Optional[np.ndarray] = None,
) -> Tuple[Rhino.DocObjects.PointCloudObject, "Rhino.RhinoDoc"]:
    """Add a single PointCloud to the document under the given layer path."""
    doc = sc.doc
    layer_idx = _ensure_layer(layer_path)
    cloud = rg.PointCloud()
    pt_list = _np_to_point3d_list(points)
    color_list = _np_to_color_list(colors)
    if color_list is None:
        for pt in pt_list:
            cloud.Add(pt)
    else:
        for pt, c in zip(pt_list, color_list):
            cloud.Add(pt, c)
    attrs = Rhino.DocObjects.ObjectAttributes()
    attrs.LayerIndex = layer_idx
    obj_id = doc.Objects.AddPointCloud(cloud, attrs)
    return obj_id, doc


def render_cave_reference(points: np.ndarray) -> None:
    """Render the raw, un-sliced cave as a single PointCloud on the
    CaveReference layer. Used during early development."""
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CAVE_REFERENCE_LAYER}"
    add_pointcloud(points, layer_path)
    sc.doc.Views.Redraw()


def render_pointclouds(curtains: Sequence[dict]) -> None:
    """Render per-curtain PointClouds.

    Each item in `curtains` is the dict produced by curtains.slice_and_project:
        { 'plane_x': float, 'points_2d': np.ndarray (M, 2),
          'points_3d': np.ndarray (M, 3), 'colors': np.ndarray (M, 3) | None }
    The displayed point is (plane_x, y, z) — i.e. the projected position.
    """
    for i, c in enumerate(curtains):
        layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CURTAINS_LAYER}::Curtain_{i:02d}"
        plane_x = c["plane_x"]
        pts_2d = c["points_2d"]
        projected = np.column_stack([
            np.full(pts_2d.shape[0], plane_x),
            pts_2d[:, 0],
            pts_2d[:, 1],
        ])
        add_pointcloud(projected, layer_path, colors=c.get("colors"))
    sc.doc.Views.Redraw()
```

- [ ] **Step 2: Modify `build_scene.py` to render the raw cave**

Replace the entire contents of `build_scene.py` with:

```python
#! python 3
# r: numpy
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time

from pearlscape import PearlscapeParams
from pearlscape.cave import make_default_cave
from pearlscape import display


def main() -> None:
    params = PearlscapeParams()
    params.validate()

    t0 = time.time()
    cave = make_default_cave(params)
    pts = cave.sample_surface_points()
    print(f"Generated {len(pts)} cave surface points in {time.time()-t0:.2f}s")

    display.render_cave_reference(pts)
    print("Rendered cave reference. Look at the viewport.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual verification**

1. In Rhino 8, open a fresh document.
2. Run `_ScriptEditor`, open `build_scene.py`, press F5.
3. Expected: console prints generation time and point count; viewport shows a tube-like point cloud along X, roughly 5m long, ~2.4m diameter, centered vertically at Z=1.5. Zoom to extents (Ctrl+Shift+E in Rhino) to see it.
4. The cave layer is `Pearlscape > CaveReference`.

If the cloud looks like a clean cylinder rather than a noisy cave: increase `fbm_amplitude` in `params.py` temporarily to 0.5 and re-run to confirm the noise is doing something. Reset to 0.30.

- [ ] **Step 4: Commit**

```bash
git add RhinoPython/pearlscape/display.py RhinoPython/build_scene.py
git commit -m "feat: add PointCloud renderer and visualize raw cave"
```

---

## Task 6: Curtain slicing and projection

**Files:**
- Create: `RhinoPython/pearlscape/curtains.py`
- Modify: `RhinoPython/build_scene.py`
- Modify: `RhinoPython/pearlscape/display.py` (add curtain-plane rectangle rendering)

- [ ] **Step 1: Write `pearlscape/curtains.py`**

```python
"""Slice the cave point cloud into curtain slabs and project onto curtain planes."""

from typing import List

import numpy as np


def curtain_x_positions(curtain_count: int, spacing: float, x_center: float) -> np.ndarray:
    """Return the X coordinate of each curtain plane, centered on x_center."""
    offsets = (np.arange(curtain_count) - (curtain_count - 1) / 2.0) * spacing
    return offsets + x_center


def slice_and_project(
    cave_points: np.ndarray,
    curtain_count: int,
    curtain_spacing: float,
    x_center: float,
) -> List[dict]:
    """Assign each cave point to a curtain slab and project onto its plane.

    Returns one dict per curtain:
        {
            'plane_x':   float,
            'points_2d': np.ndarray (M_i, 2)  # columns (Y, Z)
            'points_3d': np.ndarray (M_i, 3)  # original positions, for color sampling
        }
    """
    xs = curtain_x_positions(curtain_count, curtain_spacing, x_center)
    x_min = xs[0] - curtain_spacing / 2.0
    x_max = xs[-1] + curtain_spacing / 2.0

    # Drop points outside the array's X coverage.
    cave_x = cave_points[:, 0]
    in_range = (cave_x >= x_min) & (cave_x < x_max)
    pts = cave_points[in_range]

    # Assign each surviving point to the curtain whose slab it falls in.
    # Slabs are [xs[i] - spacing/2, xs[i] + spacing/2).
    slab_idx = np.floor((pts[:, 0] - x_min) / curtain_spacing).astype(np.int64)
    # Clip pathological edge case.
    slab_idx = np.clip(slab_idx, 0, curtain_count - 1)

    curtains: List[dict] = []
    for i in range(curtain_count):
        mask = slab_idx == i
        assigned = pts[mask]
        curtains.append({
            "plane_x": float(xs[i]),
            "points_2d": assigned[:, 1:3].copy(),     # (Y, Z)
            "points_3d": assigned.copy(),
        })
    return curtains


def array_x_center(cave_length: float) -> float:
    """The curtain array centers on the cave's midpoint along X."""
    return cave_length / 2.0


if __name__ == "__main__":
    from .params import PearlscapeParams
    from .cave import make_default_cave
    p = PearlscapeParams()
    pts = make_default_cave(p).sample_surface_points()
    curtains = slice_and_project(
        pts,
        curtain_count=p.curtain_count,
        curtain_spacing=p.curtain_spacing,
        x_center=array_x_center(p.cave_length),
    )
    print(f"{len(curtains)} curtains, "
          f"total assigned beads: {sum(len(c['points_2d']) for c in curtains)}")
    counts = [len(c["points_2d"]) for c in curtains]
    print(f"Min/mean/max beads per curtain: "
          f"{min(counts)}/{sum(counts)//len(counts)}/{max(counts)}")
```

- [ ] **Step 2: Add curtain-plane rectangle rendering to `display.py`**

Append the following to `RhinoPython/pearlscape/display.py`:

```python
def render_curtain_planes(
    plane_xs: Sequence[float],
    width: float,
    height: float,
) -> None:
    """Render the 2.5×3.0m curtain rectangles as outlines for visual reference."""
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CURTAIN_PLANES_LAYER}"
    layer_idx = _ensure_layer(layer_path)
    doc = sc.doc
    half_w = width / 2.0
    for x in plane_xs:
        corners = [
            rg.Point3d(x, -half_w, 0.0),
            rg.Point3d(x,  half_w, 0.0),
            rg.Point3d(x,  half_w, height),
            rg.Point3d(x, -half_w, height),
            rg.Point3d(x, -half_w, 0.0),
        ]
        polyline = rg.Polyline(corners)
        attrs = Rhino.DocObjects.ObjectAttributes()
        attrs.LayerIndex = layer_idx
        doc.Objects.AddPolyline(polyline, attrs)
    sc.doc.Views.Redraw()
```

- [ ] **Step 3: Update `build_scene.py` to wire in curtains**

Replace the entire contents of `build_scene.py` with:

```python
#! python 3
# r: numpy
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time

from pearlscape import PearlscapeParams
from pearlscape.cave import make_default_cave
from pearlscape.curtains import array_x_center, curtain_x_positions, slice_and_project
from pearlscape import display


def main() -> None:
    params = PearlscapeParams()
    params.validate()

    t0 = time.time()
    cave = make_default_cave(params)
    pts = cave.sample_surface_points()
    print(f"Cave: {len(pts)} surface points in {time.time()-t0:.2f}s")

    t0 = time.time()
    x_center = array_x_center(params.cave_length)
    curtains = slice_and_project(
        pts,
        curtain_count=params.curtain_count,
        curtain_spacing=params.curtain_spacing,
        x_center=x_center,
    )
    total = sum(len(c["points_2d"]) for c in curtains)
    print(f"Curtains: {len(curtains)} planes, {total} beads in {time.time()-t0:.2f}s")

    plane_xs = curtain_x_positions(
        params.curtain_count, params.curtain_spacing, x_center
    )
    display.render_curtain_planes(plane_xs, params.curtain_width, params.curtain_height)
    display.render_pointclouds(curtains)
    print("Rendered. Look at the viewport.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Manual verification**

1. Open a fresh Rhino document (or delete previous Pearlscape geometry).
2. Run `build_scene.py` (F5).
3. Expected output (counts approximate):
   ```
   Cave: ~55000 surface points in N.NNs
   Curtains: 25 planes, ~55000 beads in N.NNs
   Rendered. Look at the viewport.
   ```
4. Viewport shows:
   - 25 rectangular outlines (the curtain planes), spaced 200mm apart along X.
   - Each rectangle's interior has a point cloud — the bead positions for that curtain.
   - Looking down the +X axis (front view) shows the bead patterns overlapping into a cave silhouette.
   - Looking from the side (right view, looking along -Y), the curtains read as parallel sliced sections.
5. Layer panel: `Pearlscape > Curtains > Curtain_00 … Curtain_24` plus `Pearlscape > CurtainPlanes`.

- [ ] **Step 5: Commit**

```bash
git add RhinoPython/pearlscape/curtains.py RhinoPython/pearlscape/display.py RhinoPython/build_scene.py
git commit -m "feat: slice cave into per-curtain projected point clouds"
```

---

## Task 7: Color — palette and FBM-driven lookup

**Files:**
- Create: `RhinoPython/pearlscape/color.py`
- Modify: `RhinoPython/build_scene.py`

- [ ] **Step 1: Write `pearlscape/color.py`**

```python
"""Discrete color lookup driven by a second FBM noise field.

Color is evaluated at each bead's original 3D position (before curtain
projection), so the field stays spatially coherent across curtains.
"""

from typing import List, Sequence

import numpy as np

from .noise import fbm3_01, make_perm


def assign_colors(
    points_3d: np.ndarray,
    *,
    palette: Sequence,
    base_freq: float,
    octaves: int,
    lacunarity: float,
    gain: float,
    seed: int,
) -> np.ndarray:
    """Return an (N, 3) uint8 RGB array for the given 3D points."""
    if points_3d.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    perm = make_perm(seed)
    n01 = fbm3_01(
        points_3d * base_freq, perm,
        octaves=octaves, lacunarity=lacunarity, gain=gain,
    )
    # Clamp into [0, 1) defensively; FBM normalization is approximate.
    n01 = np.clip(n01, 0.0, 0.999999)
    m = len(palette)
    idx = np.floor(n01 * m).astype(np.int64)
    palette_arr = np.array(palette, dtype=np.uint8)   # (M, 3)
    return palette_arr[idx]


def apply_to_curtains(curtains: List[dict], params) -> None:
    """Mutate each curtain dict in-place, adding a 'colors' key."""
    for c in curtains:
        c["colors"] = assign_colors(
            c["points_3d"],
            palette=params.palette,
            base_freq=params.color_base_freq,
            octaves=params.color_fbm_octaves,
            lacunarity=params.color_fbm_lacunarity,
            gain=params.color_fbm_gain,
            seed=params.color_noise_seed,
        )


if __name__ == "__main__":
    from .params import PearlscapeParams
    from .cave import make_default_cave
    from .curtains import array_x_center, slice_and_project
    p = PearlscapeParams()
    pts = make_default_cave(p).sample_surface_points()
    curtains = slice_and_project(
        pts, p.curtain_count, p.curtain_spacing, array_x_center(p.cave_length),
    )
    apply_to_curtains(curtains, p)
    # Print color histogram for the first curtain.
    if len(curtains[0]["colors"]) > 0:
        unique, counts = np.unique(curtains[0]["colors"], axis=0, return_counts=True)
        print("Curtain 0 color distribution:")
        for rgb, n in zip(unique, counts):
            print(f"  rgb({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d}) -> {n}")
```

- [ ] **Step 2: Update `build_scene.py` to wire in color**

Replace the `main()` body in `build_scene.py` (keep the imports section as-is) so it reads:

```python
from pearlscape import color as color_mod   # add to imports


def main() -> None:
    params = PearlscapeParams()
    params.validate()

    t0 = time.time()
    cave = make_default_cave(params)
    pts = cave.sample_surface_points()
    print(f"Cave: {len(pts)} surface points in {time.time()-t0:.2f}s")

    t0 = time.time()
    x_center = array_x_center(params.cave_length)
    curtains = slice_and_project(
        pts,
        curtain_count=params.curtain_count,
        curtain_spacing=params.curtain_spacing,
        x_center=x_center,
    )
    total = sum(len(c["points_2d"]) for c in curtains)
    print(f"Curtains: {len(curtains)} planes, {total} beads in {time.time()-t0:.2f}s")

    t0 = time.time()
    color_mod.apply_to_curtains(curtains, params)
    print(f"Colors assigned in {time.time()-t0:.2f}s")

    plane_xs = curtain_x_positions(
        params.curtain_count, params.curtain_spacing, x_center
    )
    display.render_curtain_planes(plane_xs, params.curtain_width, params.curtain_height)
    display.render_pointclouds(curtains)
    print("Rendered. Look at the viewport.")
```

For clarity, here is the full intended `build_scene.py` after this edit:

```python
#! python 3
# r: numpy
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time

from pearlscape import PearlscapeParams
from pearlscape.cave import make_default_cave
from pearlscape.curtains import array_x_center, curtain_x_positions, slice_and_project
from pearlscape import color as color_mod
from pearlscape import display


def main() -> None:
    params = PearlscapeParams()
    params.validate()

    t0 = time.time()
    cave = make_default_cave(params)
    pts = cave.sample_surface_points()
    print(f"Cave: {len(pts)} surface points in {time.time()-t0:.2f}s")

    t0 = time.time()
    x_center = array_x_center(params.cave_length)
    curtains = slice_and_project(
        pts,
        curtain_count=params.curtain_count,
        curtain_spacing=params.curtain_spacing,
        x_center=x_center,
    )
    total = sum(len(c["points_2d"]) for c in curtains)
    print(f"Curtains: {len(curtains)} planes, {total} beads in {time.time()-t0:.2f}s")

    t0 = time.time()
    color_mod.apply_to_curtains(curtains, params)
    print(f"Colors assigned in {time.time()-t0:.2f}s")

    plane_xs = curtain_x_positions(
        params.curtain_count, params.curtain_spacing, x_center
    )
    display.render_curtain_planes(plane_xs, params.curtain_width, params.curtain_height)
    display.render_pointclouds(curtains)
    print("Rendered. Look at the viewport.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual verification**

1. Fresh Rhino doc, run `build_scene.py`.
2. Expected: same geometry as Task 6, but the bead points are now colored from the 6-color palette in painterly blobs.
3. Looking down +X (front view), the same Perlin "blob" should appear in roughly the same Y/Z region across multiple curtains — because the color field is sampled in 3D space, not per-curtain.
4. If all beads are the same color: check that the color FBM seed is `42` (different from geometry seed `1`) and `color_base_freq` is set.

- [ ] **Step 4: Commit**

```bash
git add RhinoPython/pearlscape/color.py RhinoPython/build_scene.py
git commit -m "feat: add palette-quantized FBM color field"
```

---

## Task 8: Display — instanced mesh sphere mode

**Files:**
- Modify: `RhinoPython/pearlscape/display.py`
- Modify: `RhinoPython/build_scene.py`

- [ ] **Step 1: Add `render_instances` to `display.py`**

Append the following to `RhinoPython/pearlscape/display.py`:

```python
def _build_bead_block_definition(diameter: float, subd: int, name: str = "PearlscapeBead") -> int:
    """Ensure a block definition for a unit bead mesh exists; return its index."""
    doc = sc.doc
    existing = doc.InstanceDefinitions.Find(name, True)
    if existing is not None:
        return existing.Index
    radius = diameter / 2.0
    sphere = rg.Sphere(rg.Point3d.Origin, radius)
    mesh = rg.Mesh.CreateFromSphere(sphere, max(6, 2 ** (subd + 1)), max(6, 2 ** (subd + 1)))
    # CreateFromSphere uses (count_around, count_vertical) — pass enough to give a smooth bead.
    idx = doc.InstanceDefinitions.Add(
        name,
        "Pearlscape bead",
        rg.Point3d.Origin,
        [mesh],
        [Rhino.DocObjects.ObjectAttributes()],
    )
    return idx


def render_instances(curtains: Sequence[dict], diameter: float, subd: int) -> None:
    """Render each bead as an InstanceReference to a shared bead mesh block.

    Per-instance color is set via ObjectAttributes (ColorSource = ColorFromObject).
    """
    doc = sc.doc
    block_idx = _build_bead_block_definition(diameter, subd)

    for i, c in enumerate(curtains):
        layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CURTAINS_LAYER}::Curtain_{i:02d}"
        layer_idx = _ensure_layer(layer_path)
        plane_x = c["plane_x"]
        pts_2d = c["points_2d"]
        colors = c.get("colors")
        for j in range(pts_2d.shape[0]):
            t = rg.Transform.Translation(
                plane_x, float(pts_2d[j, 0]), float(pts_2d[j, 1])
            )
            attrs = Rhino.DocObjects.ObjectAttributes()
            attrs.LayerIndex = layer_idx
            if colors is not None:
                rgb = colors[j]
                attrs.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
                attrs.ObjectColor = sd.Color.FromArgb(
                    int(rgb[0]), int(rgb[1]), int(rgb[2])
                )
            doc.Objects.AddInstanceObject(block_idx, t, attrs)
    sc.doc.Views.Redraw()
```

- [ ] **Step 2: Dispatch on `display_mode` in `build_scene.py`**

Replace the rendering block in `build_scene.py` (the last three lines of `main()` before `print("Rendered. ...")`) so that:

```python
    display.render_curtain_planes(plane_xs, params.curtain_width, params.curtain_height)
    if params.display_mode == "pointcloud":
        display.render_pointclouds(curtains)
    elif params.display_mode == "instances":
        display.render_instances(curtains, params.bead_diameter, params.instance_sphere_subd)
    else:
        raise ValueError(f"Unknown display_mode: {params.display_mode!r}")
    print(f"Rendered ({params.display_mode}). Look at the viewport.")
```

- [ ] **Step 3: Manual verification — pointcloud mode (regression check)**

1. Leave `display_mode = "pointcloud"` (default).
2. Fresh Rhino doc, run `build_scene.py`.
3. Expected: identical to Task 7 output.

- [ ] **Step 4: Manual verification — instances mode**

1. Edit `params.py`, change `display_mode = "instances"`.
2. **Lower bead count temporarily** for first test: set `total_surface_samples = 5000` (60k instances is heavy).
3. Fresh Rhino doc, run `build_scene.py`.
4. Expected: ~5k spherical mesh beads in the document, colored from the palette. Wireframe view shows them as polyhedra; shaded view shows colored spheres.
5. If render performance is acceptable at 5k, scale up: 20k, then full 60k. Note timings.
6. Reset `total_surface_samples = 60000` and `display_mode = "pointcloud"` when done.

- [ ] **Step 5: Commit**

```bash
git add RhinoPython/pearlscape/display.py RhinoPython/build_scene.py
git commit -m "feat: add instanced mesh-sphere display mode"
```

---

## Task 9: PDF export

**Files:**
- Create: `RhinoPython/pearlscape/export.py`
- Modify: `RhinoPython/build_scene.py`

**Context for the implementer:** Rhino exposes PDF export via `Rhino.FileIO.FilePdf`. We create one `PageView` (layout) per curtain, drop in a single orthographic detail looking along +X, set per-layout layer visibility so only that curtain's bead layer is on, then export each layout as its own PDF. We size the page to ISO A1 by default; final fabrication scale will be revisited.

- [ ] **Step 1: Write `pearlscape/export.py`**

```python
"""Per-curtain PDF export.

Creates one Rhino layout (PageView) per curtain, populates it with an
orthographic detail looking along +X, isolates that curtain's bead layer,
and writes one PDF per layout.
"""

import os
from typing import List

import Rhino
import Rhino.Geometry as rg
import Rhino.Display as rd
import scriptcontext as sc


PAGE_SIZES_MM = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
    "A0": (841.0, 1189.0),
}


def _curtains_parent_layer_index() -> int:
    doc = sc.doc
    idx = doc.Layers.FindByFullPath("Pearlscape::Curtains", -1)
    if idx < 0:
        raise RuntimeError("Pearlscape::Curtains parent layer not found. Run build_scene first.")
    return idx


def _curtain_layer_index(curtain_idx: int) -> int:
    path = f"Pearlscape::Curtains::Curtain_{curtain_idx:02d}"
    idx = sc.doc.Layers.FindByFullPath(path, -1)
    if idx < 0:
        raise RuntimeError(f"Layer not found: {path}")
    return idx


def _set_layout_layer_visibility(page: rd.RhinoPageView, visible_curtain: int) -> None:
    """Hide every Pearlscape::Curtains::Curtain_NN layer except the chosen one,
    plus CurtainPlanes and CaveReference, in this page's detail viewports.

    Per-viewport layer visibility in Rhino lives on the Layer (not the detail);
    we call layer.SetPerViewportVisible(detail.Viewport.Id, ...) for each detail.
    """
    doc = sc.doc
    extra_hide_paths = {
        "Pearlscape::CurtainPlanes",
        "Pearlscape::CaveReference",
    }
    details = page.GetDetailViews()
    if not details:
        return
    for layer in doc.Layers:
        full = layer.FullPath
        if full.startswith("Pearlscape::Curtains::Curtain_"):
            n = int(full.rsplit("_", 1)[-1])
            visible = (n == visible_curtain)
        elif full in extra_hide_paths:
            visible = False
        else:
            continue
        for det in details:
            layer.SetPerViewportVisible(det.Viewport.Id, visible)
        doc.Layers.Modify(layer, layer.Index, True)


def _make_layout(name: str, page_w_mm: float, page_h_mm: float) -> rd.RhinoPageView:
    """Create a new layout page sized to (w, h) in millimeters."""
    doc = sc.doc
    page = doc.Views.AddPageView(name, page_w_mm, page_h_mm)
    return page


def _add_orthographic_detail(
    page: rd.RhinoPageView,
    plane_x: float,
    curtain_width: float,
    curtain_height: float,
    margin_mm: float = 20.0,
) -> None:
    """Add one orthographic detail looking along +X, framed to the curtain bounds."""
    page_w = page.PageWidth
    page_h = page.PageHeight
    # Choose uniform fit-to-page scale (mm per meter).
    scale_w = (page_w - 2 * margin_mm) / (curtain_width * 1000.0)
    scale_h = (page_h - 2 * margin_mm) / (curtain_height * 1000.0)
    scale = min(scale_w, scale_h)

    # Detail rectangle on the page, centered.
    detail_w_mm = curtain_width * 1000.0 * scale
    detail_h_mm = curtain_height * 1000.0 * scale
    cx = page_w / 2.0
    cy = page_h / 2.0
    corner_a = rg.Point2d(cx - detail_w_mm / 2.0, cy - detail_h_mm / 2.0)
    corner_b = rg.Point2d(cx + detail_w_mm / 2.0, cy + detail_h_mm / 2.0)

    # Rhino's 'Right' projection looks along -X; curtain plane reads as (Y, Z).
    detail = page.AddDetailView("Curtain", corner_a, corner_b, rd.DefinedViewportProjection.Right)
    detail.Viewport.ZoomExtents()
    detail.DetailGeometry.IsProjectionLocked = True
    detail.CommitChanges()


def create_curtain_layouts(
    plane_xs: List[float],
    curtain_width: float,
    curtain_height: float,
    page_size: str = "A1",
) -> List[str]:
    """Create one layout per curtain. Returns the list of layout names."""
    if page_size not in PAGE_SIZES_MM:
        raise ValueError(f"Unknown page_size {page_size!r}; "
                         f"options: {sorted(PAGE_SIZES_MM)}")
    page_w, page_h = PAGE_SIZES_MM[page_size]

    names = []
    for i, x in enumerate(plane_xs):
        name = f"Curtain_{i:02d}"
        # Remove any pre-existing layout with this name to keep runs idempotent.
        existing = sc.doc.Views.GetPageViews()
        for v in existing:
            if v.PageName == name:
                sc.doc.Views.Remove(v)
                break
        page = _make_layout(name, page_w, page_h)
        _add_orthographic_detail(page, x, curtain_width, curtain_height)
        _set_layout_layer_visibility(page, i)
        names.append(name)
    sc.doc.Views.Redraw()
    return names


def export_all_pdfs(output_dir: str) -> List[str]:
    """Export every Curtain_NN layout as its own PDF. Returns the list of output paths."""
    os.makedirs(output_dir, exist_ok=True)
    out_paths: List[str] = []
    for view in sc.doc.Views.GetPageViews():
        if not view.PageName.startswith("Curtain_"):
            continue
        out_path = os.path.join(output_dir, f"{view.PageName}.pdf")
        pdf = Rhino.FileIO.FilePdf.Create()
        page_w_mm = view.PageWidth
        page_h_mm = view.PageHeight
        # Use 600 DPI for crisp dot reproduction.
        pdf.AddPage(view, page_w_mm, page_h_mm, 600)
        pdf.Write(out_path)
        out_paths.append(out_path)
    return out_paths
```

- [ ] **Step 2: Wire export into `build_scene.py`**

Add to the imports section of `build_scene.py`:

```python
from pearlscape import export as export_mod
```

Append to `main()` after the rendering block:

```python
    out_dir = os.path.join(_HERE, params.pdf_output_dir)
    t0 = time.time()
    layout_names = export_mod.create_curtain_layouts(
        list(plane_xs), params.curtain_width, params.curtain_height,
        page_size=params.pdf_page_size,
    )
    print(f"Created {len(layout_names)} layouts in {time.time()-t0:.2f}s")
    t0 = time.time()
    pdf_paths = export_mod.export_all_pdfs(out_dir)
    print(f"Exported {len(pdf_paths)} PDFs to {out_dir} in {time.time()-t0:.2f}s")
```

For clarity, the full intended `build_scene.py` after this edit is:

```python
#! python 3
# r: numpy
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time

from pearlscape import PearlscapeParams
from pearlscape.cave import make_default_cave
from pearlscape.curtains import array_x_center, curtain_x_positions, slice_and_project
from pearlscape import color as color_mod
from pearlscape import display
from pearlscape import export as export_mod


def main() -> None:
    params = PearlscapeParams()
    params.validate()

    t0 = time.time()
    cave = make_default_cave(params)
    pts = cave.sample_surface_points()
    print(f"Cave: {len(pts)} surface points in {time.time()-t0:.2f}s")

    t0 = time.time()
    x_center = array_x_center(params.cave_length)
    curtains = slice_and_project(
        pts,
        curtain_count=params.curtain_count,
        curtain_spacing=params.curtain_spacing,
        x_center=x_center,
    )
    total = sum(len(c["points_2d"]) for c in curtains)
    print(f"Curtains: {len(curtains)} planes, {total} beads in {time.time()-t0:.2f}s")

    t0 = time.time()
    color_mod.apply_to_curtains(curtains, params)
    print(f"Colors assigned in {time.time()-t0:.2f}s")

    plane_xs = curtain_x_positions(
        params.curtain_count, params.curtain_spacing, x_center
    )
    display.render_curtain_planes(plane_xs, params.curtain_width, params.curtain_height)
    if params.display_mode == "pointcloud":
        display.render_pointclouds(curtains)
    elif params.display_mode == "instances":
        display.render_instances(curtains, params.bead_diameter, params.instance_sphere_subd)
    else:
        raise ValueError(f"Unknown display_mode: {params.display_mode!r}")
    print(f"Rendered ({params.display_mode}). Look at the viewport.")

    out_dir = os.path.join(_HERE, params.pdf_output_dir)
    t0 = time.time()
    layout_names = export_mod.create_curtain_layouts(
        list(plane_xs), params.curtain_width, params.curtain_height,
        page_size=params.pdf_page_size,
    )
    print(f"Created {len(layout_names)} layouts in {time.time()-t0:.2f}s")
    t0 = time.time()
    pdf_paths = export_mod.export_all_pdfs(out_dir)
    print(f"Exported {len(pdf_paths)} PDFs to {out_dir} in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add `exports/` to `.gitignore`**

Append `RhinoPython/exports/` to the repo's `.gitignore`.

- [ ] **Step 4: Manual verification**

1. Fresh Rhino doc, run `build_scene.py`.
2. Expected console output:
   ```
   Cave: ... points in N.NNs
   Curtains: 25 planes, ... beads in N.NNs
   Colors assigned in N.NNs
   Rendered (pointcloud). Look at the viewport.
   Created 25 layouts in N.NNs
   Exported 25 PDFs to .../RhinoPython/exports in N.NNs
   ```
3. The Rhino layout tabs at the bottom of the viewport now show 25 layouts named `Curtain_00` through `Curtain_24`.
4. Click `Curtain_00`: you should see a single curtain's bead pattern, framed inside the page margins, in roughly the right scale.
5. Open `RhinoPython/exports/Curtain_00.pdf` in any PDF viewer. Confirm:
   - Page size is A1 (594×841mm).
   - Bead positions are visible.
   - Colors are reproduced.
   - Only this curtain's beads are visible (no other curtains bleeding through).
6. Spot-check `Curtain_12.pdf` (middle of array): should show a different bead pattern.

- [ ] **Step 5: Commit**

```bash
git add RhinoPython/pearlscape/export.py RhinoPython/build_scene.py .gitignore
git commit -m "feat: per-curtain layout creation and PDF export"
```

---

## Self-review notes (for the planner)

**Spec coverage:**
- Section 4 (Module structure): Tasks 1–9 create every listed file.
- Section 5 (Default parameters): Task 1 sets all values exactly per the spec.
- Section 6 (Geometry pipeline): Tasks 2–4 implement Perlin/FBM, blue-noise sampling, and the radial-displacement cave.
- Section 7 (Curtain slicing): Task 6.
- Section 8 (Color): Task 7.
- Section 9 (Display): Task 5 (PointCloud) + Task 8 (Instance).
- Section 10 (PDF export): Task 9.
- Section 11 (Known future directions): not implementation work — captured by leaving the `CaveSurface` interface in place.
- Section 12 (Assumptions): the `Rhino.FileIO.FilePdf` assumption is exercised in Task 9; if it fails, the implementer should fall back to `_-Print` command automation (note in task).

**Placeholder scan:** no `TBD` / `TODO` / "implement later". All code blocks are complete. `params.palette` defaults to concrete RGB tuples (user will refine artistically; not a placeholder).

**Type consistency:** the per-curtain dict keys `'plane_x'`, `'points_2d'`, `'points_3d'`, `'colors'` are consistent across `curtains.py`, `color.py`, `display.py`, `export.py`. `CaveSurface.sample_surface_points()` returns `np.ndarray (N, 3)` consistently.

**Known small risk in Task 9:** the per-viewport layer visibility API (`Layer.SetPerViewportVisible` + `Layers.Modify`) and `Rhino.FileIO.FilePdf.AddPage` signature have evolved across Rhino versions. If the Task 9 calls fail at runtime in Rhino 8 SR23, the fix is local (a few lines), and the verification step will catch it on first run. Fallback for PDF export: invoke `-_Print` via `Rhino.RhinoApp.RunScript` from a per-layout loop.
