# Thin-Shell Curtain Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `curtain_mode = "shell"` track that snaps the raw cave Poisson surface cloud onto densely-spaced thin planes, dropping per-plane overlaps, so the model becomes a thin cuttable shell that still reads as the default cave.

**Architecture:** A new param selects between the existing `build_curtains` (band) and a new `build_shell_curtains` (shell). Shell mode reuses `cave.sample_surface_points()`, snaps each bead's X to the nearest plane, and runs a deterministic greedy overlap filter per plane. Output dicts match band mode so color/display/PDF/PLY paths are unchanged.

**Tech Stack:** Python 3, numpy. Tests are headless `__main__` smoke blocks run with `python <file>` (no pytest). Rhino is only needed for the final viewport check.

**Project conventions:**
- Lengths in mm.
- Do NOT run `git commit` — the user commits. Each task ends at a review checkpoint.
- Entry-point `.py` files keep their `#! python 3` line 1 and `# r: numpy` line 2.

---

## File Structure

- `RhinoPython/pearlscape/params.py` — add `curtain_mode`, `shell_curtain_spacing`, validation. (Modify)
- `RhinoPython/pearlscape/curtains.py` — add `_shell_spacing`, `_greedy_overlap_filter`, `build_shell_curtains`; generalize `curtain_planes` for shell mode; extend `__main__` tests. (Modify)
- `RhinoPython/build_scene.py` — import and dispatch to `build_shell_curtains` when `curtain_mode == "shell"`. (Modify)

---

## Task 1: Add shell-mode parameters

**Files:**
- Modify: `RhinoPython/pearlscape/params.py`

- [ ] **Step 1: Add the two parameters to the "Curtain array" section**

In `RhinoPython/pearlscape/params.py`, find this block (around line 62-69):

```python
    curtain_count: int = 100
    curtain_spacing: float = 50.0
    curtain_width: float = 2500.0
    curtain_height: float = 3000.0
    cave_center_z: float = 1500.0   # cave centerline elevation
```

Insert immediately AFTER that block (before the `# --- Beads ---` comment):

```python
    # Curtain generation mode:
    #   "band"  -> build_curtains: a thick band sampled outward from each cross-
    #              section boundary (the original volumetric curtains).
    #   "shell" -> build_shell_curtains: snap the raw cave Poisson surface cloud
    #              onto densely-spaced thin planes (a thin cuttable shell that
    #              mirrors the default cave look). Leaves band logic untouched.
    curtain_mode: str = "band"
    # Thin-shell plane spacing (mm); used only when curtain_mode == "shell".
    # 0 = auto, resolved to 2 * bead_diameter at use.
    shell_curtain_spacing: float = 0.0
```

- [ ] **Step 2: Add validation**

In the same file, in `validate()`, find:

```python
        assert self.curtain_count >= 2
```

Insert immediately after it:

```python
        assert self.curtain_mode in ("band", "shell")
        assert self.shell_curtain_spacing >= 0.0
```

- [ ] **Step 3: Verify params still import and validate**

Run:
```bash
python -c "import sys; sys.path.insert(0, 'RhinoPython'); from pearlscape.params import PearlscapeParams; p = PearlscapeParams(); p.validate(); print('curtain_mode=', p.curtain_mode, 'shell_curtain_spacing=', p.shell_curtain_spacing)"
```
Expected: `curtain_mode= band shell_curtain_spacing= 0.0` (no assertion error).

- [ ] **Step 4: Checkpoint** — review the diff. (User commits.)

---

## Task 2: Shell builder + plane generalization + tests

**Files:**
- Modify: `RhinoPython/pearlscape/curtains.py`

This task is TDD: write the failing `__main__` test additions first, confirm they fail, then implement the three functions.

- [ ] **Step 1: Write the failing tests (add to `__main__`)**

In `RhinoPython/pearlscape/curtains.py`, find the final lines of the `__main__` block:

```python
    print(f"5 curtains, {total} beads, deterministic, seeds distinct")
    print(f"curtain_planes: nurbs->{len(pn)} planes, cylinder->{len(pc)} (filtered)")
    print(f"bead budget: target 20,000 -> {total_b} beads ({rel:.1%} off)")
    print("OK")
```

Insert the following BEFORE the `print(f"5 curtains, ...")` line:

```python
    # --- shell mode: snap the cave cloud onto thin planes, drop overlaps ---
    shell_params = PearlscapeParams()
    shell_params.curtain_mode = "shell"
    shell_params.cave_type = "cylinder"
    shell_params.shell_curtain_spacing = 0.0          # auto -> 2 * bead_diameter
    assert _shell_spacing(shell_params) == 2.0 * shell_params.bead_diameter

    shell_cave = CylinderFBMCave(
        radius=shell_params.cave_radius, length=2000.0, fbm_amplitude=900.0,
        fbm_base_freq=0.00035, fbm_octaves=6, fbm_lacunarity=2.0, fbm_gain=0.55,
        noise_seed=shell_params.noise_seed, target_samples=1000, center_z=1500.0,
        noise_type="ridged", bead_spacing=12.0,        # Poisson cloud, 12mm spacing
    )
    source = shell_cave.sample_surface_points()
    shell_planes = curtain_planes(shell_cave, shell_params)
    # planes tile the X extent [0, 2000] at 2*bead_diameter = 12mm spacing
    assert shell_planes.min() >= -1e-6 and shell_planes.max() <= 2000.0 + 1e-6
    assert np.allclose(np.diff(shell_planes), 12.0)

    shell_curtains, shell_spacing_used = build_shell_curtains(
        shell_cave, shell_planes, shell_params)
    assert shell_spacing_used == shell_params.bead_diameter

    # every kept bead lies exactly on its plane; points_2d/points_3d agree
    for c in shell_curtains:
        if c["points_3d"].shape[0]:
            assert np.allclose(c["points_3d"][:, 0], c["plane_x"])
            assert np.array_equal(c["points_3d"][:, 1], c["points_2d"][:, 0])
            assert np.array_equal(c["points_3d"][:, 2], c["points_2d"][:, 1])

    # no two kept beads on a plane are closer than bead_diameter in Y/Z
    bd = shell_params.bead_diameter
    for c in shell_curtains:
        yz = c["points_2d"]
        if yz.shape[0] >= 2:
            d2 = ((yz[:, None, :] - yz[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(d2, np.inf)
            assert d2.min() >= bd * bd - 1e-6, f"overlap on plane: {np.sqrt(d2.min()):.3f}"

    shell_total = sum(c["points_2d"].shape[0] for c in shell_curtains)
    assert 0 < shell_total <= source.shape[0], (shell_total, source.shape[0])

    # determinism: same params -> identical kept beads
    shell_curtains2, _ = build_shell_curtains(shell_cave, shell_planes, shell_params)
    assert all(np.array_equal(a["points_2d"], b["points_2d"])
               for a, b in zip(shell_curtains, shell_curtains2)), "shell not deterministic"

    print(f"shell: {len(shell_planes)} planes, "
          f"{shell_total}/{source.shape[0]} beads kept")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
python RhinoPython/pearlscape/curtains.py
```
Expected: FAIL with `NameError: name '_shell_spacing' is not defined` (the first new assertion references it).

- [ ] **Step 3: Implement `_shell_spacing`**

In `RhinoPython/pearlscape/curtains.py`, add this function immediately after `array_x_center` (before `curtain_planes`):

```python
def _shell_spacing(params) -> float:
    """Thin-shell plane spacing (mm): params.shell_curtain_spacing, or
    2 * bead_diameter when that is 0 (auto)."""
    if params.shell_curtain_spacing > 0.0:
        return float(params.shell_curtain_spacing)
    return 2.0 * float(params.bead_diameter)
```

- [ ] **Step 4: Generalize `curtain_planes` for shell mode**

Replace the body of `curtain_planes` (the function currently spanning roughly lines 33-62). Replace from `x_min, x_max = cave.x_extent()` through the `else:` branch that sets `plane_xs` (i.e. up to but NOT including the `inside = ...` line) with:

```python
    x_min, x_max = cave.x_extent()

    if params.curtain_mode == "shell":
        # Shell mode always tiles the X extent at the thin-shell spacing,
        # regardless of cave_type — an explicit curtain_count makes no sense
        # for a dense thin shell.
        spacing = _shell_spacing(params)
        use_extent_fit = True
    else:
        spacing = params.curtain_spacing
        use_extent_fit = (params.cave_type == "nurbs")

    if use_extent_fit:
        length = x_max - x_min
        gaps = int(np.floor(length / spacing + 1e-9))   # spacing intervals that fit
        count = max(1, gaps + 1)
        span = gaps * spacing
        x0 = x_min + (length - span) / 2.0              # center the planes in the extent
        plane_xs = x0 + np.arange(count) * spacing
    else:
        x_center = 0.5 * (x_min + x_max)
        plane_xs = curtain_x_positions(params.curtain_count, spacing, x_center)
```

Leave the trailing `inside = ...` / `dropped = ...` / `return plane_xs[inside]` lines exactly as they are. (This preserves band-mode behavior: nurbs still extent-fits, cylinder still uses `curtain_count`.)

- [ ] **Step 5: Implement `_greedy_overlap_filter`**

Add this function in `RhinoPython/pearlscape/curtains.py` immediately before `build_curtains`:

```python
def _greedy_overlap_filter(yz: np.ndarray, radius: float) -> np.ndarray:
    """Greedy keep-order filter over (M, 2) points. Returns a boolean mask that
    keeps a point only if no already-kept point lies within `radius` of it.

    Uses a radius-sized spatial-hash grid so each point only tests its 3x3 cell
    neighbourhood (two points within `radius` differ by at most one cell per
    axis). Deterministic in input order — the caller passes points in the cave
    sampler's fixed order, so reruns are bit-identical.
    """
    m = yz.shape[0]
    keep = np.zeros(m, dtype=bool)
    if m == 0:
        return keep
    inv = 1.0 / radius
    r2 = radius * radius
    cx_all = np.floor(yz[:, 0] * inv).astype(np.int64)
    cz_all = np.floor(yz[:, 1] * inv).astype(np.int64)
    cells = {}   # (cx, cz) -> list of kept-point row indices
    for j in range(m):
        cx, cz = int(cx_all[j]), int(cz_all[j])
        py, pz = yz[j, 0], yz[j, 1]
        clash = False
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                bucket = cells.get((cx + dx, cz + dz))
                if not bucket:
                    continue
                for k in bucket:
                    dy = py - yz[k, 0]
                    dz2 = pz - yz[k, 1]
                    if dy * dy + dz2 * dz2 < r2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            keep[j] = True
            cells.setdefault((cx, cz), []).append(j)
    return keep
```

- [ ] **Step 6: Implement `build_shell_curtains`**

Add this function in `RhinoPython/pearlscape/curtains.py` immediately after `build_curtains` (before the `if __name__ == "__main__":` block):

```python
def build_shell_curtains(cave, plane_xs: Sequence[float], params) -> Tuple[List[dict], float]:
    """Thin-shell mode: snap the cave's Poisson surface cloud onto the thin
    curtain planes (X -> nearest plane, Y/Z unchanged), then drop per-plane
    overlaps within bead_diameter so each plane is a gappy, non-continuous line.

    Returns (curtains, bead_spacing) with the SAME dict shape as build_curtains:
      {'plane_x', 'points_2d' (M, 2)=(Y, Z), 'points_3d' (M, 3)=(plane_x, Y, Z)}
    bead_spacing is the bead_diameter drop radius (reported in the run summary).
    """
    plane_xs = np.asarray(plane_xs, dtype=np.float64)
    n_planes = int(plane_xs.shape[0])
    radius = float(params.bead_diameter)

    empty = lambda px: {
        "plane_x": float(px),
        "points_2d": np.zeros((0, 2), dtype=np.float64),
        "points_3d": np.zeros((0, 3), dtype=np.float64),
    }

    pts = cave.sample_surface_points()              # (N, 3) world points
    if pts.shape[0] == 0 or n_planes == 0:
        return [empty(px) for px in plane_xs], radius

    # Snap each bead to its nearest plane. Planes are uniformly spaced, so a
    # single round gives the index; clamp ends-beads into the valid range.
    x0 = float(plane_xs[0])
    spacing = _shell_spacing(params)
    idx = np.round((pts[:, 0] - x0) / spacing).astype(np.int64)
    idx = np.clip(idx, 0, n_planes - 1)

    curtains: List[dict] = []
    for i, px in enumerate(plane_xs):
        yz = pts[idx == i][:, 1:3]                   # boolean index preserves order
        if yz.shape[0] == 0:
            curtains.append(empty(px))
            continue
        kept = _greedy_overlap_filter(yz, radius)
        kept_yz = yz[kept]
        if kept_yz.shape[0]:
            pts_3d = np.column_stack([
                np.full(kept_yz.shape[0], float(px), dtype=np.float64),
                kept_yz[:, 0],
                kept_yz[:, 1],
            ])
        else:
            pts_3d = np.zeros((0, 3), dtype=np.float64)
        curtains.append({
            "plane_x": float(px),
            "points_2d": kept_yz,
            "points_3d": pts_3d,
        })
    return curtains, radius
```

- [ ] **Step 7: Run the test to verify it passes**

Run:
```bash
python RhinoPython/pearlscape/curtains.py
```
Expected: PASS — output ends with lines like:
```
shell: 167 planes, NNNNN/MMMMM beads kept
5 curtains, ... beads, deterministic, seeds distinct
curtain_planes: nurbs->41 planes, cylinder->40 (filtered)
bead budget: target 20,000 -> ... beads (...% off)
OK
```
The exact `shell:` counts depend on the cloud, but `NNNNN <= MMMMM` and the `OK` line must appear with no assertion error. (The existing `nurbs->41` / `cylinder->40` line must be UNCHANGED, confirming band-mode plane logic still works.)

- [ ] **Step 8: Checkpoint** — review the diff. (User commits.)

---

## Task 3: Dispatch in the pipeline

**Files:**
- Modify: `RhinoPython/build_scene.py`

- [ ] **Step 1: Import the shell builder**

In `RhinoPython/build_scene.py`, find:

```python
from pearlscape.curtains import build_curtains, curtain_planes
```

Replace with:

```python
from pearlscape.curtains import build_curtains, build_shell_curtains, curtain_planes
```

- [ ] **Step 2: Dispatch on curtain_mode**

In `main()`, find:

```python
    t0 = time.time()
    plane_xs = curtain_planes(cave, params)
    curtains, bead_spacing = build_curtains(cave, plane_xs, params)
    total = sum(len(c["points_2d"]) for c in curtains)
    print(f"Curtains: {len(curtains)} planes, {total} beads in {time.time()-t0:.2f}s")
```

Replace the single `curtains, bead_spacing = build_curtains(...)` line with:

```python
    if params.curtain_mode == "shell":
        curtains, bead_spacing = build_shell_curtains(cave, plane_xs, params)
    else:
        curtains, bead_spacing = build_curtains(cave, plane_xs, params)
```

(Leave the surrounding `t0`, `plane_xs`, `total`, and `print` lines unchanged.)

- [ ] **Step 3: Verify build_scene imports cleanly (syntax/import check)**

`build_scene.py` imports Rhino modules indirectly only via the nurbs path; the
curtains import itself is pure numpy. Confirm the module parses and the new
symbol resolves:

Run:
```bash
python -c "import ast; ast.parse(open('RhinoPython/build_scene.py').read()); print('build_scene.py parses OK')"
python -c "import sys; sys.path.insert(0, 'RhinoPython'); from pearlscape.curtains import build_shell_curtains; print('build_shell_curtains importable')"
```
Expected:
```
build_scene.py parses OK
build_shell_curtains importable
```

- [ ] **Step 4: Manual Rhino verification (user-run)**

In Rhino's Script Editor, set `curtain_mode = "shell"` and `pipeline_mode = "curtains"` in `params.py`, then F5 `build_scene.py`. Confirm the viewport shows a thin shell of beads on densely-spaced planes that reads like the default cave (not fat slabs, not continuous rings). This step needs Rhino and is run by the user.

- [ ] **Step 5: Checkpoint** — review the diff. (User commits.)

---

## Self-Review Notes

- **Spec coverage:** `curtain_mode`/`shell_curtain_spacing` + validation (Task 1); `build_shell_curtains` snap + overlap drop, `_shell_spacing`, `curtain_planes` shell branch (Task 2); `build_scene` dispatch (Task 3). Headless tests assert on-plane placement, overlap-free spacing, determinism, count bounds, and auto-spacing — all spec test points covered.
- **Type consistency:** `build_shell_curtains` returns `(List[dict], float)` matching `build_curtains`; dict keys `plane_x`/`points_2d`/`points_3d` match. `_shell_spacing` used identically in `curtain_planes` and `build_shell_curtains`. `_greedy_overlap_filter` returns a boolean mask consumed by `yz[kept]`.
- **Band mode untouched:** `curtain_planes` refactor preserves nurbs→extent-fit and cylinder→`curtain_count`; the existing `nurbs->41`/`cylinder->40` assertions guard this.
- **Concern (carried from spec):** ~167 PDFs in `export` mode for shell. Not capped here.
