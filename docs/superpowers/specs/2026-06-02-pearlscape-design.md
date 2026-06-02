# Pearlscape — Design Spec

**Date:** 2026-06-02
**Status:** Approved (brainstorming phase)
**Target environment:** Rhino 8 SR23, CPython 3 runtime, numpy available

## 1. Overview

Pearlscape is a physical artwork: a series of translucent curtains hung from a gallery ceiling, embedded with colored glass beads. Viewed perpendicular to the curtain stack, the layered beads reconstruct the cross-sectional depth of a virtual cave, producing a diaphanous reading of interior space.

This document specifies the Rhino Python pipeline that generates the parametric 3D model and prepares per-curtain fabrication data.

## 2. Goals and non-goals

**Goals**
- Parametric model: every visual parameter exposed in one place, single-script regeneration.
- Algorithmically generated cave geometry as the starting point (cylinder + FBM displacement).
- Per-curtain bead point patterns derived by slicing and projecting the cave surface.
- Two display modes: fast PointCloud for iteration, instanced mesh spheres for rendering.
- Per-curtain PDF export, one layout per curtain.
- Architecture seam that allows replacing the cylinder cave with a Lidar scan or other geometry later, without touching downstream code.

**Non-goals (deferred)**
- Lidar scan ingestion (interface ready, implementation later).
- Global cave shape control beyond a straight cylinder (Catmull-Clark of an irregular polygon is a known future direction).
- Bent / curved cave centerlines.
- Title blocks, registration marks, fabrication grids, color legends on PDFs.
- Multi-page combined PDFs, CSV exports.
- Automated testing — all verification is manual by the user.

## 3. Coordinate conventions

- **X** = visitor walking / viewing axis = cave's cylinder axis.
- **Y** = lateral (left-right).
- **Z** = vertical (up). Rhino default: right-handed, Z-up.
- Curtains are YZ planes spaced along X.
- Cave centered laterally at Y = 0, vertically at Z = 1.5 m.
- Curtain array centered on cave's X midpoint.

## 4. Module structure

Flat package under `RhinoPython/`. One module per concern. No abstraction beyond the `CaveSurface` seam.

```
RhinoPython/
  pearlscape/
    __init__.py
    params.py          # PearlscapeParams dataclass, single source of truth
    noise.py           # Perlin + FBM, numpy vectorized
    sampling.py        # Blue-noise candidate generation (Mitchell best-candidate)
    cave.py            # CaveSurface interface + CylinderFBMCave implementation
    curtains.py        # Slab assignment + projection onto curtain planes
    color.py           # Discrete palette + FBM-driven lookup
    display.py         # PointCloud renderer + InstanceReference renderer
    export.py          # Per-curtain PDF layout export
  build_scene.py       # Top-level run script
  README.md
```

`build_scene.py` is the only entry point. It instantiates `PearlscapeParams`, runs the pipeline, populates the Rhino document.

`CaveSurface` is the seam where future geometries plug in:

```python
class CaveSurface:
    def sample_surface_points(self, n: int) -> np.ndarray:
        """Return (n, 3) array of points on the cave surface."""
        ...
```

`CylinderFBMCave` is the default implementation. Future `LidarCave`, `SubdivisionCave`, etc. implement the same interface; nothing downstream changes.

## 5. Default parameters

All in `PearlscapeParams` (dataclass). Initial values; expected to be retuned.

### Cave geometry
| Param | Value | Notes |
|---|---|---|
| `cave_radius` | 1.2 m | Base cylinder radius |
| `cave_length` | 5.0 m | Along X |
| `fbm_amplitude` | 0.30 m | Max inward displacement |
| `fbm_base_freq` | 0.8 | Cycles per meter at base octave |
| `fbm_octaves` | 4 | |
| `fbm_lacunarity` | 2.0 | |
| `fbm_gain` | 0.5 | |
| `noise_seed` | 1 | |

### Curtain array
| Param | Value | Notes |
|---|---|---|
| `curtain_count` | 25 | |
| `curtain_spacing` | 0.20 m | Array spans 4.8 m along X |
| `curtain_width` | 2.5 m | Y extent |
| `curtain_height` | 3.0 m | Z extent (hangs from Z=3.0 to Z=0) |

### Beads
| Param | Value | Notes |
|---|---|---|
| `total_surface_samples` | 60,000 | Before curtain slab assignment |
| `bead_diameter` | 6 mm | For instanced display + fabrication |

### Color
| Param | Value | Notes |
|---|---|---|
| `palette` | 6 placeholder RGB tuples | Override before final |
| `color_base_freq` | 0.6 cycles/m | |
| `color_fbm_octaves` | 2 | Shallower than geometry |
| `color_fbm_lacunarity` | 2.0 | |
| `color_fbm_gain` | 0.5 | |
| `color_noise_seed` | 42 | Independent of `noise_seed` |

### Display / export
| Param | Value | Notes |
|---|---|---|
| `display_mode` | `"pointcloud"` | or `"instances"` |
| `instance_sphere_subd` | 2 | Icosphere subdivisions (42 verts) |
| `pdf_page_size` | `"A1"` | 594×841 mm; revisit at fabrication time |

## 6. Geometry pipeline

Single pass, numpy-vectorized.

**Step 1 — Blue-noise candidate generation**
Generate `total_surface_samples` 2D points in parameter space `(θ ∈ [0, 2π), x ∈ [0, L])`. Mitchell's best-candidate algorithm with θ-wrap (toroidal distance metric on the θ axis). Expected runtime ≈ 1–2 s for 60k points.

**Step 2 — FBM evaluation**
For each `(θ, x)`, evaluate FBM on the 3D vector `(cos θ · r_base, sin θ · r_base, x)`. The 3D input ensures continuity across the θ=0 / θ=2π seam.

Perlin implementation: gradient-grid version (Ken Perlin's improved noise, 2002), ~80 lines numpy. FBM is `sum_k (gain^k * perlin(p * lacunarity^k))` for `k ∈ [0, octaves)`, normalized so the output is in `[0, 1)`.

**Step 3 — Radial displacement**
```
r_eff = cave_radius - fbm_amplitude * fbm01(θ, x)
px    = x
py    = r_eff * cos θ
pz_local = r_eff * sin θ
```

**Step 4 — Place in world**
Translate so cave is centered at the curtain array's X midpoint, Y=0, Z=1.5.

**Output**
`np.ndarray` of shape `(N, 3)`. Cave is open at both X ends (no caps).

**Self-intersection guard**
Assert `fbm_amplitude < cave_radius` at construction. Warn if amplitude × base frequency suggests overhang risk (heuristic threshold; refine on observation).

## 7. Curtain slicing and projection

**Slab assignment**
Curtain `i` sits at `x_i = x_center + (i - (N-1)/2) · spacing`. Each curtain owns the half-open slab `[x_i - spacing/2, x_i + spacing/2)`. Points outside the array's X range are dropped.

**Projection**
For each point `(px, py, pz)` assigned to curtain `i`, the projected position is `(x_i, py, pz)`. The original 3D position is retained alongside.

**Why retain the 3D position**
Color is sampled at the **original 3D location**, not the projected location. This keeps the color noise spatially coherent across curtains — a noise feature spans multiple curtains rather than being re-randomized per slice.

**Output**
List of dicts, one per curtain:
```python
{
    'plane_x': float,
    'points_2d': np.ndarray,    # (M_i, 2), columns (Y, Z)
    'points_3d': np.ndarray,    # (M_i, 3), original positions
}
```

**Bounds check**
At default params the cave (radius 1.2, center Z=1.5) fits inside the 2.5×3.0m curtain comfortably. A soft warning is emitted if any bead lands outside the curtain rectangle.

## 8. Color model

**Palette**
List of `M` discrete RGB tuples in `params.py` (default `M = 6`).

**Driver**
A second FBM field, independent seed and frequency, **2 octaves** (shallower than geometry's 4). Same `noise.py` implementation; only the parameters differ.

**Evaluation**
For each bead, evaluate the color FBM at its **3D pre-projection position**. Normalize the FBM output to `[0, 1)`. Compute `palette_index = floor(value * M)`. Look up `palette[palette_index]`.

**Output**
Each curtain gains a third array: `'colors': np.ndarray (M_i, 3)` of uint8 RGB.

## 9. Display pipeline

Two render modes, selected by `display_mode`. Both consume the per-curtain pipeline output.

### Mode A — PointCloud (default)

For each curtain, build a single `Rhino.Geometry.PointCloud`. Use `PointCloud.Add(point, color)` for per-vertex color. Add to per-curtain Rhino layer. Approximately one `PointCloud` per curtain (~25 objects total). Optimized for fast iteration.

### Mode B — Instanced mesh spheres

1. Build a low-poly icosphere mesh once via `Mesh.CreateIcoSphere(subdivisions=2)`, scaled to `bead_diameter`.
2. Register as a block definition: `doc.InstanceDefinitions.Add("PearlscapeBead", ...)`.
3. For each bead: build a translation `Transform`, set `ObjectAttributes.ColorSource = ColorFromObject`, `ObjectAttributes.ObjectColor = bead_color`, call `doc.Objects.AddInstanceObject(block_index, transform, attributes)`.

Heavier (60k instance refs) but renders properly with materials, lighting, depth of field.

### Layer organization (both modes)
```
Pearlscape
├── Curtains
│   ├── Curtain_00
│   ├── Curtain_01
│   └── ...
├── CurtainPlanes      # 2.5×3.0m rectangles for visualization
└── CaveReference      # un-sliced cave point cloud, hidden by default
```

Per-curtain layers are required by the PDF export.

### Switch
`build_scene.py` dispatches to either `display.render_pointclouds(curtains)` or `display.render_instances(curtains, block_def)`. No conditional logic outside this dispatch.

## 10. PDF export

**Per-curtain layout creation**
For each curtain `i`:
1. Create a `PageView` named `Curtain_{i:02d}`, sized to `pdf_page_size`.
2. Add a single detail view, orthographic, looking along +X. The curtain reads as a flat `(Y, Z)` drawing.
3. Set detail scale: 1:1 if the curtain fits the page; else uniform fit-to-page with the scale recorded in the layout name (`Curtain_00_1to2`). 2.5×3.0m on A1 → roughly 1:4 with margins.
4. In the detail's layer visibility, isolate that curtain's bead layer; hide all other `Curtain_*`, `CurtainPlanes`, `CaveReference`.

**Export call**
`export.export_all_pdfs(folder)` iterates layouts, writes one PDF per layout via `Rhino.FileIO.FilePdf`. Default output: `RhinoPython/exports/curtain_{i:02d}.pdf`.

**Deferred** (per brief: "additional labels and grids as needed later")
- Title block / page metadata
- Registration marks / fabrication grid
- Color legend
- Combined multi-page PDF
- CSV / non-printing fabrication exports

**Flagged for verification**
At default params, beads on paper are ~1.5 mm dots (6 mm beads at 1:4). Print legibility needs to be checked on a real proof before locking scale.

## 11. Known future directions

Captured here so the architecture leaves room without preemptively building for them:

- **Lidar cave**: plug a new `CaveSurface` implementation into `cave.py`. Downstream pipeline unchanged.
- **Global shape control via Catmull-Clark of an irregular polygon**: another `CaveSurface` implementation.
- **Bent / curved cave centerlines**: introduce a centerline curve and Frenet frame in the cylinder implementation; surface sampling stays parameter-space.
- **Bead size variation**: extend per-bead attributes from `(position, color)` to include `diameter`.
- **Non-uniform palette weighting**: change quantization thresholds in `color.py`, palette structure unchanged.
- **Blue-noise from texture**: swap `sampling.py` implementation, interface unchanged.
- **SDF-based geometry pipeline**: behind the same `CaveSurface` interface.

## 12. Assumptions

- numpy is available in Rhino 8 SR23's CPython runtime (ships with it).
- `Rhino.Geometry.PointCloud.Add(point, color)` supports per-vertex color (it does in Rhino 8).
- `Rhino.FileIO.FilePdf` API is usable from scripted Python in Rhino 8 (to be verified during implementation; fallback is `_Print` command automation).
- Visitors view the artwork from the perpendicular axis — they do not walk through the cave; the curtains are unpassable.
- All testing / verification is manual by the user.
