# Pearlscape

Parametric Rhino Python pipeline for the *Pearlscape* installation — a series of translucent, ceiling-hung curtains embedded with coloured glass beads. The beads on each curtain fill the plane outward from the cave's cross-section curve at that depth — a sharp inner cave edge with a soft outer volume — so that, viewed perpendicular to the stack, the layered slices reconstruct the cave's interior depth.

## Requirements

- **Rhino 8 SR23** or newer.
- Use the **new Script Editor** (`_ScriptEditor` command), *not* the legacy `_EditPythonScript`. The legacy editor only supports IronPython 2 and will fail on the Python 3 syntax this project uses.
- The CPython 3 runtime ships with `numpy`. No manual install needed — `build_scene.py` declares it via `# r: numpy` and Rhino auto-installs it on first run.

## Running

1. Open Rhino 8.
2. Make sure your document units are **millimetres** (this project's parameters are all in mm).
3. Run `_ScriptEditor`.
4. Open `RhinoPython/build_scene.py`.
5. Press **F5**.

The script regenerates the scene from scratch on every run. For the cleanest results, start in a fresh Rhino document each time — successive runs into the same document accumulate geometry and layouts.

## The three pipeline modes

Edit `pipeline_mode` in `RhinoPython/pearlscape/params.py` to control how much of the pipeline runs.

| `pipeline_mode` | What it produces                                                                                 | Fast to iterate? | Use when…                                                  |
|-----------------|--------------------------------------------------------------------------------------------------|------------------|------------------------------------------------------------|
| `"cave"`        | One unsliced, palette-coloured cave point cloud on the `Pearlscape::CaveReference` layer.        | Fastest          | Tuning the cave's geometry, noise, or colour field.        |
| `"curtains"`    | Cave cross-section is taken at 25 curtain planes; each plane's beads are sampled in-plane, outward from the cross-section curve.                    | Medium           | Tuning curtain spacing, colour palette, or display mode.   |
| `"export"`      | Curtain model **plus** 25 A1 layout pages, one PDF per curtain in `exports/`.                    | Slowest          | Producing fabrication output, or final preflight.          |

Each mode is a strict superset of the one above it.

## The `display_mode` toggle

| `display_mode`  | What you see                                                          | Use when…                                  |
|-----------------|-----------------------------------------------------------------------|--------------------------------------------|
| `"pointcloud"`  | Flat coloured point dots — one Rhino `PointCloud` per curtain.        | Fast iteration; the everyday working mode. |
| `"instances"`   | Coloured mesh spheres — one `InstanceReference` per bead.             | Material / lighting / DOF renders.         |
| `"sprites"`     | Coloured circles (`bead_diameter` mm, scale with distance), drawn by a display conduit. | Eyeballing very high bead counts fast.     |

Instances mode is heavier (60k InstanceReferences for the default bead count). For first tests in this mode, lower `total_surface_samples` to `5_000` and scale back up once you've confirmed it renders correctly.

Sprites mode is the fast path for large bead counts: all beads draw in a single batched `DrawSprites` call. Caveats — it is **screen-only** (a display conduit, not document geometry), so it does **not** appear in PDF export (`pipeline_mode="export"` rejects it) and offers no per-curtain layer toggling (it's all-beads-or-nothing). It works in `pipeline_mode="cave"` and `"curtains"`. The conduit persists across F5 runs and is cleared automatically when you switch back to `pointcloud`/`instances`. Because sprites aren't document geometry, they can't be selected or deleted with normal Rhino tools — to hide them without re-running the pipeline, open `RhinoPython/clear_sprites.py` and press F5.

## Parameter guide

All tunable parameters live in `RhinoPython/pearlscape/params.py` in the `PearlscapeParams` dataclass. Lengths are in **millimetres**, frequencies in **cycles per millimetre**.

### Cave geometry

| Parameter        | Default   | Meaning                                                     |
|------------------|-----------|-------------------------------------------------------------|
| `cave_radius`    | `1200.0`  | Base cylinder radius (mm).                                  |
| `cave_length`    | `5000.0`  | Cave length along X (mm).                                   |
| `fbm_amplitude`  | `300.0`   | Maximum inward radial displacement of the wall (mm).        |
| `fbm_base_freq`  | `0.0008`  | Noise frequency at the base octave (cycles/mm).             |
| `fbm_octaves`    | `4`       | Number of FBM octaves; more octaves = more fine detail.     |
| `fbm_lacunarity` | `2.0`     | Per-octave frequency multiplier.                            |
| `fbm_gain`       | `0.5`     | Per-octave amplitude multiplier.                            |
| `noise_type`     | `"fbm"`   | `"fbm"` (smooth, rolling) or `"ridged"` (sharp inward crevices). Reuses the `fbm_*` octave/lacunarity/gain params. |
| `noise_seed`     | `1`       | Seed for the geometry noise; change for a different cave.   |
| `cave_type`      | `"cylinder"` | `"cylinder"` (radial-noise cylinder) or `"nurbs"` (lofted irregular tube). See below. |

Tip: if you re-tune `cave_radius` and want the noise pattern to look the same, scale `fbm_base_freq` inversely (e.g. doubling radius → halve frequency).

### NURBS cave (`cave_type = "nurbs"`)

An irregular tube lofted from jittered cross-section rings, sampled and displaced
with the same `fbm_*` / `noise_type` noise. The base surface is kept as editable
document geometry on the `Pearlscape::CaveSurface` layer.

| Parameter                  | Default     | Meaning                                                            |
|----------------------------|-------------|--------------------------------------------------------------------|
| `nurbs_surface_source`     | `"rebuild"` | `"rebuild"` lofts from the params below; `"reuse"` samples the (hand-edited) surface on the `CaveSurface` layer (falls back to rebuild if none). |
| `nurbs_sections`           | `8`         | Cross-section rings along X.                                       |
| `nurbs_section_points`     | `12`        | Control points per ring (minimum 4).                               |
| `nurbs_radius_jitter`      | `0.35`      | Per-point radius variation, fraction of `cave_radius`.             |
| `nurbs_radius_jitter_freq` | `0.0006`    | Noise frequency for radius jitter (cycles/mm).                     |
| `nurbs_centerline_amp`     | `400.0`     | Centerline meander amplitude (mm, Y/Z only).                       |
| `nurbs_centerline_freq`    | `0.0003`    | Noise frequency for centerline meander (cycles/mm).               |
| `nurbs_shape_seed`         | `7`         | Seed for the shape noise (independent of `noise_seed`).            |
| `nurbs_grid_u` / `nurbs_grid_v` | `180` / `180` | Surface-eval grid resolution (X / θ). Raise for finer base form. |

To hand-edit: run once in `"rebuild"`, then `_PointsOn` the surface, move points,
`_PointsOff`, switch to `"reuse"`, and re-run to resample from your edited form.

### Curtain array

| Parameter         | Default   | Meaning                                                                |
|-------------------|-----------|------------------------------------------------------------------------|
| `curtain_count`   | `25`      | Number of curtain planes. **Ignored in `"nurbs"` mode** — there the count is derived from the surface's X extent and `curtain_spacing`. Used only in `"cylinder"` mode. |
| `curtain_spacing` | `200.0`   | Spacing between adjacent curtains along X (mm).                        |
| `curtain_width`   | `2500.0`  | Curtain width along Y (mm).                                            |
| `curtain_height`  | `3000.0`  | Curtain height along Z (mm); ceiling at Z=3000, floor at Z=0.          |
| `cave_center_z`          | `1500.0`  | Vertical position of the cave's centreline (mm).                       |
| `curtain_band_thickness` | `120.0`   | Max outward reach of beads from the inner cross-section curve (mm). The gap-bridging knob — raise to close inter-curtain gaps. |
| `curtain_band_fade`      | `1.5`     | Outward density falloff exponent: `p = (1 − d/T)^fade`. Higher = beads hug the inner edge tighter. |
| `bead_min_spacing`       | `6.0`     | In-plane blue-noise spacing (mm). Must be ≥ `bead_diameter`. |
| `target_bead_count`      | `0`       | When > 0, auto-solves `bead_min_spacing` each run so the total bead count lands near this target (given the live band shape + cave geometry). `0` uses `bead_min_spacing` as set above. The solved spacing is printed to the console. |

Note: the curtain array is fitted to the cave's **actual X extent**. In `"nurbs"` mode the plane count is derived so curtains span the surface exactly (every `curtain_spacing` mm); `curtain_count` is ignored. In `"cylinder"` mode `curtain_count` planes are placed, centered on the cave. In both modes any plane falling outside the cave's X extent is dropped (and logged), so the curtain geometry never extends past the surface.

### Beads

| Parameter               | Default   | Meaning                                                       |
|-------------------------|-----------|---------------------------------------------------------------|
| `total_surface_samples` | `60_000`  | Approximate number of beads across the whole cave.            |
| `bead_diameter`         | `6.0`     | Physical bead diameter (mm). Used in `"instances"` mode.      |

Bridson's Poisson-disk sampler tends to undershoot by ~30–40%, so the actual bead count will be lower than the target. Tighten by raising `total_surface_samples` proportionally.

Note: in `"curtains"`/`"export"` modes the bead count is **not** governed by `total_surface_samples`. Beads are sampled directly in each curtain plane, so the count emerges from the band geometry — tune it via `curtain_band_thickness` and `bead_min_spacing`. To target a specific count instead, set `target_bead_count` (> 0) and the run back-solves `bead_min_spacing` for you (landing a few percent under target). `total_surface_samples` governs only the `"cave"` reference cloud.

### Colour

| Parameter             | Default   | Meaning                                                                  |
|-----------------------|-----------|--------------------------------------------------------------------------|
| `palette`             | 6 colours | Discrete RGB palette; each bead picks one entry.                         |
| `color_base_freq`     | `0.0006`  | Colour noise frequency (cycles/mm).                                      |
| `color_fbm_octaves`   | `2`       | Octaves of FBM for the colour field. More → flatter palette distribution.|
| `color_fbm_lacunarity`| `2.0`     | Per-octave frequency multiplier (same as geometry).                      |
| `color_fbm_gain`      | `0.5`     | Per-octave amplitude multiplier (same as geometry).                      |
| `color_noise_seed`    | `42`      | Independent of `noise_seed`; change for a different colour distribution. |
| `color_dither`        | `1.0`     | Fuzziness of palette boundaries, in palette-step units. `0` = razor-sharp regions; `1` = a one-step dithered band where neighbouring colours mix; `>1` softer/wider. Stochastic per bead, deterministic per run. |

The colour noise is sampled at each bead's *original 3D position* (not its projected curtain position), so colour blobs stay spatially coherent across neighbouring curtains.

### Display / export

| Parameter              | Default        | Meaning                                                                |
|------------------------|----------------|------------------------------------------------------------------------|
| `pipeline_mode`        | `"export"`     | `"cave"` / `"curtains"` / `"export"`. See the modes table above.       |
| `display_mode`         | `"pointcloud"` | `"pointcloud"` / `"instances"` / `"sprites"`. See the display table above. |
| `instance_sphere_subd` | `2`            | Mesh density for instance-mode bead spheres.                           |
| `pdf_page_size`        | `"A1"`         | One of `"A4"`, `"A3"`, `"A2"`, `"A1"`, `"A0"`.                         |
| `pdf_output_dir`       | `"exports"`    | Output directory relative to `RhinoPython/` (gitignored).              |

## Project layout

```
pearlscape/
├── README.md                                # this file
├── RhinoPython/
│   ├── build_scene.py                       # F5 entry point — runs the pipeline
│   ├── clear_sprites.py                      # F5 helper — hides sprite beads
│   ├── exports/                             # generated PDFs (gitignored)
│   ├── README.md                            # short "how to run" pointer
│   └── pearlscape/
│       ├── __init__.py
│       ├── params.py                        # all tunable parameters
│       ├── noise.py                         # Perlin + FBM (numpy)
│       ├── sampling.py                      # Bridson Poisson-disk with θ-wrap
│       ├── cave.py                          # CaveSurface protocol + CylinderFBMCave
│       ├── curtains.py                      # slab assignment + projection
│       ├── color.py                         # palette-quantised FBM colour
│       ├── display.py                       # PointCloud and instanced-mesh renderers
│       └── export.py                        # per-curtain layouts + PDF export
└── docs/
    └── superpowers/
        ├── specs/2026-06-02-pearlscape-design.md
        └── plans/2026-06-02-pearlscape.md
```

## Extending — swapping the cave geometry

The `CaveSurface` protocol in `pearlscape/cave.py` is the seam where future geometries plug in. The default `CylinderFBMCave` is one implementation; the protocol's contract is simply:

```python
class CaveSurface(Protocol):
    def sample_surface_points(self) -> np.ndarray:
        """Return an (N, 3) array of points on the cave surface, in world coordinates."""
```

To swap in a Lidar scan or a Catmull-Clark-subdivided cave, write a new class that implements this method, then update `make_default_cave` (or wire it in directly in `build_scene.py`). Nothing downstream — slicing, colour, display, export — needs to change.

## Known gotchas

- **Use the new Script Editor (`_ScriptEditor`), not the legacy one.** Legacy = IronPython 2 = won't run.
- **Re-running in the same Rhino document** accumulates layouts and instance definitions. The script cleans up `Curtain_*` layouts on each run, but for cleanest results start in a fresh document.
- **Changing `bead_diameter` in `"instances"` mode** creates a new block definition (e.g. `PearlscapeBead_d8.0_s2`) without removing the old one. Manually delete the old block via `_BlockManager` if it bothers you.
- **The PDF export uses 300 DPI** by default. For higher resolution, pass a different `dpi` value to `export.export_all_pdfs(out_dir, dpi=600)`.
