# Point-cloud export + three.js bead preview

**Date:** 2026-06-08
**Status:** Approved

## Goal

Export the Pearlscape bead model to a standard point-cloud file, and provide a
bare-bones browser app (three.js, no build step) that previews it with the same
glass-bead sprites used in Rhino. Must stay smooth at ~500k beads.

## Background

A bead is a position `(x, y, z)` in mm plus an RGB colour (uint8). The Rhino
pipeline (`build_scene.py`) builds per-curtain bead sets (`curtains.build_curtains`),
assigns colours (`color.apply_to_curtains`), and renders them. In `sprites`
display mode all beads draw in a single batched `DrawSprites` call as procedurally
generated glass-bead bitmaps (tinted body + bottom-left colour glint + white
top-right specular, screen-aligned, world-sized). The viewer mirrors that look and
that single-draw-call performance pattern.

Rhino is Z-up; three.js defaults to Y-up. The viewer handles the up-axis so we
never transform the 500k points.

## Components

### 1. `RhinoPython/pearlscape/ply_export.py` (new, pure-Python)

No Rhino imports — headless-testable, decoupled from the PDF-centric `export.py`.

```python
def write_ply(path, points_xyz, colors_rgb, bead_diameter) -> str
```

- **Binary little-endian PLY**, one `vertex` element: `float x/y/z` + `uchar
  red/green/blue` = 15 bytes/bead (~7.5 MB at 500k).
- Header includes `comment bead_diameter <mm>` so the viewer can size beads in true
  world units (mm), matching Rhino's `sizeInWorldSpace=True`.
- Coordinates written as-is (Rhino Z-up).
- Vectorized with numpy: build one structured dtype array, write header text then
  `array.tofile(f)`. `colors_rgb` may be `None` → write white.

### 2. Pipeline trigger

In `params.py`:
- New `pipeline_mode` value `"export_ply"`.
- New field `ply_output_path: str = "web/data/pearlscape.ply"` (relative to repo
  root; `web/data/` is gitignored).

In `build_scene.py`: when `pipeline_mode == "export_ply"`, run the curtains
pipeline (beads + colours), flatten all curtains into one `(N,3)` positions +
`(N,3)` colours array (reuse the flatten logic from
`display.render_sprites_curtains`), write the PLY, AND render to the viewport so
you see what you exported.

`validate()`: allow `display_mode == "sprites"` together with
`pipeline_mode == "export_ply"` (the existing sprites/export block exists only
because sprites can't drive PDF export; PLY export is display-mode-agnostic).

### 3. Viewer — `web/` directory (no build step)

```
web/
├── index.html      # importmap + canvas + minimal HUD
├── main.js         # scene, loader, render loop
├── beads.glsl.js   # vertex + fragment shader strings
└── data/           # exported .ply lands here (gitignored)
```

- three.js + OrbitControls + PLYLoader via CDN ES-module importmap.
- Served via a static server (`python -m http.server` in `web/`); `file://`
  blocks ES-module + CDN loads.
- Z-up: `camera.up = (0,0,1)`; OrbitControls target at the cloud centroid;
  frame the bounding box on load.

### 4. Rendering — single `THREE.Points` + custom `ShaderMaterial`

The performance core: 500k beads = one draw call, all on GPU.

- `BufferGeometry` takes PLY position + colour buffers directly (no per-bead JS
  objects). PLYLoader produces these from the binary body.
- **Vertex shader** — world-sized points:
  `gl_PointSize = beadDiameter * (canvasHeight / (2·tan(fov/2))) / -viewZ`
  so beads shrink with distance (like Rhino). `beadDiameter` is a uniform
  (seeded from the PLY comment, adjustable via slider). Pass vertex colour through.
- **Fragment shader** — glass bead from `gl_PointCoord`: tinted body with a
  bottom-left colour glint, a sharp white top-right specular, and a ~2px
  antialiased rim via `discard` outside the disc. Screen-aligned. Depth-tested
  and effectively opaque (alpha-discard rim) to avoid transparency sort cost at
  500k points.

### 5. HUD (minimal)

Overlay with: **Load .ply** file button (also accepts drag-and-drop), a **bead
size** slider, and a **bead count + live FPS** readout. To read the
`bead_diameter` comment (PLYLoader doesn't expose comments), scan the short ASCII
header out of the ArrayBuffer before handing the buffer to `PLYLoader.parse`.

## Data flow

```
Rhino F5 (pipeline_mode="export_ply")
  → build_curtains → colours → flatten → ply_export.write_ply → web/data/pearlscape.ply
Browser: load .ply → header scan (bead_diameter) → PLYLoader.parse
  → THREE.Points(+shader) → orbit
```

## Error handling

- `write_ply`: validate `points_xyz` shape `(N,3)`; `colors_rgb` is `(N,3)` or
  `None`; create parent dir if missing.
- Viewer: if a loaded file isn't valid PLY, show an error in the HUD rather than a
  blank canvas. Missing `bead_diameter` comment → fall back to slider default.

## Testing

- Headless unit test for `ply_export`: round-trip a small array; assert the PLY
  header is well-formed and the binary body parses back to identical
  points/colours. No Rhino needed.
- Viewer verified manually in-browser against an exported 500k file: confirm FPS
  is interactive and the look matches Rhino.

## Out of scope (YAGNI)

No bundler, framework, or server-side component. No streaming / LOD / chunking
(500k in one buffer is well within a single draw call). No animation. No
material/lighting controls beyond bead size.
