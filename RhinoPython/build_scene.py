#! python 3
# r: numpy
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Rhino's Script Editor caches imported modules across F5 runs. Drop any
# cached pearlscape modules so source edits take effect on every run.
for _m in list(sys.modules):
    if _m == "pearlscape" or _m.startswith("pearlscape."):
        del sys.modules[_m]

import time

import numpy as np

from pearlscape import PearlscapeParams
from pearlscape.cave import make_default_cave
from pearlscape.curtains import build_curtains, curtain_planes
from pearlscape import color as color_mod
from pearlscape import display
from pearlscape import export as export_mod
from pearlscape import ply_export
from pearlscape import palettes


def _write_palettes_json(path: str, params) -> None:
    """Write every named palette from palettes.py (plus the live one, if custom)
    to a JSON the web viewer loads into its palette dropdown. The entry matching
    params.palette is flagged as the default."""
    entries = [
        {"name": key.replace("_", " ").title(),
         "colors": [list(int(v) for v in c) for c in pal]}
        for key, pal in palettes.ALL.items()
    ]
    default_name = palettes.name_of(params.palette)
    if default_name == "custom":
        # The live palette isn't one of the named constants; expose it too.
        entries.insert(0, {
            "name": "Custom (current)",
            "colors": [list(int(v) for v in c) for c in params.palette],
        })
        default_name = "Custom (current)"
    data = {"default": default_name, "palettes": entries}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_run_summary(beads: np.ndarray, params, n_curtains: int, bead_spacing: float) -> None:
    """Print a config + result summary block for the beads just generated.

    `beads` is the (N, 3) array of actual bead positions (cave cloud in "cave"
    mode, projected per-curtain beads otherwise) — not the NURBS surface.
    `n_curtains` is the actual number of curtain planes built (derived from the
    surface extent in "nurbs" mode, not necessarily params.curtain_count).
    `bead_spacing` is the in-plane spacing actually used (the budget solver may
    override params.bead_min_spacing).
    """
    n = int(beads.shape[0])
    print("")
    print("--- run summary ---")
    print("")
    print("curtain params (mm):")
    print(f"  curtain_count = {n_curtains}")
    print(f"  curtain_spacing = {params.curtain_spacing:g}")
    print(f"  curtain_band_thickness = {params.curtain_band_thickness:g}")
    print(f"  curtain_band_fade = {params.curtain_band_fade:g}")
    print(f"  bead_min_spacing = {bead_spacing:g}")
    print("")
    print(f"beads: {n:,}")
    print(f"bead_diameter: {params.bead_diameter:g}")
    print("")
    if n:
        lo = beads.min(axis=0)
        hi = beads.max(axis=0)
        print("bead bbox (mm):")
        print(f"  X {float(hi[0]-lo[0]):.0f}")
        print(f"  Y {float(hi[1]-lo[1]):.0f}")
        print(f"  Z {float(hi[2]-lo[2]):.0f}")
    else:
        print("bead bbox (mm): n/a (no beads)")
    print("")
    print(f"noise type: {params.noise_type}")
    print("")
    print("noise params:")
    print(f"  amplitude = {params.fbm_amplitude}")
    print(f"  base_freq = {params.fbm_base_freq}")
    print(f"  octaves = {params.fbm_octaves}")
    print(f"  lacunarity = {params.fbm_lacunarity}")
    print(f"  gain = {params.fbm_gain}")
    print(f"  seed = {params.noise_seed}")
    print("")
    print(f"palette: {palettes.name_of(params.palette)} ({len(params.palette)} colours)")
    print("")
    print("colour params:")
    print(f"  base_freq = {params.color_base_freq}")
    print(f"  octaves = {params.color_fbm_octaves}")
    print(f"  lacunarity = {params.color_fbm_lacunarity}")
    print(f"  gain = {params.color_fbm_gain}")
    print(f"  seed = {params.color_noise_seed}")
    print(f"  dither = {params.color_dither}")


def main() -> None:
    params = PearlscapeParams()
    params.validate()

    # Drop any sprite conduit from a previous F5 run so a mode switch (e.g.
    # sprites -> pointcloud) doesn't leave stale beads drawn on screen.
    display.clear_sprite_conduit()

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
        print_run_summary(pts, params, params.curtain_count, params.bead_min_spacing)
        return

    # Modes "curtains" and "export": sample each curtain plane directly outward
    # from the cave's cross-section boundary. The dense surface cloud is NOT
    # produced in these modes — only the cross-section boundary is needed.
    t0 = time.time()
    plane_xs = curtain_planes(cave, params)
    curtains, bead_spacing = build_curtains(cave, plane_xs, params)
    total = sum(len(c["points_2d"]) for c in curtains)
    print(f"Curtains: {len(curtains)} planes, {total} beads in {time.time()-t0:.2f}s")

    t0 = time.time()
    color_mod.apply_to_curtains(curtains, params)
    print(f"Colors assigned in {time.time()-t0:.2f}s")

    display.render_curtain_planes(plane_xs, params.curtain_width, params.curtain_height)
    if params.display_mode == "pointcloud":
        display.render_pointclouds(curtains)
    elif params.display_mode == "instances":
        display.render_instances(curtains, params.bead_diameter, params.instance_sphere_subd)
    elif params.display_mode == "sprites":
        display.render_sprites_curtains(curtains, params.bead_diameter)
    else:
        raise ValueError(f"Unknown display_mode: {params.display_mode!r}")
    print(f"Rendered ({params.display_mode}). Look at the viewport.")

    if params.pipeline_mode == "export":
        # Mode 3 — full export: per-curtain layouts + PDFs.
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

    # Combined bead positions/colours across all curtains, used for the summary
    # and (in export_ply mode) the PLY file.
    if total:
        populated = [c for c in curtains if len(c["points_2d"])]
        beads = np.vstack([display._projected_points(c) for c in populated])
        have_colors = all(c.get("colors") is not None for c in populated)
        bead_colors = np.vstack([c["colors"] for c in populated]) if have_colors else None
    else:
        beads = np.zeros((0, 3))
        bead_colors = None

    if params.pipeline_mode == "export_ply":
        # Per-bead colour field + dither source, so the web viewer can re-quantize
        # against any palette without re-running the noise. Computed from the same
        # points_3d and seeding as the baked colours, so they stay consistent.
        if total:
            field = np.concatenate([
                color_mod.color_field(
                    c["points_3d"], base_freq=params.color_base_freq,
                    octaves=params.color_fbm_octaves,
                    lacunarity=params.color_fbm_lacunarity,
                    gain=params.color_fbm_gain, seed=params.color_noise_seed,
                ) for c in populated
            ])
            dither_rand = np.concatenate([
                color_mod.dither_randoms(c["points_3d"].shape[0], params.color_noise_seed)
                for c in populated
            ])
        else:
            field = np.zeros((0,))
            dither_rand = np.zeros((0,))

        # Path is relative to the repo root (parent of RhinoPython/).
        repo_root = os.path.dirname(_HERE)
        out_path = os.path.join(repo_root, params.ply_output_path)
        t0 = time.time()
        ply_export.write_ply(
            out_path, beads, bead_colors, params.bead_diameter,
            field=field, dither_rand=dither_rand, color_dither=params.color_dither,
        )
        size_mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
        print(f"Wrote {beads.shape[0]:,} beads to {out_path} "
              f"({size_mb:.1f} MB) in {time.time()-t0:.2f}s")

        # Ship all named palettes (palettes.py is the source of truth) next to the
        # PLY so the viewer's dropdown stays in sync with the project.
        palettes_path = os.path.join(os.path.dirname(out_path), "palettes.json")
        _write_palettes_json(palettes_path, params)
        print(f"Wrote palette list to {palettes_path}")

    # Summary (final output): measure the in-plane bead positions from cross-section
    # sampling, not the raw surface cloud.
    print_run_summary(beads, params, len(plane_xs), bead_spacing)


if __name__ == "__main__":
    main()
