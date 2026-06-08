#! python 3
# r: numpy
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

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
from pearlscape.curtains import array_x_center, build_curtains, curtain_x_positions
from pearlscape import color as color_mod
from pearlscape import display
from pearlscape import export as export_mod
from pearlscape import palettes


def print_run_summary(beads: np.ndarray, params) -> None:
    """Print a config + result summary block for the beads just generated.

    `beads` is the (N, 3) array of actual bead positions (cave cloud in "cave"
    mode, projected per-curtain beads otherwise) — not the NURBS surface.
    """
    n = int(beads.shape[0])
    print("")
    print("--- run summary ---")
    print("")
    print("curtain params (mm):")
    print(f"  curtain_count = {params.curtain_count}")
    print(f"  curtain_spacing = {params.curtain_spacing:g}")
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

    # Summary (final output): measure the in-plane bead positions from cross-section
    # sampling, not the raw surface cloud.
    if total:
        beads = np.vstack([
            display._projected_points(c) for c in curtains if len(c["points_2d"])
        ])
    else:
        beads = np.zeros((0, 3))
    print_run_summary(beads, params)


if __name__ == "__main__":
    main()
