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
