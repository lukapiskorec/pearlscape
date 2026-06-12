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
from pearlscape.curtains import (
    align_curtain_strings, build_curtains, build_shell_curtains, curtain_planes,
    _shell_spacing, _string_overlap,
)
from pearlscape import color as color_mod
from pearlscape import display
from pearlscape import export as export_mod
from pearlscape import ply_export
from pearlscape import palettes
from pearlscape import sections as sections_mod


# Modes that run the section grid on top of the curtain build.
SECTION_MODES = ("sections", "export_section", "export_sections_ply")


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


def _export_ply_bundle(beads, bead_colors, field, dither_rand, params) -> None:
    """Write the bead cloud to params.ply_output_path (binary PLY) plus a
    palettes.json beside it, with the per-bead colour field + dither so the web
    viewer can re-quantize against any palette. Shared by the curtain export
    ("export_ply") and the raw-cave export ("export_cave_ply")."""
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


def print_run_summary(beads: np.ndarray, params, n_curtains: int, bead_spacing: float) -> None:
    """Print a config + result summary, listing ONLY the parameters that
    actually affected this run.

    Sections are gated by pipeline_mode, cave_type, and curtain_mode so the
    output never shows params the selected mode ignores (e.g. band-band-only
    params in shell mode, or curtain params in a raw-cave run).

    `beads` is the (N, 3) array of generated bead positions. `n_curtains` and
    `bead_spacing` are used only for curtain runs: the actual number of planes
    built (derived from the surface extent in "nurbs" mode, not necessarily
    params.curtain_count) and the in-plane spacing the budget solver resolved.
    """
    n = int(beads.shape[0])
    is_cave_run = params.pipeline_mode in ("cave", "export_cave_ply")
    is_curtain_run = params.pipeline_mode in (
        "curtains", "export", "export_ply") + SECTION_MODES
    # Cave runs AND shell-mode curtains both build the cloud from
    # cave.sample_surface_points(), so the surface-sampling params apply to both —
    # EXCEPT for the points cave, whose cloud is imported, not Poisson-sampled.
    samples_surface = (
        (is_cave_run or (is_curtain_run and params.curtain_mode == "shell"))
        and params.cave_type != "points"
    )

    print("")
    print("--- run summary ---")
    print("")
    print(f"pipeline_mode: {params.pipeline_mode}")
    if params.display_mode == "instances":
        print(f"display_mode: instances (sphere subd {params.instance_sphere_subd})")
    else:
        print(f"display_mode: {params.display_mode}")
    print("")

    # --- cave geometry ---
    # An imported point cloud (cave_type "points") defines the geometry itself, so
    # the cylinder/nurbs radius/length/center_z and the wall-displacement noise
    # below don't apply and are omitted.
    is_points_cave = params.cave_type == "points"
    print("cave geometry (mm):")
    print(f"  cave_type = {params.cave_type}")
    if is_points_cave:
        print(f"  point source = Pearlscape::CavePoints (used as-is)")
    else:
        print(f"  cave_radius = {params.cave_radius:g}")
        print(f"  cave_length = {params.cave_length:g}")
        print(f"  cave_center_z = {params.cave_center_z:g}")
    if params.cave_type == "nurbs":
        print(f"  nurbs_surface_source = {params.nurbs_surface_source}")
        print(f"  nurbs_grid = {params.nurbs_grid_u} x {params.nurbs_grid_v}")
        if params.nurbs_surface_source == "rebuild":
            print(f"  nurbs_sections = {params.nurbs_sections}")
            print(f"  nurbs_section_points = {params.nurbs_section_points}")
            print(f"  nurbs_radius_jitter = {params.nurbs_radius_jitter:g} "
                  f"@ {params.nurbs_radius_jitter_freq:g} cyc/mm")
            print(f"  nurbs_centerline_amp = {params.nurbs_centerline_amp:g} "
                  f"@ {params.nurbs_centerline_freq:g} cyc/mm")
            print(f"  nurbs_shape_seed = {params.nurbs_shape_seed}")
    print("")

    # --- wall displacement noise (cylinder/nurbs surfaces only) ---
    if not is_points_cave:
        print("wall noise:")
        print(f"  noise_type = {params.noise_type}")
        print(f"  amplitude = {params.fbm_amplitude:g}")
        print(f"  base_freq = {params.fbm_base_freq:g}")
        print(f"  octaves = {params.fbm_octaves}")
        print(f"  lacunarity = {params.fbm_lacunarity:g}")
        print(f"  gain = {params.fbm_gain:g}")
        print(f"  seed = {params.noise_seed}")
        print("")

    # --- surface sampling (raw cave cloud / shell-mode source) ---
    if samples_surface:
        print("surface sampling:")
        if params.cave_bead_spacing > 0.0:
            print(f"  cave_bead_spacing = {params.cave_bead_spacing:g} (Poisson min spacing)")
        else:
            print(f"  total_surface_samples = {params.total_surface_samples:,} "
                  f"(cave_bead_spacing = 0)")
        print("")

    # --- curtains (mode-specific) ---
    if is_curtain_run:
        print("curtains (mm):")
        print(f"  curtain_mode = {params.curtain_mode}")
        print(f"  curtain_count = {n_curtains} (planes built)")
        print(f"  curtain_width x height = "
              f"{params.curtain_width:g} x {params.curtain_height:g}")
        if params.curtain_mode == "shell":
            auto = "  (auto = 2 x bead_diameter)" if params.shell_curtain_spacing <= 0.0 else ""
            print(f"  shell_curtain_spacing = {_shell_spacing(params):g}{auto}")
            print(f"  overlap drop radius = bead_diameter ({params.bead_diameter:g})")
        else:
            print(f"  curtain_spacing = {params.curtain_spacing:g}")
            print(f"  curtain_band_thickness = {params.curtain_band_thickness:g}")
            print(f"  curtain_band_fade = {params.curtain_band_fade:g}")
            solved = "  (solved from target_bead_count)" if params.target_bead_count > 0 else ""
            print(f"  bead_min_spacing = {bead_spacing:g}{solved}")
            if params.target_bead_count > 0:
                print(f"  target_bead_count = {params.target_bead_count:,}")
        if params.string_align:
            auto = "  (auto = bead_diameter)" if params.string_align_overlap <= 0.0 else ""
            print(f"  string_align overlap = {_string_overlap(params):g}{auto}")
        print("")

    # --- result ---
    print("result:")
    print(f"  beads = {n:,}")
    print(f"  bead_diameter = {params.bead_diameter:g}")
    if n:
        lo = beads.min(axis=0)
        hi = beads.max(axis=0)
        print(f"  bbox (mm): X {float(hi[0]-lo[0]):.0f}  "
              f"Y {float(hi[1]-lo[1]):.0f}  Z {float(hi[2]-lo[2]):.0f}")
    else:
        print("  bbox (mm): n/a (no beads)")
    print("")

    # --- colour ---
    print("colour:")
    print(f"  palette = {palettes.name_of(params.palette)} ({len(params.palette)} colours)")
    print(f"  base_freq = {params.color_base_freq:g}")
    print(f"  octaves = {params.color_fbm_octaves}")
    print(f"  lacunarity = {params.color_fbm_lacunarity:g}")
    print(f"  gain = {params.color_fbm_gain:g}")
    print(f"  seed = {params.color_noise_seed}")
    print(f"  dither = {params.color_dither:g}")

    # --- export targets ---
    if params.pipeline_mode == "export":
        print("")
        print("pdf export:")
        print(f"  pdf_page_size = {params.pdf_page_size}")
        print(f"  pdf_output_dir = {params.pdf_output_dir}")
    elif params.pipeline_mode in ("export_ply", "export_cave_ply"):
        print("")
        print("ply export:")
        print(f"  ply_output_path = {params.ply_output_path}")

    # --- sectioning ---
    if params.pipeline_mode in SECTION_MODES:
        print("")
        print("sections (mm):")
        print(f"  section_size = {params.section_size:g}")
        print(f"  section_grid_shift = {params.section_grid_shift}")
        print(f"  layout spacing / margin = {params.section_layout_spacing:g} "
              f"/ {params.section_layout_margin:g}")
        if params.pipeline_mode == "export_section":
            print(f"  section_export_code = {params.section_export_code}")
            print(f"  pdf_page_size = {params.pdf_page_size}")
            print(f"  pdf_output_dir = {params.pdf_output_dir}/sections/"
                  f"{params.section_export_code}")
        elif params.pipeline_mode == "export_sections_ply":
            print(f"  sections_ply_output_path = {params.sections_ply_output_path}")


def main() -> None:
    params = PearlscapeParams()
    params.validate()

    # Drop any sprite conduit from a previous F5 run so a mode switch (e.g.
    # sprites -> pointcloud) doesn't leave stale beads drawn on screen.
    display.clear_sprite_conduit()

    cave = make_default_cave(params)

    if params.pipeline_mode in ("cave", "export_cave_ply"):
        # Mode 1 — raw cave only: single coloured PointCloud on CaveReference.
        # "export_cave_ply" additionally writes the cloud to a binary PLY.
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

        if params.pipeline_mode == "export_cave_ply":
            # Per-bead colour field + dither source (same seeding as the baked
            # colours), so the viewer can re-quantize against any palette.
            field = color_mod.color_field(
                pts, base_freq=params.color_base_freq,
                octaves=params.color_fbm_octaves,
                lacunarity=params.color_fbm_lacunarity,
                gain=params.color_fbm_gain, seed=params.color_noise_seed,
            )
            dither_rand = color_mod.dither_randoms(pts.shape[0], params.color_noise_seed)
            _export_ply_bundle(pts, colors, field, dither_rand, params)

        # n_curtains / bead_spacing are unused for raw-cave runs (the summary
        # gates the curtain section off pipeline_mode).
        print_run_summary(pts, params, 0, 0.0)
        return

    # Modes "curtains" and "export": sample each curtain plane directly outward
    # from the cave's cross-section boundary. The dense surface cloud is NOT
    # produced in these modes — only the cross-section boundary is needed.
    t0 = time.time()
    plane_xs = curtain_planes(cave, params)
    if params.curtain_mode == "shell":
        curtains, bead_spacing = build_shell_curtains(cave, plane_xs, params)
    else:
        curtains, bead_spacing = build_curtains(cave, plane_xs, params)
    total = sum(len(c["points_2d"]) for c in curtains)
    print(f"Curtains: {len(curtains)} planes, {total} beads in {time.time()-t0:.2f}s")

    if params.string_align:
        # Before colouring/sectioning, so everything downstream (sections,
        # PDFs, PLYs) inherits the aligned positions.
        t0 = time.time()
        st = align_curtain_strings(curtains, params)
        print(f"String align: {st['beads']:,} beads onto {st['strings']:,} strings "
              f"(max shift {st['max_shift']:.1f} mm) in {time.time()-t0:.2f}s")

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

    # Combined bead positions/colours across all curtains, used for the summary,
    # the section grid bbox, and (in export_ply mode) the PLY file.
    if total:
        populated = [c for c in curtains if len(c["points_2d"])]
        beads = np.vstack([display._projected_points(c) for c in populated])
        have_colors = all(c.get("colors") is not None for c in populated)
        bead_colors = np.vstack([c["colors"] for c in populated]) if have_colors else None
    else:
        beads = np.zeros((0, 3))
        bead_colors = None

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

    if params.pipeline_mode in SECTION_MODES:
        if not total:
            raise RuntimeError("Sections: no beads were generated; nothing to section.")
        display.clear_sections_layers()

        bbox_min = beads.min(axis=0)
        bbox_max = beads.max(axis=0)
        grid = sections_mod.grid_from_bbox(
            bbox_min, bbox_max, params.section_size, params.section_grid_shift)
        nx, ny, nz = grid["counts"]
        ox, oy, oz = grid["origin"]
        print(f"Section grid: {nx} x {ny} x {nz} cubes of {params.section_size:g} mm, "
              f"origin ({ox:.0f}, {oy:.0f}, {oz:.0f})")

        secs = sections_mod.assign_sections(curtains, grid)
        print(f"Sections: {len(secs)} occupied cubes")
        for s in secs:
            print(f"  {s['code']}: {s['n_beads']:,} beads "
                  f"across {len(s['curtains'])} planes")

        layout = sections_mod.layout_positions(secs, bbox_min, bbox_max, params)
        display.render_section_grid(grid)
        display.render_section_layout(secs, layout, params.section_size)

        if params.display_mode == "sprites":
            # The sprite conduit rendered above holds only the model's beads;
            # rebuild it with the review-wall copies appended so the sections
            # read as beads too (the wall PointClouds stay — they're the
            # per-section document geometry).
            wall_pts, wall_cols = sections_mod.layout_beads(secs, layout)
            sprite_pts = np.vstack([beads, wall_pts])
            sprite_cols = (np.vstack([bead_colors, wall_cols])
                           if bead_colors is not None else None)
            display.render_sprites(sprite_pts, sprite_cols, params.bead_diameter)

        if params.pipeline_mode == "export_section":
            by_code = {s["code"]: s for s in secs}
            sel = by_code.get(params.section_export_code)
            if sel is None:
                raise ValueError(
                    f"section_export_code {params.section_export_code!r} is not an "
                    f"occupied cube; choose one of: {', '.join(by_code)}")
            plane_layers = display.render_section_export_curtains(
                sel, params.section_size)
            t0 = time.time()
            layout_names = export_mod.create_section_layouts(
                sel["code"], plane_layers, sel["cube_min"], params.section_size,
                page_size=params.pdf_page_size,
            )
            print(f"Created {len(layout_names)} section layouts in {time.time()-t0:.2f}s")
            out_dir = os.path.join(_HERE, params.pdf_output_dir, "sections", sel["code"])
            t0 = time.time()
            pdf_paths = export_mod.export_all_pdfs(out_dir, prefix="Section_")
            print(f"Exported {len(pdf_paths)} PDFs to {out_dir} in {time.time()-t0:.2f}s")

        elif params.pipeline_mode == "export_sections_ply":
            pts, cols = sections_mod.bake_layout_cloud(secs, layout, params)
            repo_root = os.path.dirname(_HERE)
            out_path = os.path.join(repo_root, params.sections_ply_output_path)
            t0 = time.time()
            # No field/dither: the cube edges + labels carry fixed colours, so
            # palette re-quantization in the viewer is deliberately inert here.
            ply_export.write_ply(out_path, pts, cols, params.bead_diameter)
            size_mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
            print(f"Wrote {pts.shape[0]:,} points ({len(secs)} sections + boxes + "
                  f"labels) to {out_path} ({size_mb:.1f} MB) in {time.time()-t0:.2f}s")

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

        _export_ply_bundle(beads, bead_colors, field, dither_rand, params)

    # Summary (final output): measure the in-plane bead positions from cross-section
    # sampling, not the raw surface cloud.
    print_run_summary(beads, params, len(plane_xs), bead_spacing)


if __name__ == "__main__":
    main()
