#! python 3
"""Single source of truth for all tunable parameters.

Units: all lengths in millimeters (Rhino document units). Frequencies are in
cycles per millimeter. A value of 1200 means 1.2 metres.

Not an entry point — F5 build_scene.py to generate. The sys.path shim below only
makes a direct F5 of this file resolve the `pearlscape` package (matching the
sibling modules), so it errors-free instead of ModuleNotFoundError.
"""

import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from dataclasses import dataclass, field
from typing import List, Tuple

from pearlscape import palettes


RGB = Tuple[int, int, int]


def _default_palette() -> List[RGB]:
    return [
        ( 255, 255,   0),   # yellow
        ( 255,   0, 255),   # magenta
        (   0, 255,   0),   # green
        ( 255, 255,   0),   # yellow
    ]


@dataclass
class PearlscapeParams:
    # --- Cave geometry (mm) ---
    cave_radius: float = 1200.0
    cave_length: float = 2000.0

    fbm_amplitude: float = 900.0
    fbm_base_freq: float = 0.00035
    fbm_octaves: int = 6
    fbm_lacunarity: float = 2.0
    fbm_gain: float = 0.55

    # Wall displacement noise: "fbm" (smooth, rolling) or "ridged" (sharp,
    # craggy crevices). Reuses the fbm_* octave/lacunarity/gain params above.
    noise_type: str = "ridged"

    noise_seed: int = 4

    # --- Cave geometry (NURBS, used when cave_type == "nurbs") ---
    # "cylinder" -> CylinderFBMCave (radial noise on a cylinder).
    # "nurbs"    -> NurbsLoftCave (lofted irregular tube, sampled + displaced).
    # "points" -> PointCloudCave: beads sourced directly from the points on the
    #             Pearlscape::CavePoints layer (shell curtain_mode only).
    cave_type: str = "nurbs"
    # "rebuild" -> loft from the nurbs_* params each run (replaces the surface
    #              on the Pearlscape::CaveSurface layer).
    # "reuse"   -> sample the existing (possibly hand-edited) surface on that
    #              layer; falls back to "rebuild" with a warning if none found.
    nurbs_surface_source: str = "reuse"
    nurbs_sections: int = 8            # K: cross-section rings along X
    nurbs_section_points: int = 12     # P: control points per ring
    nurbs_radius_jitter: float = 0.35  # per-point radius variation, fraction of cave_radius
    nurbs_radius_jitter_freq: float = 0.0006   # cycles/mm
    nurbs_centerline_amp: float = 400.0        # meander amplitude (mm, Y/Z only)
    nurbs_centerline_freq: float = 0.0003      # cycles/mm
    nurbs_shape_seed: int = 7          # seed for shape noise (independent of noise_seed)
    nurbs_grid_u: int = 180            # surface-eval grid divisions along X
    nurbs_grid_v: int = 180            # surface-eval grid divisions around theta

    # --- Curtain array (mm) ---
    # In "nurbs" mode curtain_count is IGNORED: the plane count is derived from
    # the surface's actual X extent and curtain_spacing, so curtains fit the cave
    # exactly. It is used only in "cylinder" mode.
    curtain_count: int = 100
    curtain_spacing: float = 50.0
    curtain_width: float = 2500.0
    curtain_height: float = 3000.0
    cave_center_z: float = 1500.0   # cave centerline elevation
    # Curtain generation mode:
    #   "band"  -> build_curtains: a thick band sampled outward from each cross-
    #              section boundary (the original volumetric curtains).
    #   "shell" -> build_shell_curtains: snap the raw cave Poisson surface cloud
    #              onto densely-spaced thin planes (a thin cuttable shell that
    #              mirrors the default cave look). Leaves band logic untouched.
    curtain_mode: str = "shell"
    # Thin-shell plane spacing (mm); used only when curtain_mode == "shell".
    # 0 = auto, resolved to 2 * bead_diameter at use.
    shell_curtain_spacing: float = 10.0
    # --- String alignment (fabrication) ---
    # Physically the beads hang on vertical strings. When on, beads in each
    # curtain whose Y positions overlap (closer than string_align_overlap) are
    # pulled onto a shared string, resolving pairs left to right with a seeded
    # coin deciding which side shifts. String positions stay random per
    # curtain (no snap grid), so curtains never align in the view direction.
    string_align: bool = True
    # Y-overlap that forces a shared string (mm); 0 = auto -> bead_diameter.
    string_align_overlap: float = 0.0

    # --- Beads ---
    total_surface_samples: int = 300_000
    bead_diameter: float = 6.0   # mm
    # Cave-mode bead spacing (mm): true Poisson-disk MINIMUM distance between
    # beads on the raw cave surface (pipeline_mode "cave" / "export_cave_ply").
    # When > 0, beads are Poisson-sampled so none land closer than this (no
    # clustering); the count follows from the spacing + cave area. Larger spacing
    # => looser, fewer beads. 0 disables it and uses total_surface_samples with
    # the faster jittered sampler (which permits clustering).
    cave_bead_spacing: float = 8.0

    # --- Curtain band (in-plane bead placement) ---
    # Beads are sampled directly in each curtain plane, outward from the cave's
    # cross-section curve. Thickness controls how far the soft outer volume
    # reaches (and thus how much it bridges the gaps between curtains); fade is
    # the exponent of the outward density falloff p = (1 - d/T)^fade.
    curtain_band_thickness: float = 120.0   # max outward reach of beads (mm)
    curtain_band_fade: float = 1.5          # outward density falloff exponent
    bead_min_spacing: float = 6.0           # in-plane blue-noise spacing (mm)
    # Bead budget. When > 0, bead_min_spacing is auto-solved each run so the
    # total bead count lands near this target, given the live band_thickness /
    # fade / curtain layout + cave geometry. 0 disables the solver and uses
    # bead_min_spacing as set above.
    target_bead_count: int = 500_000

    # --- Color ---
    palette: List[RGB] = field(default_factory=lambda: list(palettes.FERN_AND_ORCHID_V2))
    color_base_freq: float = 0.0070   # cycles per mm
    color_fbm_octaves: int = 6
    color_fbm_lacunarity: float = 2.0
    color_fbm_gain: float = 0.5
    color_noise_seed: int = 42
    # Softens palette boundaries by stochastic dithering. Width of the per-bead
    # nudge in palette-step units: 0 = sharp, 1 = a one-step fuzzy band, >1 wider.
    color_dither: float = 1.0

    # --- Sectioning (fabrication) ---
    # A lattice of section_size cubes covering the curtain model, used to cut
    # the model into fabricable chunks. Cube faces sit at multiples of
    # section_size from the world origin; section_grid_shift (mm, per axis)
    # slides the whole lattice — nudge it so each cube holds a single bead
    # surface (top OR bottom wall) rather than both.
    section_size: float = 1000.0
    section_grid_shift: Tuple[float, float, float] = (0.0, -300.0, 0.0)
    # Review wall: copies of every occupied cube arranged beside the model
    # (+Y), rows = original Z level. Pitch between copies and the gap between
    # the model bbox and the wall, both in mm.
    section_layout_spacing: float = 1200.0
    section_layout_margin: float = 1000.0
    # Which cube "export_section" writes PDFs for — a code from the section
    # table printed by every sections run (e.g. "X0_Y1_Z2").
    section_export_code: str = "X2_Y1_Z1"
    # Display ONLY this one cube (the section_export_code one) in the viewport,
    # at its true model position — no whole-model beads, review wall or grid.
    # Sprites aren't selectable once drawn, so this is the only way to view a
    # single section alone. Affects display only, not the PDF/PLY output.
    section_isolate: bool = True
    # PLY path for pipeline_mode="export_sections_ply" (relative to repo root).
    sections_ply_output_path: str = "web/data/pearlscape_sections.ply"

    # --- Pipeline / display / export ---
    # pipeline_mode controls how much of the pipeline runs:
    #   "cave"       -> raw cave point cloud only (fastest; tune geometry)
    #   "curtains"   -> cave is sliced into curtains and coloured (no PDF I/O)
    #   "export"     -> curtains + per-curtain layouts + PDFs (full pipeline)
    #   "export_ply" -> curtains + a single binary PLY of all beads (for the
    #                   web viewer); also renders to the viewport. Works with
    #                   any display_mode.
    #   "export_cave_ply" -> raw cave cloud + a single binary PLY of it (no
    #                   curtains); also renders to the viewport.
    #   "sections"   -> curtains + the section cube grid (wireframes) + a
    #                   labelled review wall of per-cube copies beside the
    #                   model. No file I/O; tune the grid shift / seeds here.
    #   "export_section" -> "sections" + per-curtain-plane layouts and PDFs
    #                   for the ONE cube named by section_export_code.
    #   "export_sections_ply" -> "sections" + a single binary PLY of the whole
    #                   review wall (beads + cube edges + labels) for the web
    #                   viewer.
    pipeline_mode: str = "sections"
    display_mode: str = "sprites"   # "pointcloud" | "instances" | "sprites"
    instance_sphere_subd: int = 2
    pdf_page_size: str = "A1"
    pdf_output_dir: str = "exports"
    # PLY output path for pipeline_mode="export_ply" (relative to the repo root,
    # i.e. the parent of RhinoPython/). web/data/ is gitignored.
    ply_output_path: str = "web/data/pearlscape.ply"

    def validate(self) -> None:
        assert self.fbm_amplitude < self.cave_radius, (
            f"fbm_amplitude ({self.fbm_amplitude}) must be < cave_radius "
            f"({self.cave_radius}) to avoid pinching the cave shut."
        )
        assert self.curtain_count >= 2
        assert self.curtain_mode in ("band", "shell")
        assert self.shell_curtain_spacing >= 0.0
        assert self.noise_type in ("fbm", "ridged")
        assert self.pipeline_mode in (
            "cave", "curtains", "export", "export_ply", "export_cave_ply",
            "sections", "export_section", "export_sections_ply",
        )
        assert self.section_size > 0.0
        assert len(self.section_grid_shift) == 3
        assert self.section_layout_spacing >= self.section_size, (
            f"section_layout_spacing ({self.section_layout_spacing}) must be >= "
            f"section_size ({self.section_size}) or the review-wall copies overlap."
        )
        assert self.section_layout_margin >= 0.0
        assert self.section_export_code
        assert self.string_align_overlap >= 0.0
        assert self.display_mode in ("pointcloud", "instances", "sprites")
        assert not (self.display_mode == "sprites" and self.pipeline_mode == "export"), (
            "display_mode='sprites' is screen-only and cannot drive PDF export; "
            "use 'pointcloud' or 'instances' for pipeline_mode='export'."
        )
        assert self.cave_type in ("cylinder", "nurbs", "points")
        assert not (self.cave_type == "points" and self.curtain_mode == "band"), (
            "cave_type='points' supports curtain_mode='shell' only; a raw point "
            "cloud has no analytic cross-section for the band sampler."
        )
        assert self.nurbs_surface_source in ("rebuild", "reuse")
        assert self.nurbs_sections >= 2
        assert self.nurbs_section_points >= 4   # periodic degree-3 ring needs >= 4 pts
        assert 0.0 <= self.nurbs_radius_jitter < 1.0
        assert self.nurbs_grid_u >= 2 and self.nurbs_grid_v >= 3
        assert self.color_dither >= 0.0
        assert self.curtain_band_thickness > 0.0
        assert self.curtain_band_fade >= 0.0
        assert self.bead_diameter > 0.0
        assert self.bead_min_spacing >= self.bead_diameter, (
            f"bead_min_spacing ({self.bead_min_spacing}) must be >= bead_diameter "
            f"({self.bead_diameter}) so placed beads cannot physically overlap."
        )
        assert self.cave_bead_spacing == 0.0 or self.cave_bead_spacing >= self.bead_diameter, (
            f"cave_bead_spacing ({self.cave_bead_spacing}) must be 0 (use the sample "
            f"count) or >= bead_diameter ({self.bead_diameter})."
        )
        assert self.target_bead_count >= 0
        assert len(self.palette) >= 2
