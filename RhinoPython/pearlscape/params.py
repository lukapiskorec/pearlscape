"""Single source of truth for all tunable parameters.

Units: all lengths in millimeters (Rhino document units). Frequencies are in
cycles per millimeter. A value of 1200 means 1.2 metres.
"""

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
    # --- Cave geometry (mm) ---
    cave_radius: float = 1200.0
    cave_length: float = 2000.0

    fbm_amplitude: float = 950.0
    fbm_base_freq: float = 0.00035
    fbm_octaves: int = 6
    fbm_lacunarity: float = 2.0
    fbm_gain: float = 0.55

    # Wall displacement noise: "fbm" (smooth, rolling) or "ridged" (sharp,
    # craggy crevices). Reuses the fbm_* octave/lacunarity/gain params above.
    noise_type: str = "ridged"

    noise_seed: int = 1

    # --- Curtain array (mm) ---
    curtain_count: int = 25
    curtain_spacing: float = 200.0
    curtain_width: float = 2500.0
    curtain_height: float = 3000.0
    cave_center_z: float = 1500.0   # cave centerline elevation

    # --- Beads ---
    total_surface_samples: int = 60_000
    bead_diameter: float = 6.0   # mm

    # --- Color ---
    palette: List[RGB] = field(default_factory=_default_palette)
    color_base_freq: float = 0.0006   # cycles per mm
    color_fbm_octaves: int = 2
    color_fbm_lacunarity: float = 2.0
    color_fbm_gain: float = 0.5
    color_noise_seed: int = 42

    # --- Pipeline / display / export ---
    # pipeline_mode controls how much of the pipeline runs:
    #   "cave"     -> raw cave point cloud only (fastest; tune geometry)
    #   "curtains" -> cave is sliced into curtains and coloured (no PDF I/O)
    #   "export"   -> curtains + per-curtain layouts + PDFs (full pipeline)
    pipeline_mode: str = "cave"
    display_mode: str = "sprites"   # "pointcloud" | "instances" | "sprites"
    instance_sphere_subd: int = 2
    pdf_page_size: str = "A1"
    pdf_output_dir: str = "exports"

    def validate(self) -> None:
        assert self.fbm_amplitude < self.cave_radius, (
            f"fbm_amplitude ({self.fbm_amplitude}) must be < cave_radius "
            f"({self.cave_radius}) to avoid pinching the cave shut."
        )
        assert self.curtain_count >= 2
        assert self.noise_type in ("fbm", "ridged")
        assert self.pipeline_mode in ("cave", "curtains", "export")
        assert self.display_mode in ("pointcloud", "instances", "sprites")
        assert not (self.display_mode == "sprites" and self.pipeline_mode == "export"), (
            "display_mode='sprites' is screen-only and cannot drive PDF export; "
            "use 'pointcloud' or 'instances' for pipeline_mode='export'."
        )
        assert len(self.palette) >= 2
