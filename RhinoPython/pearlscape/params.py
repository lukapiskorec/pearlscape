"""Single source of truth for all tunable parameters."""

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
    # --- Cave geometry ---
    cave_radius: float = 1.2
    cave_length: float = 5.0
    fbm_amplitude: float = 0.30
    fbm_base_freq: float = 0.8
    fbm_octaves: int = 4
    fbm_lacunarity: float = 2.0
    fbm_gain: float = 0.5
    noise_seed: int = 1

    # --- Curtain array ---
    curtain_count: int = 25
    curtain_spacing: float = 0.20
    curtain_width: float = 2.5
    curtain_height: float = 3.0
    cave_center_z: float = 1.5   # cave centerline elevation

    # --- Beads ---
    total_surface_samples: int = 60_000
    bead_diameter: float = 0.006  # 6 mm

    # --- Color ---
    palette: List[RGB] = field(default_factory=_default_palette)
    color_base_freq: float = 0.6
    color_fbm_octaves: int = 2
    color_fbm_lacunarity: float = 2.0
    color_fbm_gain: float = 0.5
    color_noise_seed: int = 42

    # --- Display / export ---
    display_mode: str = "pointcloud"   # "pointcloud" | "instances"
    instance_sphere_subd: int = 2
    pdf_page_size: str = "A1"
    pdf_output_dir: str = "exports"

    def validate(self) -> None:
        assert self.fbm_amplitude < self.cave_radius, (
            f"fbm_amplitude ({self.fbm_amplitude}) must be < cave_radius "
            f"({self.cave_radius}) to avoid pinching the cave shut."
        )
        assert self.curtain_count >= 2
        assert self.display_mode in ("pointcloud", "instances")
        assert len(self.palette) >= 2
