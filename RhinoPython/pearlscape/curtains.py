#! python 3
# r: numpy
"""Build per-curtain bead point sets by sampling each curtain plane outward from
the cave's craggy cross-section boundary. Replaces the old slice-and-project
approach: beads are placed directly in the plane, so there is no projection
stacking and the inner cave edge stays sharp."""

import os
import sys
from typing import List, Sequence

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape import cross_section


def curtain_x_positions(curtain_count: int, spacing: float, x_center: float) -> np.ndarray:
    """Return the X coordinate of each curtain plane, centered on x_center."""
    offsets = (np.arange(curtain_count) - (curtain_count - 1) / 2.0) * spacing
    return offsets + x_center


def array_x_center(cave_length: float) -> float:
    """The curtain array centers on the cave's midpoint along X."""
    return cave_length / 2.0


def build_curtains(cave, plane_xs: Sequence[float], params) -> List[dict]:
    """For each curtain plane, sample beads in-plane outside the cave's cross-
    section boundary, fading outward. Returns one dict per curtain:
        {'plane_x': float,
         'points_2d': np.ndarray (M, 2)  # columns (Y, Z), the in-plane beads
         'points_3d': np.ndarray (M, 3)} # (plane_x, Y, Z), for colour sampling
    """
    # Angular resolution for the boundary curve: ~one sample per bead-spacing arc
    # along the nominal circumference (finer than bead size is wasted detail).
    n_angular = max(720, int(2.0 * np.pi * params.cave_radius / params.bead_min_spacing))

    curtains: List[dict] = []
    for i, plane_x in enumerate(plane_xs):
        plane_x = float(plane_x)
        centroid, theta, r_inner = cave.inner_boundary(plane_x, n_angular)
        pts_2d = cross_section.sample_band(
            centroid, theta, r_inner,
            band_thickness=params.curtain_band_thickness,
            fade=params.curtain_band_fade,
            spacing=params.bead_min_spacing,
            # Curtain seeds occupy [noise_seed + 1000, noise_seed + 1000 + curtain_count).
            # Keep other seed offsets derived from noise_seed outside this range.
            seed=params.noise_seed + 1000 + i,
        )
        if pts_2d.shape[0]:
            pts_3d = np.column_stack([
                np.full(pts_2d.shape[0], plane_x, dtype=np.float64),
                pts_2d[:, 0],
                pts_2d[:, 1],
            ])
        else:
            pts_3d = np.zeros((0, 3), dtype=np.float64)
        curtains.append({
            "plane_x": plane_x,
            "points_2d": pts_2d,
            "points_3d": pts_3d,
        })
    return curtains


if __name__ == "__main__":
    # Headless smoke test on the pure-numpy cylinder cave.
    from pearlscape.cave import CylinderFBMCave
    from pearlscape.params import PearlscapeParams

    params = PearlscapeParams()   # uses default band params (120 / 1.5 / 6)
    cave = CylinderFBMCave(
        radius=params.cave_radius, length=2000.0, fbm_amplitude=900.0,
        fbm_base_freq=0.00035, fbm_octaves=6, fbm_lacunarity=2.0, fbm_gain=0.55,
        noise_seed=params.noise_seed, target_samples=1000, center_z=1500.0,
        noise_type="ridged",
    )
    xs = curtain_x_positions(5, 100.0, array_x_center(2000.0))
    curtains = build_curtains(cave, xs, params)

    assert len(curtains) == 5, len(curtains)
    for c in curtains:
        assert c["points_2d"].ndim == 2 and c["points_2d"].shape[1] == 2
        assert c["points_3d"].shape == (c["points_2d"].shape[0], 3)
        if c["points_3d"].shape[0]:
            assert np.allclose(c["points_3d"][:, 0], c["plane_x"])
            assert np.allclose(c["points_3d"][:, 1], c["points_2d"][:, 0])
            assert np.allclose(c["points_3d"][:, 2], c["points_2d"][:, 1])
    total = sum(len(c["points_2d"]) for c in curtains)
    assert total > 0, "no beads generated"

    # Determinism: same params -> identical beads.
    curtains2 = build_curtains(cave, xs, params)
    same = all(np.array_equal(a["points_2d"], b["points_2d"])
               for a, b in zip(curtains, curtains2))
    assert same, "build_curtains not deterministic"

    # Per-curtain seed isolation: distinct curtains must not produce identical
    # bead patterns (would mean the +i seed offset isn't taking effect).
    no_clash = not any(
        np.array_equal(curtains[j]["points_2d"], curtains[k]["points_2d"])
        for j in range(len(curtains)) for k in range(j + 1, len(curtains))
        if curtains[j]["points_2d"].shape[0] > 0 and curtains[k]["points_2d"].shape[0] > 0
    )
    assert no_clash, "two curtains produced identical bead patterns (seed not varying)"

    print(f"5 curtains, {total} beads, deterministic, seeds distinct")
    print("OK")
