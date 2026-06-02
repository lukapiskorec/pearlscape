"""Slice the cave point cloud into curtain slabs and project onto curtain planes."""

from typing import List

import numpy as np


def curtain_x_positions(curtain_count: int, spacing: float, x_center: float) -> np.ndarray:
    """Return the X coordinate of each curtain plane, centered on x_center."""
    offsets = (np.arange(curtain_count) - (curtain_count - 1) / 2.0) * spacing
    return offsets + x_center


def slice_and_project(
    cave_points: np.ndarray,
    curtain_count: int,
    curtain_spacing: float,
    x_center: float,
) -> List[dict]:
    """Assign each cave point to a curtain slab and project onto its plane.

    Returns one dict per curtain:
        {
            'plane_x':   float,
            'points_2d': np.ndarray (M_i, 2)  # columns (Y, Z)
            'points_3d': np.ndarray (M_i, 3)  # original positions, for color sampling
        }
    """
    xs = curtain_x_positions(curtain_count, curtain_spacing, x_center)
    x_min = xs[0] - curtain_spacing / 2.0
    x_max = xs[-1] + curtain_spacing / 2.0

    # Drop points outside the array's X coverage.
    cave_x = cave_points[:, 0]
    in_range = (cave_x >= x_min) & (cave_x < x_max)
    pts = cave_points[in_range]

    # Assign each surviving point to the curtain whose slab it falls in.
    slab_idx = np.floor((pts[:, 0] - x_min) / curtain_spacing).astype(np.int64)
    slab_idx = np.clip(slab_idx, 0, curtain_count - 1)

    curtains: List[dict] = []
    for i in range(curtain_count):
        mask = slab_idx == i
        assigned = pts[mask]
        curtains.append({
            "plane_x": float(xs[i]),
            "points_2d": assigned[:, 1:3].copy(),     # (Y, Z)
            "points_3d": assigned.copy(),
        })
    return curtains


def array_x_center(cave_length: float) -> float:
    """The curtain array centers on the cave's midpoint along X."""
    return cave_length / 2.0
