"""Discrete color lookup driven by a second FBM noise field.

Color is evaluated at each bead's original 3D position (before curtain
projection), so the field stays spatially coherent across curtains.
"""

from typing import List, Sequence

import numpy as np

from .noise import fbm3_01, make_perm


def assign_colors(
    points_3d: np.ndarray,
    *,
    palette: Sequence,
    base_freq: float,
    octaves: int,
    lacunarity: float,
    gain: float,
    seed: int,
) -> np.ndarray:
    """Return an (N, 3) uint8 RGB array for the given 3D points."""
    if points_3d.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    perm = make_perm(seed)
    n01 = fbm3_01(
        points_3d * base_freq, perm,
        octaves=octaves, lacunarity=lacunarity, gain=gain,
    )
    # Clamp into [0, 1) defensively; FBM normalization is approximate.
    n01 = np.clip(n01, 0.0, 0.999999)
    m = len(palette)
    idx = np.floor(n01 * m).astype(np.int64)
    palette_arr = np.array(palette, dtype=np.uint8)   # (M, 3)
    return palette_arr[idx]


def apply_to_curtains(curtains: List[dict], params) -> None:
    """Mutate each curtain dict in-place, adding a 'colors' key."""
    for c in curtains:
        c["colors"] = assign_colors(
            c["points_3d"],
            palette=params.palette,
            base_freq=params.color_base_freq,
            octaves=params.color_fbm_octaves,
            lacunarity=params.color_fbm_lacunarity,
            gain=params.color_fbm_gain,
            seed=params.color_noise_seed,
        )
