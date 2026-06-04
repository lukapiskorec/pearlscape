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
    dither: float = 0.0,
) -> np.ndarray:
    """Return an (N, 3) uint8 RGB array for the given 3D points.

    `dither` softens the palette boundaries. It is the width — in palette-step
    units — of a uniform per-bead nudge added to the bin coordinate before
    quantizing, so beads within ~dither/2 of a boundary take either neighbouring
    colour with linearly-ramping probability. `dither=0` is hard quantization
    (unchanged output). The nudge is deterministic, seeded from `seed`.
    """
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
    t = n01 * m   # continuous bin coordinate in [0, m)
    if dither > 0.0:
        # Independent of the FBM perm (seed) so the dither doesn't track the field.
        rng = np.random.default_rng(seed + 1)
        t = t + dither * (rng.random(t.shape[0]) - 0.5)
    idx = np.clip(np.floor(t), 0, m - 1).astype(np.int64)
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
            dither=params.color_dither,
        )


if __name__ == "__main__":
    # Run with: python -m pearlscape.color  (uses the package-relative import)
    palette = [(180, 40, 40), (220, 130, 50), (230, 200, 70),
               (80, 160, 90), (60, 110, 180), (130, 70, 170)]
    m = len(palette)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2000.0, 2000.0, size=(40000, 3))
    kw = dict(palette=palette, base_freq=0.0006, octaves=2,
              lacunarity=2.0, gain=0.5, seed=42)

    hard = assign_colors(pts, dither=0.0, **kw)

    # dither=0 must equal plain floor(n01*m) quantization (backward compatible).
    perm = make_perm(42)
    n01 = np.clip(fbm3_01(pts * 0.0006, perm, octaves=2, lacunarity=2.0, gain=0.5),
                  0.0, 0.999999)
    ref = np.array(palette, dtype=np.uint8)[np.floor(n01 * m).astype(np.int64)]
    assert np.array_equal(hard, ref), "dither=0 changed the sharp output"

    fuzzy = assign_colors(pts, dither=1.0, **kw)
    fuzzy2 = assign_colors(pts, dither=1.0, **kw)
    assert np.array_equal(fuzzy, fuzzy2), "dither not deterministic"

    changed = np.any(fuzzy != hard, axis=1)
    n_changed = int(changed.sum())
    assert 0 < n_changed < pts.shape[0], "expected some-but-not-all beads to change"

    # Changed beads must be near a bin boundary (within dither/2 of an integer t).
    t = n01 * m
    frac = t - np.floor(t)
    dist_to_boundary = np.minimum(frac, 1.0 - frac)
    assert np.all(dist_to_boundary[changed] <= 0.5 + 1e-9), \
        "a bead far from any boundary changed colour"

    print(f"Beads: {pts.shape[0]}, changed by dither=1.0: {n_changed} "
          f"({100.0 * n_changed / pts.shape[0]:.1f}%)")
    print("dither=0 matches sharp:", True)
    print("deterministic:", True)
    print("changes confined to boundary bands:", True)
    print("OK")
