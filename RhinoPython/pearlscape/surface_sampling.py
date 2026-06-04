#! python 3
# r: numpy
"""Sample a NURBS surface (given as an evaluated (nu, nv, 3) grid) into a dense
displaced point cloud — pure numpy, no Rhino. Kept separate from nurbs_cave.py
so this pipeline can be tested headlessly.

Grid convention: axis 0 is u (along the cave's X, open); axis 1 is v (around the
cross-section, periodic).
"""

import os
import sys

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape.noise import fbm3_01, ridged3_01, make_perm
from pearlscape.sampling import jittered_grid


def grid_normals(grid_xyz: np.ndarray) -> np.ndarray:
    """Per-node unit normals from cross-products of grid tangents.

    u (axis 0) is open (one-sided differences at the ends); v (axis 1) is
    periodic (np.roll). Orientation is arbitrary here; sample_and_displace
    re-orients per sample relative to the centerline.
    """
    g = grid_xyz
    du = np.empty_like(g)
    du[1:-1] = g[2:] - g[:-2]
    du[0] = g[1] - g[0]
    du[-1] = g[-1] - g[-2]
    dv = np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return n / norm


def _bilinear(grid: np.ndarray, u_frac: np.ndarray, v_frac: np.ndarray) -> np.ndarray:
    """Bilinearly sample `grid` (nu, nv, C) at fractional indices.

    u clamps to [0, nu-1] (open); v wraps modulo nv (periodic).
    Returns (N, C).
    """
    nu, nv = grid.shape[0], grid.shape[1]
    uf = np.clip(u_frac, 0.0, nu - 1.0)
    u0 = np.floor(uf).astype(np.int64)
    u0 = np.clip(u0, 0, nu - 2)
    u1 = u0 + 1
    tu = (uf - u0)[:, None]

    v0 = np.floor(v_frac).astype(np.int64) % nv
    v1 = (v0 + 1) % nv
    tv = (v_frac - np.floor(v_frac))[:, None]

    g00 = grid[u0, v0]
    g01 = grid[u0, v1]
    g10 = grid[u1, v0]
    g11 = grid[u1, v1]
    return (
        (1 - tu) * (1 - tv) * g00
        + (1 - tu) * tv * g01
        + tu * (1 - tv) * g10
        + tu * tv * g11
    )


def sample_and_displace(
    grid_xyz: np.ndarray,
    base_normals: np.ndarray,
    *,
    target: int,
    fbm_amplitude: float,
    fbm_base_freq: float,
    octaves: int,
    lacunarity: float,
    gain: float,
    noise_type: str,
    noise_seed: int,
    sample_seed: int = 0,
) -> np.ndarray:
    """Blue-noise-sample the surface grid and displace inward by ridged/FBM noise.

    Returns an (N, 3) array of displaced surface points in world coordinates.
    """
    nu, nv = grid_xyz.shape[0], grid_xyz.shape[1]

    # Approximate arc-lengths so jittered spacing is ~uniform in mm.
    centerline = grid_xyz.mean(axis=1)                       # (nu, 3)
    seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    width_len = max(float(seg.sum()), 1e-9)                  # along u (~cave length)
    # Per-u ring perimeter; np.roll closes the periodic seam. Assumes the grid has
    # no duplicated seam column (Rhino surface eval does not produce one).
    ring_edges = np.linalg.norm(
        np.roll(grid_xyz, -1, axis=1) - grid_xyz, axis=2
    ).sum(axis=1)
    wrap_len = max(float(ring_edges.mean()), 1e-9)           # around v (~circumference)

    samp = jittered_grid(width=width_len, wrap=wrap_len, target_n=target, seed=sample_seed)
    if samp.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    # jittered_grid columns are (wrap_axis, width_axis): col 0 runs around the
    # ring (-> v, periodic), col 1 runs along the length (-> u, open).
    wrap_coord = samp[:, 0]
    width_coord = samp[:, 1]

    u_frac = width_coord / width_len * (nu - 1)
    v_frac = wrap_coord / wrap_len * nv                      # periodic; _bilinear mods nv

    base_pts = _bilinear(grid_xyz, u_frac, v_frac)
    nrm = _bilinear(base_normals, u_frac, v_frac)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

    # Orient each sampled normal outward (away from the cave axis). The centerline
    # is 1-D in u, so interpolate it directly along u rather than routing it
    # through the 2-D _bilinear helper (matches _bilinear's u handling exactly).
    cu = np.clip(u_frac, 0.0, nu - 1.0)
    cu0 = np.clip(np.floor(cu).astype(np.int64), 0, nu - 2)
    ctu = (cu - cu0)[:, None]
    c_at_u = (1.0 - ctu) * centerline[cu0] + ctu * centerline[cu0 + 1]
    outward = base_pts - c_at_u
    # Flip any normal that points inward. A zero dot product is degenerate
    # (tangent normal / flat spot) -> default to outward.
    sign = np.sign(np.sum(nrm * outward, axis=1, keepdims=True))
    sign = np.where(sign == 0, 1.0, sign)
    nrm = nrm * sign

    perm = make_perm(noise_seed)
    noise_fn = ridged3_01 if noise_type == "ridged" else fbm3_01
    n01 = noise_fn(
        base_pts * fbm_base_freq, perm,
        octaves=octaves, lacunarity=lacunarity, gain=gain,
    )

    return base_pts - fbm_amplitude * n01[:, None] * nrm


if __name__ == "__main__":
    import time

    # Synthetic analytic cylinder grid (radius R along X), to exercise the
    # numpy pipeline without Rhino.
    nu, nv = 120, 120
    R, length = 1000.0, 5000.0
    u = np.linspace(0.0, length, nu)
    v = np.linspace(0.0, 2 * np.pi, nv, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    grid = np.stack([uu, R * np.cos(vv), R * np.sin(vv)], axis=2)

    normals = grid_normals(grid)
    assert np.allclose(np.linalg.norm(normals, axis=2), 1.0, atol=1e-6), "normals not unit"

    target = 200_000
    t0 = time.time()
    pts = sample_and_displace(
        grid, normals,
        target=target, fbm_amplitude=300.0, fbm_base_freq=0.0008,
        octaves=4, lacunarity=2.0, gain=0.5, noise_type="ridged", noise_seed=1,
    )
    dt = time.time() - t0
    print(f"Sampled {len(pts)} pts (target {target}) in {dt:.2f}s")
    assert abs(len(pts) - target) < 0.2 * target, "count off"
    assert not np.isnan(pts).any(), "NaNs in output"

    # Every displaced point must lie within [R - amplitude, R] of the axis
    # (inward displacement only), with a small tolerance for grid discretization.
    radial = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
    assert radial.max() <= R + 1.0, f"point outside base surface: {radial.max()}"
    assert radial.min() >= R - 300.0 - 1.0, f"over-displaced: {radial.min()}"

    pts2 = sample_and_displace(
        grid, normals,
        target=target, fbm_amplitude=300.0, fbm_base_freq=0.0008,
        octaves=4, lacunarity=2.0, gain=0.5, noise_type="ridged", noise_seed=1,
    )
    print("Deterministic:", bool(np.array_equal(pts, pts2)))
    print("X spans cave:", float(pts[:, 0].min()), "->", float(pts[:, 0].max()))
    assert sample_and_displace(
        grid, normals, target=0, fbm_amplitude=300.0, fbm_base_freq=0.0008,
        octaves=4, lacunarity=2.0, gain=0.5, noise_type="ridged", noise_seed=1,
    ).shape == (0, 3), "empty target should yield (0, 3)"
    print("OK")
