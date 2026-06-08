#! python 3
# r: numpy
"""Place curtain beads directly in the curtain plane, bounded by the cave's
craggy cross-section curve and fading outward. Pure numpy, no Rhino — testable
headlessly (see __main__).

Public functions:
  displaced_ring_boundary(base_xyz, normals, n01, *, fbm_amplitude)
      -> (centroid_yz (2,), theta (n,), r_inner (n,))
      Apply the cave's per-point inward noise displacement to a ring of base-
      surface points and return the resulting craggy inner boundary in polar
      form about the ring's Y/Z centroid. The caller supplies n01 (the noise
      value per point) using its own noise convention, so this stays pure
      geometry and matches whichever cave produced the ring.

  sample_band(centroid_yz, theta, r_inner, *, band_thickness, fade, spacing, seed)
      -> (M, 2) bead (Y, Z) positions
      Stratified blue-noise sampling of the band outside the boundary, with
      density full at the boundary and fading to zero at band_thickness.
"""

import os
import sys
from typing import Tuple

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape.sampling import jittered_grid


def displaced_ring_boundary(
    base_xyz: np.ndarray,
    normals: np.ndarray,
    n01: np.ndarray,
    *,
    fbm_amplitude: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Displace a base ring inward by fbm_amplitude * n01 along -normals and
    return (centroid_yz, theta, r_inner). theta is each displaced point's angle
    about the ring's Y/Z centroid, sorted ascending; r_inner its distance from
    that centroid. base_xyz, normals: (n, 3) world points / unit outward normals."""
    displaced = base_xyz - fbm_amplitude * n01[:, None] * normals
    yz = displaced[:, 1:3]
    centroid = yz.mean(axis=0)
    rel = yz - centroid
    theta = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)   # [0, 2pi)
    r_inner = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
    order = np.argsort(theta)
    return centroid, theta[order], r_inner[order]


def sample_band(
    centroid_yz: np.ndarray,
    theta: np.ndarray,
    r_inner: np.ndarray,
    *,
    band_thickness: float,
    fade: float,
    spacing: float,
    seed: int,
) -> np.ndarray:
    """Blue-noise-sample the band outside the inner boundary, fading outward.
    Returns (M, 2) bead positions in the curtain's (Y, Z) plane."""
    n = theta.shape[0]
    if n == 0 or band_thickness <= 0.0:
        return np.empty((0, 2), dtype=np.float64)

    cy, cz = float(centroid_yz[0]), float(centroid_yz[1])
    by = cy + r_inner * np.cos(theta)
    bz = cz + r_inner * np.sin(theta)
    # Closed-loop chord lengths -> cumulative arc-length knots (n + 1 entries).
    dby = np.diff(np.append(by, by[0]))
    dbz = np.diff(np.append(bz, bz[0]))
    seg = np.sqrt(dby ** 2 + dbz ** 2)
    arc_knots = np.concatenate([[0.0], np.cumsum(seg)])
    perimeter = float(arc_knots[-1])
    if perimeter <= 0.0:
        return np.empty((0, 2), dtype=np.float64)

    # Periodic knot arrays aligned with arc_knots: close theta to 2pi, r to r[0].
    theta_knots = np.concatenate([theta, [2.0 * np.pi]])
    r_knots = np.concatenate([r_inner, [r_inner[0]]])

    # Stratified blue-noise in (arc, radial offset): columns (wrap=arc, width=offset).
    target_n = max(1, int(round(perimeter * band_thickness / (spacing * spacing))))
    samp = jittered_grid(width=band_thickness, wrap=perimeter, target_n=target_n, seed=seed)
    if samp.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    arc = samp[:, 0]
    offset = samp[:, 1]

    th = np.interp(arc, arc_knots, theta_knots)
    ri = np.interp(arc, arc_knots, r_knots)

    # Outward fade: keep with p = (1 - offset/T)^fade. Use a derived child seed
    # so the keep/drop draws are independent of jittered_grid's own RNG stream
    # (which is also seeded from `seed`) — otherwise fade correlates with each
    # bead's arc-cell position and produces arc-direction striping.
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    p_keep = np.clip(1.0 - offset / band_thickness, 0.0, 1.0) ** fade
    keep = rng.random(arc.shape[0]) < p_keep
    th, ri, offset = th[keep], ri[keep], offset[keep]

    r = ri + offset
    y = cy + r * np.cos(th)
    z = cz + r * np.sin(th)
    return np.column_stack([y, z])


if __name__ == "__main__":
    # --- displaced_ring_boundary: zero noise -> exact circle ---
    n = 720
    th0 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R, cz = 1000.0, 1500.0
    base = np.column_stack([np.zeros(n), R * np.cos(th0), cz + R * np.sin(th0)])
    nrm = np.column_stack([np.zeros(n), np.cos(th0), np.sin(th0)])
    centroid, theta, r_inner = displaced_ring_boundary(
        base, nrm, np.zeros(n), fbm_amplitude=300.0)
    assert np.allclose(centroid, [0.0, cz], atol=1e-6), centroid
    assert np.allclose(r_inner, R, atol=1e-6), (r_inner.min(), r_inner.max())
    assert np.all(np.diff(theta) >= 0), "theta not sorted"

    # constant inward noise shrinks the radius by amplitude * n01
    _, _, r_in2 = displaced_ring_boundary(base, nrm, np.full(n, 0.5), fbm_amplitude=300.0)
    assert np.allclose(r_in2, R - 0.5 * 300.0, atol=1e-6), r_in2.mean()

    # --- sample_band: beads outside boundary, density fades outward ---
    T, spacing = 120.0, 6.0
    pts = sample_band(centroid, theta, r_inner, band_thickness=T, fade=1.5,
                      spacing=spacing, seed=7)
    assert pts.shape[0] > 0, "no beads"
    r = np.sqrt((pts[:, 0] - centroid[0]) ** 2 + (pts[:, 1] - centroid[1]) ** 2)
    assert r.min() >= R - 1e-6, f"bead inside inner edge: {r.min()} < {R}"
    assert r.max() <= R + T + 1.0, f"bead beyond band: {r.max()}"
    inner_half = int(((r - R) < T / 2).sum())
    outer_half = int(((r - R) >= T / 2).sum())
    assert inner_half > outer_half, f"fade not outward: {inner_half} vs {outer_half}"

    # determinism
    pts2 = sample_band(centroid, theta, r_inner, band_thickness=T, fade=1.5,
                       spacing=spacing, seed=7)
    assert np.array_equal(pts, pts2), "sample_band not deterministic"

    # no piles: no spacing-cell holds many beads
    keys = np.floor(pts / spacing).astype(np.int64)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    assert counts.max() <= 4, f"pile detected: max {counts.max()} per cell"

    print(f"sample_band: {pts.shape[0]} beads, "
          f"r in [{r.min():.1f}, {r.max():.1f}], max/cell {counts.max()}")

    # --- craggy inner edge via CylinderFBMCave.inner_boundary ---
    from pearlscape.cave import CylinderFBMCave
    cave = CylinderFBMCave(
        radius=1200.0, length=2000.0, fbm_amplitude=900.0, fbm_base_freq=0.00035,
        fbm_octaves=6, fbm_lacunarity=2.0, fbm_gain=0.55, noise_seed=3,
        target_samples=1000, center_z=1500.0, noise_type="ridged",
    )
    c2, th2, ri2 = cave.inner_boundary(1000.0, 1440)
    assert ri2.shape == (1440,), ri2.shape
    assert ri2.max() <= 1200.0 + 1e-6, ri2.max()
    assert ri2.min() >= 1200.0 - 900.0 - 1e-6, ri2.min()
    band = sample_band(c2, th2, ri2, band_thickness=120.0, fade=1.5, spacing=6.0, seed=11)
    rel = band - c2
    bth = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
    br = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
    ri_at = np.interp(bth, np.concatenate([th2, [2.0 * np.pi]]),
                      np.concatenate([ri2, [ri2[0]]]))
    assert np.all(br >= ri_at - 1e-6), f"bead inside craggy edge: {(br - ri_at).min()}"
    print(f"cylinder inner_boundary: r_inner in [{ri2.min():.1f}, {ri2.max():.1f}], "
          f"{band.shape[0]} band beads")
    print("OK")
