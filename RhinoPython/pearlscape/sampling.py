#! python 3
# r: numpy
"""Blue-noise (Poisson-disk) sampling in a 2D domain with one wrapped axis.

We sample in (theta_arc, x), where theta_arc = theta * cave_radius is in meters.
The first axis wraps modulo `wrap`; the second is bounded [0, width).
"""

import math
from typing import List, Tuple

import numpy as np


def estimate_radius_for_count(width: float, wrap: float, target_n: int) -> float:
    """Estimate the Poisson-disk minimum distance that yields ~target_n points.

    Uses the hexagonal close-packing density bound: max points ~ area / (r^2 * sqrt(3)/2).
    Bridson tends to undershoot this bound; treat the result as an upper bound on r.
    """
    area = max(width, 1e-9) * max(wrap, 1e-9)
    return math.sqrt(2.0 * area / (target_n * math.sqrt(3.0)))


def bridson_torus(
    width: float,
    wrap: float,
    r: float,
    *,
    k: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Bridson Poisson-disk sampling in 2D, first axis wrapped modulo `wrap`.

    Returns an (N, 2) array with columns (theta_arc, x). N is determined by the
    domain and minimum distance r; it is not a parameter.
    """
    if r <= 0:
        raise ValueError("r must be positive")

    rng = np.random.default_rng(seed)
    cell = r / math.sqrt(2.0)

    ncols = max(1, int(math.ceil(wrap / cell)))
    nrows = max(1, int(math.ceil(width / cell)))
    cell_wrap = wrap / ncols
    cell_width = width / nrows

    grid = np.full((ncols, nrows), -1, dtype=np.int64)
    points: List[Tuple[float, float]] = []
    active: List[int] = []

    def add_point(theta_arc: float, x: float) -> None:
        idx = len(points)
        points.append((theta_arc, x))
        active.append(idx)
        grid[int(theta_arc / cell_wrap), int(x / cell_width)] = idx

    def fits(theta_arc: float, x: float) -> bool:
        if x < 0 or x >= width:
            return False
        gc = int(theta_arc / cell_wrap)
        gr = int(x / cell_width)
        r2 = r * r
        for dc in range(-2, 3):
            cc = (gc + dc) % ncols
            for dr in range(-2, 3):
                cr = gr + dr
                if cr < 0 or cr >= nrows:
                    continue
                idx = grid[cc, cr]
                if idx == -1:
                    continue
                ot, ox = points[idx]
                dt = abs(ot - theta_arc)
                if dt > wrap - dt:
                    dt = wrap - dt
                dx = ox - x
                if dt * dt + dx * dx < r2:
                    return False
        return True

    # Seed point.
    add_point(float(rng.uniform(0, wrap)), float(rng.uniform(0, width)))

    while active:
        ai = int(rng.integers(0, len(active)))
        idx = active[ai]
        ap_theta, ap_x = points[idx]
        placed = False
        # Sample k candidates in the annulus [r, 2r] around the active point.
        angles = rng.uniform(0.0, 2.0 * math.pi, size=k)
        radii = rng.uniform(r, 2.0 * r, size=k)
        cand_theta = (ap_theta + radii * np.cos(angles)) % wrap
        cand_x = ap_x + radii * np.sin(angles)
        for j in range(k):
            t = float(cand_theta[j])
            x = float(cand_x[j])
            if fits(t, x):
                add_point(t, x)
                placed = True
                break
        if not placed:
            # Remove from active list (swap-remove for O(1)).
            active[ai] = active[-1]
            active.pop()

    return np.array(points, dtype=np.float64)


if __name__ == "__main__":
    # Smoke test: sample a 2*pi*1.2 x 5.0 domain for ~60k points.
    radius = 1.2
    length = 5.0
    target = 60_000
    wrap = 2.0 * math.pi * radius
    r = estimate_radius_for_count(length, wrap, target)
    print(f"Estimated r: {r:.4f}m for target {target}")
    import time
    t0 = time.time()
    pts = bridson_torus(width=length, wrap=wrap, r=r, seed=0)
    dt = time.time() - t0
    print(f"Generated {len(pts)} points in {dt:.2f}s")
    print(f"theta_arc range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}] (wrap = {wrap:.3f})")
    print(f"x range:         [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}] (length = {length:.3f})")
