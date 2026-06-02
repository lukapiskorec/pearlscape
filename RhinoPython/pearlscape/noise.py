#! python 3
# r: numpy
"""Perlin noise and FBM, numpy-vectorized. 3D inputs / 1D outputs."""

import numpy as np


def make_perm(seed: int) -> np.ndarray:
    """Build a 512-element permutation table from a seed.

    The doubled length lets us index `perm[x+1]` without wrap arithmetic.
    """
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)
    return np.concatenate([p, p])


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _grad3(h: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Ken Perlin's improved-noise gradient selector (12 edge directions of a cube)."""
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


def perlin3(p: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Perlin noise at points p of shape (N, 3). Returns (N,) values in ~[-1, 1]."""
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    fx = np.floor(x)
    fy = np.floor(y)
    fz = np.floor(z)

    X = fx.astype(np.int32) & 255
    Y = fy.astype(np.int32) & 255
    Z = fz.astype(np.int32) & 255

    x = x - fx
    y = y - fy
    z = z - fz

    u = _fade(x)
    v = _fade(y)
    w = _fade(z)

    A  = perm[X]     + Y
    AA = perm[A]     + Z
    AB = perm[A + 1] + Z
    B  = perm[X + 1] + Y
    BA = perm[B]     + Z
    BB = perm[B + 1] + Z

    n000 = _grad3(perm[AA],     x,     y,     z)
    n100 = _grad3(perm[BA],     x - 1, y,     z)
    n010 = _grad3(perm[AB],     x,     y - 1, z)
    n110 = _grad3(perm[BB],     x - 1, y - 1, z)
    n001 = _grad3(perm[AA + 1], x,     y,     z - 1)
    n101 = _grad3(perm[BA + 1], x - 1, y,     z - 1)
    n011 = _grad3(perm[AB + 1], x,     y - 1, z - 1)
    n111 = _grad3(perm[BB + 1], x - 1, y - 1, z - 1)

    x00 = _lerp(n000, n100, u)
    x10 = _lerp(n010, n110, u)
    x01 = _lerp(n001, n101, u)
    x11 = _lerp(n011, n111, u)

    y0 = _lerp(x00, x10, v)
    y1 = _lerp(x01, x11, v)

    return _lerp(y0, y1, w)


def fbm3(
    p: np.ndarray,
    perm: np.ndarray,
    *,
    octaves: int,
    lacunarity: float,
    gain: float,
) -> np.ndarray:
    """Fractal Brownian motion. Returns (N,) values in ~[-1, 1]."""
    total = np.zeros(p.shape[0], dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    norm = 0.0
    for _ in range(octaves):
        total = total + amplitude * perlin3(p * frequency, perm)
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm if norm > 0 else total


def fbm3_01(
    p: np.ndarray,
    perm: np.ndarray,
    *,
    octaves: int,
    lacunarity: float,
    gain: float,
) -> np.ndarray:
    """FBM mapped to ~[0, 1) for convenience."""
    return 0.5 * (fbm3(p, perm, octaves=octaves, lacunarity=lacunarity, gain=gain) + 1.0)


if __name__ == "__main__":
    # Smoke test: sample at non-integer coordinates (Perlin is zero at integer
    # lattice corners, so a 0..2 integer grid would produce all 0.5s).
    perm = make_perm(seed=1)
    xs = np.linspace(0.25, 1.75, 3)
    ys = np.linspace(0.25, 1.75, 3)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(9)])
    vals = fbm3_01(pts, perm, octaves=4, lacunarity=2.0, gain=0.5)
    print("Perlin/FBM smoke test (3x3 grid at z=0):")
    print(vals.reshape(3, 3).round(3))
    print(f"Min/max: {vals.min():.3f} / {vals.max():.3f}  "
          f"(varies: {bool(vals.max() - vals.min() > 0.01)})")
    vals2 = fbm3_01(pts, perm, octaves=4, lacunarity=2.0, gain=0.5)
    print("Deterministic:", bool(np.allclose(vals, vals2)))
