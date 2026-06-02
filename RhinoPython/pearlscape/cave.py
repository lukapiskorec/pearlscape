#! python 3
# r: numpy
"""Cave surface generation.

`CaveSurface` is the seam where future geometries (Lidar, subdivision)
plug in without changing downstream code. The default implementation
displaces a cylinder's surface inward by FBM in cylindrical parameter
space — fast, deterministic, no self-intersection while
fbm_amplitude < cave_radius.
"""

import math
import os
import sys
from typing import Protocol

import numpy as np

# Make sibling-package imports resolve when this file is F5-run directly
# (Rhino loads it as __main__, with no parent package context). When imported
# normally as `pearlscape.cave`, this sys.path adjustment is a no-op.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape.noise import fbm3_01, make_perm
from pearlscape.sampling import bridson_torus, estimate_radius_for_count


class CaveSurface(Protocol):
    """Anything that can produce surface points of a cave shape."""

    def sample_surface_points(self) -> np.ndarray:
        """Return an (N, 3) array of points on the cave surface, in world coordinates."""
        ...


class CylinderFBMCave:
    """A cylinder along X, with surface points blue-noise-sampled and
    radially displaced inward by FBM noise."""

    def __init__(
        self,
        *,
        radius: float,
        length: float,
        fbm_amplitude: float,
        fbm_base_freq: float,
        fbm_octaves: int,
        fbm_lacunarity: float,
        fbm_gain: float,
        noise_seed: int,
        target_samples: int,
        center_z: float,
        sampling_seed: int = 0,
        radius_shrink: float = 0.93,
    ) -> None:
        if fbm_amplitude >= radius:
            raise ValueError(
                f"fbm_amplitude ({fbm_amplitude}) must be < radius ({radius})"
            )
        self.radius = radius
        self.length = length
        self.fbm_amplitude = fbm_amplitude
        self.fbm_base_freq = fbm_base_freq
        self.fbm_octaves = fbm_octaves
        self.fbm_lacunarity = fbm_lacunarity
        self.fbm_gain = fbm_gain
        self.noise_seed = noise_seed
        self.target_samples = target_samples
        self.center_z = center_z
        self.sampling_seed = sampling_seed
        # Bridson underfills the theoretical bound. Shrink the estimated r
        # to land closer to the target. Empirical 0.93 typically lands ~60k
        # when target is 60k.
        self.radius_shrink = radius_shrink

    def sample_surface_points(self) -> np.ndarray:
        wrap = 2.0 * math.pi * self.radius
        r = estimate_radius_for_count(self.length, wrap, self.target_samples)
        r *= self.radius_shrink

        param_pts = bridson_torus(
            width=self.length,
            wrap=wrap,
            r=r,
            seed=self.sampling_seed,
        )

        theta = param_pts[:, 0] / self.radius   # back to radians
        x = param_pts[:, 1]

        # Evaluate FBM on the base-cylinder position so the noise input
        # is continuous around the theta seam.
        base = np.column_stack([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            x,
        ]) * self.fbm_base_freq

        perm = make_perm(self.noise_seed)
        n01 = fbm3_01(
            base, perm,
            octaves=self.fbm_octaves,
            lacunarity=self.fbm_lacunarity,
            gain=self.fbm_gain,
        )

        r_eff = self.radius - self.fbm_amplitude * n01

        # Place in world: X = x, Y = r_eff*cos, Z = center_z + r_eff*sin.
        pts = np.column_stack([
            x,
            r_eff * np.cos(theta),
            self.center_z + r_eff * np.sin(theta),
        ])
        return pts


def make_default_cave(params) -> CylinderFBMCave:
    """Build a CylinderFBMCave from a PearlscapeParams instance."""
    return CylinderFBMCave(
        radius=params.cave_radius,
        length=params.cave_length,
        fbm_amplitude=params.fbm_amplitude,
        fbm_base_freq=params.fbm_base_freq,
        fbm_octaves=params.fbm_octaves,
        fbm_lacunarity=params.fbm_lacunarity,
        fbm_gain=params.fbm_gain,
        noise_seed=params.noise_seed,
        target_samples=params.total_surface_samples,
        center_z=params.cave_center_z,
    )


if __name__ == "__main__":
    from pearlscape.params import PearlscapeParams
    p = PearlscapeParams()
    cave = make_default_cave(p)
    import time
    t0 = time.time()
    pts = cave.sample_surface_points()
    print(f"Generated {len(pts)} surface points in {time.time()-t0:.2f}s")
    print(f"X range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"Y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"Z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
