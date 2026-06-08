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

from pearlscape.noise import fbm3_01, ridged3_01, make_perm
from pearlscape.sampling import bridson_torus, estimate_radius_for_count


class CaveSurface(Protocol):
    """Anything that can produce surface points of a cave shape."""

    def sample_surface_points(self) -> np.ndarray:
        """Return an (N, 3) array of points on the cave surface, in world coordinates."""
        ...

    def inner_boundary(self, plane_x: float, n_angular: int):
        """Return (centroid_yz (2,), theta (n_angular,), r_inner (n_angular,)) for
        the cave's craggy cross-section at X = plane_x, in the curtain's Y/Z plane.
        theta is sorted ascending in [0, 2*pi); r_inner is measured from centroid_yz."""
        ...


class CylinderFBMCave:
    """A cylinder along X, with surface points blue-noise-sampled and
    radially displaced inward by noise. The noise is either smooth FBM or
    sharp ridged multifractal, selected by `noise_type`."""

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
        noise_type: str = "fbm",
        sampling_seed: int = 0,
        radius_shrink: float = 0.93,
    ) -> None:
        if fbm_amplitude >= radius:
            raise ValueError(
                f"fbm_amplitude ({fbm_amplitude}) must be < radius ({radius})"
            )
        if noise_type not in ("fbm", "ridged"):
            raise ValueError(f"noise_type must be 'fbm' or 'ridged', got {noise_type!r}")
        self.radius = radius
        self.length = length
        self.fbm_amplitude = fbm_amplitude
        self.fbm_base_freq = fbm_base_freq
        self.fbm_octaves = fbm_octaves
        self.fbm_lacunarity = fbm_lacunarity
        self.fbm_gain = fbm_gain
        self.noise_seed = noise_seed
        self.noise_type = noise_type
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
        noise_fn = ridged3_01 if self.noise_type == "ridged" else fbm3_01
        n01 = noise_fn(
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

    def inner_boundary(self, plane_x: float, n_angular: int):
        """Craggy cross-section boundary at X = plane_x. Reuses the exact noise
        convention of sample_surface_points so the boundary matches the wall."""
        from pearlscape.cross_section import displaced_ring_boundary

        th = np.linspace(0.0, 2.0 * math.pi, n_angular, endpoint=False)
        cos, sin = np.cos(th), np.sin(th)
        xcol = np.full(n_angular, float(plane_x))
        base_xyz = np.column_stack([
            xcol,
            self.radius * cos,
            self.center_z + self.radius * sin,
        ])
        # Outward normals: radial in Y/Z, zero along X.
        normals = np.column_stack([np.zeros(n_angular), cos, sin])
        # Noise sampled exactly as sample_surface_points: (R cos, R sin, x) * freq.
        base_noise = np.column_stack([
            self.radius * cos, self.radius * sin, xcol,
        ]) * self.fbm_base_freq
        perm = make_perm(self.noise_seed)
        noise_fn = ridged3_01 if self.noise_type == "ridged" else fbm3_01
        n01 = noise_fn(
            base_noise, perm,
            octaves=self.fbm_octaves, lacunarity=self.fbm_lacunarity, gain=self.fbm_gain,
        )
        return displaced_ring_boundary(
            base_xyz, normals, n01, fbm_amplitude=self.fbm_amplitude,
        )


def make_default_cave(params):
    """Build the cave selected by params.cave_type.

    "cylinder" -> CylinderFBMCave (pure numpy, Rhino-free).
    "nurbs"    -> NurbsLoftCave (imported lazily; needs Rhino for the loft).
    """
    if params.cave_type == "nurbs":
        # Lazy import so the cylinder path stays importable outside Rhino.
        from pearlscape.nurbs_cave import make_nurbs_cave
        return make_nurbs_cave(params)

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
        noise_type=params.noise_type,
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
