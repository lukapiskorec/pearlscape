#! python 3
# r: numpy
"""Irregular lofted-NURBS cave: build a tube from jittered cross-section rings,
evaluate it on a grid, then hand off to surface_sampling for the dense displaced
point cloud. Rhino-dependent (loft + surface eval); the numpy mass-work lives in
surface_sampling.py.
"""

import math
import os
import sys

import numpy as np

import Rhino.Geometry as rg

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape.noise import fbm3_01, ridged3_01, make_perm
from pearlscape import surface_sampling


def build_loft_surface(params):
    """Loft K jittered closed rings into an irregular NURBS tube; return its
    NurbsSurface. Centerline meanders in Y/Z only (X stays the slicing axis)."""
    K = params.nurbs_sections
    P = params.nurbs_section_points
    R = params.cave_radius
    perm = make_perm(params.nurbs_shape_seed)

    def noise01_at(pt_xyz, freq, tag):
        # One scalar of smooth FBM noise at a tagged location, in [0, 1].
        q = np.array([[pt_xyz[0] * freq + tag * 13.7,
                       pt_xyz[1] * freq + tag * 7.1,
                       pt_xyz[2] * freq + tag * 3.3]])
        return float(fbm3_01(q, perm, octaves=3, lacunarity=2.0, gain=0.5)[0])

    rings = []
    for k in range(K):
        # K >= 2 guaranteed by validate(), so K - 1 >= 1.
        x = params.cave_length * k / (K - 1)
        cy = (noise01_at((x, 0.0, 0.0), params.nurbs_centerline_freq, 1) * 2 - 1) \
            * params.nurbs_centerline_amp
        cz = (noise01_at((x, 0.0, 0.0), params.nurbs_centerline_freq, 2) * 2 - 1) \
            * params.nurbs_centerline_amp
        pts = []
        for p in range(P):
            theta = 2.0 * math.pi * p / P
            base = (R * math.cos(theta), R * math.sin(theta), x)
            jit = noise01_at(base, params.nurbs_radius_jitter_freq, 3) * 2 - 1
            r = R * (1.0 + params.nurbs_radius_jitter * jit)
            pts.append(rg.Point3d(
                x,
                cy + r * math.cos(theta),
                params.cave_center_z + cz + r * math.sin(theta),
            ))
        # Closed periodic degree-3 curve through the ring control points.
        # A periodic degree-3 curve needs >= 4 control points (validate() enforces
        # nurbs_section_points >= 4); guard anyway for a clear error.
        ring = rg.NurbsCurve.Create(True, 3, pts)
        if ring is None:
            raise RuntimeError(
                f"NurbsCurve.Create returned None for ring {k} "
                f"(need nurbs_section_points >= 4 for a periodic degree-3 curve)"
            )
        rings.append(ring)

    breps = rg.Brep.CreateFromLoft(
        rings, rg.Point3d.Unset, rg.Point3d.Unset,
        rg.LoftType.Normal, False,
    )
    if not breps:
        raise RuntimeError("Brep.CreateFromLoft returned no result")
    brep = breps[0]
    if brep.Faces.Count > 1:
        print(f"WARNING: loft returned {brep.Faces.Count} faces; using Faces[0]")
    srf = brep.Faces[0].UnderlyingSurface().ToNurbsSurface()
    if srf is None:
        raise RuntimeError("ToNurbsSurface() returned None — degenerate loft face")
    return srf


def eval_surface_grid(surface, nu: int, nv: int) -> np.ndarray:
    """Evaluate `surface` on a regular (nu x nv) grid -> (nu, nv, 3) array.

    The loft may assign U or V to the ring direction, so detect which surface
    direction is CLOSED (the periodic ring) and map it to axis 1 (v); the OPEN
    direction (along X) maps to axis 0 (u). u spans its domain inclusively; v
    spans [v0, v1) with the endpoint excluded so there's no duplicate seam."""
    if not surface.IsClosed(0) and not surface.IsClosed(1):
        print("WARNING: neither surface direction is closed; "
              "assuming direction 1 is the periodic ring.")
    closed_dir = 1 if surface.IsClosed(1) else (0 if surface.IsClosed(0) else 1)
    open_dir = 1 - closed_dir
    od = surface.Domain(open_dir)
    cd = surface.Domain(closed_dir)
    u_open = np.linspace(od.T0, od.T1, nu)
    v_closed = np.linspace(cd.T0, cd.T1, nv, endpoint=False)
    grid = np.empty((nu, nv, 3), dtype=np.float64)
    for i in range(nu):
        for j in range(nv):
            uv = [0.0, 0.0]
            uv[open_dir] = float(u_open[i])
            uv[closed_dir] = float(v_closed[j])
            pt = surface.PointAt(uv[0], uv[1])
            grid[i, j, 0] = pt.X
            grid[i, j, 1] = pt.Y
            grid[i, j, 2] = pt.Z
    return grid


class NurbsLoftCave:
    """CaveSurface backed by a lofted NURBS tube `surface`."""

    def __init__(self, surface, params) -> None:
        self.surface = surface
        self.params = params
        self._grid = None       # (nu, nv, 3) base-surface eval, cached
        self._normals = None    # (nu, nv, 3) per-node unit normals, cached
        # Note: only inner_boundary uses this cache; sample_surface_points
        # evaluates its own grid. The two run in different pipeline modes, so
        # the grid is never evaluated twice in one run.

    def _ensure_grid(self):
        """Evaluate (and cache) the base surface grid + normals once."""
        if self._grid is None:
            p = self.params
            self._grid = eval_surface_grid(self.surface, p.nurbs_grid_u, p.nurbs_grid_v)
            self._normals = surface_sampling.grid_normals(self._grid)
        return self._grid, self._normals

    def x_extent(self):
        """(x_min, x_max) of the actual lofted surface along X, from the cached
        grid — so it reflects a hand-edited 'reuse' surface, not a nominal length."""
        grid, _ = self._ensure_grid()
        xs = grid[:, :, 0]
        return (float(xs.min()), float(xs.max()))

    def sample_surface_points(self) -> np.ndarray:
        p = self.params
        grid = eval_surface_grid(self.surface, p.nurbs_grid_u, p.nurbs_grid_v)
        normals = surface_sampling.grid_normals(grid)
        return surface_sampling.sample_and_displace(
            grid, normals,
            target=p.total_surface_samples,
            fbm_amplitude=p.fbm_amplitude,
            fbm_base_freq=p.fbm_base_freq,
            octaves=p.fbm_octaves,
            lacunarity=p.fbm_lacunarity,
            gain=p.fbm_gain,
            noise_type=p.noise_type,
            noise_seed=p.noise_seed,
        )

    def inner_boundary(self, plane_x: float, n_angular: int):
        """Craggy cross-section boundary at X = plane_x, derived from the cached
        base grid + the same noise convention as sample_and_displace."""
        from pearlscape.cross_section import displaced_ring_boundary

        grid, normals = self._ensure_grid()
        nu, nv = grid.shape[0], grid.shape[1]

        # Interpolate the base ring at the u where X == plane_x. X is ~monotonic
        # along u (centerline meanders in Y/Z only), so per-u mean X is a safe key.
        # np.interp requires xu non-decreasing; warn if the loft ever violates that
        # (it would otherwise return a silently-wrong u with no error).
        xu = grid[:, :, 0].mean(axis=1)                      # (nu,)
        if np.any(np.diff(xu) < 0.0):
            print("WARNING: inner_boundary: per-u mean X is not monotonic; "
                  "cross-section u-interpolation may be wrong.")
        u_frac = float(np.interp(plane_x, xu, np.arange(nu)))
        u0 = int(np.clip(np.floor(u_frac), 0, nu - 2))
        tu = u_frac - u0
        ring = (1.0 - tu) * grid[u0] + tu * grid[u0 + 1]     # (nv, 3)
        ring_n = (1.0 - tu) * normals[u0] + tu * normals[u0 + 1]

        # Resample the periodic ring to n_angular points.
        v_idx = np.linspace(0.0, nv, n_angular, endpoint=False)
        v0 = np.floor(v_idx).astype(np.int64) % nv
        v1 = (v0 + 1) % nv
        tv = (v_idx - np.floor(v_idx))[:, None]
        base_xyz = (1.0 - tv) * ring[v0] + tv * ring[v1]     # (n_angular, 3)
        nrm = (1.0 - tv) * ring_n[v0] + tv * ring_n[v1]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)

        # Orient normals outward from the ring's center (match sample_and_displace).
        center3 = base_xyz.mean(axis=0)
        outward = base_xyz - center3
        sign = np.sign(np.sum(nrm * outward, axis=1, keepdims=True))
        sign = np.where(sign == 0, 1.0, sign)
        nrm = nrm * sign

        # Noise at world coordinates (matches sample_and_displace: base_pts * freq).
        p = self.params
        perm = make_perm(p.noise_seed)
        noise_fn = ridged3_01 if p.noise_type == "ridged" else fbm3_01
        n01 = noise_fn(
            base_xyz * p.fbm_base_freq, perm,
            octaves=p.fbm_octaves, lacunarity=p.fbm_lacunarity, gain=p.fbm_gain,
        )
        return displaced_ring_boundary(
            base_xyz, nrm, n01, fbm_amplitude=p.fbm_amplitude,
        )


def make_nurbs_cave(params) -> NurbsLoftCave:
    """Resolve the base surface (rebuild or reuse) and return the cave.

    rebuild: loft from params, render onto Pearlscape::CaveSurface (replacing).
    reuse:   read the surface from that layer; fall back to rebuild if absent.
    """
    from pearlscape import display

    source = params.nurbs_surface_source
    if source == "reuse":
        srf = display.find_cave_surface()
        if srf is None:
            print("nurbs_surface_source='reuse' but no surface on "
                  "Pearlscape::CaveSurface; rebuilding.")
            source = "rebuild"
    if source == "rebuild":
        srf = build_loft_surface(params)
        display.render_cave_surface(srf)
    return NurbsLoftCave(srf, params)
