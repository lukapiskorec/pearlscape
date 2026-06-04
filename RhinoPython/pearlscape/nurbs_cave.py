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

from pearlscape.noise import fbm3_01, make_perm
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
