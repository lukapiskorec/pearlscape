#! python 3
# r: numpy
"""Build per-curtain bead point sets by sampling each curtain plane outward from
the cave's craggy cross-section boundary. Replaces the old slice-and-project
approach: beads are placed directly in the plane, so there is no projection
stacking and the inner cave edge stays sharp."""

import math
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from pearlscape import cross_section


def curtain_x_positions(curtain_count: int, spacing: float, x_center: float) -> np.ndarray:
    """Return the X coordinate of each curtain plane, centered on x_center."""
    offsets = (np.arange(curtain_count) - (curtain_count - 1) / 2.0) * spacing
    return offsets + x_center


def array_x_center(cave_length: float) -> float:
    """The curtain array centers on the cave's midpoint along X."""
    return cave_length / 2.0


def _shell_spacing(params) -> float:
    """Thin-shell plane spacing (mm): params.shell_curtain_spacing, or
    2 * bead_diameter when that is 0 (auto)."""
    if params.shell_curtain_spacing > 0.0:
        return float(params.shell_curtain_spacing)
    return 2.0 * float(params.bead_diameter)


def curtain_planes(cave, params) -> np.ndarray:
    """Curtain plane X positions, fitted to the cave's actual X extent.

    In ``"nurbs"`` mode the plane count is DERIVED from the surface's X extent and
    ``curtain_spacing`` (``curtain_count`` is ignored), so curtains span the cave
    exactly and never hang off the ends. In ``"cylinder"`` mode the explicit
    ``curtain_count`` is kept, centered on the cave. In both modes any plane that
    falls outside the cave's X extent is dropped (and the drop is logged), so the
    curtain geometry can never extend past the actual surface.
    """
    x_min, x_max = cave.x_extent()

    if params.curtain_mode == "shell":
        # Shell mode always tiles the X extent at the thin-shell spacing,
        # regardless of cave_type — an explicit curtain_count makes no sense
        # for a dense thin shell.
        spacing = _shell_spacing(params)
        use_extent_fit = True
    else:
        spacing = params.curtain_spacing
        use_extent_fit = (params.cave_type == "nurbs")

    if use_extent_fit:
        length = x_max - x_min
        gaps = int(np.floor(length / spacing + 1e-9))   # spacing intervals that fit
        count = max(1, gaps + 1)
        span = gaps * spacing
        x0 = x_min + (length - span) / 2.0              # center the planes in the extent
        plane_xs = x0 + np.arange(count) * spacing
    else:
        x_center = 0.5 * (x_min + x_max)
        plane_xs = curtain_x_positions(params.curtain_count, spacing, x_center)

    inside = (plane_xs >= x_min - 1e-6) & (plane_xs <= x_max + 1e-6)
    dropped = int((~inside).sum())
    if dropped:
        print(f"Curtains: dropped {dropped} plane(s) outside the cave X extent "
              f"[{x_min:.0f}, {x_max:.0f}] mm.")
    return plane_xs[inside]


def _resolve_bead_spacing(boundaries: Sequence[tuple], params) -> float:
    """Pick the in-plane bead spacing.

    If params.target_bead_count > 0, solve the spacing from the total band area
    so the bead count lands near the target; otherwise use params.bead_min_spacing
    as set. Band area is perimeter * band_thickness summed over curtains, and the
    fade keeps a mean fraction 1/(fade+1) of the candidates, so
        beads ~= total_area / spacing^2 * keep   ->   spacing = sqrt(area*keep/target).
    Spacing is clamped to >= bead_diameter (beads cannot pack tighter than touching).
    """
    if params.target_bead_count <= 0:
        return params.bead_min_spacing

    keep = 1.0 / (params.curtain_band_fade + 1.0)
    total_area = params.curtain_band_thickness * sum(
        cross_section.boundary_perimeter(c, th, ri) for c, th, ri in boundaries
    )
    if total_area <= 0.0:
        return params.bead_min_spacing

    s = math.sqrt(total_area * keep / params.target_bead_count)
    s_clamped = max(s, params.bead_diameter)
    if s_clamped > s + 1e-9:
        # Clamping the spacing UP (to the bead_diameter floor) means fewer beads,
        # so the count falls SHORT of the target — the cave's band can't hold that
        # many beads without overlapping them.
        est = int(round(total_area * keep / (s_clamped * s_clamped)))
        print(f"Bead budget: target {params.target_bead_count:,} needs spacing "
              f"{s:.2f}mm < bead_diameter {params.bead_diameter:g} (the tightest "
              f"valid packing); clamped to {s_clamped:.2f}mm, so the count falls "
              f"short (~{est:,}). Raise curtain_band_thickness, lower "
              f"curtain_spacing, or reduce bead_diameter to fit more.")
    else:
        print(f"Bead budget: target {params.target_bead_count:,} -> "
              f"bead_min_spacing {s_clamped:.2f}mm.")
    return s_clamped


def _greedy_overlap_filter(yz: np.ndarray, radius: float) -> np.ndarray:
    """Greedy keep-order filter over (M, 2) points. Returns a boolean mask that
    keeps a point only if no already-kept point lies within `radius` of it.

    Uses a radius-sized spatial-hash grid so each point only tests its 3x3 cell
    neighbourhood (two points within `radius` differ by at most one cell per
    axis). Deterministic in input order — the caller passes points in the cave
    sampler's fixed order, so reruns are bit-identical.
    """
    m = yz.shape[0]
    keep = np.zeros(m, dtype=bool)
    if m == 0:
        return keep
    inv = 1.0 / radius
    r2 = radius * radius
    cx_all = np.floor(yz[:, 0] * inv).astype(np.int64)
    cz_all = np.floor(yz[:, 1] * inv).astype(np.int64)
    cells = {}   # (cx, cz) -> list of kept-point row indices
    for j in range(m):
        cx, cz = int(cx_all[j]), int(cz_all[j])
        py, pz = yz[j, 0], yz[j, 1]
        clash = False
        for dx in (-1, 0, 1):
            for dcz in (-1, 0, 1):
                bucket = cells.get((cx + dx, cz + dcz))
                if not bucket:
                    continue
                for k in bucket:
                    dy = py - yz[k, 0]
                    dz = pz - yz[k, 1]
                    if dy * dy + dz * dz < r2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            keep[j] = True
            cells.setdefault((cx, cz), []).append(j)
    return keep


def _string_overlap(params) -> float:
    """Y-overlap (mm) that forces two beads onto a shared string:
    params.string_align_overlap, or bead_diameter when that is 0 (auto)."""
    if params.string_align_overlap > 0.0:
        return float(params.string_align_overlap)
    return float(params.bead_diameter)


def _align_strings_y(y: np.ndarray, threshold: float, rng) -> np.ndarray:
    """Pull vertically-overlapping beads onto shared strings; returns new Ys.

    Sweep the beads left to right (sorted by Y), growing a "current string".
    A bead closer than `threshold` to the string overlaps it: a coin decides
    which side shifts — the bead joins the string at its Y, or the whole
    string shifts out to the bead's Y (for a two-bead string that is exactly
    "randomly shift the upper or the lower bead"). The string may only shift
    while every bead on it stays within `threshold` of its original Y, so no
    bead ever moves a full threshold; chains that would drag farther split
    into separate strings instead. Final string positions are random (they
    fall out of the bead layout + coin flips, not a snap grid), and distinct
    strings always end >= `threshold` apart.
    """
    out = np.array(y, dtype=np.float64, copy=True)
    if out.shape[0] == 0:
        return out
    order = np.argsort(out, kind="stable")
    members = [order[0]]
    s = float(out[order[0]])    # the string's current Y
    m_min = s                   # leftmost ORIGINAL y on the string (drag limit)
    for idx in order[1:]:
        yi = float(out[idx])
        if yi - s < threshold:
            if yi - m_min < threshold and rng.random() < 0.5:
                s = yi
            members.append(idx)
        else:
            out[members] = s
            members = [idx]
            s = yi
            m_min = yi
    out[members] = s
    return out


def _space_strings_z(y: np.ndarray, z: np.ndarray, diameter: float) -> np.ndarray:
    """De-overlap beads vertically within each string; returns new Zs.

    `_align_strings_y` collapses beads onto shared strings by equalising their
    Y, which removes the horizontal gap that had been keeping some of them
    apart — so two beads on one string can end up closer than a bead diameter
    in Z (the PDFs showed these overlapping). For each string (the beads that
    now share one aligned Y) this sweeps top to bottom and slides every bead
    that overlaps the one above it down until they just touch (gap == diameter),
    so no two beads on a string ever overlap. The topmost bead never moves, and
    sliding down mirrors how beads rest on each other on a real hanging string.
    There is no per-bead travel cap, so the gap clears completely; drift stays
    small in practice because the cloud is pre-spaced >= a diameter in 2D, so
    overlapping runs are short (measured worst-case drift ~1.6 diameters) — and
    a bead sitting clear of the one above resets the sweep, so drift never
    propagates past a non-overlapping gap.
    """
    out = np.array(z, dtype=np.float64, copy=True)
    n = out.shape[0]
    if n < 2:
        return out
    order = np.argsort(y, kind="stable")   # walk beads grouped by their string Y
    ys = y[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ys[j + 1] == ys[i]:
            j += 1
        if j > i:
            idxs = order[i:j + 1]
            zi = out[idxs]                              # copy (fancy index)
            top = np.argsort(zi, kind="stable")[::-1]   # high Z (top) -> low (bottom)
            prev = zi[top[0]]                           # topmost bead stays put
            for t in top[1:]:
                ceil = prev - diameter                  # must sit >= diameter below prev
                zt = zi[t]
                placed = zt if zt <= ceil else ceil     # slide down to clear (no cap)
                zi[t] = placed
                prev = placed
            out[idxs] = zi
        i = j + 1
    return out


def align_curtain_strings(curtains: Sequence[dict], params) -> dict:
    """In-place per-curtain string alignment (see _align_strings_y): updates
    points_2d AND points_3d so the curtain dicts stay consistent. Call AFTER
    the curtains are built and BEFORE colouring/sectioning, so everything
    downstream inherits the aligned positions.

    Two passes per curtain: align Y to pull overlapping beads onto shared
    strings, then de-overlap Z within each string (see _space_strings_z) so
    beads sharing a string never sit closer than a bead diameter vertically.

    Per-curtain seeds occupy [noise_seed + 5000, noise_seed + 5000 + n) —
    distinct strings per curtain, clear of the +1000 curtain-sampler range.
    Returns {'beads', 'strings', 'max_shift', 'max_z_shift'} for the run log.
    """
    overlap = _string_overlap(params)
    diameter = float(params.bead_diameter)
    total = 0
    n_strings = 0
    max_shift = 0.0
    max_z_shift = 0.0
    for i, c in enumerate(curtains):
        pts = c["points_2d"]
        if pts.shape[0] == 0:
            continue
        rng = np.random.default_rng(params.noise_seed + 5000 + i)
        y_new = _align_strings_y(pts[:, 0], overlap, rng)
        max_shift = max(max_shift, float(np.abs(y_new - pts[:, 0]).max()))
        # De-overlap each string vertically using the physical bead size (not
        # the Y merge `overlap`, which may be set wider than the beads).
        z_new = _space_strings_z(y_new, pts[:, 1], diameter)
        max_z_shift = max(max_z_shift, float(np.abs(z_new - pts[:, 1]).max()))
        n_strings += int(np.unique(y_new).size)
        total += int(pts.shape[0])
        pts[:, 0] = y_new
        pts[:, 1] = z_new
        c["points_3d"][:, 1] = y_new
        c["points_3d"][:, 2] = z_new
    return {"beads": total, "strings": n_strings,
            "max_shift": max_shift, "max_z_shift": max_z_shift}


def build_curtains(cave, plane_xs: Sequence[float], params) -> Tuple[List[dict], float]:
    """For each curtain plane, sample beads in-plane outside the cave's cross-
    section boundary, fading outward.

    Returns (curtains, bead_spacing):
      curtains: one dict per curtain
        {'plane_x': float,
         'points_2d': np.ndarray (M, 2)  # columns (Y, Z), the in-plane beads
         'points_3d': np.ndarray (M, 3)} # (plane_x, Y, Z), for colour sampling
      bead_spacing: the in-plane spacing (mm) actually used — equal to
        params.bead_min_spacing, or the value the budget solver resolved when
        target_bead_count > 0.
    """
    # Angular resolution for the boundary curve: ~one sample per bead-diameter arc
    # along the nominal circumference (finer than bead size is wasted detail). Keyed
    # to bead_diameter, not bead_min_spacing, so the curve stays crisp even when the
    # budget solver widens the spacing.
    n_angular = max(720, int(2.0 * np.pi * params.cave_radius / params.bead_diameter))

    # Compute every cross-section boundary once (the geometry cost; for the NURBS
    # cave the surface grid is evaluated and cached on the first call), then resolve
    # the bead spacing from them if a bead budget is set.
    boundaries = [cave.inner_boundary(float(px), n_angular) for px in plane_xs]
    spacing = _resolve_bead_spacing(boundaries, params)

    curtains: List[dict] = []
    for i, (plane_x, (centroid, theta, r_inner)) in enumerate(zip(plane_xs, boundaries)):
        plane_x = float(plane_x)
        pts_2d = cross_section.sample_band(
            centroid, theta, r_inner,
            band_thickness=params.curtain_band_thickness,
            fade=params.curtain_band_fade,
            spacing=spacing,
            # Curtain seeds occupy [noise_seed + 1000, noise_seed + 1000 + curtain_count).
            # Keep other seed offsets derived from noise_seed outside this range.
            seed=params.noise_seed + 1000 + i,
        )
        if pts_2d.shape[0]:
            pts_3d = np.column_stack([
                np.full(pts_2d.shape[0], plane_x, dtype=np.float64),
                pts_2d[:, 0],
                pts_2d[:, 1],
            ])
        else:
            pts_3d = np.zeros((0, 3), dtype=np.float64)
        curtains.append({
            "plane_x": plane_x,
            "points_2d": pts_2d,
            "points_3d": pts_3d,
        })
    return curtains, spacing


def build_shell_curtains(cave, plane_xs: Sequence[float], params) -> Tuple[List[dict], float]:
    """Thin-shell mode: snap the cave's Poisson surface cloud onto the thin
    curtain planes (X -> nearest plane, Y/Z unchanged), then drop per-plane
    overlaps within bead_diameter so each plane is a gappy, non-continuous line.

    Returns (curtains, bead_spacing) with the SAME dict shape as build_curtains:
      {'plane_x', 'points_2d' (M, 2)=(Y, Z), 'points_3d' (M, 3)=(plane_x, Y, Z)}
    bead_spacing is the bead_diameter drop radius (reported in the run summary).
    """
    plane_xs = np.asarray(plane_xs, dtype=np.float64)
    n_planes = int(plane_xs.shape[0])
    radius = float(params.bead_diameter)

    empty = lambda px: {
        "plane_x": float(px),
        "points_2d": np.zeros((0, 2), dtype=np.float64),
        "points_3d": np.zeros((0, 3), dtype=np.float64),
    }

    pts = cave.sample_surface_points()              # (N, 3) world points
    if pts.shape[0] == 0 or n_planes == 0:
        return [empty(px) for px in plane_xs], radius

    # Snap each bead to its nearest plane. Planes are uniformly spaced, so a
    # single round gives the index; clamp ends-beads into the valid range.
    # The snap trusts a uniform grid at _shell_spacing(params) — build plane_xs
    # via curtain_planes (shell mode), which tiles the extent at exactly this
    # spacing. Guard turns a mismatched/non-uniform grid into a clear error
    # rather than silent bead misassignment.
    x0 = float(plane_xs[0])
    spacing = _shell_spacing(params)
    if n_planes >= 2:
        assert np.allclose(np.diff(plane_xs), spacing), (
            "build_shell_curtains expects plane_xs uniformly spaced at "
            "_shell_spacing(params)"
        )
    idx = np.round((pts[:, 0] - x0) / spacing).astype(np.int64)
    idx = np.clip(idx, 0, n_planes - 1)

    curtains: List[dict] = []
    for i, px in enumerate(plane_xs):
        yz = pts[idx == i][:, 1:3]                   # boolean index preserves order
        if yz.shape[0] == 0:
            curtains.append(empty(px))
            continue
        kept = _greedy_overlap_filter(yz, radius)
        kept_yz = yz[kept]
        if kept_yz.shape[0]:
            pts_3d = np.column_stack([
                np.full(kept_yz.shape[0], float(px), dtype=np.float64),
                kept_yz[:, 0],
                kept_yz[:, 1],
            ])
        else:
            pts_3d = np.zeros((0, 3), dtype=np.float64)
        curtains.append({
            "plane_x": float(px),
            "points_2d": kept_yz,
            "points_3d": pts_3d,
        })
    return curtains, radius


if __name__ == "__main__":
    # Headless smoke test on the pure-numpy cylinder cave.
    from pearlscape.cave import CylinderFBMCave, PointCloudCave
    from pearlscape.params import PearlscapeParams

    params = PearlscapeParams()   # uses default band params (120 / 1.5 / 6)
    params.target_bead_count = 0  # test the plain bead_min_spacing path here
    cave = CylinderFBMCave(
        radius=params.cave_radius, length=2000.0, fbm_amplitude=900.0,
        fbm_base_freq=0.00035, fbm_octaves=6, fbm_lacunarity=2.0, fbm_gain=0.55,
        noise_seed=params.noise_seed, target_samples=1000, center_z=1500.0,
        noise_type="ridged",
    )
    xs = curtain_x_positions(5, 100.0, array_x_center(2000.0))
    curtains, spacing_used = build_curtains(cave, xs, params)
    assert spacing_used == params.bead_min_spacing   # no budget -> param spacing

    assert len(curtains) == 5, len(curtains)
    for c in curtains:
        assert c["points_2d"].ndim == 2 and c["points_2d"].shape[1] == 2
        assert c["points_3d"].shape == (c["points_2d"].shape[0], 3)
        if c["points_3d"].shape[0]:
            assert np.allclose(c["points_3d"][:, 0], c["plane_x"])
            assert np.allclose(c["points_3d"][:, 1], c["points_2d"][:, 0])
            assert np.allclose(c["points_3d"][:, 2], c["points_2d"][:, 1])
    total = sum(len(c["points_2d"]) for c in curtains)
    assert total > 0, "no beads generated"

    # Determinism: same params -> identical beads.
    curtains2, _ = build_curtains(cave, xs, params)
    same = all(np.array_equal(a["points_2d"], b["points_2d"])
               for a, b in zip(curtains, curtains2))
    assert same, "build_curtains not deterministic"

    # Per-curtain seed isolation: distinct curtains must not produce identical
    # bead patterns (would mean the +i seed offset isn't taking effect).
    no_clash = not any(
        np.array_equal(curtains[j]["points_2d"], curtains[k]["points_2d"])
        for j in range(len(curtains)) for k in range(j + 1, len(curtains))
        if curtains[j]["points_2d"].shape[0] > 0 and curtains[k]["points_2d"].shape[0] > 0
    )
    assert no_clash, "two curtains produced identical bead patterns (seed not varying)"

    # --- string alignment ---
    rng_y = np.random.default_rng(0)
    y_rand = rng_y.uniform(0.0, 500.0, size=400)
    aligned = _align_strings_y(y_rand, 6.0, np.random.default_rng(1))
    shifts = np.abs(aligned - y_rand)
    assert shifts.max() < 6.0, shifts.max()       # every move stays under the overlap
    uniq = np.unique(aligned)
    assert uniq.size < y_rand.size                # merges happened
    assert np.all(np.diff(uniq) >= 6.0 - 1e-9)    # strings >= overlap apart
    # deterministic per seed, different across seeds (random strings, no grid)
    assert np.array_equal(aligned, _align_strings_y(y_rand, 6.0, np.random.default_rng(1)))
    assert not np.array_equal(aligned, _align_strings_y(y_rand, 6.0, np.random.default_rng(2)))

    # --- vertical de-overlap within a string (_space_strings_z) ---
    one = np.zeros(3)                                   # one string (shared Y)
    # resolvable: lower bead slides down to clear, top bead never moves
    z_out = _space_strings_z(one, np.array([0.0, 3.0, 20.0]), 6.0)
    assert np.allclose(np.sort(z_out), [-3.0, 3.0, 20.0]), z_out
    # coincident pair: separates by exactly one diameter
    z2 = _space_strings_z(np.zeros(2), np.array([5.0, 5.0]), 6.0)
    assert abs(abs(z2[0] - z2[1]) - 6.0) < 1e-9, z2
    # already spaced: untouched
    z3 = _space_strings_z(np.zeros(3), np.array([0.0, 6.0, 12.0]), 6.0)
    assert np.allclose(z3, [0.0, 6.0, 12.0]), z3
    # strings are independent: a second Y group does not affect the first
    y2 = np.array([0.0, 0.0, 9.0, 9.0])
    z4 = _space_strings_z(y2, np.array([0.0, 1.0, 100.0, 101.0]), 6.0)
    assert abs(z4[1] - z4[0]) >= 6.0 - 1e-9 and abs(z4[3] - z4[2]) >= 6.0 - 1e-9, z4
    # every string fully de-overlaps (no cap) and a clear gap resets drift
    rng_z = np.random.default_rng(7)
    z_rand = rng_z.uniform(0.0, 300.0, size=30)
    zr = _space_strings_z(np.zeros(30), z_rand, 6.0)
    assert (np.diff(np.sort(zr)) >= 6.0 - 1e-9).all()   # zero overlaps remain
    assert zr.max() <= z_rand.max() + 1e-9              # only slides down, never up
    # (near-total resolution on realistic, pre-spaced beads is checked below)

    # align_curtain_strings on the production shell path: beads are pre-spaced
    # >= a bead diameter in 2D (the greedy filter), so Y-merge + Z de-overlap
    # leaves almost no residual. Band mode is NOT used here — its sampler permits
    # 2D overlaps, so beads would be overlapping before alignment even runs.
    sa = PearlscapeParams()
    sa.curtain_mode = "shell"
    sa.cave_type = "cylinder"
    sa.string_align = True
    sa_cave = CylinderFBMCave(
        radius=sa.cave_radius, length=2000.0, fbm_amplitude=900.0,
        fbm_base_freq=0.00035, fbm_octaves=6, fbm_lacunarity=2.0, fbm_gain=0.55,
        noise_seed=sa.noise_seed, target_samples=1000, center_z=1500.0,
        noise_type="ridged", bead_spacing=sa.cave_bead_spacing)
    sa_planes = curtain_planes(sa_cave, sa)
    sa_curtains, _ = build_shell_curtains(sa_cave, sa_planes, sa)
    stats = align_curtain_strings(sa_curtains, sa)
    assert stats["beads"] == sum(len(c["points_2d"]) for c in sa_curtains)
    assert 0 < stats["strings"] < stats["beads"]
    assert stats["max_shift"] < sa.bead_diameter
    assert stats["max_z_shift"] >= 0.0
    sa_pairs = sa_overlap = 0
    for c in sa_curtains:
        if c["points_2d"].shape[0]:
            assert np.array_equal(c["points_3d"][:, 1], c["points_2d"][:, 0])
            assert np.array_equal(c["points_3d"][:, 2], c["points_2d"][:, 1])
            ys = np.unique(c["points_2d"][:, 0])
            assert np.all(np.diff(ys) >= sa.bead_diameter - 1e-9)
            # vertical de-overlap: every bead on a string is >= a diameter apart
            order = np.argsort(c["points_2d"][:, 0], kind="stable")
            ys_s = c["points_2d"][order, 0]; zs_s = c["points_2d"][order, 1]
            i = 0; n = ys_s.shape[0]
            while i < n:
                j = i
                while j + 1 < n and ys_s[j + 1] == ys_s[i]:
                    j += 1
                if j > i:
                    g = np.diff(np.sort(zs_s[i:j + 1]))
                    sa_pairs += g.size
                    sa_overlap += int((g < sa.bead_diameter - 1e-9).sum())
                i = j + 1
    assert sa_pairs > 0 and sa_overlap == 0, (sa_overlap, sa_pairs)   # zero overlaps
    u0 = np.unique(sa_curtains[0]["points_2d"][:, 0])
    u1 = np.unique(sa_curtains[1]["points_2d"][:, 0])
    assert not np.array_equal(u0, u1), "two curtains share a string layout"
    print(f"string align: {stats['beads']} beads -> {stats['strings']} strings, "
          f"max Y shift {stats['max_shift']:.2f} mm, max Z shift {stats['max_z_shift']:.2f} mm, "
          f"{sa_overlap}/{sa_pairs} string pairs still overlapping")

    # curtain_planes: nurbs derives the count from the X extent; cylinder keeps
    # curtain_count; both drop planes outside the cave (extent here is [0, 2000]).
    geom = PearlscapeParams()
    geom.curtain_mode = "band"   # shell default would tile at the thin-shell spacing
    geom.curtain_spacing = 50.0
    geom.cave_type = "nurbs"
    pn = curtain_planes(cave, geom)
    assert len(pn) == 41, len(pn)                      # 2000/50 gaps + 1, spanning [0, 2000]
    assert pn.min() >= -1e-6 and pn.max() <= 2000.0 + 1e-6, (pn.min(), pn.max())
    geom.cave_type = "cylinder"
    geom.curtain_count = 100
    pc = curtain_planes(cave, geom)
    assert len(pc) == 40, len(pc)                      # 60 of 100 fall outside [0, 2000]
    assert pc.min() >= -1e-6 and pc.max() <= 2000.0 + 1e-6

    # Bead budget: with target_bead_count set, the total lands near the target.
    budget = PearlscapeParams()
    budget.target_bead_count = 20_000
    xs_b = curtain_x_positions(10, 50.0, array_x_center(2000.0))
    curtains_b, _ = build_curtains(cave, xs_b, budget)
    total_b = sum(len(c["points_2d"]) for c in curtains_b)
    rel = abs(total_b - budget.target_bead_count) / budget.target_bead_count
    assert rel < 0.10, f"budget off: got {total_b}, target {budget.target_bead_count} ({rel:.1%})"

    # --- shell mode: snap the cave cloud onto thin planes, drop overlaps ---
    shell_params = PearlscapeParams()
    shell_params.curtain_mode = "shell"
    shell_params.cave_type = "cylinder"
    shell_params.shell_curtain_spacing = 0.0          # auto -> 2 * bead_diameter
    assert _shell_spacing(shell_params) == 2.0 * shell_params.bead_diameter

    shell_cave = CylinderFBMCave(
        radius=shell_params.cave_radius, length=2000.0, fbm_amplitude=900.0,
        fbm_base_freq=0.00035, fbm_octaves=6, fbm_lacunarity=2.0, fbm_gain=0.55,
        noise_seed=shell_params.noise_seed, target_samples=1000, center_z=1500.0,
        noise_type="ridged", bead_spacing=12.0,        # Poisson cloud, 12mm spacing
    )
    source = shell_cave.sample_surface_points()
    shell_planes = curtain_planes(shell_cave, shell_params)
    # planes tile the X extent [0, 2000] at 2*bead_diameter = 12mm spacing
    assert shell_planes.min() >= -1e-6 and shell_planes.max() <= 2000.0 + 1e-6
    assert np.allclose(np.diff(shell_planes), 12.0)

    shell_curtains, shell_spacing_used = build_shell_curtains(
        shell_cave, shell_planes, shell_params)
    assert shell_spacing_used == shell_params.bead_diameter

    # every kept bead lies exactly on its plane; points_2d/points_3d agree
    for c in shell_curtains:
        if c["points_3d"].shape[0]:
            assert np.allclose(c["points_3d"][:, 0], c["plane_x"])
            assert np.array_equal(c["points_3d"][:, 1], c["points_2d"][:, 0])
            assert np.array_equal(c["points_3d"][:, 2], c["points_2d"][:, 1])

    # no two kept beads on a plane are closer than bead_diameter in Y/Z
    bd = shell_params.bead_diameter
    for c in shell_curtains:
        yz = c["points_2d"]
        if yz.shape[0] >= 2:
            d2 = ((yz[:, None, :] - yz[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(d2, np.inf)
            assert d2.min() >= bd * bd - 1e-6, f"overlap on plane: {np.sqrt(d2.min()):.3f}"

    shell_total = sum(c["points_2d"].shape[0] for c in shell_curtains)
    assert 0 < shell_total <= source.shape[0], (shell_total, source.shape[0])

    # determinism: same params -> identical kept beads
    shell_curtains2, _ = build_shell_curtains(shell_cave, shell_planes, shell_params)
    assert all(np.array_equal(a["points_2d"], b["points_2d"])
               for a, b in zip(shell_curtains, shell_curtains2)), "shell not deterministic"

    print(f"shell: {len(shell_planes)} planes, "
          f"{shell_total}/{source.shape[0]} beads kept")

    # --- points cave: an imported cloud routed through the shell path ---
    # Reuse the shell_cave's generated cloud as a stand-in for the CavePoints
    # layer, then drive build_shell_curtains through PointCloudCave. The point of
    # this test is that an arbitrary cloud snaps onto the thin planes with the same
    # invariants (on-plane, non-overlapping, deterministic) as the generated cave.
    pts_params = PearlscapeParams()
    pts_params.cave_type = "points"
    pts_params.curtain_mode = "shell"
    pts_params.shell_curtain_spacing = 0.0            # auto -> 2 * bead_diameter

    pts_cave = PointCloudCave(source)                 # `source` = shell_cave's cloud
    assert pts_cave.x_extent() == (float(source[:, 0].min()), float(source[:, 0].max()))

    pts_planes = curtain_planes(pts_cave, pts_params)
    assert np.allclose(np.diff(pts_planes), 2.0 * pts_params.bead_diameter)
    x_lo, x_hi = pts_cave.x_extent()
    assert pts_planes.min() >= x_lo - 1e-6 and pts_planes.max() <= x_hi + 1e-6

    pts_curtains, pts_spacing = build_shell_curtains(pts_cave, pts_planes, pts_params)
    assert pts_spacing == pts_params.bead_diameter

    bd_p = pts_params.bead_diameter
    for c in pts_curtains:
        yz = c["points_2d"]
        if c["points_3d"].shape[0]:
            assert np.allclose(c["points_3d"][:, 0], c["plane_x"])
        if yz.shape[0] >= 2:
            d2 = ((yz[:, None, :] - yz[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(d2, np.inf)
            assert d2.min() >= bd_p * bd_p - 1e-6, f"overlap: {np.sqrt(d2.min()):.3f}"

    pts_curtains2, _ = build_shell_curtains(pts_cave, pts_planes, pts_params)
    assert all(np.array_equal(a["points_2d"], b["points_2d"])
               for a, b in zip(pts_curtains, pts_curtains2)), "points cave not deterministic"

    pts_total = sum(c["points_2d"].shape[0] for c in pts_curtains)
    assert 0 < pts_total <= source.shape[0]

    # sample_surface_points returns a copy (mutating it must not touch the source)
    grabbed = pts_cave.sample_surface_points()
    grabbed[0, 0] += 123.0
    assert pts_cave.points[0, 0] != grabbed[0, 0]

    # inner_boundary is unsupported for a raw cloud; empty clouds are rejected.
    try:
        pts_cave.inner_boundary(0.0, 720)
        raise AssertionError("inner_boundary should raise for PointCloudCave")
    except NotImplementedError:
        pass
    try:
        PointCloudCave(np.zeros((0, 3)))
        raise AssertionError("empty cloud should raise")
    except ValueError:
        pass

    print(f"points cave: {len(pts_planes)} planes, "
          f"{pts_total}/{source.shape[0]} beads kept (shell-routed)")

    print(f"5 curtains, {total} beads, deterministic, seeds distinct")
    print(f"curtain_planes: nurbs->{len(pn)} planes, cylinder->{len(pc)} (filtered)")
    print(f"bead budget: target 20,000 -> {total_b} beads ({rel:.1%} off)")
    print("OK")
