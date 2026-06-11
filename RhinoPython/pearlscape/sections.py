#! python 3
# r: numpy
"""Section the curtain model into a lattice of section_size cubes for
fabrication.

Pure numpy — no Rhino imports, so the grid math, bead assignment, layout and
the baked PLY cloud (beads + cube-edge dots + dot-matrix labels) all run and
test headless. Rhino-side rendering of the grid/layout lives in display.py.

A "section" is one occupied cube cell of the grid:
    {'code': str,            # e.g. "X0_Y1_Z2" — cell index per axis
     'ijk': (i, j, k),
     'cube_min': (3,) array, # world min corner of the cell
     'n_beads': int,
     'curtains': [           # per curtain plane intersecting the cell
        {'plane_index': int, # index into the FULL model's curtain list
         'plane_x': float,
         'points_2d': (M, 2), 'points_3d': (M, 3), 'colors': (M, 3) | None}]}
"""

import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


def grid_from_bbox(bbox_min: np.ndarray, bbox_max: np.ndarray,
                   size: float, shift: Sequence[float]) -> dict:
    """Cube lattice covering [bbox_min, bbox_max].

    Cell faces sit at shift + n * size per axis (world-anchored, so the grid
    stays put when the model changes slightly); the shift params slide it.
    Returns {'origin': (3,), 'counts': (nx, ny, nz), 'size': float}.
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    origin = shift + size * np.floor((bbox_min - shift) / size)
    counts = np.floor((bbox_max - origin) / size).astype(np.int64) + 1
    return {"origin": origin, "counts": tuple(int(c) for c in counts),
            "size": float(size)}


def section_code(i: int, j: int, k: int) -> str:
    return f"X{i}_Y{j}_Z{k}"


def assign_sections(curtains: Sequence[dict], grid: dict) -> List[dict]:
    """Split each curtain's beads into the grid cells they fall in.

    Returns only the OCCUPIED cells, sorted by (k, j, i) — i.e. by Z level
    first, which is also the layout row order. Bead order within a section is
    deterministic (curtain order, then the sampler's fixed in-plane order).
    """
    size = grid["size"]
    ox, oy, oz = grid["origin"]
    nx, ny, nz = grid["counts"]

    buckets: Dict[Tuple[int, int, int], List[dict]] = {}
    for ci, c in enumerate(curtains):
        pts = c["points_2d"]
        if pts.shape[0] == 0:
            continue
        i = int(np.clip(np.floor((c["plane_x"] - ox) / size), 0, nx - 1))
        jj = np.clip(np.floor((pts[:, 0] - oy) / size), 0, ny - 1).astype(np.int64)
        kk = np.clip(np.floor((pts[:, 1] - oz) / size), 0, nz - 1).astype(np.int64)
        colors = c.get("colors")
        for j, k in sorted(set(zip(jj.tolist(), kk.tolist()))):
            mask = (jj == j) & (kk == k)
            buckets.setdefault((i, j, k), []).append({
                "plane_index": ci,
                "plane_x": float(c["plane_x"]),
                "points_2d": pts[mask],
                "points_3d": c["points_3d"][mask],
                "colors": colors[mask] if colors is not None else None,
            })

    sections: List[dict] = []
    for (i, j, k) in sorted(buckets, key=lambda t: (t[2], t[1], t[0])):
        plates = buckets[(i, j, k)]
        sections.append({
            "code": section_code(i, j, k),
            "ijk": (i, j, k),
            "cube_min": np.array([ox + i * size, oy + j * size, oz + k * size]),
            "n_beads": int(sum(p["points_2d"].shape[0] for p in plates)),
            "curtains": plates,
        })
    return sections


def layout_positions(sections: Sequence[dict], model_bbox_min: np.ndarray,
                     model_bbox_max: np.ndarray, params) -> Dict[str, np.ndarray]:
    """2D wall of section copies beside the model, read in the Right view.

    Rows correspond to the section's original Z level (ground level = bottom
    row), columns enumerate the sections within a level ordered by (j, i).
    The wall starts section_layout_margin beyond the model's +Y extent and
    sits on Z=0; copies keep the model's min-X slab. Returns code -> world
    min corner of the copied cube.
    """
    pitch = float(params.section_layout_spacing)
    x0 = float(model_bbox_min[0])
    y0 = float(model_bbox_max[1]) + float(params.section_layout_margin)

    levels = sorted({s["ijk"][2] for s in sections})
    row_of = {k: r for r, k in enumerate(levels)}
    next_col: Dict[int, int] = {}
    layout: Dict[str, np.ndarray] = {}
    # `sections` arrives sorted (k, j, i) from assign_sections; iterate as-is
    # so columns within a row read in (j, i) order.
    for s in sections:
        k = s["ijk"][2]
        col = next_col.get(k, 0)
        next_col[k] = col + 1
        layout[s["code"]] = np.array(
            [x0, y0 + col * pitch, row_of[k] * pitch], dtype=np.float64)
    return layout


# --- dot-matrix labels + cube edges (for the points-only PLY/web viewer) ----

# 5x7 bitmap font, top row first. Covers the section-code alphabet only.
_FONT_5X7 = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
}


def dot_text_points(text: str, height: float) -> np.ndarray:
    """Dot positions for `text` as an (N, 2) array of (u, v), u rightward and
    v up from the baseline-left of the first character. Characters outside the
    font are skipped. Dot pitch is height / 7; advance is 6 dots per char."""
    step = height / 7.0
    pts: List[List[float]] = []
    for ci, ch in enumerate(text):
        rows = _FONT_5X7.get(ch)
        if rows is None:
            continue
        u0 = ci * 6 * step
        for r, row in enumerate(rows):          # r = 0 is the TOP row
            for cidx, bit in enumerate(row):
                if bit == "1":
                    pts.append([u0 + cidx * step, (6 - r) * step])
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(pts, dtype=np.float64)


def cube_edge_points(size: float, spacing: float) -> np.ndarray:
    """Dots along the 12 edges of a size-cube with min corner at the origin,
    spaced <= `spacing` apart (corners shared by edges repeat — harmless)."""
    n = max(2, int(np.ceil(size / spacing)) + 1)
    t = np.linspace(0.0, size, n)
    edges = []
    for a in (0.0, size):
        for b in (0.0, size):
            ca = np.full_like(t, a)
            cb = np.full_like(t, b)
            edges.append(np.column_stack([t, ca, cb]))   # 4 edges along X
            edges.append(np.column_stack([ca, t, cb]))   # 4 edges along Y
            edges.append(np.column_stack([ca, cb, t]))   # 4 edges along Z
    return np.vstack(edges)


_BOX_RGB = (90, 90, 90)
_LABEL_RGB = (255, 255, 255)


def layout_beads(sections: Sequence[dict], layout: Dict[str, np.ndarray],
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Every section copy's beads translated to its layout slot — beads ONLY,
    no box/label markers (Rhino draws those as curves and TextDots; this feeds
    the sprite conduit). Returns (points (N, 3) float64, colors (N, 3) uint8;
    colourless plates fall back to white)."""
    pts_list: List[np.ndarray] = []
    col_list: List[np.ndarray] = []
    for s in sections:
        shift = layout[s["code"]] - s["cube_min"]
        for p in s["curtains"]:
            m = p["points_3d"].shape[0]
            if m == 0:
                continue
            pts_list.append(p["points_3d"] + shift)
            if p["colors"] is not None:
                col_list.append(p["colors"])
            else:
                col_list.append(np.full((m, 3), 255, dtype=np.uint8))
    if not pts_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(pts_list), np.vstack(col_list).astype(np.uint8)


def bake_layout_cloud(sections: Sequence[dict], layout: Dict[str, np.ndarray],
                      params) -> Tuple[np.ndarray, np.ndarray]:
    """The whole sections layout as one bead cloud for the web viewer:
    every section's beads translated to its layout slot, plus the cube edges
    (gray dots) and the section-code label (white dots, in the YZ plane below
    each cube so it reads when viewed along the X axis).

    Returns (points (N, 3) float64, colors (N, 3) uint8).
    """
    size = float(params.section_size)
    box_off = cube_edge_points(size, 4.0 * float(params.bead_diameter))
    label_h = 0.12 * size
    label_gap = 0.5 * label_h

    bead_pts, bead_cols = layout_beads(sections, layout)
    pts_list: List[np.ndarray] = [bead_pts]
    col_list: List[np.ndarray] = [bead_cols]
    for s in sections:
        tmin = layout[s["code"]]
        pts_list.append(tmin + box_off)
        col_list.append(np.full((box_off.shape[0], 3), _BOX_RGB, dtype=np.uint8))

        uv = dot_text_points(s["code"], label_h)
        if uv.shape[0]:
            label = np.column_stack([
                np.full(uv.shape[0], tmin[0]),
                tmin[1] + uv[:, 0],
                tmin[2] - label_gap - label_h + uv[:, 1],
            ])
            pts_list.append(label)
            col_list.append(np.full((uv.shape[0], 3), _LABEL_RGB, dtype=np.uint8))

    if not pts_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(pts_list), np.vstack(col_list).astype(np.uint8)


if __name__ == "__main__":
    # Headless smoke test: synthetic curtains spanning two cells per axis.
    from pearlscape.params import PearlscapeParams

    params = PearlscapeParams()
    params.section_size = 1000.0
    params.section_grid_shift = (0.0, 0.0, 0.0)

    # grid_from_bbox: snapping + shift
    g = grid_from_bbox(np.array([50.0, -900.0, 0.0]),
                       np.array([1950.0, 900.0, 2500.0]), 1000.0, (0.0, 0.0, 0.0))
    assert np.allclose(g["origin"], [0.0, -1000.0, 0.0]), g["origin"]
    assert g["counts"] == (2, 2, 3), g["counts"]
    g2 = grid_from_bbox(np.array([50.0, -900.0, 0.0]),
                        np.array([1950.0, 900.0, 2500.0]), 1000.0, (100.0, 0.0, 0.0))
    assert np.allclose(g2["origin"], [-900.0, -1000.0, 0.0]), g2["origin"]
    assert g2["counts"] == (3, 2, 3), g2["counts"]

    # Two curtains; the second has beads in two Z cells.
    def curtain(plane_x, yz):
        yz = np.asarray(yz, dtype=np.float64)
        pts3 = np.column_stack([np.full(yz.shape[0], plane_x), yz[:, 0], yz[:, 1]])
        cols = np.tile(np.array([[10, 20, 30]], dtype=np.uint8), (yz.shape[0], 1))
        return {"plane_x": plane_x, "points_2d": yz, "points_3d": pts3, "colors": cols}

    curtains = [
        curtain(100.0, [[100.0, 100.0], [200.0, 200.0]]),
        curtain(1500.0, [[100.0, 100.0], [100.0, 1500.0], [300.0, 1600.0]]),
    ]
    grid = grid_from_bbox(np.array([100.0, 100.0, 100.0]),
                          np.array([1500.0, 300.0, 1600.0]), 1000.0, (0, 0, 0))
    secs = assign_sections(curtains, grid)
    codes = [s["code"] for s in secs]
    assert codes == ["X0_Y0_Z0", "X1_Y0_Z0", "X1_Y0_Z1"], codes   # sorted (k, j, i)
    assert secs[0]["n_beads"] == 2 and secs[1]["n_beads"] == 1 and secs[2]["n_beads"] == 2
    assert secs[2]["curtains"][0]["plane_index"] == 1
    assert np.allclose(secs[2]["cube_min"], [1000.0, 0.0, 1000.0])
    # plate points carry their colours
    assert secs[0]["curtains"][0]["colors"].shape == (2, 3)

    # determinism
    secs_b = assign_sections(curtains, grid)
    for a, b in zip(secs, secs_b):
        assert a["code"] == b["code"]
        for pa, pb in zip(a["curtains"], b["curtains"]):
            assert np.array_equal(pa["points_3d"], pb["points_3d"])

    # layout: rows by Z level, columns in (j, i) order, wall beyond +Y
    lp = layout_positions(secs, np.array([0.0, -500.0, 0.0]),
                          np.array([2000.0, 500.0, 2000.0]), params)
    assert set(lp) == set(codes)
    y_wall = 500.0 + params.section_layout_margin
    assert np.allclose(lp["X0_Y0_Z0"], [0.0, y_wall, 0.0])
    assert np.allclose(lp["X1_Y0_Z0"],
                       [0.0, y_wall + params.section_layout_spacing, 0.0])
    assert np.allclose(lp["X1_Y0_Z1"], [0.0, y_wall, params.section_layout_spacing])

    # dot text: every section-code character renders; unknown chars skipped
    txt = dot_text_points("X12_Y0_Z9", 140.0)
    assert txt.shape[0] > 0 and txt.shape[1] == 2
    assert dot_text_points("??", 140.0).shape == (0, 2)
    assert txt[:, 1].min() >= 0.0 and txt[:, 1].max() <= 140.0

    # cube edges: 12 edges, all dots on the cube boundary
    edges = cube_edge_points(1000.0, 25.0)
    n_per = max(2, int(np.ceil(1000.0 / 25.0)) + 1)
    assert edges.shape == (12 * n_per, 3)
    on_face = np.isclose(edges, 0.0) | np.isclose(edges, 1000.0)
    assert np.all(on_face.sum(axis=1) >= 2), "edge dot not on a cube edge"

    # layout_beads: beads only, translated, colours carried through
    wall_pts, wall_cols = layout_beads(secs, lp)
    assert wall_pts.shape[0] == sum(s["n_beads"] for s in secs)
    assert wall_cols.shape == wall_pts.shape and wall_cols.dtype == np.uint8
    assert np.all(wall_cols == np.array([10, 20, 30], dtype=np.uint8))

    # baked cloud: beads + boxes + labels, colours aligned
    pts, cols = bake_layout_cloud(secs, lp, params)
    assert pts.shape[0] == cols.shape[0] and pts.shape[0] > 5
    assert cols.dtype == np.uint8
    n_beads = sum(s["n_beads"] for s in secs)
    assert np.count_nonzero((cols == np.array([10, 20, 30])).all(axis=1)) == n_beads
    assert np.count_nonzero((cols == np.array(_LABEL_RGB)).all(axis=1)) > 0
    # bake spaces box dots at 4 * bead_diameter — count against that, not `edges`
    bake_edges = cube_edge_points(params.section_size, 4.0 * params.bead_diameter)
    assert np.count_nonzero((cols == np.array(_BOX_RGB)).all(axis=1)) == len(secs) * bake_edges.shape[0]
    # beads landed inside their layout cubes
    bead_rows = (cols == np.array([10, 20, 30])).all(axis=1)
    bead_pts = pts[bead_rows]
    s_size = params.section_size
    ok = np.zeros(bead_pts.shape[0], dtype=bool)
    for code, tmin in lp.items():
        inside = np.all((bead_pts >= tmin - 1e-9) & (bead_pts <= tmin + s_size + 1e-9), axis=1)
        ok |= inside
    assert ok.all(), "a translated bead fell outside every layout cube"

    print(f"sections: {len(secs)} cells, {n_beads} beads, layout + bake OK")
    print("OK")
