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


# --- per-sheet fabrication data (strings + colour tally, for the PDF export) -

def _cluster_strings(points_2d: np.ndarray, tol: float) -> List[List[int]]:
    """Group bead ROW-INDICES into strings, ordered left -> right (ascending Y).

    Beads are sorted by Y; a new string starts whenever the Y gap to the previous
    bead exceeds `tol` (so a run of beads each within `tol` of the last forms one
    string — the same rule the old string_columns used). Empty in -> []."""
    n = points_2d.shape[0]
    if n == 0:
        return []
    order = np.argsort(points_2d[:, 0], kind="stable")
    ys = points_2d[order, 0]
    clusters: List[List[int]] = []
    cur = [int(order[0])]
    prev_y = float(ys[0])
    for t in range(1, n):
        y = float(ys[t])
        if y - prev_y <= tol:
            cur.append(int(order[t]))
        else:
            clusters.append(cur)
            cur = [int(order[t])]
        prev_y = y
    clusters.append(cur)
    return clusters


def string_columns(points_2d: np.ndarray, tol: float) -> np.ndarray:
    """Representative Y of each vertical string in a curtain plane.

    Beads physically hang on shared vertical strings (see params.string_align):
    beads whose Y positions sit within `tol` of each other belong to one string.
    Returns the sorted (1,) array of one Y per string (the cluster mean), so the
    PDF export can draw a vertical line per string. Empty in -> empty out."""
    clusters = _cluster_strings(points_2d, tol)
    if not clusters:
        return np.zeros((0,), dtype=np.float64)
    means = [float(np.mean(points_2d[idxs, 0])) for idxs in clusters]
    return np.array(means, dtype=np.float64)


def string_layout(points_2d: np.ndarray, colors, cube_min, section_size: float,
                  tol: float, letters: Dict[Tuple[int, int, int], str]) -> List[dict]:
    """Per-string fabrication data for ONE curtain plane within a section cube.

    Returns a list ordered left -> right (one dict per string):
        {'number': int,        # 1-based, resets per call (i.e. per curtain)
         'offset_mm': int,      # round(mean_Y - cube_min_y), from the left edge
         'beads': [(letter, pos_mm), ...]}   # top -> bottom

    pos_mm = round((cube_min_z + section_size) - Z), i.e. distance DOWN from the
    cube's top edge. `letters` maps rgb -> placeholder letter (palette_letters);
    a colour not in the map, or a colourless plane, yields '?'."""
    y0 = float(cube_min[1])
    z_top = float(cube_min[2]) + float(section_size)
    clusters = _cluster_strings(points_2d, tol)
    out: List[dict] = []
    for number, idxs in enumerate(clusters, start=1):
        mean_y = float(np.mean(points_2d[idxs, 0]))
        zs = points_2d[idxs, 1]
        order = np.argsort(zs, kind="stable")[::-1]    # top (high Z) -> bottom
        beads: List[Tuple[str, int]] = []
        for bi in order:
            ridx = idxs[int(bi)]
            if colors is not None:
                rgb = (int(colors[ridx, 0]), int(colors[ridx, 1]), int(colors[ridx, 2]))
                letter = letters.get(rgb, "?")
            else:
                letter = "?"
            pos_mm = int(round(z_top - float(points_2d[ridx, 1])))
            beads.append((letter, pos_mm))
        out.append({"number": number,
                    "offset_mm": int(round(mean_y - y0)),
                    "beads": beads})
    return out


def curtain_summary(colors, letters: Dict[Tuple[int, int, int], str]) -> dict:
    """Bead total + per-letter counts for ONE curtain plane, most-common first.

    Returns {'n_beads': int, 'by_letter': [(letter, count), ...]}. Built on
    color_counts, so ordering/tie-breaks match the title-strip legend. A
    colourless plane -> {'n_beads': 0, 'by_letter': []}."""
    cc = color_counts(colors)        # [((r, g, b), count), ...] most-common first
    by_letter = [(letters.get(rgb, "?"), int(count)) for rgb, count in cc]
    return {"n_beads": int(sum(count for _, count in cc)), "by_letter": by_letter}


def wrap_text(text: str, max_chars: int) -> List[str]:
    """Word-wrap `text` to lines of at most `max_chars` characters, breaking at
    spaces. A single word longer than `max_chars` is hard-split so nothing is
    ever lost. Empty text -> ['']. Used by the PDF document so long string lines
    wrap instead of running off the page."""
    if max_chars < 1:
        max_chars = 1
    lines: List[str] = []
    cur = ""
    for w in text.split(" "):
        cand = w if not cur else cur + " " + w
        if len(cand) <= max_chars:
            cur = cand
            continue
        if cur:
            lines.append(cur)
            cur = ""
        while len(w) > max_chars:
            lines.append(w[:max_chars])
            w = w[max_chars:]
        cur = w
    if cur or not lines:
        lines.append(cur)
    return lines


def section_document_rows(section, params) -> List[dict]:
    """Logical rows for the section fabrication document, shared by the PDF and
    the .txt export so the two never drift. Each row is
        {'text': str, 'swatch': (r,g,b)|None, 'indent': int, 'checkbox': bool}
    where `indent` is a hierarchy LEVEL (0 = flush, 1 = nested), not millimetres.
    Legend rows carry a swatch colour; string rows carry an (empty) fabricator
    checkbox. Pure data — no Rhino."""
    from pearlscape import palettes
    letters = palette_letters(params.palette)
    section_size = float(params.section_size)
    string_tol = (float(params.string_align_overlap)
                  if params.string_align_overlap > 0.0 else float(params.bead_diameter))
    rows: List[dict] = []

    def add(text="", swatch=None, indent=0, checkbox=False):
        rows.append({"text": text, "swatch": swatch, "indent": indent,
                     "checkbox": checkbox})

    add(f"Section {section['code']}")
    add(f"Palette: {palettes.name_of(params.palette)}    "
        f"{int(section.get('n_beads', 0))} beads total")
    # Curtain count is beads-only — assign_sections never keeps an empty curtain.
    # Spacing comes from the plane_index gap, so skipped (empty) curtains in the
    # middle don't distort it.
    beaded = section["curtains"]
    if len(beaded) >= 2 and (beaded[-1]["plane_index"] - beaded[0]["plane_index"]):
        dx = float(beaded[-1]["plane_x"] - beaded[0]["plane_x"])
        di = int(beaded[-1]["plane_index"] - beaded[0]["plane_index"])
        add(f"Curtains (with beads): {len(beaded)}    "
            f"curtain spacing: {int(round(dx / di))}mm")
    else:
        add(f"Curtains (with beads): {len(beaded)}")
    add()
    add("Colour key (placeholder letters):")
    for c in params.palette:
        rgb = (int(c[0]), int(c[1]), int(c[2]))
        add(f"{letters.get(rgb, '?')} = RGB({rgb[0]}, {rgb[1]}, {rgb[2]})",
            swatch=rgb, indent=1)
    add()
    add("Bead layout: each bead's number is its offset in mm from the TOP of the section.")
    add()

    for plane in section["curtains"]:
        colors = plane["colors"]
        summ = curtain_summary(colors, letters)
        tally = ", ".join(f"{ltr} {cnt}" for ltr, cnt in summ["by_letter"])
        head = f"Curtain C{plane['plane_index']:03d} - {summ['n_beads']} beads"
        if tally:
            head += f": {tally}"
        add(head)
        layout = string_layout(plane["points_2d"], colors, section["cube_min"],
                               section_size, string_tol, letters)
        for s in layout:
            n = len(s["beads"])
            beads = ", ".join(f"{ltr}-{pos}mm" for ltr, pos in s["beads"])
            noun = "bead" if n == 1 else "beads"
            add(f"String {s['number']:03d} - {s['offset_mm']}mm - "
                f"{n} {noun}, layout: {beads} ///",
                indent=1, checkbox=True)
        add()
    return rows


def section_document_text(rows) -> List[str]:
    """Render section_document_rows as plain-text lines for the .txt export:
    each `indent` level becomes a two-space step and string rows get a '[ ] '
    checkbox. Lines are NOT wrapped (the text editor handles that). Blank rows
    stay empty (no trailing spaces)."""
    out: List[str] = []
    for r in rows:
        if not r["text"]:
            out.append("")
            continue
        prefix = "  " * int(r["indent"])
        if r["checkbox"]:
            prefix += "[ ] "
        out.append(prefix + r["text"])
    return out


def color_counts(colors) -> List[Tuple[Tuple[int, int, int], int]]:
    """Bead tally per colour as [((r, g, b), count), ...], most-common first.

    Bead colours are exact discrete palette entries, so unique RGB == palette
    colour. Ties break by RGB so the order is deterministic. None/empty -> []."""
    if colors is None or colors.shape[0] == 0:
        return []
    rgb = np.asarray(colors).reshape(-1, 3).astype(np.int64)
    uniq, counts = np.unique(rgb, axis=0, return_counts=True)
    order = sorted(range(uniq.shape[0]),
                   key=lambda i: (-int(counts[i]), tuple(int(v) for v in uniq[i])))
    return [(tuple(int(v) for v in uniq[i]), int(counts[i])) for i in order]


_PLACEHOLDER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def palette_letters(palette) -> Dict[Tuple[int, int, int], str]:
    """Map each palette RGB to a placeholder letter by palette order:
    palette[0] -> 'A', palette[1] -> 'B', ... The same colour gets the same
    letter on every sheet and in every document. Palettes hold <= 6 colours, so
    'Z' is never reached; a colour past the alphabet maps to '?'. Duplicate RGBs
    keep their first (lowest-index) letter. Returns {(r, g, b): letter}."""
    out: Dict[Tuple[int, int, int], str] = {}
    for i, c in enumerate(palette):
        rgb = (int(c[0]), int(c[1]), int(c[2]))
        if rgb not in out:
            out[rgb] = _PLACEHOLDER_LETTERS[i] if i < len(_PLACEHOLDER_LETTERS) else "?"
    return out


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

    # string_columns: near-equal Y merge into one string; far ones stay split
    cols = string_columns(np.array([[10.0, 0.0], [10.4, 5.0], [50.0, 1.0],
                                     [10.2, 9.0], [49.6, 3.0]]), tol=1.0)
    assert cols.shape == (2,), cols
    assert np.isclose(cols[0], (10.0 + 10.4 + 10.2) / 3.0) and np.isclose(cols[1], 49.8)
    assert string_columns(np.zeros((0, 2)), 1.0).shape == (0,)

    # color_counts: tally by exact RGB, most-common first, deterministic ties
    cc = color_counts(np.array([[1, 2, 3], [1, 2, 3], [9, 9, 9], [1, 2, 3], [4, 5, 6]],
                               dtype=np.uint8))
    assert cc[0] == ((1, 2, 3), 3), cc
    assert {c for c, _ in cc} == {(1, 2, 3), (9, 9, 9), (4, 5, 6)}
    assert sum(n for _, n in cc) == 5
    assert color_counts(None) == [] and color_counts(np.zeros((0, 3))) == []

    # palette_letters: letters by palette order, dupes keep first, lookups miss -> caller's default
    pl = palette_letters([(10, 20, 30), (40, 50, 60), (70, 80, 90), (10, 20, 30)])
    assert pl[(10, 20, 30)] == "A" and pl[(40, 50, 60)] == "B" and pl[(70, 80, 90)] == "C"
    assert len(pl) == 3, pl                     # duplicate (10,20,30) not re-added
    assert pl.get((1, 1, 1), "?") == "?"        # colour not in palette

    # string_layout: two strings, numbered left->right, beads top->bottom, mm from top
    sl_letters = palette_letters([(10, 20, 30), (40, 50, 60)])   # A, B
    sl_pts = np.array([[100.0, 100.0],    # string 1 (Y~100): low bead
                       [100.4, 800.0],    # string 1: high bead (top)
                       [500.0, 400.0]])   # string 2 (Y~500)
    sl_cols = np.array([[10, 20, 30],     # A
                        [40, 50, 60],     # B
                        [40, 50, 60]], dtype=np.uint8)   # B
    sl = string_layout(sl_pts, sl_cols, cube_min=np.array([0.0, 0.0, 0.0]),
                       section_size=1000.0, tol=1.0, letters=sl_letters)
    assert [s["number"] for s in sl] == [1, 2], sl
    assert sl[0]["offset_mm"] == 100 and sl[1]["offset_mm"] == 500, sl
    # string 1 beads top->bottom: Z=800 (B, 1000-800=200mm) then Z=100 (A, 900mm)
    assert sl[0]["beads"] == [("B", 200), ("A", 900)], sl[0]["beads"]
    assert sl[1]["beads"] == [("B", 600)], sl[1]["beads"]
    # colourless plane -> '?' letters, still positioned
    sl_none = string_layout(sl_pts, None, np.array([0.0, 0.0, 0.0]), 1000.0, 1.0, sl_letters)
    assert all(ltr == "?" for s in sl_none for ltr, _ in s["beads"]), sl_none
    assert string_layout(np.zeros((0, 2)), None, np.zeros(3), 1000.0, 1.0, sl_letters) == []

    # curtain_summary: total + per-letter, most-common first
    cs_letters = palette_letters([(1, 2, 3), (4, 5, 6), (9, 9, 9)])   # A, B, C
    cs = curtain_summary(np.array([[1, 2, 3], [1, 2, 3], [9, 9, 9], [1, 2, 3], [4, 5, 6]],
                                  dtype=np.uint8), cs_letters)
    assert cs["n_beads"] == 5, cs
    assert cs["by_letter"][0] == ("A", 3), cs            # most common first
    assert dict(cs["by_letter"]) == {"A": 3, "C": 1, "B": 1}, cs
    assert curtain_summary(None, cs_letters) == {"n_beads": 0, "by_letter": []}

    # wrap_text: word wrap at spaces, hard-split overlong words, empty -> ['']
    assert wrap_text("a bb ccc", 5) == ["a bb", "ccc"], wrap_text("a bb ccc", 5)
    assert wrap_text("short", 80) == ["short"]
    assert wrap_text("", 10) == [""]
    assert wrap_text("abcdefgh", 3) == ["abc", "def", "gh"]      # overlong single word
    _long = "String 001 - offset 236mm - 2 beads, layout: A 100mm, B 200mm /"
    _w = wrap_text(_long, 25)
    assert all(len(_l) <= 25 for _l in _w), _w                   # every line fits
    assert " ".join(_w).split() == _long.split()                # no words dropped/added

    # section_document_rows + section_document_text: structure + txt formatting
    sdr = section_document_rows(secs[0], params)
    assert sdr[0]["text"].startswith("Section "), sdr[0]
    assert any(r["swatch"] is not None for r in sdr), "no legend swatch rows"
    assert any(r["checkbox"] for r in sdr), "no string rows carry a checkbox"
    assert all(not r["checkbox"] for r in sdr
               if r["text"].startswith(("Curtain ", "Section ", "Palette"))), sdr
    sdt = section_document_text(sdr)
    assert len(sdt) == len(sdr)                                   # one text line per row
    assert any(l.startswith("  [ ] String ") for l in sdt), sdt  # checkbox + nested indent
    assert "" in sdt                                             # blank separator rows survive

    # string line format: no "offset", dashed bead codes, triple-slash end, singular noun
    _str_lines = [l for l in sdt if "String " in l]
    assert _str_lines and all(l.rstrip().endswith("///") for l in _str_lines), _str_lines
    assert all("offset" not in l for l in _str_lines), _str_lines
    assert all("-" in l.split("layout:")[1] for l in _str_lines), _str_lines  # dashed codes
    assert any(" 1 bead," in l for l in _str_lines), _str_lines               # singular

    # header: beads-only curtain count + inter-curtain spacing from plane_index gaps
    _multi = {"code": "X0_Y0_Z0", "cube_min": np.array([0.0, 0.0, 0.0]), "n_beads": 3,
              "curtains": [
                  {"plane_index": 5, "plane_x": 50.0,
                   "points_2d": np.array([[10.0, 10.0]]), "colors": None},
                  {"plane_index": 8, "plane_x": 80.0,
                   "points_2d": np.array([[10.0, 10.0], [20.0, 20.0]]), "colors": None}]}
    _hrow = next(r["text"] for r in section_document_rows(_multi, params)
                 if r["text"].startswith("Curtains"))
    assert "2" in _hrow and "10mm" in _hrow, _hrow      # 2 curtains, (80-50)/(8-5)=10mm

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
