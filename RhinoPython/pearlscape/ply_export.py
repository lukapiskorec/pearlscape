"""Write a bead point cloud to a binary little-endian PLY file.

Pure numpy + file I/O — no Rhino imports, so it runs (and tests) headless. The
viewer reads the `comment bead_diameter` / `comment color_dither` lines to size
beads in true world units and re-quantize colours.

Base layout (one `vertex` element, 15 bytes each):
    property float x / y / z          (12 bytes, little-endian)
    property uchar red / green / blue  (3 bytes)   # baked default-palette colour

When `field` is given, two extra properties (5 bytes) let a viewer re-colour
against any palette without re-running the noise:
    property float field               (4 bytes)   # [0,1) colour-field value
    property uchar dither              (1 byte)    # dither source, quantized *255
"""

import os

import numpy as np


# Packed (unaligned) so the on-disk stride matches PLY's tight layout exactly.
_BASE_FIELDS = [
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
]
_FIELD_FIELDS = [("field", "<f4"), ("dither", "u1")]

_VERTEX_DTYPE = np.dtype(_BASE_FIELDS)              # 15 bytes (base only)
_VERTEX_DTYPE_FIELD = np.dtype(_BASE_FIELDS + _FIELD_FIELDS)  # 20 bytes


def write_ply(path, points_xyz, colors_rgb, bead_diameter,
              field=None, dither_rand=None, color_dither=0.0):
    """Write `points_xyz` (N, 3 float) + `colors_rgb` (N, 3 uint8 | None) to a
    binary PLY at `path`. `bead_diameter` (mm) is stored as a header comment.

    If `field` (N,) and `dither_rand` (N,) are given, they are written as extra
    per-vertex properties (with `color_dither` recorded as a header comment) so a
    viewer can re-quantize colours against any palette. Returns the path written.
    """
    points_xyz = np.asarray(points_xyz)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N, 3); got {points_xyz.shape}")
    n = points_xyz.shape[0]

    if colors_rgb is None:
        colors_rgb = np.full((n, 3), 255, dtype=np.uint8)
    else:
        colors_rgb = np.asarray(colors_rgb)
        if colors_rgb.shape != (n, 3):
            raise ValueError(
                f"colors_rgb must match points: ({n}, 3); got {colors_rgb.shape}"
            )

    has_field = field is not None
    if has_field:
        field = np.asarray(field)
        dither_rand = np.asarray(dither_rand)
        if field.shape != (n,) or dither_rand.shape != (n,):
            raise ValueError(
                f"field and dither_rand must be ({n},); got {field.shape}, "
                f"{dither_rand.shape}"
            )

    dtype = _VERTEX_DTYPE_FIELD if has_field else _VERTEX_DTYPE
    verts = np.empty(n, dtype=dtype)
    verts["x"] = points_xyz[:, 0]
    verts["y"] = points_xyz[:, 1]
    verts["z"] = points_xyz[:, 2]
    verts["red"] = colors_rgb[:, 0]
    verts["green"] = colors_rgb[:, 1]
    verts["blue"] = colors_rgb[:, 2]
    if has_field:
        verts["field"] = field.astype("<f4")
        verts["dither"] = np.clip(np.round(dither_rand * 255.0), 0, 255).astype("u1")

    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment bead_diameter {float(bead_diameter):g}",
    ]
    if has_field:
        lines.append(f"comment color_dither {float(color_dither):g}")
    lines += [
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if has_field:
        lines += ["property float field", "property uchar dither"]
    lines.append("end_header\n")
    header = "\n".join(lines)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        verts.tofile(f)
    return path


if __name__ == "__main__":
    # Headless round-trip: write a small cloud, read the header + binary back, and
    # assert the points/colours survive exactly. No Rhino needed.
    import struct
    import tempfile

    rng = np.random.default_rng(0)
    pts = rng.uniform(-1000.0, 1000.0, size=(1000, 3)).astype(np.float32)
    cols = rng.integers(0, 256, size=(1000, 3), dtype=np.uint8)

    path = os.path.join(tempfile.gettempdir(), "pearlscape_ply_test.ply")
    write_ply(path, pts, cols, bead_diameter=6.0)

    with open(path, "rb") as f:
        raw = f.read()

    end_token = b"end_header\n"
    hdr_end = raw.index(end_token) + len(end_token)
    header_txt = raw[:hdr_end].decode("ascii")
    assert "format binary_little_endian 1.0" in header_txt
    assert "comment bead_diameter 6" in header_txt
    assert "element vertex 1000" in header_txt

    body = np.frombuffer(raw[hdr_end:], dtype=_VERTEX_DTYPE)
    assert body.shape[0] == 1000, body.shape[0]
    back_xyz = np.column_stack([body["x"], body["y"], body["z"]])
    back_rgb = np.column_stack([body["red"], body["green"], body["blue"]])
    assert np.array_equal(back_xyz, pts), "positions did not round-trip"
    assert np.array_equal(back_rgb, cols), "colours did not round-trip"

    # bytes-per-vertex must be exactly 15 (no struct padding).
    assert _VERTEX_DTYPE.itemsize == 15, _VERTEX_DTYPE.itemsize
    assert len(raw) - hdr_end == 1000 * 15

    # colors_rgb=None -> all white.
    path2 = os.path.join(tempfile.gettempdir(), "pearlscape_ply_white.ply")
    write_ply(path2, pts, None, bead_diameter=8.0)
    with open(path2, "rb") as f:
        raw2 = f.read()
    body2 = np.frombuffer(raw2[raw2.index(end_token) + len(end_token):], dtype=_VERTEX_DTYPE)
    assert np.all(body2["red"] == 255) and np.all(body2["green"] == 255)

    # Extended format: field + dither properties round-trip, 20 bytes each.
    field = rng.random(1000).astype(np.float64)
    dr = rng.random(1000)
    path3 = os.path.join(tempfile.gettempdir(), "pearlscape_ply_field.ply")
    write_ply(path3, pts, cols, bead_diameter=6.0, field=field, dither_rand=dr,
              color_dither=1.0)
    with open(path3, "rb") as f:
        raw3 = f.read()
    hdr3 = raw3[:raw3.index(end_token) + len(end_token)].decode("ascii")
    assert "comment color_dither 1" in hdr3
    assert "property float field" in hdr3 and "property uchar dither" in hdr3
    body3 = np.frombuffer(raw3[raw3.index(end_token) + len(end_token):], dtype=_VERTEX_DTYPE_FIELD)
    assert _VERTEX_DTYPE_FIELD.itemsize == 20, _VERTEX_DTYPE_FIELD.itemsize
    assert np.allclose(body3["field"], field.astype("<f4")), "field did not round-trip"
    # dither stored as uint8 *255: within one quantization step.
    assert np.all(np.abs(body3["dither"].astype(float) / 255.0 - dr) <= 1.0 / 255.0 + 1e-6)

    print(f"PLY round-trip OK: base {_VERTEX_DTYPE.itemsize}B, "
          f"extended {_VERTEX_DTYPE_FIELD.itemsize}B; header + binary parse back identical.")
    print("OK")
