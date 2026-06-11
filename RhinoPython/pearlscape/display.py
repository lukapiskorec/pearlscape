"""Render bead positions into the Rhino document.

Two modes:
- pointcloud: one Rhino.Geometry.PointCloud per curtain (fast, default).
- instances: one InstanceReference per bead (heavy, for rendering — added later).

At this task we only implement the PointCloud mode and a single-cloud
helper used to visualize the raw cave during early development.
"""

from typing import List, Optional, Sequence

import numpy as np

import Rhino
import Rhino.Geometry as rg
import Rhino.Display as rd
import scriptcontext as sc
import System.Drawing as sd


PEARLSCAPE_PARENT_LAYER = "Pearlscape"
CURTAINS_LAYER = "Curtains"
CAVE_REFERENCE_LAYER = "CaveReference"
CURTAIN_PLANES_LAYER = "CurtainPlanes"
CAVE_SURFACE_LAYER = "CaveSurface"
CAVE_POINTS_LAYER = "CavePoints"
SECTIONS_LAYER = "Sections"


def _ensure_layer(path: str, color: Optional[sd.Color] = None) -> int:
    """Ensure a layer at `path` (e.g. 'Pearlscape::Curtains::Curtain_00') exists.
    Returns the layer index."""
    doc = sc.doc
    layer_table = doc.Layers
    idx = layer_table.FindByFullPath(path, -1)
    if idx >= 0:
        return idx
    parts = path.split("::")
    parent_id = None
    current = ""
    for p in parts:
        current = p if not current else f"{current}::{p}"
        i = layer_table.FindByFullPath(current, -1)
        if i < 0:
            layer = Rhino.DocObjects.Layer()
            layer.Name = p
            if parent_id is not None:
                layer.ParentLayerId = parent_id
            if color is not None and current == path:
                layer.Color = color
            i = layer_table.Add(layer)
        parent_id = layer_table[i].Id
    return layer_table.FindByFullPath(path, -1)


def _np_to_point3d_list(pts: np.ndarray) -> List[rg.Point3d]:
    return [rg.Point3d(float(p[0]), float(p[1]), float(p[2])) for p in pts]


def _np_to_color_list(colors: Optional[np.ndarray]) -> Optional[List[sd.Color]]:
    if colors is None:
        return None
    return [sd.Color.FromArgb(int(c[0]), int(c[1]), int(c[2])) for c in colors]


def _projected_points(c: dict) -> np.ndarray:
    """Project a curtain's 2D bead positions onto its plane: (plane_x, y, z)."""
    plane_x = c["plane_x"]
    pts_2d = c["points_2d"]
    return np.column_stack([
        np.full(pts_2d.shape[0], plane_x),
        pts_2d[:, 0],
        pts_2d[:, 1],
    ])


def add_pointcloud(
    points: np.ndarray,
    layer_path: str,
    colors: Optional[np.ndarray] = None,
):
    """Add a single PointCloud to the document under the given layer path."""
    doc = sc.doc
    layer_idx = _ensure_layer(layer_path)
    cloud = rg.PointCloud()
    pt_list = _np_to_point3d_list(points)
    color_list = _np_to_color_list(colors)
    if color_list is None:
        for pt in pt_list:
            cloud.Add(pt)
    else:
        for pt, c in zip(pt_list, color_list):
            cloud.Add(pt, c)
    attrs = Rhino.DocObjects.ObjectAttributes()
    attrs.LayerIndex = layer_idx
    return doc.Objects.AddPointCloud(cloud, attrs)


def render_cave_reference(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """Render the raw, un-sliced cave as a single PointCloud on the
    CaveReference layer. Pass `colors` (N, 3 uint8) for per-point colour."""
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CAVE_REFERENCE_LAYER}"
    add_pointcloud(points, layer_path, colors=colors)
    sc.doc.Views.Redraw()


def render_pointclouds(curtains: Sequence[dict]) -> None:
    """Render per-curtain PointClouds.

    Each item in `curtains` is the dict produced by curtains.build_curtains:
        { 'plane_x': float, 'points_2d': np.ndarray (M, 2),
          'points_3d': np.ndarray (M, 3), 'colors': np.ndarray (M, 3) | None }
    The displayed point is (plane_x, y, z) — i.e. the projected position.
    """
    for i, c in enumerate(curtains):
        layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CURTAINS_LAYER}::Curtain_{i:02d}"
        add_pointcloud(_projected_points(c), layer_path, colors=c.get("colors"))
    sc.doc.Views.Redraw()


def render_curtain_planes(
    plane_xs: Sequence[float],
    width: float,
    height: float,
) -> None:
    """Render the curtain rectangles as outlines for visual reference."""
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CURTAIN_PLANES_LAYER}"
    layer_idx = _ensure_layer(layer_path)
    doc = sc.doc
    half_w = width / 2.0
    for x in plane_xs:
        corners = [
            rg.Point3d(x, -half_w, 0.0),
            rg.Point3d(x,  half_w, 0.0),
            rg.Point3d(x,  half_w, height),
            rg.Point3d(x, -half_w, height),
            rg.Point3d(x, -half_w, 0.0),
        ]
        polyline = rg.Polyline(corners)
        attrs = Rhino.DocObjects.ObjectAttributes()
        attrs.LayerIndex = layer_idx
        doc.Objects.AddPolyline(polyline, attrs)
    sc.doc.Views.Redraw()


# --- Sectioning (fabrication) ------------------------------------------------
# Everything sections-related lives under Pearlscape::Sections and is cleared
# at the start of every sections run, so F5 reruns (e.g. after shifting the
# grid) never stack stale geometry.


def clear_sections_layers() -> None:
    """Delete all objects AND layers under (and including) Pearlscape::Sections.

    Section codes change when the grid shifts, so stale per-code sublayers must
    go too — deleting the whole subtree is the only clean slate."""
    doc = sc.doc
    root = f"{PEARLSCAPE_PARENT_LAYER}::{SECTIONS_LAYER}"
    doomed = [layer for layer in doc.Layers
              if not layer.IsDeleted and (layer.FullPath == root or
                                          layer.FullPath.startswith(root + "::"))]
    for layer in doomed:
        for obj in doc.Objects.FindByLayer(layer) or []:
            doc.Objects.Delete(obj, True)
    # Children before parents, or the parent delete fails on non-empty.
    for layer in sorted(doomed, key=lambda l: l.FullPath.count("::"), reverse=True):
        doc.Layers.Delete(layer.Index, True)


def _add_box_wireframe(layer_idx: int, min_pt, size: float) -> None:
    """12 cube edges as 2 rectangle polylines + 4 vertical lines (wireframe in
    every display mode — an AddBox brep would shade over the beads)."""
    doc = sc.doc
    x0, y0, z0 = float(min_pt[0]), float(min_pt[1]), float(min_pt[2])
    attrs = Rhino.DocObjects.ObjectAttributes()
    attrs.LayerIndex = layer_idx
    for z in (z0, z0 + size):
        ring = [
            rg.Point3d(x0, y0, z),
            rg.Point3d(x0 + size, y0, z),
            rg.Point3d(x0 + size, y0 + size, z),
            rg.Point3d(x0, y0 + size, z),
            rg.Point3d(x0, y0, z),
        ]
        doc.Objects.AddPolyline(rg.Polyline(ring), attrs)
    for dx in (0.0, size):
        for dy in (0.0, size):
            doc.Objects.AddLine(
                rg.Line(rg.Point3d(x0 + dx, y0 + dy, z0),
                        rg.Point3d(x0 + dx, y0 + dy, z0 + size)),
                attrs,
            )


def render_section_grid(grid: dict) -> None:
    """Wireframe cube for every cell of the section lattice, overlaid on the
    model (Pearlscape::Sections::Grid). `grid` is sections.grid_from_bbox()."""
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{SECTIONS_LAYER}::Grid"
    layer_idx = _ensure_layer(layer_path, color=sd.Color.FromArgb(128, 128, 128))
    size = grid["size"]
    ox, oy, oz = grid["origin"]
    nx, ny, nz = grid["counts"]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                _add_box_wireframe(
                    layer_idx,
                    (ox + i * size, oy + j * size, oz + k * size),
                    size,
                )
    sc.doc.Views.Redraw()


def render_section_layout(sections: Sequence[dict], layout: dict, size: float) -> None:
    """The review wall: each occupied cube copied to its layout slot — beads as
    a coloured PointCloud, the cube as a wireframe box, and the section code as
    a TextDot below it. One layer per section: Pearlscape::Sections::Layout::<code>."""
    doc = sc.doc
    for s in sections:
        code = s["code"]
        tmin = layout[code]
        shift = tmin - s["cube_min"]
        layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{SECTIONS_LAYER}::Layout::{code}"
        layer_idx = _ensure_layer(layer_path)

        plates = [p for p in s["curtains"] if p["points_3d"].shape[0]]
        if plates:
            pts = np.vstack([p["points_3d"] for p in plates]) + shift
            have_colors = all(p["colors"] is not None for p in plates)
            cols = np.vstack([p["colors"] for p in plates]) if have_colors else None
            add_pointcloud(pts, layer_path, colors=cols)

        _add_box_wireframe(layer_idx, tmin, size)

        attrs = Rhino.DocObjects.ObjectAttributes()
        attrs.LayerIndex = layer_idx
        dot_pt = rg.Point3d(float(tmin[0]), float(tmin[1]) + size / 2.0,
                            float(tmin[2]) - 0.06 * size)
        doc.Objects.AddTextDot(code, dot_pt, attrs)
    sc.doc.Views.Redraw()


def render_section_export_curtains(section: dict, size: float):
    """Document geometry for the PDF export of ONE section: each curtain plane
    inside the cube gets its own layer (…::Export::C<global plane index>) with
    that plane's beads plus the cube's YZ outline at the plane (so every PDF
    page carries the section frame). Returns [(plane_index, layer_path), ...]
    for the layout builder."""
    doc = sc.doc
    base = f"{PEARLSCAPE_PARENT_LAYER}::{SECTIONS_LAYER}::Export::{section['code']}"
    cube_min = section["cube_min"]
    y0, z0 = float(cube_min[1]), float(cube_min[2])
    out = []
    for p in section["curtains"]:
        layer_path = f"{base}::C{p['plane_index']:03d}"
        layer_idx = _ensure_layer(layer_path)
        if p["points_3d"].shape[0]:
            add_pointcloud(p["points_3d"], layer_path, colors=p["colors"])
        px = p["plane_x"]
        frame = [
            rg.Point3d(px, y0, z0),
            rg.Point3d(px, y0 + size, z0),
            rg.Point3d(px, y0 + size, z0 + size),
            rg.Point3d(px, y0, z0 + size),
            rg.Point3d(px, y0, z0),
        ]
        attrs = Rhino.DocObjects.ObjectAttributes()
        attrs.LayerIndex = layer_idx
        doc.Objects.AddPolyline(rg.Polyline(frame), attrs)
        out.append((p["plane_index"], layer_path))
    sc.doc.Views.Redraw()
    return out


def render_cave_surface(surface: rg.Surface) -> None:
    """Add the base NURBS cave surface to Pearlscape::CaveSurface, replacing any
    surface already on that layer. Real document geometry, so it persists across
    F5 runs and can be hand-edited (see nurbs_surface_source="reuse")."""
    doc = sc.doc
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CAVE_SURFACE_LAYER}"
    layer_idx = _ensure_layer(layer_path)
    # Clear any existing objects on the layer first.
    layer = doc.Layers[layer_idx]
    existing = doc.Objects.FindByLayer(layer)
    if existing:
        for obj in existing:
            doc.Objects.Delete(obj, True)
    attrs = Rhino.DocObjects.ObjectAttributes()
    attrs.LayerIndex = layer_idx
    doc.Objects.AddSurface(surface, attrs)
    sc.doc.Views.Redraw()


def find_cave_surface() -> Optional[rg.NurbsSurface]:
    """Return the first surface on Pearlscape::CaveSurface, or None.

    Accepts a raw Surface or a single-face Brep (what AddSurface round-trips to);
    returns a NurbsSurface for uniform downstream evaluation.
    """
    doc = sc.doc
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CAVE_SURFACE_LAYER}"
    idx = doc.Layers.FindByFullPath(layer_path, -1)
    if idx < 0:
        return None
    layer = doc.Layers[idx]
    objs = doc.Objects.FindByLayer(layer)
    if not objs:
        return None
    geo = objs[0].Geometry
    if isinstance(geo, rg.Brep):
        if geo.Faces.Count == 0:
            return None
        return geo.Faces[0].UnderlyingSurface().ToNurbsSurface()
    if isinstance(geo, rg.Surface):
        return geo.ToNurbsSurface()
    return None


def find_cave_points() -> Optional[np.ndarray]:
    """Return all points on Pearlscape::CavePoints as an (N, 3) array, or None.

    Aggregates every point on the layer regardless of how it was authored: each
    PointCloud object is expanded to its member points, and each individual Point
    object contributes its location. Coordinates are world coordinates, used as-is.
    Returns None if the layer is missing or holds no point geometry.
    """
    doc = sc.doc
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CAVE_POINTS_LAYER}"
    idx = doc.Layers.FindByFullPath(layer_path, -1)
    if idx < 0:
        return None
    layer = doc.Layers[idx]
    objs = doc.Objects.FindByLayer(layer)
    if not objs:
        return None

    rows: List[List[float]] = []
    for obj in objs:
        geo = obj.Geometry
        if isinstance(geo, rg.PointCloud):
            for item in geo:
                p = item.Location
                rows.append([p.X, p.Y, p.Z])
        elif isinstance(geo, rg.Point):
            p = geo.Location
            rows.append([p.X, p.Y, p.Z])
        # Other geometry on the layer is ignored.

    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def _build_bead_block_definition(diameter: float, subd: int, name: str = "PearlscapeBead") -> int:
    """Ensure a block definition for a unit bead mesh exists; return its index.

    The block name encodes (diameter, subd) so that changing either param
    in `params.py` produces a new block rather than silently reusing the
    cached one in the Rhino document.
    """
    doc = sc.doc
    full_name = f"{name}_d{diameter}_s{subd}"
    existing = doc.InstanceDefinitions.Find(full_name, True)
    if existing is not None:
        return existing.Index
    radius = diameter / 2.0
    sphere = rg.Sphere(rg.Point3d.Origin, radius)
    # CreateFromSphere takes (count_around, count_vertical). subd controls density.
    count = max(6, 2 ** (subd + 1))
    mesh = rg.Mesh.CreateFromSphere(sphere, count, count)
    # The mesh inside the block must defer to the InstanceReference's color
    # for per-bead colour overrides to render.
    mesh_attrs = Rhino.DocObjects.ObjectAttributes()
    mesh_attrs.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromParent
    idx = doc.InstanceDefinitions.Add(
        full_name,
        "Pearlscape bead",
        rg.Point3d.Origin,
        [mesh],
        [mesh_attrs],
    )
    return idx


def render_instances(curtains: Sequence[dict], diameter: float, subd: int) -> None:
    """Render each bead as an InstanceReference to a shared bead mesh block.

    Per-instance color is set via ObjectAttributes (ColorSource = ColorFromObject).
    """
    doc = sc.doc
    block_idx = _build_bead_block_definition(diameter, subd)

    for i, c in enumerate(curtains):
        layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CURTAINS_LAYER}::Curtain_{i:02d}"
        layer_idx = _ensure_layer(layer_path)
        plane_x = c["plane_x"]
        pts_2d = c["points_2d"]
        colors = c.get("colors")
        for j in range(pts_2d.shape[0]):
            t = rg.Transform.Translation(
                plane_x, float(pts_2d[j, 0]), float(pts_2d[j, 1])
            )
            attrs = Rhino.DocObjects.ObjectAttributes()
            attrs.LayerIndex = layer_idx
            if colors is not None:
                rgb = colors[j]
                attrs.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
                attrs.ObjectColor = sd.Color.FromArgb(
                    int(rgb[0]), int(rgb[1]), int(rgb[2])
                )
            doc.Objects.AddInstanceObject(block_idx, t, attrs)
    sc.doc.Views.Redraw()


# --- Sprite display mode -----------------------------------------------------
# Screen-aligned colored circles drawn by a DisplayConduit. Fast: all beads are
# drawn in one batched DrawSprites call per frame. The conduit is viewport-only
# (not document geometry), so it does NOT appear in PDF export.

_SPRITE_CONDUIT_KEY = "pearlscape_sprite_conduit"


def _make_bead_sprites(px: int = 64):
    """Build the two stacked sprite bitmaps for a glass bead.

    Returns (body_bitmap, specular_bitmap), drawn as two passes:

    1. Body — grayscale + alpha, *multiplied* by each bead's colour
       (DisplayBitmapDrawList tint). An opaque disc (transparent only *outside*
       the circle, with a ~2px antialiased edge), a darker/richer core, and a
       soft colour glint at the bottom-left (the brightest version of the bead
       colour, since multiply can't brighten past it).
    2. Specular — pure white + alpha, drawn untinted on top: a sharp-edged
       highlight at the top-right that multiply-tinting could never produce.
       Masked to the disc so it can't spill past the rim, and it sits over the
       opaque body so its antialiased edge blends against the bead, not the
       background.

    Built once per conduit, so the per-pixel cost is irrelevant at draw time.
    """
    n = px
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cx = cy = (n - 1) / 2.0
    radius = n / 2.0
    t = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius   # 0 centre .. 1 rim

    # Opaque disc: alpha 1 inside, ~2px antialiased edge, 0 outside.
    aa = 2.0 / n
    disc_a = np.clip((1.0 - t) / aa, 0.0, 1.0)

    # Body luminance: darker/richer core nudged toward the top-right (opposite the
    # colour glint), brightening outward.
    bcx, bcy = cx + 0.08 * radius, cy - 0.08 * radius
    bt = np.clip(np.sqrt((xx - bcx) ** 2 + (yy - bcy) ** 2) / radius, 0.0, 1.0)
    lum = 0.45 + (0.88 - 0.45) * bt
    # Colour glint, bottom-left.
    gcx, gcy = cx - 0.40 * radius, cy + 0.40 * radius
    gd = np.sqrt((xx - gcx) ** 2 + (yy - gcy) ** 2) / radius
    lum = np.clip(lum + 0.6 * np.clip(1.0 - gd / 0.32, 0.0, 1.0) ** 2, 0.0, 1.0)

    # White specular, top-right: a sharp-edged disc (~2px AA, same as the body),
    # masked to the bead so it can't spill past the rim.
    scx, scy = cx + 0.40 * radius, cy - 0.40 * radius
    sd_ = np.sqrt((xx - scx) ** 2 + (yy - scy) ** 2) / radius
    spec_radius = 0.15
    spec_a = np.clip((spec_radius - sd_) / aa, 0.0, 1.0) * disc_a

    g = (lum * 255.0).astype(np.uint8)
    body_alpha = (disc_a * 255.0).astype(np.uint8)
    spec_alpha = (spec_a * 255.0).astype(np.uint8)

    body = sd.Bitmap(n, n)
    spec = sd.Bitmap(n, n)
    for y in range(n):
        for x in range(n):
            gv = int(g[y, x])
            body.SetPixel(x, y, sd.Color.FromArgb(int(body_alpha[y, x]), gv, gv, gv))
            spec.SetPixel(x, y, sd.Color.FromArgb(int(spec_alpha[y, x]), 255, 255, 255))
    return body, spec


class SpriteConduit(rd.DisplayConduit):
    """Draw bead positions as world-sized, screen-aligned glass-bead sprites.

    Two stacked DisplayBitmaps (body + white specular) are drawn as two batched
    DrawSprites passes. The body is tinted per-point by the bead colours; the
    specular pass is tinted all-white so the highlight stays pure white. Both
    point lists are built once at construction and reused across redraws.
    """

    def __init__(self, points: np.ndarray, colors: Optional[np.ndarray], diameter: float) -> None:
        # Explicit base init is the documented RhinoCommon-Python pattern for
        # subclassing DisplayConduit (avoids pythonnet MRO surprises).
        rd.DisplayConduit.__init__(self)
        self._diameter = float(diameter)
        body_bmp, spec_bmp = _make_bead_sprites()
        self._bitmap = rd.DisplayBitmap(body_bmp)
        self._spec_bitmap = rd.DisplayBitmap(spec_bmp)

        pt_list = _np_to_point3d_list(points)
        color_list = _np_to_color_list(colors)
        if color_list is None:
            color_list = [sd.Color.White] * len(pt_list)
        self._sprites = rd.DisplayBitmapDrawList()
        self._sprites.SetPoints(pt_list, color_list)
        # Specular pass: same points, all-white tint so white × white stays white
        # regardless of bead colour. [White] * n is N cheap references, not N objects.
        self._spec_list = rd.DisplayBitmapDrawList()
        self._spec_list.SetPoints(pt_list, [sd.Color.White] * len(pt_list))

        bbox = rg.BoundingBox(pt_list)
        bbox.Inflate(self._diameter / 2.0)
        self._bbox = bbox

    def CalculateBoundingBox(self, e) -> None:
        # Keep the sprites inside the view's clipping bounds and let ZoomExtents
        # frame them — without this the conduit would be culled.
        e.IncludeBoundingBox(self._bbox)

    def PostDrawObjects(self, e) -> None:
        # DrawSprites' world-space size is the sprite RADIUS (half-extent), not
        # the diameter — passing the full diameter draws beads at 2x their true
        # size. Pass half so the on-screen bead diameter equals bead_diameter,
        # matching render_instances' real spheres and the three.js viewer.
        # Pass 1: bead body, tinted per-point. Pass 2: pure-white specular on top.
        half = self._diameter / 2.0
        e.Display.DrawSprites(self._bitmap, self._sprites, half, True)
        e.Display.DrawSprites(self._spec_bitmap, self._spec_list, half, True)


def clear_sprite_conduit() -> None:
    """Disable and drop any sprite conduit left over from a previous run.

    Called at the start of every render so switching away from sprites mode
    leaves no stale beads on screen.
    """
    existing = sc.sticky.get(_SPRITE_CONDUIT_KEY)
    if existing is not None:
        existing.Enabled = False
        del sc.sticky[_SPRITE_CONDUIT_KEY]


def render_sprites(points: np.ndarray, colors: Optional[np.ndarray], diameter: float) -> None:
    """Render beads as colored circle sprites via a persistent conduit.

    Stored in scriptcontext.sticky so the conduit survives after the F5 script
    returns (otherwise it is garbage-collected and the beads vanish).
    """
    clear_sprite_conduit()
    conduit = SpriteConduit(points, colors, diameter)
    conduit.Enabled = True
    sc.sticky[_SPRITE_CONDUIT_KEY] = conduit
    sc.doc.Views.Redraw()


def render_sprites_curtains(curtains: Sequence[dict], diameter: float) -> None:
    """Flatten every curtain's projected beads into one sprite conduit.

    A single combined conduit (one draw call) is the fast path; per-curtain
    layer visibility is not available in sprites mode.
    """
    all_pts = []
    all_cols = []
    have_colors = True
    for c in curtains:
        all_pts.append(_projected_points(c))
        cc = c.get("colors")
        if cc is None:
            have_colors = False
        else:
            all_cols.append(cc)
    combined_pts = np.vstack(all_pts) if all_pts else np.empty((0, 3))
    combined_cols = np.vstack(all_cols) if (have_colors and all_cols) else None
    render_sprites(combined_pts, combined_cols, diameter)
