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
import scriptcontext as sc
import System.Drawing as sd


PEARLSCAPE_PARENT_LAYER = "Pearlscape"
CURTAINS_LAYER = "Curtains"
CAVE_REFERENCE_LAYER = "CaveReference"
CURTAIN_PLANES_LAYER = "CurtainPlanes"


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


def render_cave_reference(points: np.ndarray) -> None:
    """Render the raw, un-sliced cave as a single PointCloud on the
    CaveReference layer. Used during early development."""
    layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CAVE_REFERENCE_LAYER}"
    add_pointcloud(points, layer_path)
    sc.doc.Views.Redraw()


def render_pointclouds(curtains: Sequence[dict]) -> None:
    """Render per-curtain PointClouds.

    Each item in `curtains` is the dict produced by curtains.slice_and_project:
        { 'plane_x': float, 'points_2d': np.ndarray (M, 2),
          'points_3d': np.ndarray (M, 3), 'colors': np.ndarray (M, 3) | None }
    The displayed point is (plane_x, y, z) — i.e. the projected position.
    """
    for i, c in enumerate(curtains):
        layer_path = f"{PEARLSCAPE_PARENT_LAYER}::{CURTAINS_LAYER}::Curtain_{i:02d}"
        plane_x = c["plane_x"]
        pts_2d = c["points_2d"]
        projected = np.column_stack([
            np.full(pts_2d.shape[0], plane_x),
            pts_2d[:, 0],
            pts_2d[:, 1],
        ])
        add_pointcloud(projected, layer_path, colors=c.get("colors"))
    sc.doc.Views.Redraw()
