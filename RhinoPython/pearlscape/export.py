"""Per-curtain PDF export.

Creates one Rhino layout (PageView) per curtain, populates it with an
orthographic detail looking along +X, isolates that curtain's bead layer,
and writes one PDF per layout.
"""

import os
from typing import List

import Rhino
import Rhino.Geometry as rg
import Rhino.Display as rd
import scriptcontext as sc
import System.Drawing as sd


PAGE_SIZES_MM = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
    "A0": (841.0, 1189.0),
}


def _set_layout_layer_visibility(page: rd.RhinoPageView, visible_curtain: int) -> None:
    """Hide every Pearlscape::Curtains::Curtain_NN layer except the chosen one,
    plus CurtainPlanes and CaveReference, in this page's detail viewports.

    Per-viewport layer visibility in Rhino lives on the Layer (not the detail);
    we call layer.SetPerViewportVisible(detail.Viewport.Id, ...) for each detail.
    """
    doc = sc.doc
    extra_hide_paths = {
        "Pearlscape::CurtainPlanes",
        "Pearlscape::CaveReference",
    }
    details = page.GetDetailViews()
    if not details:
        return
    for layer in doc.Layers:
        full = layer.FullPath
        if full.startswith("Pearlscape::Curtains::Curtain_"):
            n = int(full.rsplit("_", 1)[-1])
            visible = (n == visible_curtain)
        elif full in extra_hide_paths:
            visible = False
        else:
            continue
        for det in details:
            layer.SetPerViewportVisible(det.Viewport.Id, visible)
        doc.Layers.Modify(layer, layer.Index, True)


def _make_layout(name: str, page_w_mm: float, page_h_mm: float) -> rd.RhinoPageView:
    """Create a new layout page sized to (w, h) in millimeters."""
    doc = sc.doc
    page = doc.Views.AddPageView(name, page_w_mm, page_h_mm)
    return page


def _add_orthographic_detail(
    page: rd.RhinoPageView,
    plane_x: float,
    curtain_width: float,
    curtain_height: float,
    margin_mm: float = 20.0,
) -> None:
    """Add one orthographic detail looking along +X, framed to the curtain bounds."""
    page_w = page.PageWidth
    page_h = page.PageHeight
    # Both page and curtain are in mm, so the scale is just a ratio.
    scale_w = (page_w - 2 * margin_mm) / curtain_width
    scale_h = (page_h - 2 * margin_mm) / curtain_height
    scale = min(scale_w, scale_h)

    # Detail rectangle on the page, centered.
    detail_w_mm = curtain_width * scale
    detail_h_mm = curtain_height * scale
    cx = page_w / 2.0
    cy = page_h / 2.0
    corner_a = rg.Point2d(cx - detail_w_mm / 2.0, cy - detail_h_mm / 2.0)
    corner_b = rg.Point2d(cx + detail_w_mm / 2.0, cy + detail_h_mm / 2.0)

    # Rhino's 'Right' projection looks along -X; curtain plane reads as (Y, Z).
    detail = page.AddDetailView("Curtain", corner_a, corner_b, rd.DefinedViewportProjection.Right)
    detail.Viewport.ZoomExtents()
    detail.DetailGeometry.IsProjectionLocked = True
    detail.CommitChanges()


def create_curtain_layouts(
    plane_xs: List[float],
    curtain_width: float,
    curtain_height: float,
    page_size: str = "A1",
) -> List[str]:
    """Create one layout per curtain. Returns the list of layout names."""
    if page_size not in PAGE_SIZES_MM:
        raise ValueError(f"Unknown page_size {page_size!r}; "
                         f"options: {sorted(PAGE_SIZES_MM)}")
    page_w, page_h = PAGE_SIZES_MM[page_size]

    # Idempotency: drop any pre-existing Curtain_NN layouts before creating
    # the new set. Rhino 8's RhinoCommon API for page-view deletion from
    # Python is awkward: ViewTable exposes neither Remove nor Close, and
    # the `-_DeleteLayout` macro via RunScript runs asynchronously (so the
    # layouts persist for the rest of the script). The instance-level
    # RhinoView.Close() is the only call that removes a page view
    # synchronously from inside a Python script.
    for v in list(sc.doc.Views.GetPageViews()):
        if v.PageName.startswith("Curtain_"):
            v.Close()

    names = []
    for i, x in enumerate(plane_xs):
        name = f"Curtain_{i:02d}"
        page = _make_layout(name, page_w, page_h)
        _add_orthographic_detail(page, x, curtain_width, curtain_height)
        _set_layout_layer_visibility(page, i)
        names.append(name)
    sc.doc.Views.Redraw()
    return names


def _isolate_pearlscape_layer(page: rd.RhinoPageView, keep_path: str) -> None:
    """In this page's details, hide EVERY layer under Pearlscape except
    `keep_path` (its ancestors stay visible so the kept leaf isn't masked by a
    hidden parent). Used by the section layouts, where unrelated Pearlscape
    geometry (grid cubes, review wall, model curtains) must never print."""
    doc = sc.doc
    keep = {keep_path}
    parts = keep_path.split("::")
    for n in range(1, len(parts)):
        keep.add("::".join(parts[:n]))
    details = page.GetDetailViews()
    if not details:
        return
    for layer in doc.Layers:
        if layer.IsDeleted:
            continue
        full = layer.FullPath
        if not (full == "Pearlscape" or full.startswith("Pearlscape::")):
            continue
        visible = full in keep
        for det in details:
            layer.SetPerViewportVisible(det.Viewport.Id, visible)
        doc.Layers.Modify(layer, layer.Index, True)


def _add_section_detail(
    page: rd.RhinoPageView,
    cube_min,
    section_size: float,
    margin_mm: float = 20.0,
) -> None:
    """One orthographic detail looking along +X, framed to the section cube's
    YZ square (NOT ZoomExtents) so every page of a section prints at the same
    scale regardless of how its beads happen to spread."""
    page_w = page.PageWidth
    page_h = page.PageHeight
    side = min(page_w, page_h) - 2 * margin_mm   # the framed region is square
    cx = page_w / 2.0
    cy = page_h / 2.0
    corner_a = rg.Point2d(cx - side / 2.0, cy - side / 2.0)
    corner_b = rg.Point2d(cx + side / 2.0, cy + side / 2.0)

    detail = page.AddDetailView("Section", corner_a, corner_b,
                                rd.DefinedViewportProjection.Right)
    bbox = rg.BoundingBox(
        rg.Point3d(float(cube_min[0]), float(cube_min[1]), float(cube_min[2])),
        rg.Point3d(float(cube_min[0]) + section_size,
                   float(cube_min[1]) + section_size,
                   float(cube_min[2]) + section_size),
    )
    bbox.Inflate(0.02 * section_size)   # keep the frame rect off the detail edge
    detail.Viewport.ZoomBoundingBox(bbox)
    detail.DetailGeometry.IsProjectionLocked = True
    detail.CommitChanges()


def create_section_layouts(
    code: str,
    plane_layers: List[tuple],
    cube_min,
    section_size: float,
    page_size: str = "A1",
) -> List[str]:
    """One layout per curtain plane of the selected section, named
    Section_<code>_C<NNN> (NNN = the plane's index in the FULL model, so pages
    trace back to the uncut curtains). `plane_layers` is the
    [(plane_index, layer_path), ...] list from
    display.render_section_export_curtains. Returns the layout names."""
    if page_size not in PAGE_SIZES_MM:
        raise ValueError(f"Unknown page_size {page_size!r}; "
                         f"options: {sorted(PAGE_SIZES_MM)}")
    page_w, page_h = PAGE_SIZES_MM[page_size]

    # Idempotency, same Close() story as create_curtain_layouts.
    for v in list(sc.doc.Views.GetPageViews()):
        if v.PageName.startswith("Section_"):
            v.Close()

    names = []
    for plane_index, layer_path in plane_layers:
        name = f"Section_{code}_C{plane_index:03d}"
        page = _make_layout(name, page_w, page_h)
        _add_section_detail(page, cube_min, section_size)
        _isolate_pearlscape_layer(page, layer_path)
        names.append(name)
    sc.doc.Views.Redraw()
    return names


def export_all_pdfs(output_dir: str, dpi: float = 300.0,
                    prefix: str = "Curtain_") -> List[str]:
    """Export every layout whose name starts with `prefix` as its own PDF.
    Returns the list of output paths.

    Uses Rhino 8's ViewCaptureSettings to specify per-page size and DPI; the
    earlier `pdf.AddPage(view, w, h, dpi)` 4-arg overload was removed in
    favour of this settings-object pattern.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_paths: List[str] = []
    seen_names = set()
    for view in sc.doc.Views.GetPageViews():
        if not view.PageName.startswith(prefix):
            continue
        if view.PageName in seen_names:
            continue
        seen_names.add(view.PageName)
        out_path = os.path.join(output_dir, f"{view.PageName}.pdf")
        # Page dimensions in mm -> pixels at chosen DPI (1 inch = 25.4 mm).
        pw_px = int(round(view.PageWidth / 25.4 * dpi))
        ph_px = int(round(view.PageHeight / 25.4 * dpi))
        size = sd.Size(pw_px, ph_px)
        settings = Rhino.Display.ViewCaptureSettings(view, size, dpi)
        pdf = Rhino.FileIO.FilePdf.Create()
        pdf.AddPage(settings)
        pdf.Write(out_path)
        out_paths.append(out_path)
    return out_paths
