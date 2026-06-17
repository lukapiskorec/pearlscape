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


_TITLE_STRIP_MM = 80.0

# The whole sheet is drawn directly in PAGE space at 1:1 — no detail view. The
# page is section_size wide and section_size + strip tall; a bead at in-plane
# (Y, Z) maps to page (Y - cube_min_y, strip + Z - cube_min_z), so a 1x1 m cube
# prints as a 1x1 m drawing exactly, identically on every sheet. (DetailView +
# SetScale could not be made to hold a reliable 1:1 — it kept blanking.)

_SHEET_LAYER = "SectionSheet"
_TITLE_LAYER = "SectionTitleBlock"
_TEXT_RGB = sd.Color.FromArgb(20, 20, 20)
_BORDER_RGB = sd.Color.FromArgb(20, 20, 20)
_STRING_RGB = sd.Color.FromArgb(150, 150, 150)
_FRAME_RGB = sd.Color.FromArgb(20, 20, 20)


def _page_attrs(page: rd.RhinoPageView, layer_idx: int, color=None):
    attrs = Rhino.DocObjects.ObjectAttributes()
    attrs.LayerIndex = layer_idx
    attrs.Space = Rhino.DocObjects.ActiveSpace.PageSpace
    attrs.ViewportId = page.MainViewport.Id
    if color is not None:
        # Both display AND plot colour — FilePdf prints via PlotColor, which
        # defaults to black, so ObjectColor alone prints everything black.
        attrs.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
        attrs.ObjectColor = color
        attrs.PlotColorSource = Rhino.DocObjects.ObjectPlotColorSource.PlotColorFromObject
        attrs.PlotColor = color
    return attrs


def _page_text(page, layer_idx, text, x, y, height, color=_TEXT_RGB):
    doc = sc.doc
    plane = rg.Plane(rg.Point3d(x, y, 0.0), rg.Vector3d.ZAxis)
    te = rg.TextEntity.Create(text, plane, doc.DimStyles.Current, False, 0.0, 0.0)
    te.TextHeight = height
    doc.Objects.AddText(te, _page_attrs(page, layer_idx, color))


def _page_text_vertical(page, layer_idx, text, x, y, height, color=_TEXT_RGB):
    """Page-space text rotated 90 deg CCW (reads bottom-to-top). Baseline runs
    up +Y from (x, y); used for the string labels along the bottom of the field."""
    doc = sc.doc
    plane = rg.Plane(rg.Point3d(x, y, 0.0),
                     rg.Vector3d(0.0, 1.0, 0.0), rg.Vector3d(-1.0, 0.0, 0.0))
    te = rg.TextEntity.Create(text, plane, doc.DimStyles.Current, False, 0.0, 0.0)
    te.TextHeight = height
    doc.Objects.AddText(te, _page_attrs(page, layer_idx, color))


def _page_filled_rect(page, layer_idx, x, y, w, h, color, hatch_idx, tol):
    doc = sc.doc
    poly = rg.Polyline([rg.Point3d(x, y, 0), rg.Point3d(x + w, y, 0),
                        rg.Point3d(x + w, y + h, 0), rg.Point3d(x, y + h, 0),
                        rg.Point3d(x, y, 0)])
    for hatch in rg.Hatch.Create(poly.ToNurbsCurve(), hatch_idx, 0.0, 1.0, tol):
        doc.Objects.AddHatch(hatch, _page_attrs(page, layer_idx, color))


def _page_line(page, layer_idx, x0, y0, x1, y1, color):
    sc.doc.Objects.AddLine(rg.Line(rg.Point3d(x0, y0, 0), rg.Point3d(x1, y1, 0)),
                           _page_attrs(page, layer_idx, color))


def _page_rect_outline(page, layer_idx, x, y, w, h, color):
    poly = rg.Polyline([rg.Point3d(x, y, 0), rg.Point3d(x + w, y, 0),
                        rg.Point3d(x + w, y + h, 0), rg.Point3d(x, y + h, 0),
                        rg.Point3d(x, y, 0)])
    sc.doc.Objects.AddPolyline(poly, _page_attrs(page, layer_idx, color))


def _page_filled_circle(page, layer_idx, cx, cy, r, color, hatch_idx, tol):
    doc = sc.doc
    circle = rg.Circle(rg.Plane(rg.Point3d(cx, cy, 0), rg.Vector3d.ZAxis), r)
    for hatch in rg.Hatch.Create(circle.ToNurbsCurve(), hatch_idx, 0.0, 1.0, tol):
        doc.Objects.AddHatch(hatch, _page_attrs(page, layer_idx, color))


def _draw_section_field(page, plane, cube_min, section_size, radius, string_tol,
                        sheet_layer, hatch_idx, tol, letters):
    """Draw one curtain plane's beads, strings, string labels and frame in page
    space at 1:1. Page origin (0, strip) corresponds to the cube's (Y, Z) min
    corner. `letters` maps rgb -> placeholder letter, passed through so the string
    numbering here matches the summary document."""
    from pearlscape import sections as smod
    y0, z0 = float(cube_min[1]), float(cube_min[2])
    pts2d = plane["points_2d"]
    colors = plane["colors"]

    # Vertical strings + a label (number + offset) along the bottom of each. Lines
    # use the exact cluster-mean Y (string_columns); labels use string_layout,
    # which shares the same _cluster_strings ordering, so the i-th line and i-th
    # label are the same string by construction.
    cols = smod.string_columns(pts2d, string_tol)
    strings = smod.string_layout(pts2d, colors, cube_min, section_size,
                                 string_tol, letters)
    for yv, s in zip(cols, strings):
        x = float(yv) - y0
        _page_line(page, sheet_layer, x, _TITLE_STRIP_MM,
                   x, _TITLE_STRIP_MM + section_size, _STRING_RGB)
        label = f"{s['number']:03d}  {s['offset_mm']}mm"
        _page_text_vertical(page, sheet_layer, label, x + 1.5,
                            _TITLE_STRIP_MM + 8.0, 5.0, _TEXT_RGB)

    # Beads as filled discs at true diameter, fill = bead colour.
    for bi in range(pts2d.shape[0]):
        cx = float(pts2d[bi, 0]) - y0
        cy = _TITLE_STRIP_MM + (float(pts2d[bi, 1]) - z0)
        col = (sd.Color.FromArgb(int(colors[bi, 0]), int(colors[bi, 1]),
                                 int(colors[bi, 2]))
               if colors is not None else sd.Color.Black)
        _page_filled_circle(page, sheet_layer, cx, cy, radius, col, hatch_idx, tol)

    # Cube outline frames the 1x1 m field.
    _page_rect_outline(page, sheet_layer, 0, _TITLE_STRIP_MM,
                       section_size, section_size, _FRAME_RGB)


def _add_section_title_strip(page, code, plane_index, legend, section_size):
    """Title strip on the paper below the bead field: section/curtain code,
    scale, total bead count, a colour-circle + count legend, and a 100 mm scale
    bar (5 mm ticks). `legend` is the [((r,g,b), count), ...] list from
    sections.color_counts."""
    from pearlscape import display
    doc = sc.doc
    layer_idx = display._ensure_layer(_TITLE_LAYER)
    hatch_idx = display._solid_hatch_index()
    tol = doc.ModelAbsoluteTolerance
    w = section_size
    h = _TITLE_STRIP_MM
    total = sum(n for _, n in legend)

    # The strip sits in the bottom h mm of the page (y in [0, h]); the field is
    # above it, so no opaque background is needed. A top border rules them off.
    doc.Objects.AddLine(rg.Line(rg.Point3d(0, h, 0), rg.Point3d(w, h, 0)),
                        _page_attrs(page, layer_idx, _BORDER_RGB))

    # Left block: code, scale, bead total. The code sits half a font-height higher
    # than before so it lines up with the legend circles.
    _page_text(page, layer_idx, f"{code}  /  C{plane_index:03d}", 16, 56, 20)
    _page_text(page, layer_idx, "Scale 1:1", 16, 24, 12)
    _page_text(page, layer_idx, f"{total} beads", 130, 24, 12)

    # 100 mm scale bar with 5 mm ticks (the sheet is 1:1) on the right of the
    # strip; taller ticks every 10 mm.
    bar_len = 100.0
    bar_x1 = w - 16.0
    bar_x0 = bar_x1 - bar_len
    bar_y = 30.0
    _page_line(page, layer_idx, bar_x0, bar_y, bar_x1, bar_y, _BORDER_RGB)
    for t in range(int(round(bar_len / 5.0)) + 1):
        tx = bar_x0 + t * 5.0
        th = 8.0 if t % 2 == 0 else 4.0
        _page_line(page, layer_idx, tx, bar_y, tx, bar_y + th, _BORDER_RGB)
    _page_text(page, layer_idx, "100 mm", bar_x0 + 30.0, bar_y + 20.0, 11)

    # Legend: colour circle + count, laid left-to-right from x0, wrapping up a
    # row. Counts sit half a font-height higher so they centre on the circles,
    # and the legend stops before the scale bar so the two never overlap.
    sw = 22.0          # circle diameter (its bounding box side)
    slot = 96.0        # horizontal pitch per entry
    x0 = 340.0         # start well right of the title block so it can breathe
    rows_y = (44.0, 14.0)   # two rows inside the strip
    per_row = max(1, int((bar_x0 - 16.0 - x0) / slot))
    r = sw / 2.0
    for i, (rgb, n) in enumerate(legend):
        col = i % per_row
        row = (i // per_row) % len(rows_y)
        x = x0 + col * slot
        y = rows_y[row]
        cx, cy = x + r, y + r
        _page_filled_circle(page, layer_idx, cx, cy, r,
                            sd.Color.FromArgb(rgb[0], rgb[1], rgb[2]), hatch_idx, tol)
        doc.Objects.AddCircle(
            rg.Circle(rg.Plane(rg.Point3d(cx, cy, 0.0), rg.Vector3d.ZAxis), r),
            _page_attrs(page, layer_idx, _BORDER_RGB))
        _page_text(page, layer_idx, f"x{n}", x + sw + 6, y + 9.5, 13)


def create_section_layouts(section: dict, section_size: float, params) -> List[str]:
    """One layout per curtain plane of the selected section, named
    Section_<code>_C<NNN> (NNN = the plane's index in the FULL model, so pages
    trace back to the uncut curtains). Each page is a section_size-wide,
    (section_size + strip)-tall sheet whose 1x1 m field is drawn directly in page
    space at 1:1 — beads, strings and frame — with a title strip below. `section`
    is one entry from sections.assign_sections. Returns the layout names."""
    from pearlscape import display, sections as smod
    doc = sc.doc
    code = section["code"]
    cube_min = section["cube_min"]
    page_w = float(section_size)
    page_h = float(section_size) + _TITLE_STRIP_MM
    radius = float(params.bead_diameter) / 2.0
    string_tol = (float(params.string_align_overlap)
                  if params.string_align_overlap > 0.0 else float(params.bead_diameter))
    letters = smod.palette_letters(params.palette)
    sheet_layer = display._ensure_layer(_SHEET_LAYER)
    hatch_idx = display._solid_hatch_index()
    tol = doc.ModelAbsoluteTolerance

    # Idempotency, same Close() story as create_curtain_layouts.
    for v in list(sc.doc.Views.GetPageViews()):
        if v.PageName.startswith("Section_"):
            v.Close()

    names = []
    for plane in section["curtains"]:
        plane_index = plane["plane_index"]
        name = f"Section_{code}_C{plane_index:03d}"
        page = _make_layout(name, page_w, page_h)
        _draw_section_field(page, plane, cube_min, section_size, radius,
                            string_tol, sheet_layer, hatch_idx, tol, letters)
        _add_section_title_strip(page, code, plane_index,
                                 smod.color_counts(plane["colors"]), section_size)
        names.append(name)
    sc.doc.Views.Redraw()
    return names


_DOC_LAYER = "SectionDocText"
_DOC_PAGE_W = 210.0          # A4 portrait, mm
_DOC_PAGE_H = 297.0
_DOC_MARGIN = 15.0
_DOC_TEXT_H = 2.25          # body text height, mm (50% of the original 4.5)
_DOC_LEADING = 3.5          # line pitch, mm
_DOC_SWATCH = 2.5           # legend swatch side, mm
_DOC_BOX = 2.5              # fabricator checkbox side, mm
_DOC_GAP = 1.5             # gap from a row marker (checkbox/swatch) to its text, mm
_DOC_INDENT = 4.0         # hierarchy indent for legend/string rows, mm
_DOC_CHAR_W = 0.75       # est. glyph width as a fraction of text height (for wrap;
                         #   over-estimates the real font so wrapped lines clear
                         #   the right margin — lower it if wrapping looks early)


def create_section_document(section, params, out_path, dpi: float = 300.0) -> str:
    """Write a multi-page A4 PDF listing every curtain and string in `section`.

    Rows come from sections.section_document_rows (shared with the .txt export).
    Long lines are word-wrapped to the page width (nothing is clipped, a right
    margin is kept) and each string row carries an empty checkbox for the
    fabricator. All pages are added to ONE FilePdf and written once to
    `out_path`. Temporary page views (named _SectionDoc_<code>_pNN) are closed
    afterwards. Returns out_path."""
    from pearlscape import display, sections as smod
    doc = sc.doc
    code = section["code"]
    layer_idx = display._ensure_layer(_DOC_LAYER)
    hatch_idx = display._solid_hatch_index()
    tol = doc.ModelAbsoluteTolerance

    rows = smod.section_document_rows(section, params)

    # Expand each logical row into physical lines: reserve its marker column
    # (checkbox or swatch) and the indent, then word-wrap the text to the page
    # width LESS a right margin (_DOC_MARGIN reserved on both sides), with a
    # hanging indent so continuation lines align under the text (no marker).
    char_w = _DOC_TEXT_H * _DOC_CHAR_W
    phys = []
    for row in rows:
        marker_x = _DOC_MARGIN + row["indent"] * _DOC_INDENT
        x_text = marker_x
        if row["checkbox"]:
            x_text += _DOC_BOX + _DOC_GAP
        elif row["swatch"] is not None:
            x_text += _DOC_SWATCH + _DOC_GAP
        avail = _DOC_PAGE_W - _DOC_MARGIN - x_text
        max_chars = max(1, int(avail / char_w))
        for li, sub in enumerate(smod.wrap_text(row["text"], max_chars)):
            phys.append({
                "text": sub,
                "x_text": x_text,
                "marker_x": marker_x,
                "checkbox": row["checkbox"] and li == 0,
                "swatch": row["swatch"] if (row["swatch"] is not None and li == 0) else None,
            })

    per_page = max(1, int((_DOC_PAGE_H - 2.0 * _DOC_MARGIN) / _DOC_LEADING))
    pages = [phys[i:i + per_page] for i in range(0, len(phys), per_page)] or [[]]

    # Idempotency: drop any stale doc pages for this code (same Close() story as
    # create_section_layouts).
    prefix = f"_SectionDoc_{code}_"
    for v in list(doc.Views.GetPageViews()):
        if v.PageName.startswith(prefix):
            v.Close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pdf = Rhino.FileIO.FilePdf.Create()
    created = []
    for pi, page_lines in enumerate(pages):
        page = doc.Views.AddPageView(f"{prefix}{pi:02d}", _DOC_PAGE_W, _DOC_PAGE_H)
        created.append(page)
        y = _DOC_PAGE_H - _DOC_MARGIN
        for pl in page_lines:
            # Rhino top-anchors the TextEntity at its insertion point and draws
            # the glyphs DOWNWARD, so a marker drawn at the row top floats above
            # the text. Centre the marker on the text body instead.
            text_org_y = y - _DOC_TEXT_H
            text_mid_y = text_org_y - _DOC_TEXT_H / 2.0
            if pl["checkbox"]:
                _page_rect_outline(page, layer_idx, pl["marker_x"],
                                   text_mid_y - _DOC_BOX / 2.0,
                                   _DOC_BOX, _DOC_BOX, _BORDER_RGB)
            if pl["swatch"] is not None:
                r, g, b = pl["swatch"]
                _page_filled_rect(page, layer_idx, pl["marker_x"],
                                  text_mid_y - _DOC_SWATCH / 2.0,
                                  _DOC_SWATCH, _DOC_SWATCH,
                                  sd.Color.FromArgb(r, g, b), hatch_idx, tol)
            if pl["text"]:
                _page_text(page, layer_idx, pl["text"], pl["x_text"],
                           text_org_y, _DOC_TEXT_H)
            y -= _DOC_LEADING
        pw_px = int(round(_DOC_PAGE_W / 25.4 * dpi))
        ph_px = int(round(_DOC_PAGE_H / 25.4 * dpi))
        settings = Rhino.Display.ViewCaptureSettings(page, sd.Size(pw_px, ph_px), dpi)
        pdf.AddPage(settings)

    pdf.Write(out_path)
    for page in created:
        page.Close()
    doc.Views.Redraw()
    return out_path


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
