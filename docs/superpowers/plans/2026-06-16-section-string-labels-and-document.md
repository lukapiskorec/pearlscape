# Section String Labels + Fabrication Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Number every string in each section curtain, label each string (number + horizontal offset in mm) vertically on the section PDF plans, and emit a separate per-section PDF document listing every curtain and string with its beads.

**Architecture:** All string numbering, offsets and bead positions are computed once in `sections.py` (pure numpy, headless-tested) and consumed by both the Rhino plan-drawing code and the new document writer in `export.py`, so plan and document can never drift. `build_scene.py` wires the document into the existing `export_section` flow.

**Tech Stack:** Python 3, numpy (headless logic), RhinoCommon (`Rhino.Geometry`, `Rhino.FileIO.FilePdf`) for drawing/PDF.

---

## Conventions for this plan

- **Tests** live in each module's `if __name__ == "__main__"` block as `assert`
  statements (the existing pattern in `sections.py` / `curtains.py`). "Run the
  test" means run the whole module headless and confirm it prints `OK`.
- **Headless run command** (from `C:\Users\lukap\Documents\GitHub\pearlscape\RhinoPython`):
  ```
  "C:/Users/lukap/AppData/Local/Python/pythoncore-3.14-64/python.exe" pearlscape/sections.py
  ```
  (The plain `python` launcher mis-reads the `#! python 3` shebang on line 1, so
  use the explicit interpreter path above.)
- **Commits are the user's.** Do NOT run `git commit`. Each "Checkpoint" step is
  where the user reviews and commits.
- `export.py` and `build_scene.py` are RhinoCommon code with **no headless
  harness**; they are verified by running the pipeline in Rhino and inspecting the
  PDFs. Keep all testable logic in `sections.py`.

## File Structure

- **Modify** `pearlscape/sections.py` — add `palette_letters`, `_cluster_strings`
  (refactored out of `string_columns`), `string_layout`, `curtain_summary`, and
  their tests. One responsibility: pure fabrication data.
- **Modify** `pearlscape/export.py` — add `_page_text_vertical`, draw string
  labels in `_draw_section_field`, add `create_section_document` +
  `_section_document_lines`. One responsibility: Rhino page/PDF drawing.
- **Modify** `build_scene.py` — call `create_section_document` in the
  `export_section` block.

---

## Task 1: `palette_letters` — colour → letter by palette order

**Files:**
- Modify: `pearlscape/sections.py` (add function after `color_counts`, ~line 161)
- Test: `pearlscape/sections.py` `__main__` block

- [ ] **Step 1: Add the function**

After `color_counts` (ends ~line 161), add:

```python
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
```

- [ ] **Step 2: Add the test**

In the `__main__` block, after the `color_counts` test (~line 356, after the
`assert color_counts(None) == [] ...` line), add:

```python
    # palette_letters: letters by palette order, dupes keep first, lookups miss -> caller's default
    pl = palette_letters([(10, 20, 30), (40, 50, 60), (70, 80, 90), (10, 20, 30)])
    assert pl[(10, 20, 30)] == "A" and pl[(40, 50, 60)] == "B" and pl[(70, 80, 90)] == "C"
    assert len(pl) == 3, pl                     # duplicate (10,20,30) not re-added
    assert pl.get((1, 1, 1), "?") == "?"        # colour not in palette
```

- [ ] **Step 3: Run the test**

Run: `"C:/Users/lukap/AppData/Local/Python/pythoncore-3.14-64/python.exe" pearlscape/sections.py`
Expected: prints `OK` (no AssertionError).

- [ ] **Step 4: Checkpoint** — user reviews and commits.

---

## Task 2: `_cluster_strings` — extract the clustering, keep `string_columns` identical

**Files:**
- Modify: `pearlscape/sections.py` (`string_columns`, lines 127-146)
- Test: existing `string_columns` test in `__main__` (lines 343-348) is the
  regression guard — it must still pass unchanged.

- [ ] **Step 1: Replace `string_columns` with a clustering helper + thin wrapper**

Replace the whole current `string_columns` function (lines 127-146) with:

```python
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
```

- [ ] **Step 2: Run the regression test**

Run: `"C:/Users/lukap/AppData/Local/Python/pythoncore-3.14-64/python.exe" pearlscape/sections.py`
Expected: prints `OK`. In particular the existing assertion
`assert cols.shape == (2,)` / `np.isclose(cols[0], (10.0 + 10.4 + 10.2) / 3.0)` /
`np.isclose(cols[1], 49.8)` (lines 344-347) still passes — proving the refactor
preserved `string_columns` exactly.

- [ ] **Step 3: Checkpoint** — user reviews and commits.

---

## Task 3: `string_layout` — per-string fabrication data

**Files:**
- Modify: `pearlscape/sections.py` (add after `string_columns`)
- Test: `pearlscape/sections.py` `__main__`

- [ ] **Step 1: Add the function**

After the new `string_columns` (from Task 2), add:

```python
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
```

- [ ] **Step 2: Add the test**

In `__main__`, after the Task 1 `palette_letters` test, add:

```python
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
```

- [ ] **Step 3: Run the test**

Run: `"C:/Users/lukap/AppData/Local/Python/pythoncore-3.14-64/python.exe" pearlscape/sections.py`
Expected: prints `OK`.

- [ ] **Step 4: Checkpoint** — user reviews and commits.

---

## Task 4: `curtain_summary` — per-curtain bead total + per-letter counts

**Files:**
- Modify: `pearlscape/sections.py` (add after `string_layout`)
- Test: `pearlscape/sections.py` `__main__`

- [ ] **Step 1: Add the function**

```python
def curtain_summary(colors, letters: Dict[Tuple[int, int, int], str]) -> dict:
    """Bead total + per-letter counts for ONE curtain plane, most-common first.

    Returns {'n_beads': int, 'by_letter': [(letter, count), ...]}. Built on
    color_counts, so ordering/tie-breaks match the title-strip legend. A
    colourless plane -> {'n_beads': 0, 'by_letter': []}."""
    cc = color_counts(colors)        # [((r, g, b), count), ...] most-common first
    by_letter = [(letters.get(rgb, "?"), int(count)) for rgb, count in cc]
    return {"n_beads": int(sum(count for _, count in cc)), "by_letter": by_letter}
```

- [ ] **Step 2: Add the test**

In `__main__`, after the Task 3 test, add:

```python
    # curtain_summary: total + per-letter, most-common first
    cs_letters = palette_letters([(1, 2, 3), (4, 5, 6), (9, 9, 9)])   # A, B, C
    cs = curtain_summary(np.array([[1, 2, 3], [1, 2, 3], [9, 9, 9], [1, 2, 3], [4, 5, 6]],
                                  dtype=np.uint8), cs_letters)
    assert cs["n_beads"] == 5, cs
    assert cs["by_letter"][0] == ("A", 3), cs            # most common first
    assert dict(cs["by_letter"]) == {"A": 3, "C": 1, "B": 1}, cs
    assert curtain_summary(None, cs_letters) == {"n_beads": 0, "by_letter": []}
```

- [ ] **Step 3: Run the test**

Run: `"C:/Users/lukap/AppData/Local/Python/pythoncore-3.14-64/python.exe" pearlscape/sections.py`
Expected: prints `OK`.

- [ ] **Step 4: Checkpoint** — user reviews and commits.

---

## Task 5: Vertical string labels on the section PDF plan

**Files:**
- Modify: `pearlscape/export.py` (`_page_text_vertical` new; `_draw_section_field`
  lines 219-245; `create_section_layouts` lines 292-328)
- Verification: in Rhino only (no headless test).

- [ ] **Step 1: Add a vertical page-text helper**

After `_page_text` (ends ~line 188), add:

```python
def _page_text_vertical(page, layer_idx, text, x, y, height, color=_TEXT_RGB):
    """Page-space text rotated 90 deg CCW (reads bottom-to-top). Baseline runs
    up +Y from (x, y); used for the string labels along the bottom of the field."""
    doc = sc.doc
    plane = rg.Plane(rg.Point3d(x, y, 0.0),
                     rg.Vector3d(0.0, 1.0, 0.0), rg.Vector3d(-1.0, 0.0, 0.0))
    te = rg.TextEntity.Create(text, plane, doc.DimStyles.Current, False, 0.0, 0.0)
    te.TextHeight = height
    doc.Objects.AddText(te, _page_attrs(page, layer_idx, color))
```

- [ ] **Step 2: Add string labels in `_draw_section_field`**

Change the signature to accept `letters` and `section`-derived data, and replace
the string-line loop. The current function header + string loop (lines 219-232):

```python
def _draw_section_field(page, plane, cube_min, section_size, radius, string_tol,
                        sheet_layer, hatch_idx, tol):
    """Draw one curtain plane's beads, strings and frame in page space at 1:1.
    Page origin (0, strip) corresponds to the cube's (Y, Z) min corner."""
    from pearlscape import sections as smod
    y0, z0 = float(cube_min[1]), float(cube_min[2])
    pts2d = plane["points_2d"]
    colors = plane["colors"]

    # Vertical strings (full field height) at each shared bead column.
    for yv in smod.string_columns(pts2d, string_tol):
        x = float(yv) - y0
        _page_line(page, sheet_layer, x, _TITLE_STRIP_MM,
                   x, _TITLE_STRIP_MM + section_size, _STRING_RGB)
```

becomes:

```python
def _draw_section_field(page, plane, cube_min, section_size, radius, string_tol,
                        sheet_layer, hatch_idx, tol, letters):
    """Draw one curtain plane's beads, strings, string labels and frame in page
    space at 1:1. Page origin (0, strip) corresponds to the cube's (Y, Z) min
    corner. `letters` maps rgb -> placeholder letter (unused for the plan beads,
    passed through to keep string numbering identical to the document)."""
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
```

(The bead loop and frame outline below this, lines 234-245, are unchanged.)

- [ ] **Step 3: Pass `letters` from `create_section_layouts`**

In `create_section_layouts` (lines 292-328): compute the letters once and pass
them into `_draw_section_field`. After the line
`string_tol = (...)` (lines 306-307), add:

```python
    letters = smod.palette_letters(params.palette)
```

and change the `_draw_section_field(...)` call (lines 322-323) from:

```python
        _draw_section_field(page, plane, cube_min, section_size, radius,
                            string_tol, sheet_layer, hatch_idx, tol)
```

to:

```python
        _draw_section_field(page, plane, cube_min, section_size, radius,
                            string_tol, sheet_layer, hatch_idx, tol, letters)
```

- [ ] **Step 4: Verify in Rhino**

Run the pipeline in Rhino with `pipeline_mode = "export_section"` and a valid
`section_export_code`. Open a generated `Section_<code>_C<NNN>.pdf` and confirm:
each string line has a small vertical `NNN  <offset>mm` label at its bottom, and
the offsets increase left to right.

- [ ] **Step 5: Checkpoint** — user reviews and commits.

---

## Task 6: `create_section_document` — the per-section summary PDF

**Files:**
- Modify: `pearlscape/export.py` (add imports + two functions before
  `export_all_pdfs`, ~line 331)
- Verification: in Rhino only.

- [ ] **Step 1: Add the palettes import**

At the top of `export.py`, the package imports are pulled in lazily inside
functions (e.g. `from pearlscape import display, sections as smod`). Follow that
pattern — do not add a module-level import. (Used inside the functions below.)

- [ ] **Step 2: Add the document constants + line builder**

Before `export_all_pdfs` (~line 331), add:

```python
_DOC_LAYER = "SectionDocText"
_DOC_PAGE_W = 210.0          # A4 portrait, mm
_DOC_PAGE_H = 297.0
_DOC_MARGIN = 15.0
_DOC_TEXT_H = 4.5            # body text height, mm
_DOC_LEADING = 7.0          # line pitch, mm
_DOC_SWATCH = 5.0           # legend swatch side, mm


def _section_document_lines(section, params, letters, section_size, string_tol):
    """Flatten the section into a list of row dicts for the document:
        {'text': str, 'swatch': (r, g, b) | None, 'indent': float}
    One row per printed line; legend rows carry a swatch colour."""
    from pearlscape import palettes, sections as smod
    rows = []

    def add(text="", swatch=None, indent=0.0):
        rows.append({"text": text, "swatch": swatch, "indent": indent})

    add(f"Section {section['code']}")
    add(f"Palette: {palettes.name_of(params.palette)}    "
        f"{int(section.get('n_beads', 0))} beads total")
    add()
    add("Colour key (placeholder letters):")
    for c in params.palette:
        rgb = (int(c[0]), int(c[1]), int(c[2]))
        add(f"{letters.get(rgb, '?')} = RGB({rgb[0]}, {rgb[1]}, {rgb[2]})",
            swatch=rgb, indent=8.0)
    add()

    for plane in section["curtains"]:
        colors = plane["colors"]
        summ = smod.curtain_summary(colors, letters)
        tally = ", ".join(f"{ltr} {cnt}" for ltr, cnt in summ["by_letter"])
        head = f"Curtain C{plane['plane_index']:03d} - {summ['n_beads']} beads"
        if tally:
            head += f": {tally}"
        add(head)
        layout = smod.string_layout(plane["points_2d"], colors,
                                    section["cube_min"], section_size,
                                    string_tol, letters)
        for s in layout:
            beads = ", ".join(f"{ltr} {pos}mm" for ltr, pos in s["beads"])
            add(f"String {s['number']:03d} - offset {s['offset_mm']}mm - "
                f"{len(s['beads'])} beads, layout: {beads} /", indent=8.0)
        add()
    return rows
```

- [ ] **Step 3: Add the document writer**

Immediately after `_section_document_lines`, add:

```python
def create_section_document(section, params, out_path, dpi: float = 300.0) -> str:
    """Write a multi-page A4 PDF listing every curtain and string in `section`.

    All pages are added to ONE FilePdf and written once to `out_path`. Temporary
    page views (named _SectionDoc_<code>_pNN) are closed afterwards. Returns
    out_path."""
    from pearlscape import display, sections as smod
    doc = sc.doc
    code = section["code"]
    letters = smod.palette_letters(params.palette)
    section_size = float(params.section_size)
    string_tol = (float(params.string_align_overlap)
                  if params.string_align_overlap > 0.0 else float(params.bead_diameter))
    layer_idx = display._ensure_layer(_DOC_LAYER)
    hatch_idx = display._solid_hatch_index()
    tol = doc.ModelAbsoluteTolerance

    rows = _section_document_lines(section, params, letters, section_size, string_tol)
    per_page = max(1, int((_DOC_PAGE_H - 2.0 * _DOC_MARGIN) / _DOC_LEADING))
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)] or [[]]

    # Idempotency: drop any stale doc pages for this code (same Close() story as
    # create_section_layouts).
    prefix = f"_SectionDoc_{code}_"
    for v in list(doc.Views.GetPageViews()):
        if v.PageName.startswith(prefix):
            v.Close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pdf = Rhino.FileIO.FilePdf.Create()
    created = []
    for pi, page_rows in enumerate(pages):
        page = doc.Views.AddPageView(f"{prefix}{pi:02d}", _DOC_PAGE_W, _DOC_PAGE_H)
        created.append(page)
        y = _DOC_PAGE_H - _DOC_MARGIN
        for row in page_rows:
            x = _DOC_MARGIN + row["indent"]
            if row["swatch"] is not None:
                r, g, b = row["swatch"]
                _page_filled_rect(page, layer_idx, _DOC_MARGIN, y - _DOC_SWATCH,
                                  _DOC_SWATCH, _DOC_SWATCH,
                                  sd.Color.FromArgb(r, g, b), hatch_idx, tol)
                x = _DOC_MARGIN + _DOC_SWATCH + 3.0
            if row["text"]:
                _page_text(page, layer_idx, row["text"], x, y - _DOC_TEXT_H,
                           _DOC_TEXT_H)
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
```

- [ ] **Step 4: Verify in Rhino**

After Task 7 wires it in, run `export_section` in Rhino and open
`Section_<code>_strings.pdf`. Confirm: header with section code + palette name +
bead total; a colour key with swatches; one `Curtain C<NNN> - N beads: ...` line
per curtain; and one `String 001 - offset ...mm - N beads, layout: ... /` line
per string, matching the numbers/offsets on the plan from Task 5.

- [ ] **Step 5: Checkpoint** — user reviews and commits.

---

## Task 7: Wire the document into `export_section`

**Files:**
- Modify: `build_scene.py` (`export_section` block, lines 422-437)
- Verification: in Rhino only.

- [ ] **Step 1: Call the document writer after the plan PDFs**

In `build_scene.py`, the `export_section` block currently ends (lines 434-437):

```python
            out_dir = os.path.join(_HERE, params.pdf_output_dir, "sections", sel["code"])
            t0 = time.time()
            pdf_paths = export_mod.export_all_pdfs(out_dir, prefix="Section_")
            print(f"Exported {len(pdf_paths)} PDFs to {out_dir} in {time.time()-t0:.2f}s")
```

Append immediately after the `print`:

```python
            t0 = time.time()
            doc_path = os.path.join(out_dir, f"Section_{sel['code']}_strings.pdf")
            export_mod.create_section_document(sel, params, doc_path)
            print(f"Exported section document to {doc_path} in {time.time()-t0:.2f}s")
```

- [ ] **Step 2: Verify the full flow in Rhino**

Run the pipeline with `pipeline_mode = "export_section"` and a valid
`section_export_code`. Confirm `.../sections/<code>/` contains both the
`Section_<code>_C<NNN>.pdf` plan sheets (with vertical string labels) and a single
`Section_<code>_strings.pdf` document, and that string numbers/offsets agree
between them.

- [ ] **Step 3: Checkpoint** — user reviews and commits.

---

## Final verification

- [ ] Headless logic green:
  `"C:/Users/lukap/AppData/Local/Python/pythoncore-3.14-64/python.exe" pearlscape/sections.py`
  prints `OK` (covers Tasks 1-4 + the `string_columns` regression).
- [ ] Rhino `export_section` run produces labelled plan sheets + one summary
  document per section, with matching numbering (Tasks 5-7).

## Known limitations (out of scope; revisit if needed)

- **Long string lines** (a string with many beads) can run past the A4 right
  margin in the document — no wrapping yet. If it bites, reduce `_DOC_TEXT_H` or
  add wrapping.
- **Label crowding** on the plan where strings sit ~bead_diameter apart — labels
  kept small (5 mm); add a skip/stagger rule later if dense.
