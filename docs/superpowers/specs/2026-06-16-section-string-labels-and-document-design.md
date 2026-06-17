# Section String Labels + Fabrication Document — Design

**Date:** 2026-06-16
**Scope:** Section PDF exports only. The same methodology will later be extended
to the whole-structure exports, but that is explicitly out of scope here.

## Goal

Add precise, fabrication-ready string-level information to the per-section PDF
exports:

1. Number every string within each curtain (today only the curtain itself is
   labelled, e.g. `C185`–`C267`).
2. On each section-curtain PDF plan, draw each string's number and its
   horizontal offset (mm) "from the left edge", written vertically along the
   string at the bottom.
3. Produce a separate, standalone PDF document per section that lists every
   curtain and every string with its beads, in this format:

   ```
   Curtain C185 — 47 beads: A 20, B 18, C 9
   String 001 - offset 236mm - 4 beads, layout: B 156mm, A 358mm, A 431mm, B 789mm /
   String 002 - offset 241mm - 3 beads, layout: A 274mm, A 390mm, B 672mm /
   ...
   ```

   `A`, `B`, … are placeholder colour letters.

## Decisions

- **Colour → letter mapping:** by palette order. `A = palette[0]`,
  `B = palette[1]`, … following the gradient order in `palettes.py`. The same
  colour maps to the same letter on every sheet and in every document.
- **Per-bead mm value:** measured from the **top** edge of the cube,
  `pos_mm = round((cube_min_z + section_size) − Z)`. Beads are listed
  top → bottom (how beads hang from a fixed point).
- **String numbering:** resets per curtain. Each curtain's strings are numbered
  `001, 002, …` left → right (ascending Y).
- **String offset:** `offset_mm = round(mean_Y − cube_min_y)` — distance from the
  cube's left edge. This is the same value already used to position the string
  line on the plan, so plan and document agree by construction.
- **Rounding:** all displayed mm values are rounded to whole millimetres.
- **Document rendering:** one multi-page PDF per section (approach A), built by
  adding pages to a single `Rhino.FileIO.FilePdf` and writing once.

## Architecture

Single source of truth: all string numbering, offsets and bead positions are
computed once in `sections.py` (pure numpy, headless-testable) and consumed by
both the plan-drawing code and the document writer, so the two can never drift.

### 1. Data layer — `pearlscape/sections.py` (pure numpy)

- `palette_letters(palette) -> Dict[Tuple[int,int,int], str]`
  Maps each palette RGB to a letter (`A`, `B`, …) by palette order. Colours not
  in the palette map to `"?"` (should not occur — bead colours are exact palette
  entries).

- `_cluster_strings(points_2d, tol) -> List[List[int]]`
  Extracted from the existing `string_columns`: groups bead row-indices into
  strings (beads whose Y sit within `tol` of the running cluster). Both
  `string_columns` and `string_layout` call this, so the clustering rule is
  defined once. `string_columns` keeps its current return type (one mean-Y per
  string) by mapping over the clusters.

- `string_layout(points_2d, colors, cube_min, section_size, tol, letters) -> List[dict]`
  One entry per string, ordered left → right:
  ```
  {'number': int,          # 1-based, resets per call (per curtain)
   'offset_mm': int,       # round(mean_Y - cube_min_y)
   'beads': [(letter, pos_mm), ...]}   # sorted top->bottom, pos_mm from top edge
  ```

- `curtain_summary(colors, letters) -> dict`
  `{'n_beads': int, 'by_letter': [(letter, count), ...]}` (reuses
  `color_counts`, mapped through `letters`, most-common first).

### 2. Plan labels — `pearlscape/export.py`

- `_page_text_vertical(page, layer_idx, text, x, y, height, color)` — page-space
  text rotated 90° (reads bottom-to-top), via a rotated `rg.Plane`.

- In `_draw_section_field`, after drawing the string lines, draw each string's
  `number` (zero-padded 3 digits) and `"<offset>mm"` as vertical text anchored at
  the bottom edge of that string's line. Driven by the same `string_layout`
  output used by the document. Label text height ~4–5 mm.

### 3. Summary document — `pearlscape/export.py` + `build_scene.py`

- `create_section_document(section, params, out_path)` builds a multi-page A4
  portrait PDF:
  - **Header:** section code, palette name, and a letter → swatch → RGB legend.
  - **Per curtain:** a stats line, e.g. `C185 — 47 beads: A 20, B 18, C 9`.
  - **Per string:** one line in the target format
    `String 001 - offset 236mm - N beads, layout: <letter> <pos>mm, ... /`.
  - Paginated by line count; all pages added to one `FilePdf`, written once to
    `out_path`. Temporary page views are closed after writing (same
    `RhinoView.Close()` idempotency story as the existing layouts).

- `build_scene.py`, `export_section` block: after `create_section_layouts` +
  `export_all_pdfs`, call `create_section_document` and save alongside the plan
  PDFs as `.../sections/<code>/Section_<code>_strings.pdf`.

## Data flow

```
curtains (aligned, coloured)
   -> assign_sections            -> section dict {code, cube_min, curtains:[plane...]}
   -> palette_letters(palette)   -> {rgb: letter}                (once per run)
   for each plane in section:
     string_layout(plane.points_2d, plane.colors, cube_min, size, tol, letters)
        -> [{number, offset_mm, beads:[(letter,pos_mm)]}]
     curtain_summary(plane.colors, letters) -> {n_beads, by_letter}
   plan:     export._draw_section_field  consumes string_layout (vertical labels)
   document: export.create_section_document consumes string_layout + curtain_summary
```

## Testing

- `sections.py __main__` (headless): `palette_letters` (order + unknown → `?`);
  `string_layout` (per-curtain numbering, left→right order, `offset_mm`,
  top→bottom bead order, `pos_mm` from top edge, letter mapping); `curtain_summary`
  counts; and that the refactored `string_columns` still returns identical values
  to before (regression guard).
- `export.py` is Rhino-only drawing code with no headless harness; verified in
  Rhino by inspecting the generated plan PDFs and the summary document.

## Out of scope

- Whole-structure (non-section) exports.
- Real palette/colour names beyond the placeholder letters.
- Label de-crowding (skip/stagger) for tightly spaced strings — keep labels
  small for now; add a spacing rule later only if the PDFs read as too dense.

## Risk / concerns

- **Label crowding:** where strings sit ~bead_diameter apart, vertical labels can
  overlap horizontally at 1:1. Mitigated by small text; revisit if dense.
- **Sectioning interaction:** unchanged from current behaviour — beads already
  shift slightly in Y/Z before sectioning (string alignment), so string offsets
  are computed on the post-alignment positions that the plan also uses.
