# Pearlscape — Rhino Python

## Running

1. Open Rhino 8.
2. Run the `_ScriptEditor` command.
3. Open `build_scene.py`.
4. Press F5.

All parameters live in `pearlscape/params.py`. Edit and re-run.

## Python runtime

Rhino 8's Script Editor supports both IronPython 2 and CPython 3. This project requires **CPython 3** (for type annotations, dataclasses, and numpy). Every entry-point script begins with `#! python 3` to select the right runtime. If you create a new script you intend to run directly with F5, add that same first line.

## Layout

- `pearlscape/` — library modules (one per concern).
- `build_scene.py` — entry point.
- `exports/` — generated PDFs (created on first export).
