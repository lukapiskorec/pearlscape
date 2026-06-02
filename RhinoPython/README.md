# Pearlscape — Rhino Python

## Running

1. Open Rhino 8.
2. Run the `_ScriptEditor` command.
3. Open `build_scene.py`.
4. Press F5.

All parameters live in `pearlscape/params.py`. Edit and re-run.

## Python runtime

Rhino 8's Script Editor supports both IronPython 2 and CPython 3. This project requires **CPython 3** (for type annotations, dataclasses, and numpy). Every entry-point script begins with two directive lines:

```
#! python 3
# r: numpy
```

`#! python 3` selects the CPython 3 runtime (the default for `.py` is IronPython 2). `# r: numpy` tells the Script Editor to ensure numpy is installed in Rhino's per-project Python environment. If you create a new script you intend to run directly with F5, add both lines at the top.

## Layout

- `pearlscape/` — library modules (one per concern).
- `build_scene.py` — entry point.
- `exports/` — generated PDFs (created on first export).
