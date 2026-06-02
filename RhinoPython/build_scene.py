#! python 3
# r: numpy
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Rhino's Script Editor caches imported modules across F5 runs. Drop any
# cached pearlscape modules so source edits take effect on every run.
for _m in list(sys.modules):
    if _m == "pearlscape" or _m.startswith("pearlscape."):
        del sys.modules[_m]

import time

from pearlscape import PearlscapeParams
from pearlscape.cave import make_default_cave
from pearlscape import display


def main() -> None:
    params = PearlscapeParams()
    params.validate()

    t0 = time.time()
    cave = make_default_cave(params)
    pts = cave.sample_surface_points()
    print(f"Generated {len(pts)} cave surface points in {time.time()-t0:.2f}s")

    display.render_cave_reference(pts)
    print("Rendered cave reference. Look at the viewport.")


if __name__ == "__main__":
    main()
