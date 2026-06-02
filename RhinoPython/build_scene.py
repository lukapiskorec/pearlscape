#! python 3
"""Top-level entry point. Run this from Rhino's Script Editor (F5)."""

import os
import sys

# Make the pearlscape package importable when running from Rhino.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pearlscape import PearlscapeParams


def main() -> None:
    params = PearlscapeParams()
    params.validate()
    print(f"Pearlscape params loaded: {params.curtain_count} curtains, "
          f"{params.total_surface_samples} beads target.")


if __name__ == "__main__":
    main()
