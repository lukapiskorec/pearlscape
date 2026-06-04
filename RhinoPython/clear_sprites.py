#! python 3
# r: numpy
"""Hide the sprite beads. Open in Rhino's Script Editor and press F5.

Sprites are drawn by a DisplayConduit, not document geometry, so they can't be
selected or deleted with the normal Rhino tools. This disables the conduit
(stored in scriptcontext.sticky by display.render_sprites) and redraws.

Reuses display.clear_sprite_conduit() so the sticky key and disable logic stay
in one place.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Match build_scene.py: drop cached pearlscape modules so edits take effect.
for _m in list(sys.modules):
    if _m == "pearlscape" or _m.startswith("pearlscape."):
        del sys.modules[_m]

import scriptcontext as sc

from pearlscape import display

display.clear_sprite_conduit()
sc.doc.Views.Redraw()
print("Sprite conduit cleared.")
