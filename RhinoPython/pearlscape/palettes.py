"""Ready-made colour palettes for Pearlscape.

Each constant is a list of RGB tuples ordered low -> high along the colour-noise
field, so the list reads as a gradient. All are slightly desaturated with lifted
lightness, nature-inspired, and contain no browns (the "orange" entries are light
coral/apricot on purpose).

To use one, in params.py:

    from pearlscape import palettes
    ...
    palette: List[RGB] = field(default_factory=lambda: list(palettes.LAGOON))

Reorder a list to change which colours sit adjacent; trim to 4 entries for broader
blocks of each hue, or keep 5-6 for more transition bands (plays well with
`color_dither`).
"""

from typing import Dict, List, Tuple

RGB = Tuple[int, int, int]


# Deep sea -> foam.
OCEAN_BLUE: List[RGB] = [
    ( 52,  95, 140),
    ( 74, 128, 168),
    (104, 168, 196),
    (140, 200, 206),
    (186, 224, 216),
]

# Forest sage -> pale lichen.
MOSS_GREEN: List[RGB] = [
    ( 92, 120,  86),
    (124, 154, 100),
    (160, 182, 128),
    (192, 206, 158),
    (214, 224, 186),
]

# Coral & apricot vs lilac & violet (bright orange + purple).
ORCHID_SUNSET: List[RGB] = [
    (242, 146,  92),
    (247, 178, 120),
    (214, 160, 196),
    (176, 130, 190),
    (140, 104, 170),
]

# Green / purple.
FERN_AND_ORCHID: List[RGB] = [
    ( 96, 150, 104),
    (140, 182, 132),
    (190, 160, 200),
    (160, 124, 180),
    (126,  96, 150),
]

# Green / purple.
FERN_AND_ORCHID_V2: List[RGB] = [
    (185, 245, 85),
    (170, 55, 140),
]

# Seafoam green -> ocean blue (green / blue).
LAGOON: List[RGB] = [
    (128, 196, 160),
    (108, 190, 192),
    ( 92, 162, 194),
    ( 78, 130, 180),
    ( 70, 104, 158),
]

# Northern-lights green / teal / blue / violet.
AURORA: List[RGB] = [
    (120, 202, 158),
    ( 92, 178, 180),
    ( 96, 142, 196),
    (134, 128, 198),
    (168, 140, 204),
]

# Meadow green, sky, lavender, blossom, butter, soft teal.
WILDFLOWER: List[RGB] = [
    (150, 190, 128),
    (138, 194, 212),
    (186, 166, 210),
    (242, 166, 140),
    (244, 220, 150),
    (124, 176, 150),
]


# Name -> palette, for browsing or iterating (e.g. a palette-cycling preview).
ALL: Dict[str, List[RGB]] = {
    "ocean_blue": OCEAN_BLUE,
    "moss_green": MOSS_GREEN,
    "orchid_sunset": ORCHID_SUNSET,
    "fern_and_orchid": FERN_AND_ORCHID,
    "fern_and_orchid_v2": FERN_AND_ORCHID_V2,
    "lagoon": LAGOON,
    "aurora": AURORA,
    "wildflower": WILDFLOWER,
}


def name_of(palette) -> str:
    """Return a display name for `palette` if it matches a known constant in ALL,
    else 'custom'. Compares by RGB values, order-sensitive."""
    target = [tuple(int(v) for v in c) for c in palette]
    for key, pal in ALL.items():
        if [tuple(int(v) for v in c) for c in pal] == target:
            return key.replace("_", " ").title()
    return "custom"
