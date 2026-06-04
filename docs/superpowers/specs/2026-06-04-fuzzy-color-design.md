# Fuzzy Color (Dithered Palette) — Design Spec

**Date:** 2026-06-04
**Status:** Approved (brainstorming phase)
**Target environment:** Rhino 8 SR23, CPython 3 runtime, numpy available

## 1. Overview

Soften the hard boundaries between palette colors. Today the FBM colour field is
quantized with `idx = floor(n01 * M)`, so the line where `n01` crosses a bin
boundary is a razor-sharp edge between two palette colours. This adds *stochastic
(dithered) quantization*: near a boundary each bead has a probability of taking
either neighbouring colour, so the transition becomes a dithered band that — over
sparse, spaced beads — reads as a soft gradient while every bead still uses a
discrete palette colour (no interpolated RGB).

## 2. Goals and non-goals

**Goals**
- A `color_dither` parameter (band width in palette-step units) controlling fuzziness.
- `color_dither = 0` reproduces the current sharp output exactly (backward compatible).
- Deterministic per run; pure numpy; negligible cost.

**Non-goals (deferred)**
- Interpolated/blended RGB between palette entries — the user explicitly wants
  discrete colours chosen probabilistically, not new in-between colours.
- Gaussian/smooth dither profile — linear band (uniform dither) only.
- Position-hash dither (identical colours across `"cave"` vs `"curtains"` modes) —
  not needed; the two are separate views.

## 3. Mechanism

In `color.py` `assign_colors`, after computing `n01` (FBM in [0,1)):

```
t = n01 * M                                  # continuous bin coordinate, M = len(palette)
if color_dither > 0:
    u = rng.random(N)                        # per-bead uniform [0,1); rng = default_rng(seed + 1)
    t = t + color_dither * (u - 0.5)         # uniform nudge of width color_dither (step units)
idx = clip(floor(t), 0, M - 1)
return palette[idx]
```

- `color_dither` is in **palette-step units**: `0` = sharp; `1` = a one-step-wide
  band where beads within ±½ step of a boundary mix with linearly-ramping
  probability; `>1` widens the band and can reach past adjacent colours.
- The clip keeps indices valid when the nudge pushes below 0 or to/above M.

## 4. Determinism

The per-bead uniform is drawn from `np.random.default_rng(seed + 1)`, where `seed`
is the existing `color_noise_seed`. The `+ 1` keeps the dither stream independent
of the FBM permutation (`make_perm(seed)`). Same params → identical output.

Note: `assign_colors` is called once over all points in `"cave"` mode and
per-curtain in `"curtains"`/`"export"` modes, so a given bead's dither draw differs
between those modes (different arrays). This is irrelevant — they are separate
views, and the underlying FBM colour field (the spatial structure) is identical
because it is sampled at each bead's 3D position regardless of call grouping.

## 5. Changes

- `color.py`: add `dither: float` keyword to `assign_colors`; implement the
  mechanism above; pass `params.color_dither` through in `apply_to_curtains`.
- `params.py`: add `color_dither: float = 1.0`; `validate()` asserts `>= 0`.
- `build_scene.py`: pass `dither=params.color_dither` to the `"cave"`-mode
  `color_mod.assign_colors(...)` call (the `"curtains"`/`"export"` path goes
  through `apply_to_curtains`, already covered).
- `README.md`: document `color_dither` in the Colour parameter table.

## 6. Verification

Headless (pure numpy; add to a `color.py` `__main__` smoke test or a one-off):
- `color_dither = 0` yields output identical to `floor(n01 * M)` quantization.
- `color_dither > 0` changes some bead colours only near boundaries (interior
  regions far from any boundary are unchanged); all returned indices map to valid
  palette entries.
- Deterministic: two identical calls produce identical arrays.

Manual (Rhino F5): `pipeline_mode="cave"`, raise `color_dither` from 0 → 1 → 2 and
confirm the colour boundaries soften into dithered bands; `color_dither=0` matches
the previous look.
