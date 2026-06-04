# Ridged Noise — Design Spec

**Date:** 2026-06-04
**Status:** Approved (brainstorming phase)
**Target environment:** Rhino 8 SR23, CPython 3 runtime, numpy available

## 1. Overview

Add ridged-multifractal noise as an alternative to FBM for the cave wall
displacement. Current FBM produces smooth, rolling walls; ridged noise produces
sharp, craggy ridges for a more alien/eroded look. This is the first of three
independent feature additions (the others — sprite display, irregular base
geometry — are separate specs).

## 2. Goals and non-goals

**Goals**
- A `ridged3_01` noise function in `noise.py`, drop-in compatible with `fbm3_01`.
- A `noise_type` parameter (`"fbm" | "ridged"`) toggling the cave wall displacement.
- Reuse existing `fbm_octaves` / `fbm_lacunarity` / `fbm_gain` params — no new knobs.
- Default `"fbm"` so existing caves render identically.

**Non-goals (deferred)**
- Ridged option for the color field (geometry only for now).
- A sign flip to make ridges protrude inward as fins (standard inward-crevice
  convention only; flip can be exposed later if wanted).

## 3. Algorithm

Faithful Musgrave weighted ridged multifractal, matching the supplied reference.
Per octave, with `value=0`, `amplitude=1`, `frequency=1`, `weight=1`:

```
n      = 1 - |perlin3(p * frequency)|   # perlin3 is already [-1,1]; no sub(0.5).mul(2) step
signal = n * n * weight                 # square sharpens ridges; weight feeds back
value += signal * amplitude
weight  = clamp(signal, 0, 1)           # previous octave's signal attenuates the next
frequency *= lacunarity
amplitude *= gain
```

Output normalized by `Σ amplitudeᵢ` (the theoretical max, reached when every
octave's `n=1`), giving a true `[0, 1]` range. This preserves the meaning of
`fbm_amplitude` as the maximum inward radial displacement, identical to FBM.

Note the starting `amplitude=1` (vs the reference's `0.5`); since the output is
normalized by the amplitude sum, the absolute start value is irrelevant — only
`gain` shapes the per-octave falloff.

## 4. Visual consequence

Wall displacement is inward (`r_eff = radius − fbm_amplitude · n01`). Ridged
peaks (where Perlin crosses zero coherently across octaves → `n≈1`) therefore
become **sharp inward crevices**, with flatter rock between them — gouged, craggy
walls rather than smooth rolling FBM.

## 5. Changes

- `noise.py`: add `ridged3_01(p, perm, *, octaves, lacunarity, gain)`.
- `params.py`: add `noise_type: str = "fbm"`; assert membership in `validate()`.
- `cave.py`: `CylinderFBMCave` takes `noise_type`, selects the noise fn for wall
  displacement; `make_default_cave` passes `params.noise_type` through.
- `noise.py` `__main__`: extend smoke test — `ridged3_01` output in `[0,1]`,
  differs from FBM, deterministic.

## 6. Verification

Manual: run `noise.py` directly (smoke test prints), then run `build_scene.py`
in `pipeline_mode="cave"` with `noise_type="ridged"` and confirm the cave wall
shows sharp crevices. FBM default path must look unchanged.
