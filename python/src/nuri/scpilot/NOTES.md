# Shape complementarity pilot — notes

Python pilot for a clean-room Lawrence & Colman (1993) shape complementarity
score. Spec sources: Connolly (1983) for the molecular surface, Quan & Stamm
(2016) for the SES validity predicates, Lawrence & Colman (1993) / CCP4 `sc`
documentation for the statistic and defaults. Rosetta's `sc` app is used only
as a black-box numeric oracle (`rosetta.py`).

## Layout

| file | role |
|---|---|
| `io.py` | `nuri.fmt.pdb` loader → `Structure` (heavy atoms, no HETATM), chain selection |
| `radii.py` | united-atom radii from residue templates (see below) |
| `arrangement.py` | arrangement of spherical caps on one sphere: caps as `(axis, cos α, sin α)`, arcs, vertices, dart-array loop walk, Gauss–Bonnet area |
| `anal.py` | `prepare` (overlaps, contained balls, need-first atom order); analytic SAS (`SasGeometry`) and SES (`SesGeometry`): circles, caps gathered from circle rows, probes, torus arcs, saddle ranges, concave faces, exact areas |
| `surface.py` | SES dot sampler driven by `SesGeometry`: convex / toroidal / concave dots, normals, weights |
| `sc.py` | one `_Side` per molecule: active atoms, buried dots, peripheral trim, nearest pairing, medians |
| `rosetta.py` | run + parse `sc.linuxgccrelease -sc:verbose` |
| `cli.py` | `run`, `sweep` (typer) |

Run: `PYTHONPATH=$PWD/python/src $B/python -m nuri.scpilot.cli run test/test_data/1ar1.pdb --pairs H:L,A:HL`
(`B=~/anaconda3/envs/nk-dev/bin`). Tests: `python/test/scpilot/`. The oracle
structures (`1ar1`, `2ptc`, `1brs`, `1vfb`, `1cho`) are in `test/test_data/`.

## Pipeline

`ses_dots` runs `prepare` first: coincident atoms raise, contained balls are
dropped, and atoms are permuted to `[active | need | occluders]` (`need` =
active ∪ neighbours of active). Every kernel after that assumes clean input
and reads state from index ranges instead of masks: spheres `< n_solve` get
arrangements, spheres `< n_active` own dots, a circle or arc is active iff
its smaller atom is `< n_active`, a probe iff its smallest atom is. Dot
owners are mapped back through `order` at the end. Exceptional cases that
would otherwise be branches inside kernels are data instead: a `-1`
appended to the probe map so full-circle ends (`-1`) read back `-1`,
self-successors so a full circle is its own loop, zero-width saddle ranges
where a spindle part is absent.

## Geometry core

Every sphere (atom SAS sphere or probe sphere) is an independent
*arrangement of caps* (`arrangement.py`). A cap is `(axis, cos α, sin α)`;
the accessible region is the sphere minus the union of caps. No angle is
recovered by inverse trig: comparisons are made on trig values (`contains`
is `n·x > cos α`, nested is `cos γ ≥ c₁c₂ + s₁s₂`), and `atan2` appears only
where an angle is consumed as an angle (φ along a circle, dart angles).

- Coincident caps (5-vectors `R·(axis, cos α, sin α)` within `TAU_C`) are
  merged first. Cap pairs are then classified by one set of trig-value
  comparisons: nested (`cos γ ≥ c₁c₂ + s₁s₂`, dropped), apart
  (`cos γ ≤ c₁c₂ − s₁s₂`; a cover of the whole sphere when `c₁ + c₂ < 0`,
  which ends the solve as `covered`, disjoint otherwise), crossing in
  between. Crossing points are written in the `n₁ ± n₂` basis so nearly
  parallel caps lose no precision. Crossing points of the same pair are
  never merged; coincident points from *different* pairs are clustered
  (`TAU_C = 1e-6 Å`). Accessibility of a cluster is an exact comparison
  against every cap except those whose own crossing points merged into it.
- Arcs are the accessible pieces of each cap circle, built for all caps of a
  sphere in one pass from the (vertex, cap) incidences. A candidate arc is
  kept iff its midpoint is outside every cap **that crosses its circle**:
  a disjoint cap cannot contain any of the circle, and testing it anyway at
  an exact tangency is a rounding coin flip that contradicted the crossing
  decision (no vertex there, yet the arc vetoed).
- Loops: every arc is an out-dart at `v_end` and a reversed in-dart at
  `v_beg`. All darts of the sphere are sorted once by `(vertex, angle,
  curvature)`; darts within `1e-9 rad` share one snapped angle and are
  ordered by signed geodesic curvature so the wedge between them is exactly
  zero (tangent circles, pinches). The successor of an in-dart is its
  predecessor in that order; a full circle has no darts and is its own loop.
  Non-alternating darts raise `DegenerateGeometryError`.
- Area: `A = R² [2πχ − Σ turn + Σ Δφ cos α]`, `χ = 2·n_patches − n_loops`,
  `n_patches = 1 + n_loops − n_cap_components` (cap components from the same
  crossing decisions, computed for all spheres in one connected-components
  call).

`anal.py` builds on that:

- Circles are the single source of pair geometry. The caps of a solved
  sphere are a slice of one array gathered from the circle rows:
  `cos α = a/R`, `sin α = rl/R`, axis `u` and frame `(e1, e2)` for the
  smaller sphere, axis `−u` and frame `(e1, −e2)` for the larger. Caps are
  tagged by circle id, so the arcs of the smaller sphere *are* the torus
  arcs, with φ carried over unchanged.
- SAS: triple vertices are intersected **once** per sorted triple `(i<j<k)`
  (candidates from the neighbour lists, circle `(i,j)` against sphere `k`,
  points by algebra without trig) and the identical points are handed to all
  three spheres, then clustered globally. Without this, exact tangencies
  gave a vertex on one sphere and none on the others (phantom faces). A
  cluster's accessibility is decided once, on the sphere of its smallest
  atom against that sphere's caps; every ball that could contain a point of
  the sphere is one of its caps, so this equals the test against all SAS
  balls without a KD-tree sweep.
- Spindle tori (`rl < rp`) are the hard case: every probe on such a circle
  is exactly `rp` from the two cusp points, so k probe spheres and the
  adjacent face sides all pass through one point. Clustering plus dart
  walking handles this; nothing is merged heuristically, and inconsistent
  geometry raises `DegenerateGeometryError`.
- Saddle per circle: generating angle β from the inward radial direction,
  `θ = atan2(a, rl)` per side, valid ranges `[-θ_i, -β0] ∪ [β0, θ_j]` with
  `β0 = atan2(√(rp² − rl²), rl)` (zero for `rl ≥ rp`, so the two ranges tile
  the arc); absent parts are zero-width. Stored as one `(n_circles, 2, 2)`
  array; closed-form area per arc.
- Concave face per probe: arrangement on the probe sphere whose caps are the
  neighbouring probes within `2rp` (`cos α = d/2rp`) **plus one hemisphere
  per accessible arc leaving the probe** (axis = departure tangent
  `u × radial`). For an ordinary vertex these
  are the three side planes of the contact triangle; merged k-fold vertices
  and probes that are the only vertex of a circle (zero face) follow from the
  same rule. Same-circle rolling probes need no extra cut: their caps are
  bounded by planes through the torus axis, a pencil whose union over an arc
  is side-hemisphere ∪ cap(end vertex).

## Sampler

Dots live inside the exact patch domains; weights are normalised so that
Σ weights per patch equals the analytic area (plus a tiny `dropped_area` for
patches too small to receive a dot). Dot **counts** are the sampled quantity.

- Convex: Fibonacci lattice with `round(4πr²·density)` points, keep directions
  outside every cap of the atom's arrangement.
- Toroidal: per arc and valid β range, `round(rp·|range|·√density)` rings of
  equal β width; per ring `k_φ = round(ring_area·density)` (exact ring
  integral), φ uniform inside the arc; ranges whose rings all round to zero get
  one ring. All (arc, range) rows are expanded to rings and dots in one
  vectorised pass. A uniform (φ, θ) grid would scale density as 1/ρ and
  over-count saddles; rings sized by exact area avoid that.
- Concave: Fibonacci lattice on the probe sphere rotated to the contact
  centroid, keep directions outside every cap of the face arrangement.

## Validation

Analytic areas vs sampled dots (`assert_dots_match`, `assert_valid_ses`):
weights reproduce the analytic areas to 1e-9 per kind, and dot counts follow
the density within 3 % (1 % on the protein fragments) for Stamm Table 1
(`32.23514`, exact), the cusp case, 3 spheres, 8 random spheres and 1ar1 H
fragments of 88 and 171 atoms. No dot lies inside any atom and no dot's
probe inside any SAS ball.

Degenerate sweeps (`anal_test.py`): a fourth sphere at ε ∈ ±[1e-1 … 1e-12, 0]
from a triple vertex, and tangent to a torus circle. Areas are finite and
converge on each side; below `TAU_C` the geometry is treated as exactly
coincident and equals the ε = 0 value. The area is genuinely discontinuous
across ε = 0 in the four-sphere case (8 probes for ε > 0, 4 for ε < 0).

Sampler on 1ar1 H (active side toward L), density 15:

| kind | dots/Å² | weight×density 10–90 % |
|---|---|---|
| convex | 14.96 | 0.98–1.03 |
| toroidal | 15.04 | 0.90–1.10 |
| concave | 14.96 | 0.94–1.07 |

No dot lies inside any other probe ball, sampled or from the full-molecule
SES (0 of 495k over the eleven interfaces); no cusp slivers; exact tangency
gives no phantom face (`test_exact_tangency_*`).

## Oracle comparison (Rosetta defaults: rp 1.7, density 15, w 0.5, band 1.5, sep 8)

Active-atom counts equal Rosetta's "buried atoms" in every case (differences
of ≤7 come from OXT atoms Rosetta adds at C-termini and altloc handling).

United-atom radii, current sampler (PDB files fetched from RCSB, chains as
deposited):

| interface | Sc ours / Rosetta | Δ | sep Δ Å | area Δ |
|---|---|---|---|---|
| 1ar1 H\|L | 0.7042 / 0.7062 | −0.002 | −0.003 | −0.4 % |
| 1ar1 A\|HL | 0.2514 / 0.2460 | +0.005 | −0.021 | −4.2 % |
| 1ar1 A\|H (133 Å²) | 0.3070 / 0.3204 | −0.013 | +0.000 | −8.6 % |
| 1ar1 A\|L | 0.2426 / 0.2343 | +0.008 | −0.061 | −4.7 % |
| 2PTC E\|I | 0.7662 / 0.7665 | −0.000 | +0.001 | −0.4 % |
| 1BRS A\|D | 0.7073 / 0.7200 | −0.013 | +0.012 | +1.8 % |
| 1BRS B\|E | 0.7260 / 0.7283 | −0.002 | +0.006 | −1.0 % |
| 1BRS C\|F | 0.7304 / 0.7312 | −0.001 | +0.010 | −2.1 % |
| 1VFB AB\|C | 0.7306 / 0.7230 | +0.008 | −0.004 | −4.6 % |
| 1VFB A\|B | 0.7749 / 0.7709 | +0.004 | −0.007 | −0.9 % |
| 1CHO EFG\|I | 0.7121 / 0.7048 | +0.007 | −0.013 | −1.3 % |

Summary: ΔSc mean +0.000, rms 0.007, max 0.013. Rosetta's own density
5→30 spread is ≈0.015, so the remaining difference is sampling noise.
`oracle_test.py` checks all eleven interfaces against Rosetta when the
binary is available.

### Element vdW radii (Alvarez 2013)

Element radii alone run about 0.05 low in Sc and 0.14 Å high in separation
against Rosetta; a uniform multiplier removes the drift.

Multiplier sweep over all eleven interfaces (density 15, Rosetta at its
defaults), for atoms without names:

| k | ΔSc mean | ΔSc rms | ΔSc max | Δsep mean | Δsep rms | Δarea mean |
|---|---|---|---|---|---|---|
| 1.00 | −0.053 | 0.069 | 0.104 | +0.138 | 0.168 | −13.6 % |
| 1.04 | −0.014 | 0.028 | 0.039 | +0.040 | 0.065 | −8.1 % |
| 1.06 | +0.000 | 0.017 | 0.028 | −0.004 | 0.041 | −5.0 % |
| 1.065 | +0.002 | 0.011 | 0.023 | −0.008 | 0.026 | −4.0 % |
| **1.07** | +0.002 | **0.007** | 0.018 | −0.009 | **0.012** | −2.5 % |
| 1.075 | +0.004 | 0.010 | 0.020 | −0.017 | 0.020 | −1.3 % |
| 1.08 | +0.008 | 0.012 | 0.022 | −0.027 | 0.030 | −1.0 % |
| 1.10 | +0.018 | 0.023 | 0.040 | −0.053 | 0.061 | +2.0 % |

Sc and separation are both closest to Rosetta at `k = 1.07` (area alone
would prefer ≈ 1.085); the minimum is sharp at 0.005 resolution. At 1.07 the
per-interface ΔSc is within ±0.01 except 1ar1 A|H (+0.018, the 130 Å²
interface) and matches the united-atom set's rms of 0.007.

### United-atom radii (default) — parity

Radii from the "Biosym MS" united-atom set for molecular-surface calculation
(table of radii sets for Connolly's MS, J. E. Wampler, UGA): CH₃ 1.95, CH₂ 1.90,
aromatic CH 1.90, C 1.80; N 1.65 (≤1 H), 1.70 (2 H), 1.75 (3 H); O 1.60,
OH 1.70; S 1.90; P 1.80. Hydrogen counts come from standard-residue templates
(`radii.py`), so no radii file is copied from any program. CCP4 documents that
the original `sc` program ships Connolly-MS united-atom radii, which this set
reproduces.

Parity numbers are the table above.

### Density and median estimator

| interface | density 5 | 15 | 30 |
|---|---|---|---|
| 1ar1 H\|L ours / Rosetta | 0.694 / 0.696 | 0.704 / 0.706 | 0.709 / 0.710 |
| 2PTC E\|I ours / Rosetta | 0.755 / 0.754 | 0.766 / 0.767 | 0.771 / 0.769 |
| 1ar1 A\|HL ours / Rosetta | 0.244 / 0.245 | 0.251 / 0.246 | 0.251 / 0.252 |

The median is exact; a 0.02-bin interpolated median (Rosetta's estimator)
differs by less than 0.001.
Area-weighted vs unweighted median on 1ar1 H|L: 0.7064 vs 0.7055.

### Runtime (python, 1ar1 chain H, 921 atoms, 266 active)

1.0 s per side: `SasGeometry.build` 0.65 s, `SesGeometry.build` 0.27 s,
sampling 0.04 s. Full analytic
SES of the chain with every atom active: 1.7 s (SAS 6366 Å², SES 5007 Å²).
Rosetta: 1.9 s per interface (brute-force O(N²) C++). What remains is one
Python iteration per solved sphere and per concave face; everything inside
is vectorised.

## Known deviations from Rosetta (intentional)

- Rosetta builds missing C-terminal OXT atoms; we score the file as given.
- At torus cusps (probe-circle radius < rp) Rosetta emits only one of the two
  arc halves; we keep both valid halves (paper-correct, tiny area effect).
- Rosetta uses `float`; we use double. Sampling patterns differ (Fibonacci
  lattice and equal-area rings vs Rosetta's grid) → per-interface noise
  ≈ ±0.01.
- Peripheral band and buried tests are identical in definition.

## Settled design for the C++ port

- Pipeline: a preparation stage (coincident, contained, need-first order)
  followed by kernels that assume clean input; state lives in index ranges
  and sentinels, not masks and guards.
- Geometry core: cap arrangement per sphere as above (`Caps` as
  `(axis, cos α, sin α)`, crossing points, global vertex clustering with
  per-cluster excusal, arcs tested against crossing caps only, one sorted
  dart array per sphere with curvature tie-break and snapped tie angles,
  Gauss–Bonnet with patch count from cap components). Same solver serves
  atom spheres and probe spheres.
- Trig: store `(cos, sin)`, compare trig values, `atan2` only for angles that
  are consumed as angles (circle φ, dart angles, saddle β ranges).
- Public surface API: `ses_dots(coords, radii, rp, density, active_mask)` →
  points, outward normals, per-dot area, owner atom, patch kind, dropped area;
  analytic areas available from the same geometry at no extra cost. Inactive
  atoms occlude but own no convex patch; a toroidal or concave patch is
  emitted when at least one of its atoms is active. Only active atoms and
  their overlap neighbours are solved; this is exact for convex and toroidal
  patches by construction, and a concave face could in principle miss the cut
  of a competing probe hosted by atoms up to `2rp` beyond the overlap shell.
  On the eleven oracle interfaces no such probe exists: 0 of 495k masked
  dots lie inside any probe ball of the full-molecule SES (25.6k vertex
  probes against 11.8k in the masked runs).
- Sc API: two atom sets (coords + radii each), params `{rp 1.7, density 15,
  weight 0.5, band 1.5, sep 8, clamp 0.999}` → `{sc, sc_a, sc_b, d_median,
  trimmed_area, counts}`. Exact medians. Sc = mean of per-side medians,
  distance = mean of per-side median separations, area = sum of trimmed areas.
  Active atoms are those within `sep` of any partner atom (Rosetta's "buried
  atoms"; counts agree); every other atom of the molecule occludes, as
  Rosetta's "blocked atoms" (its count is all non-buried atoms) do.
- Radii: caller supplies radii. Convenience providers: united-atom template
  (needs residue/atom names, PDB models) with element-vdW fallback; plain
  element vdW × 1.07 as a name-free approximation.
- Tolerances: `TAU_C = 1e-6 Å` (coincidence clustering of vertices and of
  caps; any value in `[1e-8, 1e-4]` passes the suite), direction tie
  `1e-9 rad`. Every other comparison is exact. Any positive accessibility
  slack breaks consistency between vertex acceptance and arc tests; do not
  reintroduce one.
