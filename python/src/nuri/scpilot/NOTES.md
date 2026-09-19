# Shape complementarity pilot — notes

Python pilot for a clean-room Lawrence & Colman (1993) shape complementarity
score. Spec sources: Connolly (1983) for the molecular surface, Quan & Stamm
(2016) for the SES validity predicates, Lawrence & Colman (1993) / CCP4 `sc`
documentation for the statistic and defaults. Rosetta's `sc` app is used only
as a black-box numeric oracle (`rosetta.py`). The algorithms and their proofs
are in ALGORITHMS.md.

## Provenance

Primary source: C. Quan, B. Stamm, "Mathematical analysis and calculation of
molecular surfaces", J. Comput. Phys. 322 (2016) 760–782,
doi:10.1016/j.jcp.2016.07.007. The pilot implements the paper's
characterisation of the SES; ALGORITHMS.md carries its own derivations, and
the paper's Table 1 (`Ases = 32.23514`) is a test fixture.

The authors' Matlab implementation, MolSurfComp
(<https://github.com/quanchaoyu/MolSurfComp>, LGPL-3.0), is distributed under
terms incompatible with this project's Apache-2.0. No code, formula, constant,
tolerance, table, identifier or file structure from it appears in this source
tree. The Python pilot was written from the paper; the C++ port is to be
written from the paper, from this repository and from ALGORITHMS.md, and not
with the Matlab source open. MolSurfComp was read once for a correctness audit
against the pilot. It prunes the concave cut set in four ways that the paper
does not state (Theorem 5.1 removes `B_rp(x)` for every SAS intersection point
`x` within `2rp` and imposes no filter):

| MolSurfComp | There | Here |
|---|---|---|
| high probe ⇒ uncut face, by height over the plane of the three atom centres | asserted | not adopted. Lemma 1(c), proved, on distance to the contact *triangle* |
| only low probes may cut | asserted | not adopted. Corollary of Lemma 4 (cut symmetry), proved |
| first-atom neighbourhood restriction | asserted | not adopted. Unproved, and unsound in a narrow cleft by the `R_a + R_d + 2rp` bound (ALGORITHMS.md §3). Lemma 6 prunes the same caps with a proof |
| keep only the largest departure angle among probes sharing two atoms | asserted | proved (pencil argument), not a code path |

`dist(x, T) ≥ h`, so strictly more faces are uncut here and the
implementations disagree observably (on 1brs, 3782 high faces here against
3520 there; on 2ptc, 1686 against 1555). The closed-form area of an uncut
face is the paper's Gauss–Bonnet formula (5.28) with `χ = 1` and great-circle
edges; the general area formula is (4.19)/(5.25).

Third-party material used as black-box numeric oracles: Rosetta's `sc`
application and MolSurfComp. Only their numeric output is used; no harness,
parser, fixture or I/O code from either is vendored. Rosetta is invoked as an
external binary and is neither linked nor redistributed. United-atom radii
come from a published table (below); no radii file is copied from any program.

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

## Radii

### United-atom radii (default)

Radii from the "Biosym MS" united-atom set for molecular-surface calculation
(table of radii sets for Connolly's MS, J. E. Wampler, UGA): CH₃ 1.95, CH₂ 1.90,
aromatic CH 1.90, C 1.80; N 1.65 (≤1 H), 1.70 (2 H), 1.75 (3 H); O 1.60,
OH 1.70; S 1.90; P 1.80. Hydrogen counts come from standard-residue templates
(`radii.py`). CCP4 documents that the original `sc` program ships Connolly-MS
united-atom radii, which this set reproduces.

### Element vdW radii (Alvarez 2013) × multiplier

For atoms without names. Element radii alone run about 0.05 low in Sc and
0.14 Å high in separation against Rosetta; a uniform multiplier removes the
drift. Sweep over the eleven interfaces below (density 15, Rosetta at its
defaults):

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

Sc and separation are both closest to Rosetta at `k = 1.07` (area alone would
prefer ≈ 1.085); the minimum is sharp at 0.005 resolution. At 1.07 the
per-interface ΔSc is within ±0.01 except 1ar1 A|H (+0.018, the 130 Å²
interface) and matches the united-atom set's rms of 0.007.

## Oracle comparison

Rosetta defaults: rp 1.7, density 15, w 0.5, band 1.5, sep 8. United-atom
radii; PDB files as deposited. Active-atom counts equal Rosetta's "buried
atoms" in every case (differences of ≤7 come from OXT atoms Rosetta adds at
C-termini and altloc handling). `oracle_test.py` checks all eleven interfaces
against Rosetta when the binary is available.

| interface | Sc ours / Rosetta | Δ | sep Δ Å | area Δ |
|---|---|---|---|---|
| 1ar1 H\|L | 0.7034 / 0.7062 | −0.003 | −0.001 | −0.4 % |
| 1ar1 A\|HL | 0.2509 / 0.2460 | +0.005 | −0.016 | −4.0 % |
| 1ar1 A\|H (133 Å²) | 0.3086 / 0.3204 | −0.012 | +0.002 | −8.7 % |
| 1ar1 A\|L | 0.2421 / 0.2343 | +0.008 | −0.055 | −4.6 % |
| 2PTC E\|I | 0.7655 / 0.7665 | −0.001 | +0.002 | −0.5 % |
| 1BRS A\|D | 0.7083 / 0.7200 | −0.012 | +0.010 | +1.7 % |
| 1BRS B\|E | 0.7264 / 0.7283 | −0.002 | +0.006 | −1.0 % |
| 1BRS C\|F | 0.7296 / 0.7312 | −0.002 | +0.009 | −2.1 % |
| 1VFB AB\|C | 0.7301 / 0.7230 | +0.007 | −0.002 | −4.8 % |
| 1VFB A\|B | 0.7756 / 0.7709 | +0.005 | −0.007 | −0.8 % |
| 1CHO EFG\|I | 0.7121 / 0.7048 | +0.007 | −0.012 | −1.4 % |

ΔSc mean +0.000, rms 0.007, max 0.012. Rosetta's own density 5→30 spread is
≈0.015, so the remaining difference is sampling noise.

Density dependence:

| interface | density 5 | 15 | 30 |
|---|---|---|---|
| 1ar1 H\|L ours / Rosetta | 0.699 / 0.696 | 0.703 / 0.706 | 0.708 / 0.710 |
| 2PTC E\|I ours / Rosetta | 0.752 / 0.754 | 0.765 / 0.767 | 0.770 / 0.769 |
| 1ar1 A\|HL ours / Rosetta | 0.249 / 0.245 | 0.251 / 0.246 | 0.252 / 0.252 |

The median is exact; a 0.02-bin interpolated median (Rosetta's estimator)
differs by less than 0.001. Area-weighted vs unweighted median on 1ar1 H|L:
0.7064 vs 0.7055.

Sampler on 1ar1 H (active side toward L), density 15:

| kind | dots/Å² | weight×density 10–90 % |
|---|---|---|
| convex | 14.96 | 0.98–1.03 |
| toroidal | 15.04 | 0.90–1.10 |
| concave | 14.96 | 0.94–1.07 |

## Known deviations from Rosetta (intentional)

- Rosetta builds missing C-terminal OXT atoms; we score the file as given.
- At torus cusps (probe-circle radius < rp) Rosetta emits only one of the two
  arc halves; we keep both valid halves (paper-correct, tiny area effect).
- Rosetta uses `float`; we use double. Sampling patterns differ (Fibonacci
  lattice and equal-area rings vs Rosetta's grid) → per-interface noise
  ≈ ±0.01.
- Peripheral band and buried tests are identical in definition.

## Known limitations

- Two circles of one sphere that coincide exactly (collinear centres with
  matched radii, codimension 2) are merged into one cap whose tag is the
  first member's circle; the other circle then gets no torus arc. If a
  structure ever hits this, the merged label from `_merge_coincident` is
  where the fix goes.
- Under an `active` mask a concave face can in principle miss the cut of a
  probe hosted by atoms up to `2rp` beyond the overlap shell (ALGORITHMS.md
  §3, preparation step 4); not observed on the oracle interfaces.

## Settled design for the C++ port

- Pipeline: a preparation stage (coincident, contained, need-first order)
  followed by kernels that assume clean input; state lives in index ranges
  and sentinels, not masks and guards. One predicate per decision class
  (overlap, containment, coincident caps, hidden/covered, crossing,
  accessibility, arc validity, circle side, active prefixes); consumers read
  the stored result and never re-derive it by another formula or default.
  Crossing on SAS spheres is the triple discriminant; edges survive hiding
  elsewhere, vertices do not.
- Geometry core: cap arrangement per sphere (`Caps` as `(axis, cos α, sin α)`,
  crossing points in the `n₁ ± n₂` basis, global vertex clustering with
  per-cluster excusal, arcs tested against crossing caps only, one sorted
  dart array per sphere with curvature tie-break and snapped tie angles,
  Gauss–Bonnet with patch count from cap components). The same solver serves
  atom spheres and probe spheres.
- Concave cutters: one height pass over all probes, closed-form high faces,
  one pair pass that applies the both-low, beyond-plane and triangle tests
  from both sides and keeps a pair for both faces or neither; every test is
  a necessary condition, so the region solved is the same. Port the pilot's
  brute-force equality test alongside.
- Trig: store `(cos, sin)`, compare trig values or their squares, `atan2`
  only for angles that are consumed as angles (circle φ, dart angles, saddle
  β ranges, spherical excess).
- Public surface API: `ses_dots(coords, radii, rp, density, active_mask)` →
  points, outward normals, per-dot area, owner atom, patch kind, dropped area;
  analytic areas available from the same geometry at no extra cost. Inactive
  atoms occlude but own no convex patch; a toroidal or concave patch is
  emitted when at least one of its atoms is active.
- Sc API: two atom sets (coords + radii each), params `{rp 1.7, density 15,
  weight 0.5, band 1.5, sep 8, clamp 0.999}` → `{sc, sc_a, sc_b, d_median,
  trimmed_area, counts}`. Exact medians. Sc = mean of per-side medians,
  distance = mean of per-side median separations, area = sum of trimmed areas.
  Active atoms are those within `sep` of any partner atom (Rosetta's "buried
  atoms"); every other atom of the molecule occludes (Rosetta's "blocked
  atoms").
- Radii: caller supplies radii. Convenience providers: united-atom template
  (needs residue/atom names, PDB models) with element-vdW fallback; plain
  element vdW × 1.07 as a name-free approximation.
- Tolerances: `TAU_C = 1e-6 Å` (coincidence clustering of vertices and of
  caps; any value in `[1e-8, 1e-4]` passes the suite), direction tie
  `1e-9 rad`. Every other comparison is exact. Any positive accessibility
  slack breaks consistency between vertex acceptance and arc tests; do not
  reintroduce one.
- Square roots: the circle radius `√(R_i² − a²)` is clamped with `max(·, 0)`
  in the pilot only as a NaN guard; the overlap and containment margins make
  the negative case unreachable (`rl ≥ 1.8e-3 Å` at the margin), so in C++ it
  is an `ABSL_DCHECK_GE(x, 0)` before the `sqrt`, not a `max`. If a tolerance
  is ever loosened toward rounding scale, the fix is a filter in the
  preparation stage, never a clamp in a kernel. The probe-pair sine
  `√(1 − cos² α)` is positive because the pair filter `cos α < 1` reads the
  very value the cap carries, and the contact-plane sine `√(1 − (h/rp)²)` is
  computed for low probes only, where `h < rp`. The spindle root
  `max(rp² − rl², 0)` and the crossing half-chord clamp are different: there
  the zero branch is a real case (ordinary torus, pinch) that the formulas
  handle.
