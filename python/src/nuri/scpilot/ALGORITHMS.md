# Algorithms: `arrangement.py`, `anal.py`, `surface.py`

How the pilot computes the analytic solvent-accessible surface (SAS) and
solvent-excluded surface (SES) and places dots on the SES. Written for the C++
port: every step is stated as an exact geometric predicate, and the reason
behind each tolerance is given.

Two conventions run through everything. **Filter first**: exceptional input
(coincident atoms, contained balls, inactive atoms, `rp ≤ 0`) is removed or
partitioned in a preparation stage, and no kernel afterwards carries a guard
for it. **Trig values, not angles**: caps store `(cos α, sin α)`, predicates
compare trig values (`acos(x) ≤ a` is written `x ≥ cos a`), and `atan2` is
used only where an angle is consumed as an angle.

## 1. Definitions

- Atom `i` has centre `c_i`, van der Waals radius `r_i`, and SAS radius
  `R_i = r_i + rp`, where `rp` is the probe radius.
- `B(c, R)` is the closed ball, `B°(c, R)` its interior, `S(c, R)` the
  sphere. `U = ⋃_i B(c_i, R_i)` is closed. The **SAS** is the boundary of
  `U`; its points are the accessible probe centres. An SAS point `s`
  *touches* the atoms `a` with `|s − c_a| = R_a`.
- The **SES** is the boundary of `{p : B(p, rp) ⊂ U}`: the points of `U` at
  distance exactly `rp` from the SAS. A point closer than `rp` to some SAS
  point `s` is *cut* by `s`.
- The SES decomposes into three patch families (Connolly 1983):
  - **convex**: the probe touches one atom; the patch is the atom's vdW sphere
    scaled from the accessible part of its SAS sphere;
  - **toroidal (saddle)**: the probe rolls on two atoms; its centre traces an
    arc of the circle where the two SAS spheres meet;
  - **concave**: the probe rests on three (or more) atoms; its centre is a
    **vertex** of the SAS and the patch is a region of the probe sphere.
- A **cap** on a unit sphere is `{x : n · x > cos α}` for a unit axis `n`,
  stored as `(n, cos α, sin α)`. The cap that a ball `B(y, rp)` cuts on the
  probe sphere `S(x, rp)` has axis `(y − x)/|y − x|` and
  `cos α = |x − y| / 2rp`; it is empty unless `|x − y| < 2rp`, and `α ≤ π/2`.

Every patch family is derived from **arrangements of caps**: the accessible
part of a SAS sphere is the sphere minus the caps cut by neighbouring SAS
spheres, and the concave face of a vertex is the probe sphere minus the caps
cut by neighbouring vertex probes and by the arcs it rolls off along.

## 2. `arrangement.py` — caps on one sphere

### Input and output

The solver takes a sphere radius `R` and a list of caps and returns

- `arcs`: for each cap circle the accessible sub-arcs, as records
  `Arc(cap, v_beg, v_end, phi_beg, dphi)` in the circle's own frame
  `(e1, e2, n)`; `v_* = -1` marks a full circle;
- `n_loops`, `n_patches`, `area`.

Caps are records `Cap(axis, cos α, sin α)`, arcs `Arc(...)`, and the same
holds for every entity below: storage is one record per geometric object,
and a kernel that evaluates one expression over many records (containment
of directions in all caps of a sphere, the Gram matrix of cap axes)
gathers the field it needs into an array at the call site.

The kernel (`solve`) takes an `ArrangementProblem`: the hygienic caps with
their circle frames `(e1, e2)`, the crossing matrix and its number of
components, and the clustered vertices (unit `reps`, the incident caps of
each vertex as `excused`, and `accessible`). Two preparations build it:
`local_problem` for a probe sphere, where steps 1–4 below run on the sphere
alone with the default frame `e1 = any_perpendicular(n)`, `e2 = n × e1`;
and the SAS build (§3), where the crossing decision, the vertices and the
frames come from the circles shared between spheres. The kernel itself
(steps 5–7) has no branch on the source.

### Step 1 — cap hygiene (`prepare_caps`)

On a probe sphere, caps whose circles coincide to within `TAU_C` — the
5-vectors `R · (n, cos α, sin α)` closer than `TAU_C`, clustered exactly like
vertices in step 3 — are one circle computed through different routes (the
departure hemispheres of tangent arcs at a pinch, for instance) and are
merged into their renormalised mean. On a SAS sphere no two caps share a
circle: the preparation stage (§3) removes the atoms that would cause it.

Every remaining pair is then classified once, from trig values only
(`pair_predicates`). With `cos γ = n_j · n_k` and `c, s` the stored cosines
and sines:

- **nested**: `cos γ ≥ c_j c_k + s_j s_k` (`γ ≤ |α_j − α_k|`): the smaller cap
  lies inside the larger and is dropped;
- **apart**: `cos γ ≤ c_j c_k − s_j s_k` (`γ ≥ α_j + α_k` or
  `γ ≥ 2π − α_j − α_k`). With `c_j + c_k < 0` the two caps **cover** the
  sphere: the solve ends with area 0 and no patches. Otherwise they are
  **disjoint**;
- **crossing**: everything in between.

On a probe sphere these three comparisons are the only source of the
classification. On a SAS sphere the crossing decision is made elsewhere, once
per sphere triple (§3), and is passed in; the remaining pairs are split into
nested and apart by `cos γ > c_j c_k`, which lies `s_j s_k` away from either
boundary and therefore never contradicts a crossing decision reached by
another route. Hiding is applied before the covered exit. Hiding, the covered
exit, the crossing graph of step 7 and the arc test of step 5 all read this
one classification.

### Step 2 — crossing points (`crossing_points`)

For a crossing pair the two circle intersections solve `n_1 · x = c_1`,
`n_2 · x = c_2`, `|x| = 1`. The point of the intersection line nearest the
origin is written in the basis `m = n_1 + n_2`, `w = n_1 − n_2` (orthogonal,
`|m|² = 2(1 + g)`, `|w|² = 2(1 − g)`):

```
base = (c_1 + c_2)/|m|² · m + (c_1 − c_2)/|w|² · w
h² = max(1 − |base|², 0),   x± = base ± h (n_1 × n_2)/|n_1 × n_2|
```

The textbook form `A n_1 + B n_2` with `A = (c_1 − c_2 g)/(1 − g²)` amplifies
a rounding error `ε` in the cosines to `ε/γ²` for nearly parallel axes; the
geometry itself only moves by `ε/γ`, which is what the `m, w` form gives. The
clamp on `h²` handles a pair the predicate calls crossing but rounding puts
exactly at tangency: both points coincide and step 3 merges them into a pinch.

### Step 3 — vertex clustering (`cluster_points`)

Raw crossing points closer than `TAU_C = 1e-6 Å` are merged into one cluster
(connected components of the KD-tree pair graph). This merges points that are
the *same* geometric vertex produced by different cap pairs: `k` circles
through one point yield `k(k−1)/2` raw points. The two crossing points of a
*single* pair are never closer than `2h`, so they merge only when the circles
are tangent to within the tolerance, which is the intended "pinch" case.

The cluster representative is the renormalised mean. Its incident caps are the
union of the generating pairs of its members. Callers may supply clustering
labels computed elsewhere (§3, global SAS clustering).

### Step 4 — vertex accessibility

A cluster is accessible iff it lies outside every cap **except those whose own
crossing points merged into it**: `n_c · x ≤ cos α_c` for all non-incident
`c`, compared exactly. Excusing exactly the generating caps makes the decision
deterministic at k-fold points, where the sign of the residual for an incident
cap is noise. No epsilon is needed: a non-incident cap passing through the
point within noise would have deposited its own crossing points within noise
of it, so clustering would already have made it incident.

Any positive slack breaks consistency: a vertex inside cap `l` by `δ` would be
accepted while the arc it starts is rejected by the midpoint test of step 5,
leaving a vertex with an odd number of darts.

### Step 5 — arcs (`_cap_arcs`)

Cap by cap: the accessible vertices incident to the cap get the angle
`φ = atan2(x · e2, x · e1)` in the cap frame and are sorted by it;
consecutive vertices bound the candidate arcs (wrapping around by `2π`; a
cap with exactly one vertex yields one arc from the vertex around to itself
with `dphi = 2π`). A cap with no vertices contributes one full-circle
candidate with `-1` ends.

A candidate is kept iff its midpoint is outside every cap **that crosses its
circle** (exact comparison). A disjoint cap cannot contain any point of the
circle, and a cap containing the whole circle would have made it nested
(step 1). Consulting them anyway is harmful: at an exact tangency the midpoint
can sit exactly on the tangent cap's boundary, and the comparison becomes a
rounding coin flip that can veto an arc although the crossing decision
produced no vertex there. The arc test and the crossing decision must be the
same decision. A sliver arc between the two crossing points of a near-tangent
pair lies inside the other cap by about `h²/2R > 0` and is always rejected;
once `2h < TAU_C` the two points merge into a pinch instead.

### Step 6 — face walking with dart rings (`_walk`)

Arcs are stored with increasing `φ`, which runs counter-clockwise around the
cap axis with the cap on the left. The accessible region is therefore
traversed from `v_end` to `v_beg`. Each arc is one *dart* leaving `v_end` and
arriving at `v_beg`.

Every arc with vertices contributes two darts, `Dart(arc, angle, κ, is_in)`,
appended to the ring of their vertex: an out-dart at `v_end` with departure
tangent `t = −(n × u)/|n × u|` and a reversed in-dart at `v_beg` with tangent
`+(n × u)/|n × u|`, projected into the tangent plane of the vertex `u` and
measured as an angle in a frame there. Signed geodesic curvature is `−cot α`
for out-darts and `+cot α` for reversed in-darts, with `cot α = cos α / sin α`
from the stored values.

Each vertex ring is then ordered on its own (`_dart_ring`): sort by angle;
darts whose raw angles differ from their predecessor by less than
`_TAU_DIR = 1e-9 rad` form a group (tangent circles, pinches) and take the
group's first angle; a group straddling the `−π/π` seam is recognised by the
wrap gap between the raw first and last angle and takes the first angle too.
A second sort by `(snapped angle, curvature)` puts right-curving darts first,
so the wedge between grouped darts is exactly zero. In the resulting cyclic
order, reversed-in and out darts must strictly alternate; otherwise
`DegenerateGeometryError` is raised (nothing is merged or dropped silently).

The successor of an in-dart is the previous dart in that order (the first
out-dart clockwise from its reversed tangent). The interior angle of the
region at that corner is `ι = angle(rev-in) − angle(out)` taken in `[0, 2π)`
and the turning angle is `π − ι` (a pinch has `ι = 0`, turn `+π`). Successors
default to the arc itself, so a full circle is its own loop; loops are the
cycles of the successor permutation, counted by one visited-flag walk.

### Step 7 — Gauss–Bonnet area

For a region with the boundary on its left, on a sphere of radius `R`:

```
A = R² · [ 2π χ − Σ_vertices turn + Σ_arcs dphi · cos α ]
```

The last term is `−∮ κ_g ds` for a small-circle arc traversed with its cap on
the right (`κ_g = −cot α / R`, `ds = R sin α dφ`); great circles contribute
nothing. `χ = 2·n_patches − n_loops`.

`n_patches` is obtained without grouping loops into patches. The inaccessible
region is the union of the caps; its connected components are the components
of the graph whose edges are crossing pairs (nested caps were removed, mutual
cover exited in step 1). On the sphere the loops are disjoint or pinched
simple cycles, so the faces number `1 + n_loops`, they alternate
accessible/inaccessible across every loop, and each inaccessible face is
exactly one cap component. Hence

```
n_patches = 1 + n_loops − n_cap_components
```

(a pinch visited twice by one walk counts as one loop). No caps: no loops, no
components, one patch of area `4πR²`. Three or more caps covering the sphere
with no covering pair: no loops, one component, no patch, area 0.

### Tolerances

| symbol | value | role |
|---|---|---|
| `TAU_C` | 1e-6 Å | merge coincident vertices from different pairs; merge coincident caps |
| `_TAU_DIR` | 1e-9 rad | dart tangents treated as parallel |

`TAU_C` only has to exceed the floating-point scatter of one geometric point
computed through different cap pairs (about `1e-12 × |coords|`) and stay below
the smallest gap that must remain a gap; the test suite passes for any value in
`[1e-8, 1e-4]`. All other comparisons are exact.

`TAU_C` also bounds the circle radius from below: at the overlap threshold
`d = R_i + R_j − TAU_C` the circle has `rl ≈ √(TAU_C · 2 R_i R_j / d) ≈
1.8e-3 Å` and `R_i² − a² ≈ 3e-6`, far above rounding, so the `max(·, 0)` under
that square root is a NaN guard the margin makes unreachable (a `DCHECK` in
C++), not a case the kernels can survive.

## 3. `anal.py` — SAS and SES from arrangements

### Preparation (`prepare`)

1. **Pairs.** KD-tree pairs `(i < j)` with `d ≤ R_i + R_j + 2 TAU_C` are the
   *near* pairs; those with `d < R_i + R_j − TAU_C` are the **overlaps**,
   decided once, and carry circles. A pair tangent to within `TAU_C` is near
   but not an overlap: it has no circle, but it still counts as a neighbour
   below, because two hosts of one vertex cluster can be that far apart.
   Their contact points are two members of the cluster, and the SAS build
   checks that every member lies within `TAU_C` of the cluster mean, so they
   differ by at most `2 TAU_C`. Coincident centres (`d < 1e-3 Å`) raise.
   `rp ≤ 0`, a non-positive radius, or SAS radii with
   `R_min² < 2rp² + 2 TAU_C R_max` (the hypothesis of Lemma 7 below; a vdW
   radius just above `(√2 − 1) rp`, about 0.7 Å for water) raise.
2. **Contained balls.** If `d ≤ |R_i − R_j| + TAU_C` the smaller ball is
   contained: it has no surface and generates no caps. These atoms are
   dropped from everything that follows.
3. **Shared circles.** If a third sphere passes through the circle where two
   others meet (collinear centres, radii matched to within `TAU_C`: the two
   circles it makes with either of them have the same centre and parallel
   axes), the sphere whose centre lies between the other two on the axis
   lies inside their union — each of its two caps sits inside the
   neighbouring sphere's larger cap, and the disc of the circle is inside
   both. It has no surface and is dropped like a contained ball. Without
   this, one sphere would carry two caps on the same circle, and the middle
   sphere's two caps would be exact complements, a tie for the covered test.
   With it, every solved sphere's caps lie on distinct circles.
4. **Order.** `active` is the caller's mask minus dropped atoms; `need` is
   `active ∪ neighbours(active)` and `shell` is `neighbours(need)`, where
   neighbours are the near pairs of step 1. Atoms
   are permuted to `[active | need | shell | occluders]` and the overlapping
   pairs among kept atoms are remapped and sorted by `(i, j)`. From here on
   state is read from index ranges: a sphere owns dots iff its index is
   `< n_active`, has an arrangement and torus arcs iff `< n_solve`, has caps
   and hosts vertices iff `< n_enum`; a pair touches a sphere with caps iff
   its smaller index is `< n_enum`. Vertex clusters are relabelled by their
   smallest atom, so probes are owner-sorted; active probes, active circles
   and active torus arcs are prefixes whose lengths (`n_active_probes`,
   `n_active_circles`, `n_active_arcs`) are counted once in `build` and read
   everywhere else. Dot owners are mapped back through the permutation at the
   very end.
5. **What the mask makes exact.** Every atom occludes, so the caps of any
   sphere below `n_enum` are complete and the convex patches of active atoms
   are exact. An active circle's first sphere is its active atom (active
   indices are the lowest), so active torus arcs come from solved spheres.
   The hosts of a probe on an active atom are all near that atom (their
   contact points lie within one cluster, `2 TAU_C` apart), so they are in
   `need`, every circle between two of them is solved, and the probe's
   departure tangents are complete. By Lemma 7, a probe that cuts the face
   of a probe on an active atom has a host that overlaps one of that probe's
   hosts, hence a host in `shell`: it is enumerated, its accessibility is
   decided on its owner sphere as for every other cluster, and its cap is
   present. Its own
   departure tangents may be incomplete (a circle between two shell atoms is
   never solved); it then fails the triangle predicate of Lemma 6 and is
   tested unfiltered, which is always allowed. One shell of neighbours is not
   enough: the cutter's hosts need only touch a *neighbour* of the active
   atom, and a probe wedged between two nearly opposite atoms and touched by
   a third from above can be cut by a shallow probe on the other side whose
   hosts overlap neither the third atom nor anything active.

### SAS (`SasGeometry.build`)

1. **Circles.** For each overlapping pair `(i, j)` with `i < n_enum` (a
   prefix of the sorted pairs), unit axis `u = (c_j − c_i)/d`,
   `a = (d² + R_i² − R_j²)/2d`, centre `t = c_i + a u`, radius
   `rl = √(R_i² − a²)`, frame `(e1, e2, u)`, and `a`, `d` themselves. The
   half-angles `θ_i = atan2(a, rl)`, `θ_j = atan2(d − a, rl)` are derived only
   where the saddle formulas need them.
2. **Caps per sphere** (`_sphere_caps`). Each sphere `s < n_enum` holds its
   overlap partners sorted by atom and one cap per partner in that order:
   circle `(s, j)` cuts `s` with axis `u`, `cos α = a/R_s`, `sin α = rl/R_s`,
   frame `(e1, e2)` (side 0); circle `(i, s)` cuts it with axis `−u`,
   `cos α = (d − a)/R_s`, `sin α = rl/R_s`, frame `(e1, −e2)` (right-handed
   about `−u`; side 1). A cap is named by the atom on the other side of its
   circle: `Sphere.slot(atom)` is a binary search in the partner-sorted
   caps, `−1` when there is no such cap, and a triple's atoms are that key
   directly. No cap geometry is recomputed, and a side-0 arc's `φ` *is* the
   circle's `φ`.
3. **Triple vertices, computed once** (`_crossing_triples`). Candidates are
   `(i < j < k)` with `(i, j)` a circle and `k` a partner of `i` after `j`
   that is also a partner of `j` (circle by circle, a binary search in `j`'s
   partner list; near pairs without a circle take no part). The circle
   `(i, j)` is intersected with sphere `k`: with `w = t − c_k`,
   `g = (R_k² − |w|² − rl²)/2rl`, `A = w·e1`, `B = w·e2`, `amp² = A² + B²`,
   the points exist iff `h² = amp² − g² > 0` and are

   ```
   cos φ± = (A g ∓ B h)/amp²,   sin φ± = (B g ± A h)/amp²,
   x± = t + rl (cos φ± e1 + sin φ± e2)
   ```

   which is `φ_0 ± acos(g/amp)` with `φ_0 = atan2(B, A)` written without
   inverse trig. This is the **only** crossing decision on SAS spheres: a
   candidate with `h² > 0` records a `(triple, corner)` incidence on each of
   its spheres below `n_enum`, and on that sphere the two caps of the triple
   (found by slot) are a crossing pair; these pairs form the crossing graphs
   that step 1 of §2 takes as given. Hiding then runs sphere by sphere
   (`_hide_caps`): a triple whose cap is hidden on some sphere gets
   `edge_on` false at that corner and has no vertex, but keeps its crossing
   edges where both caps survive: dropping the edge would let a cap whose
   remaining vertices are all inaccessible pass as a full circle tested
   against too few caps. The points are computed once per triple with a
   vertex and shared by all three spheres.
4. **Global clustering.** All raw points are clustered at `TAU_C`. A cluster's
   atoms are the union of its triples; k-fold coincidences give probes with
   four or more atoms. Every member must lie within `TAU_C` of the cluster
   mean (a `DCHECK` in C++, `DegenerateGeometryError` in the pilot): the
   scatter of one geometric point is about `1e-12 Å`, and the near-pair
   margin of the preparation stage relies on the bound. Clusters are sorted
   by their smallest atom, the owner, and every triple records the cluster
   ids of its two points.
5. **Accessibility on the owner sphere.** Each cluster is decided once, on the
   sphere of its smallest atom, with the test of §2 step 4 against that
   sphere's caps, excusing the caps its member triples make there: the owner
   is the first atom of every member triple that contains it, so those caps
   are the slots of the triple's other two atoms (the per-sphere solve uses
   the same incidence for its arcs). Every ball that can contain a point of
   the sphere is one of its caps or nested inside one, so this equals the
   test against all SAS balls. Covered owner spheres are included (two
   covering caps touching at one point host a legitimate probe). Accessible
   clusters are the **probes**.
6. **Per-sphere solve.** Each non-covered sphere assembles its
   `ArrangementProblem` from its incident triples (`_sphere_problem`):
   crossing edges from the incidences whose caps both survived, the vertices
   and their excused caps from those with a vertex, representatives
   projected onto the sphere, frames from the circles of its caps, and the
   cap components of its own crossing graph; then steps 5–7 of §2 run. A
   covered sphere gets an empty arrangement.
7. **Torus arcs.** The arcs of a sphere on its side-0 caps are the torus arcs
   as they stand (same frame, same `φ`); the other sphere of each circle
   reports nothing. Local vertex ids map to probe ids; full-circle ends stay
   `-1`.

### SES (`SesGeometry.build`)

- **Convex area** of atom `i`: `A_sas(i) · (r_i/R_i)²` (radial scaling maps the
  SAS patch onto the vdW sphere).
- **Saddle** of an arc on circle `c` with span `dphi`. The generating arc on the
  probe sphere is parametrised by `β`, the angle from the inward radial
  direction (toward the axis); `β < 0` leans toward atom `i`. The valid range is
  `[−θ_i, θ_j]`. The distance from the axis is `ρ(β) = rl − rp cos β`; for a
  **spindle torus** (`rl < rp`) it vanishes at `β0 = atan2(√(rp² − rl²), rl)`,
  and the part `|β| < β0` lies beyond the axis. Those points are inside every
  other probe ball on the same circle, because for a probe at angular offset
  `Δφ`

  ```
  |p − q(φ + Δφ)|² = rp² + 2 ρ rl (1 − cos Δφ),
  ```

  which is below `rp²` for all `Δφ ≠ 0` exactly when `ρ < 0`. Nothing else
  cuts a saddle (Lemma 1(b) below), so this cut is complete. With `β0 = 0`
  for `rl ≥ rp`, the two ranges

  ```
  [−θ_i, max(−θ_i, min(θ_j, −β0))]   and   [min(θ_j, max(−θ_i, β0)), θ_j]
  ```

  cover every case: they tile the arc for an ordinary torus, cut out the
  spindle, and collapse to zero width where a part is absent (which also
  happens when `a < 0` or `d − a < 0`, since `θ_i > β0` iff `a > 0`). They
  are stored per active circle as one `(n_circles, 2, 2)` array, and the
  area is

  ```
  A = dphi · rp · Σ_ranges [ rl (β_hi − β_lo) − rp (sin β_hi − sin β_lo) ].
  ```

  Endpoints are selected together with their sines (`sin θ_i = a/R_i`,
  `sin θ_j = (d − a)/R_j`, `sin β0 = √(rp² − rl²)/rp`; all `β` lie in
  `(−π/2, π/2)`), and the bracketed integral of each range is stored once per
  circle; the arc areas and the sampler's row totals are both gathers of it.
  The range is never empty once contained balls are gone: `θ_i + θ_j` is the
  angle at the probe in the triangle `(c_i, q, c_j)`.
- **Concave face** of probe `q` (only probes whose smallest atom is active).
  Caps on the probe sphere:
  1. one cap for every other vertex probe `q'` strictly within `2rp` (a
     tangent probe cuts nothing), as defined in §1; every probe pair is
     measured once and read from both sides with opposite axes;
  2. one **hemisphere per accessible arc leaving `q`**, axis = the arc's
     departure tangent `±(u × radial)` at `q` (`+` when the arc leaves toward
     increasing `φ`). Rolling along that arc, the probe sweeps the half of its
     sphere facing the tangent, so that half is not SES. Departure tangents
     are computed arc by arc from the cluster mean, normalised (the mean is
     off the circle by up to `TAU_C`) and appended to both end probes
     (`Probe.tangents`).

  For an ordinary three-atom vertex, rule 2 gives exactly the three side planes
  of the contact triangle (the departure tangent of circle `(a, b)` is normal
  to the plane through `q`, `c_a`, `c_b`). The same rule handles merged k-fold
  vertices (one hemisphere per surviving arc), a vertex that is the only
  vertex on a circle (two opposite hemispheres, zero face), and nearly
  coplanar contacts (zero face). These two kinds of caps are all that is
  needed: whatever SAS point cuts a face point, some vertex probe within `2rp`
  cuts it too (Lemma 3). The face arrangement is solved with `solve_caps` for
  the faces and caps that survive the filters below.

### Which probes can cut a face

Notation for a vertex `x` on atoms `a` with contact directions
`ĉ_a = (c_a − x)/R_a`: the **face cone** is `C_x = {Σ λ_a ĉ_a : λ_a ≥ 0}` (for
an ordinary three-atom vertex, the complement of the three departure
hemispheres), the **contact triangle** is `T_x = conv{c_a}`, and `t_x(d)` is
the distance from `x` along the ray `d ∈ C_x` to `T_x`. The face is cut by a
probe `y` where it enters the open ball `B°(y, rp)`. Merged k-fold probes take
the unfiltered path; the filters are for three-atom probes.

**Lemma 1 (contact hull).** Points that lie between an SAS point and the atoms
it touches are closer to that SAS point than to any other point outside `U`.
Precisely: let `s` be an SAS point touching atoms `A(s)`, `v` any point not
inside `U` (every SAS point qualifies), `q` any point of the convex hull of
`{c_a : a ∈ A(s)}`, and `p = (1 − τ) s + τ q` with `0 ≤ τ ≤ 1`. Then
`|p − v| ≥ |p − s|`, strictly for `τ < 1` and `v ≠ s`.

*Proof.* `v` is outside the SAS ball of `a` and `s` is on it, so
`|v − c_a| ≥ |s − c_a|`, which expands to `(c_a − s)·(v − s) ≤ |v − s|²/2`.
Averaging over the weights of `q` gives the same bound for `(q − s)·(v − s)`,
and then `|p − v|² − |p − s|² = |v − s|² − 2τ (q − s)·(v − s) ≥ (1 − τ)
|v − s|²`. ∎

Three consequences:

- **(a) Cuts lie beyond the contact plane.** Every `d ∈ C_x` does meet `T_x`:
  radial rescaling by `R_a` leaves the cone alone, so
  `C_x = cone{ĉ_a} = cone{c_a − x}`, which is exactly the cone over
  `conv{c_a}` from the apex `x`. Take `s = x`, `q` the point where the ray `d`
  meets `T_x`, `τ = rp / t_x(d)`: a face point `x + rp d` with
  `t_x(d) ≥ rp` is cut by nothing. Since `t_x(d) · (d·n) = h`, where `n` is the
  unit normal of the contact plane pointing from `x` toward it and `h` the
  distance to the plane, the cut part of the face lies inside the
  **beyond-plane cap** `D_x = {d : d·n > cos β}`, `cos β = h/rp`. This is a cap
  only when `h < rp`, which holds on every low vertex since `h ≤ dist(x, T_x)`;
  high vertices never reach any code that uses `β`. The bound is
  attained: `D_x` is exactly the cap cut by the mirror triple point
  `x + 2h n`, the second point where the three SAS spheres meet (a vertex
  only if it is accessible).
- **(b) Saddles are cut only in the spindle.** For a probe centre on the
  circle of atoms `i, j`, the generating directions are the cone of
  `ĉ_i, ĉ_j`, and the ray at angle `β` from the inward radial direction meets
  the segment `[c_i, c_j]`, which lies on the torus axis, at distance
  `rl / cos β`. So a saddle point with `rl ≥ rp cos β`, i.e. `ρ(β) ≥ 0`, is
  cut by nothing, and the spindle cut of the saddle ranges above is the whole
  cut.
- **(c) High and low vertices.** `min_{d∈C_x} t_x(d) = dist(x, T_x)`: the
  plane distance `h` when the foot of the perpendicular lies inside `T_x`,
  the distance to the nearest edge otherwise. A vertex with
  `dist(x, T_x) ≥ rp` — its own ball does not reach its contact triangle —
  is **high**: nothing cuts its face, which is the spherical triangle
  `{d : d·t_m ≤ 0}` bounded by the three departure great circles, with area
  `rp²` times the spherical excess,

  ```
  A = rp² (2π − Σ_m ∠(t_m, t_{m+1})),   ∠(t_m, t_{m+1}) = atan2(|t_m × t_{m+1}|, t_m · t_{m+1}).
  ```

  Every other vertex is **low** (k-fold probes count as low). `_probe_heights`
  decides this once for every probe; high active faces never reach the pair
  query or `solve_caps`.

**Lemma 2 (a ball inside `U` that touches a vertex points into its cone).**
If `B(q, ρ) ⊂ U` and the vertex `y` lies on its boundary, then the direction
from `y` to `q` lies in `C_y`.

*Proof.* Suppose not. Then some direction `w` points away from every touched
centre (`w·(c_b − y) ≤ 0` for all `b ∈ A(y)`) yet toward `q`
(`w·(q − y) > 0`). Moving from `y` a short way along `w` leaves every SAS ball
of the touched atoms (`|y + εw − c_b|² = R_b² − 2ε w·(c_b − y) + ε² > R_b²`)
and stays outside the SAS balls `y` does not touch, so the point is outside
`U`; yet it is inside `B(q, ρ)` (`|y + εw − q|² = ρ² − 2ε w·(q − y) + ε² < ρ²`
for small `ε`), which lies in `U`. ∎

**Lemma 3 (the first SAS point an inflating ball touches is a vertex).** Let
`x` be a vertex whose contact directions span space (they fail to only when
`x` lies in the plane of its atom centres, and then its face has no area)
and `d` a unit direction strictly inside `C_x` (`d = Σ λ_a ĉ_a` with every
`λ_a > 0`, scaled to `|d| = 1`). Inflate the
balls `B_t = B(x + t d, t)`, `t > 0`, which are nested and all pass through
`x`, and let `t₁` be the largest `t` for which the open
ball `B°_t` contains no SAS point. Then `t₁ > 0`, `B(x + t₁ d, t₁) ⊂ U`, and a
second SAS point `s₁ ≠ x` lies on its boundary; for all but a null set of
directions `d`, `s₁` is a vertex. Consequently, if `t₁ < rp`, the face point
`x + rp d` is cut by the vertex `s₁`.

*Proof.* *Small balls stay inside `U`.* A point of `B°_t` is `x + t(d + e)`
with `|e| < 1`. It is inside the SAS ball of atom `a` iff
`ĉ_a·(d + e) > t |d + e|² / 2R_a`. With `d = Σ λ_a ĉ_a`,
`Σ λ_a ĉ_a·(d + e) = 1 + d·e > |d + e|²/2`, so the largest `ĉ_a·(d + e)`
exceeds `|d + e|²/(2 Σ λ_a)`, which is at least `t |d + e|²/(2R_a)` for the
maximising `a` once `t ≤ min_b R_b / Σ_b λ_b`.

*SAS points near `x` are never inside.* Near `x` the SAS consists of the
sphere patches leaving `x`; a tangent direction `w` of the patch of atom `a`
satisfies `w ⊥ ĉ_a` and `w·ĉ_b ≤ 0` for every other touched atom `b`. Hence
`d·w = Σ_{b≠a} λ_b w·ĉ_b ≤ 0`, with equality only for `w = 0` because the
contact directions span space by hypothesis. So `d·(s − x) < 0` for SAS
points `s` close to `x`, and
`|s − (x + t d)|² − t² = |s − x|² − 2t d·(s − x) > 0` for every `t`: no such
`s` is in any `B°_t`.

*The first touch.* Hence `t₁ > 0`, and it is finite whenever some SAS point
lies in `B°_rp` (the only case used). The union of the open balls `B°_t`,
`t < t₁`, is the open ball `B°(p₁, t₁)`, `p₁ = x + t₁ d`; it is connected,
avoids the SAS and contains points inside `U`, so it lies inside `U`, and the
closed ball lies in `U`. Slightly larger balls contain SAS points, which
accumulate on the boundary of `B(p₁, t₁)` at an SAS point `s₁`, and `s₁ ≠ x`
by the previous paragraph. So `x` and `s₁` are both nearest SAS points of
`p₁`, at distance `t₁`, and `p₁` lies on the bisector plane of `x` and `s₁`.

*`s₁` is not inside a sphere patch.* If it were, on the patch of atom `a`, the
ball `B(p₁, t₁) ⊂ U` would touch the sphere `S(c_a, R_a)` from inside at
`s₁`, so `p₁` lies on the radius from `c_a` to `s₁` at distance `R_a − t₁`
from `c_a`. Then `|x − c_a| ≤ |x − p₁| + |p₁ − c_a| = R_a`, while `x`, an SAS
point, has `|x − c_a| ≥ R_a`; equality puts `p₁` on the segment from `x` to
`c_a`, so the radius through `p₁` is the radius through `x` and `s₁ = x`. (If
`p₁ = c_a`, then `d = ĉ_a`, a cone edge, excluded.)

*`s₁` is not inside an arc.* If it were, on the circle of atoms `a, b`, then
near `s₁` the region `U` is the union of the two SAS balls, and a ball inside
`U` touching their common boundary at `s₁` must have its centre in the wedge
spanned at `s₁` by the directions to `c_a` and `c_b` — otherwise some
direction leads out of both balls yet into `B(p₁, t₁)`, exactly as in
Lemma 2. So `p₁` lies in the plane through `s₁` and the axis `c_a c_b`,
within the angle at `s₁` of the triangle `(s₁, c_a, c_b)`. By Lemma 1 with
`s = s₁`, `v = x`, no point strictly inside that triangle is equidistant from
`x` and `s₁`, so `p₁` is on or beyond the segment `[c_a, c_b]`: on the axis
or across it. Across the axis, `s₁` is the *farthest* point of the circle from
`p₁`: writing `p₁ = t + z u + ξ r̂` with `r̂` the radial direction of `s₁`,
`|p₁ − s(θ)|² = z² + ξ² + rl² − 2 ξ rl cos θ`, and across the axis `ξ < 0`,
so the distance is largest at `θ = 0`. The end vertices of the arc through
`s₁` are therefore strictly closer than `t₁`, contradicting that no SAS point
is closer than `t₁`. On the axis the whole circle is equidistant and an end
vertex is also a nearest point; take it as `s₁`. (A vertex-free circle would
need the ray of `d` to meet a fixed line: a null set of directions.)

*The cut.* `p₁` is on the bisector plane of `x` and `s₁`, and `x + rp d` lies
beyond `p₁` on the ray from `x` when `t₁ < rp`, so it is closer to `s₁` than
to `x`: `|x + rp d − s₁| < rp`. ∎

**Corollary (vertex caps suffice).** If any SAS point at all cuts a face point
`x + rp d`, that point lies in `B°_rp`, so `t₁ < rp` and Lemma 3 supplies a
vertex within `rp` of the face point. The concave face of `x` is therefore
`C_x` minus the caps of the vertex probes within `2rp`, and nothing else, up
to a null set of directions (the cone boundary and the exceptions of
Lemma 3), which affects neither area nor dots. For the probes rolling along
the arcs that leave `x` this is also visible directly: their bisector planes
with `x` all contain the torus axis, and the union of their caps is the
departure hemisphere together with the cap of the arc's end vertex.

This is Quan & Stamm's Theorem 5.1, `P₋ = P₀ \ ⋃_{x∈K} B_rp(x)`: the concave
patch `P₋` is the uncut spherical triangle `P₀` minus the open probe balls of
the other SAS intersection points `K` within `2rp`. The paper's Appendix A
proves it in the plane (its Lemma 1.1) and states that space adds no
essential difficulty; Lemma 3 is the proof in `R³`.

**Lemma 4 (cutting is symmetric).** Let `x`, `y` be vertices with
`|x − y| < 2rp`, and suppose the cap of `y` removes a region of positive area
from the face of `x` computed without `y` (the cone minus the caps of all
other vertices within `2rp`). Then the ball of `x` reaches the face region of
`y` beyond `y`'s contact plane: there is `e ∈ C_y` with
`|y + rp e − x| < rp`, and `t_y(e) < rp`.

*Proof.* Pick a removed face point `k = x + rp d` with `d` strictly inside the
cone and outside the null set of Lemma 3. It is cut by `y` and by no other
vertex. Since `y ∈ B°(k, rp) = B°_rp`, the first-touch radius of Lemma 3
satisfies `t₁ < rp`, and its vertex `s₁` is within `rp` of `k`; the only such
vertex is `y`. So `q = p₁` is equidistant from `x` and `y`, at distance
`ρ = t₁ < rp`, and `B(q, ρ) ⊂ U`. Both `x` and `y` lie on its boundary, so
Lemma 2 applies at both: `u = (q − x)/ρ ∈ C_x` and `e = (q − y)/ρ ∈ C_y`. Then
`|y + rp e − x| ≤ |y + rp e − q| + |q − x| = (rp − ρ) + ρ = rp`, with equality
only if `q − x` and `q − y` point the same way, which with equal lengths means
`x = y`. Finally Lemma 1(a) at `y`, with the SAS point `x`, gives
`t_y(e) < rp`. ∎

**Corollary (both ends low, symmetric drop).** If `y` cuts `x`, then `x`
reaches `y`'s face beyond its plane: `y` is low (Lemma 1(c)), and the cap of
`x` on `y`'s sphere meets both `y`'s beyond-plane cap and `y`'s spherical
triangle. Both filters below are *necessary* conditions for `y` to cut `x`: the
x-side test is immediate from Lemma 1(a) (the removed direction lies in
`cap(y) ∩ D_x ∩ triangle_x`), the y-side test is Lemma 4. By symmetry the same
two conditions are necessary for `x` to cut `y`. So a failure on either side
proves there is no cutting in either direction, and the pair is dropped
**for both faces**.

The hypothesis "computed without `y`" cannot be dropped: a point of `x`'s
beyond-plane cap can lie inside `y`'s ball while `x`'s ball misses `y`'s cone
entirely; such a point is always inside some other vertex ball, which is what
Lemma 3 finds.

**Lemma 5 (the cap must reach the beyond-plane cap).** By Lemma 1(a) the cut
part of the face lies inside `D_x = {d : d·n > cos β}`. A cap
`{d : d·u > cos α}` meets it only if the angle between `u` and `n` is below
`α + β`:

```
u · n > cos(α + β) = cos α cos β − sin α sin β.
```

All four values are stored; no angle is recovered. The step `γ < α + β ⟺
u·n > cos(α + β)` needs `α + β ≤ π`, which holds throughout: probe-on-probe
caps have `cos α = |x − y|/2rp ≥ 0` and the beyond-plane cap has
`cos β = h/rp ≥ 0`, so both are at most `π/2`. Under that bound the test is
not merely necessary but exact for `cap ∩ D_x ≠ ∅`.

**Lemma 6 (the cap must meet the spherical triangle).** The cap meets the
closed triangle `S = {d : d·t_m ≤ 0}` iff `u ∈ S` or the angular distance
from `u` to the boundary of `S` is below `α`. Edge `m` is the arc of the
great circle `t_m · d = 0` between the corners `p_{m+1}` and `p_{m+2}`,
`p_m = ±(t_{m+1} × t_{m+2})/|·|` signed so that `t_m · p_m ≤ 0`; the corners
are perpendicular to `t_m`. The foot of `u` on that great circle is
`f = u − (u·t_m) t_m`, the squared cosine of the angle from `u` to `f` is
`|f|² = 1 − (u·t_m)²`, and `f` lies on the arc iff it is on the arc's side of
the plane through the origin and the corners' bisector `p + q`. Because
`p, q ⊥ t_m`, `f · (p + q) = u · (p + q)`, and the test is

```
u·(p + q) ≥ 0   and   (u·(p + q))² ≥ (1 − (u·t_m)²) · (p·(p + q))².
```

So the cap meets the triangle iff

```
u·t_m ≤ 0 for all m                                        (axis inside S), or
(f on arc m  and  1 − (u·t_m)² > cos² α)  for some m,      (edge reached), or
max_m u·p_m > cos α                                        (corner reached),
```

the corner test covering feet off their arcs. The first disjunct is not
redundant: a neighbour probe whose cap lies entirely in the interior of the
triangle — a hole punched in the middle of a face — meets `S` without meeting
`∂S`, and dropping it would delete the hole.

No square root is taken; all comparisons are on cosines or their squares. The
triangle exists only when the three tangents are linearly independent
(`t_1 · (t_2 × t_3) ≠ 0`); this also keeps every edge shorter than π, so
`p + q ≠ 0 and p·(p + q) = 1 + p·q > 0`, which is what makes the squaring above
sign-safe. Probes failing independence, like k-fold probes, pass the test and
are solved unfiltered. The condition is necessary for the cap to cut anything
and, for a face with no other caps, sufficient.

**Lemma 7 (host overlap).** Whenever one probe trims another's face, some
atom under the first touches some atom under the second, provided no atom is
smaller than about 0.41 probe radii. Precisely: let `y` be an effective
cutter of `x` (Lemma 4), `R_x` and `R_y` the smallest SAS radii among the
hosts of `x` and `y`, `R_max` the largest SAS radius of the structure, and
call two atoms overlapping as the preparation does, `|c_a − c_d| < R_a + R_d
− TAU_C`. If no host of `y` overlaps any host of `x`, then
`|x − y|² > 2 R_x R_y − 4 TAU_C R_max`. Since `|x − y| < 2rp`, some host pair
overlaps whenever `R_x R_y ≥ 2rp² + 2 TAU_C R_max`, which the preparation
stage enforces for the smallest SAS radius; without the tolerance the bound
is `|x − y|² > 2 R_x R_y` and the threshold is `R ≥ √2 rp`, i.e. a vdW
radius of `(√2 − 1) rp`.

*Proof.* Take the point `q` from the proof of Lemma 4: `x = q − ρu`,
`y = q − ρe` with
`u ∈ C_x`, `e ∈ C_y`, `ρ < rp`. Write `w = x − y = ρ(e − u)`, `ℓ = |w|`,
`γ = u·e < 1`, so `ℓ² = 2ρ²(1 − γ)`, `w·u = −ρ(1 − γ)`, `w·e = ρ(1 − γ)`. Let
`u = Σ_a λ_a ĉ_a` and `e = Σ_d μ_d ĉ_d` with non-negative weights over the
hosts, and abbreviate `Λ = Σ λ_a`, `Λ' = Σ λ_a/R_a`, `M = Σ μ_d`,
`M' = Σ μ_d/R_d`, `P = ρΛ'`, `Q = ρM'`.

*Both probes are outside the other's balls.* `|y − c_a| ≥ R_a` with
`y − c_a = −w − R_a ĉ_a` gives `w·ĉ_a ≥ −ℓ²/2R_a`; weighting by `λ_a` and
summing, `−ρ(1 − γ) ≥ −(ℓ²/2) Λ' = −ρ²(1 − γ) Λ'`, i.e. `P ≥ 1`. Likewise
`|x − c_d| ≥ R_d` gives `Q ≥ 1`.

*A non-overlapping host pair.* `c_a − c_d = w + R_a ĉ_a − R_d ĉ_d`, so
`|c_a − c_d| ≥ R_a + R_d − TAU_C` expands to
`ℓ² + 2R_a w·ĉ_a − 2R_d w·ĉ_d ≥ 2R_a R_d (1 + ĉ_a·ĉ_d) − 2 TAU_C (R_a + R_d)`
(the `TAU_C²` term only helps and is dropped).

*Average over both cones.* Multiply each pair's inequality by
`λ_a μ_d / (R_a R_d)` and sum. The left side becomes
`ℓ² Λ'M' + 2M' (w·u) − 2Λ' (w·e) = 2(1 − γ)(PQ − P − Q)`, the right side
`2(ΛM + γ) − 2 TAU_C (ΛM' + Λ'M)`. Since `Λ ≤ R_max Λ'` and `M ≤ R_max M'`,
the tolerance term is at most `4 TAU_C R_max Λ'M'`; hence
`(1 − γ)(PQ − P − Q) ≥ (ΛM − 2 TAU_C R_max Λ'M') + γ`.

*Conclude.* `Λ ≥ R_x Λ'` and `M ≥ R_y M'` give
`ΛM − 2 TAU_C R_max Λ'M' ≥ (R_x R_y − 2 TAU_C R_max) PQ / ρ²`, and
`P + Q ≥ 2` gives `(1 − γ)(PQ − 2) ≥ (1 − γ)(PQ − P − Q)`. Together,
`PQ [(1 − γ) − (R_x R_y − 2 TAU_C R_max)/ρ²] ≥ 2 − γ > 0`, so
`R_x R_y − 2 TAU_C R_max < ρ²(1 − γ) = ℓ²/2`. ∎

**Corollary (two neighbour shells suffice).** The hosts of a vertex cluster
`x` are pairwise *near* (§3, preparation step 1): two hosts from one triple
share a point, and two hosts from different triples of a merged cluster have
contact points within the cluster's diameter, at most `2 TAU_C` (§3, SAS
step 4), so `|c_a − c_d| ≤ R_a + R_d + 2 TAU_C`. An
overlap is a near pair, so a host of an effective cutter of `x` is within
two near hops of every host of `x`, and enumerating the vertices that have a
host in `active ∪ N(active) ∪ N²(active)`, with `N` the near neighbourhood,
captures every cutter of every face on an active atom (§3, preparation
step 5). One hop does not suffice, and the
radius threshold is not vacuous: with vdW radii of 0.03 Å at `rp ≈ 1` a
cutter exists none of whose hosts overlaps any host of the cut face.

**Order and exactness.** Per structure: heights of all probes (Lemma 1(c))
→ closed-form areas of the high active faces → probe pairs within `2rp`
→ drop pairs with a high end (both ends low) → drop pairs failing Lemma 5 on
either side → drop pairs failing Lemma 6 on either side, tested on the
survivors only → `solve_caps` on the low active faces with the surviving
caps. Dropped caps remove no area, so the arrangement's accessible
region, its area and the dots are unchanged. The filters are necessary
conditions evaluated in floating point: a rounding flip can only drop a cap
that reaches the face by a rounding-scale sliver, an area effect far below
the `1e-9` that the brute-force test (`anal_test.py`, every face solved
against every probe within `2rp`) enforces, never a topological one.

### What is discontinuous

The SES area is genuinely discontinuous at a four-sphere coincidence: a fourth
sphere passing at distance `ε` from an existing vertex splits that vertex into
the four triple points of the four spheres; all four are accessible for
`ε > 0` and none for `ε < 0`, so the two limits differ. Below `TAU_C` the
merged geometry reproduces the `ε → 0⁺` limit. Areas are continuous within
each sign, and exact tangency gives the same area as a distant fourth sphere.

## 4. `surface.py` — dots on the SES

Dots are placed **inside the exact patch domains** described above. Their
weights are normalised so that the sum over a patch equals the patch's analytic
area; dot counts follow the requested density. Only the count is a sampled
quantity.

- **Convex** (atom `i < n_active`). A Fibonacci lattice with
  `round(4π r_i² · density)` directions is generated on the unit sphere (one
  lattice per distinct count, reused); a direction is kept iff it is outside
  every cap of the atom's arrangement (`Arrangement.contains`). Dot =
  `c_i + r_i d`, normal `d`, weight `A_convex(i) / n_kept`, owner `i`.
- **Toroidal**, row by row, one row per active arc and cusp side
  (`SaddleRow`). A circle with `rl ≥ rp` has no cusp, so its two ranges are
  merged into one row spanning `[lo_0, hi_1]`: rounding the ring count on
  each half separately would double its error on narrow saddles and leave a
  seam at `β = 0`.
  1. `k_β = max(round(rp (hi − lo) √density), 1)` rings of equal `β` width;
     ring `m` spans `[β_m, β_{m+1}]` and is sampled at its middle;
  2. exact ring area `a_m = rp · dphi · [rl Δβ − rp (sin β_{m+1} − sin β_m)]`;
  3. `k_φ,m = round(a_m · density)` dots on ring `m`, uniformly spaced in `φ`
     inside `[phi_beg, phi_beg + dphi]` at `φ = phi_beg + (n + ¼ or ¾) ·
     dphi / k_φ,m`, the quarter alternating with `m` so neighbouring rings
     interleave; weight `a_m / k_φ,m`. Ring areas are
     rescaled so that the row's weights sum to its exact area; a row whose
     rings all round to zero is collapsed to one ring with
     `round(area · density)` dots, and only rows that still round to zero are
     dropped (recorded in `Dots.dropped_area`). An absent spindle side is a
     zero-width row: it drops zero area and emits nothing.
  4. Dot position `p = q(φ) + rp (−cos β · radial(φ) + sin β · u)`, normal
     `(q − p)/rp`, owner the atom whose vdW surface is nearer. `cos β`,
     `sin β` and the owner are per ring: `|p − c_i|² = R_i² + rp² −
     2 rp (rl cos β − a sin β)` and `|p − c_j|² = R_j² + rp² − 2 rp (rl cos β +
     (d − a) sin β)` do not depend on `φ`. Ring edge sines are shared between
     neighbouring rings.

- **Concave** (face of probe `q`). One probe-sphere Fibonacci lattice
  (`round(4π rp² · density)` directions, not rotated per face: weights are
  `A_face / n_kept`, so orientation is unbiased) is tested against the face
  arrangement; a direction is kept iff it is outside every cap, which encodes
  both the polygon of departure hemispheres and the neighbour-probe cuts.
  Dot = `q + rp d`, normal `−d`, weight `A_face / n_kept`, owner the atom whose
  vdW surface is nearest, from one matmul with the contact directions:
  `|p − c_a|² = R_a² + rp² − 2 rp R_a (d · ĉ_a)`.

Normals point from the SES into the solvent; for toroidal and concave dots that
is toward the probe centre, so `Dots.probes = pts + rp · normals` recovers the
probe positions used by the buried/trim tests in `sc.py`.

## 5. Complexity

Pair enumeration uses one KD-tree; triple candidates come from sorted
overlap lists and a `searchsorted` pair lookup. Each sphere's arrangement is
`O(m²)` in its cap count `m` (10–40 for proteins) and independent of all other
spheres. Global steps: one KD-tree clustering of the raw vertices, one
label-propagation pass for all cap graphs, one vectorised accessibility test
for all clusters, one height pass over all probes, one probe-pair query for
all faces. Per sphere and per face there remain a KD-tree query for coincident
caps and one for local vertex clustering (≤ 40 points, so a pairwise test
suffices in C++). Connected components everywhere are minimum-label
propagation with pointer jumping. Sampling is linear in the number of dots.
