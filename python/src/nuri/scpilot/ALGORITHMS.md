# Algorithms: `arrangement.py`, `anal.py`, `surface.py`

This document explains how the pilot computes the analytic solvent-accessible
surface (SAS) and solvent-excluded surface (SES) and how it places dots on the
SES. It is written for the C++ port: every step is stated in terms of exact
geometric predicates, and the reasons behind the tolerance choices are given.

Two conventions run through everything. **Filter first**: exceptional input
(coincident atoms, contained balls, inactive atoms, `rp ≤ 0`) is removed or
partitioned in a preparation stage, and no kernel afterwards carries a guard
for it. **Trig values, not angles**: caps store `(cos α, sin α)`, predicates
compare trig values (`acos(x) ≤ a` is written `x ≥ cos a`), and `atan2` is
used only where an angle is consumed as an angle.

## 1. Definitions

- Atom `i` has centre `c_i`, van der Waals radius `r_i`, and SAS radius
  `R_i = r_i + rp`, where `rp` is the probe radius.
- The **SAS** is the boundary of `U = ⋃_i B(c_i, R_i)`. A point of the SAS is
  an *accessible probe centre*.
- The **SES** is the boundary of the rp-erosion of `U`, that is of
  `E = {p : B(p, rp) ⊂ U}`. Equivalently a point `p` is on the SES iff its
  distance to the SAS is exactly `rp`.
- The SES decomposes into three patch families (Connolly 1983):
  - **convex**: the probe touches one atom; the patch is the atom's vdW sphere
    scaled from the accessible part of its SAS sphere;
  - **toroidal (saddle)**: the probe rolls on two atoms; its centre traces an
    arc of the circle where the two SAS spheres meet;
  - **concave**: the probe rests on three (or more) atoms; its centre is a
    vertex of the SAS and the patch is a region of the probe sphere.

Every family is derived from **arrangements of spherical caps**: the accessible
part of a SAS sphere is the sphere minus the caps cut by neighbouring SAS
spheres, and the concave face of a probe is the probe sphere minus the caps
cut by neighbouring probe balls and by the tori it rolls off along.

## 2. `arrangement.py` — caps on one sphere

### Input and output

A cap is `(axis n, cos α, sin α)`: the set of unit directions `x` with
`n · x > cos α`. The solver takes a sphere radius `R` and a list of caps and
returns:

- `arcs`: for each cap circle the accessible sub-arcs, as `(cap, v_beg, v_end,
  phi_beg, dphi)` in the circle's own frame `(e1, e2, n)`; `v_* = -1` marks a
  full circle;
- `verts`: cluster representatives (unit directions);
- `loops`: sequences of arcs traversed with the accessible region on the left
  (seen from outside the sphere);
- `n_patches`, `area`.

The frame defaults to `e1 = any_perpendicular(n)`, `e2 = n × e1`; callers
may supply frames (the SAS passes the circle frames, §3).

### Step 1 — cap hygiene (`prepare_caps`)

Empty caps (`sin α ≤ 0`) are dropped. Caps whose circles coincide to within
`TAU_C` — the 5-vectors `R · (n, cos α, sin α)` closer than `TAU_C`, clustered
exactly like vertices in step 3 — are one circle computed through different
routes (the departure hemispheres of tangent arcs at a pinch, for instance)
and are merged into their renormalised mean.

Every remaining pair is then classified once, from trig values only
(`pair_predicates`). With `cos γ = n_j · n_k` and `c, s` the stored cosines
and sines:

- **nested**: `cos γ ≥ c_j c_k + s_j s_k` (`γ ≤ |α_j − α_k|`): the smaller cap
  lies inside the larger and is dropped; it adds nothing to the union;
- **apart**: `cos γ ≤ c_j c_k − s_j s_k` (`γ ≥ α_j + α_k` or
  `γ ≥ 2π − α_j − α_k`). With `c_j + c_k < 0` (`α_j + α_k > π`) the two caps
  **cover** the sphere: the solve ends immediately with area 0 and no
  patches, and nothing downstream sees this case. Otherwise they are
  **disjoint**;
- **crossing**: everything in between.

On a probe sphere these three comparisons are the only source of the
classification. On a SAS sphere the crossing decision is made elsewhere, once
per sphere triple (§3 step 3), and is passed in; the remaining pairs are split
into nested and apart by `cos γ > c_j c_k`, which lies `s_j s_k` away from
either boundary and therefore can never contradict a crossing decision reached
by another route. Hiding is applied before the covered exit (a cap that covers
the sphere together with a smaller cap covers it together with the larger
one). Everything downstream sees only caps that can contribute boundary, and
hiding, the covered exit, the crossing graph of step 7 and the arc test of
step 5 all read one classification.

### Step 2 — crossing points (`crossing_points`)

For a crossing pair the two circle intersections solve
`n_1 · x = c_1`, `n_2 · x = c_2`, `|x| = 1`. The point of the intersection
line nearest the origin is written in the basis `m = n_1 + n_2`,
`w = n_1 − n_2` (orthogonal, `|m|² = 2(1 + g)`, `|w|² = 2(1 − g)`):

```
base = (c_1 + c_2)/|m|² · m + (c_1 − c_2)/|w|² · w
h² = max(1 − |base|², 0),   x± = base ± h (n_1 × n_2)/|n_1 × n_2|
```

The textbook form `A n_1 + B n_2` with `A = (c_1 − c_2 g)/(1 − g²)` is the
same point, but for nearly parallel axes it computes `c_1 − c_2 g` with
cancellation and divides by `1 − g² ≈ γ²`, so a rounding error `ε` in the
cosines moves the point by `ε/γ²`; the geometry itself only moves by `ε/γ`
(two nearly parallel planes at offsets differing by `ε` meet `ε/γ` away),
which is what the `m, w` form gives. The clamp on `h²` handles a pair the
predicate calls crossing but rounding puts exactly at tangency: both points
coincide and step 3 merges them into a pinch.

### Step 3 — vertex clustering (`cluster_points`)

Raw crossing points closer than `TAU_C = 1e-6 Å` are merged into one cluster
(connected components of the KD-tree pair graph). This handles points that are
the *same* geometric vertex produced by different cap pairs: k circles through
one point yield `k(k−1)/2` raw points. The two crossing points of a *single*
pair are never closer than `2h`, so they merge only when the circles are
tangent to within the tolerance, which is the intended "pinch" case.

The cluster representative is the renormalised mean. Its incident caps are the
union of the generating pairs of its members. Callers may supply clustering
labels computed elsewhere (see §3, global SAS clustering).

### Step 4 — vertex accessibility

A cluster is accessible iff it lies outside every cap **except those whose own
crossing points merged into it**: `n_c · x ≤ cos α_c` for all non-incident
`c`, compared exactly. Excusing exactly the generating caps makes the decision
deterministic at k-fold points, where the sign of the residual for an incident
cap is noise. No epsilon is needed: a non-incident cap passing through the
point within noise would have deposited its own crossing points within noise
of it, so clustering would already have made it incident.

Any larger, "permissive" slack breaks consistency: a vertex inside cap `l` by
`δ` would be accepted while the arc it starts is rejected by the midpoint test
of step 5 once its exit point from `l` lies farther than `TAU_C` (tangential
approach), leaving a vertex with an odd number of darts. A slack of `1e-7 Å`
produces exactly this failure.

### Step 5 — arcs (`_build_arcs`)

The accessible (vertex, cap) incidences of the sphere are processed together.
Each incidence gets the angle `φ = atan2(x · e2, x · e1)` of its vertex in the
cap frame; sorting by `(cap, φ)` makes consecutive incidences of a cap the
candidate arcs (wrapping around by `2π`; a cap with exactly one vertex yields
one arc from the vertex around to itself with `dphi = 2π`). Caps with no
vertices contribute one full-circle candidate with `-1` ends. Nothing here
branches on the case.

A candidate is kept iff its midpoint is outside every cap **that crosses its
circle** (exact comparison). Caps that do not cross the circle are not
consulted: a disjoint cap cannot contain any point of the circle, and a cap
containing the whole circle would have made it nested (step 1). Consulting
them anyway is not harmless: at an exact tangency the midpoint can sit exactly
on the tangent cap's boundary, and the comparison is then a rounding coin
flip that can veto an arc although the crossing decision produced no vertex
there. The arc test and the crossing decision must be the same decision.

Consistency with step 3: a sliver arc between the two crossing points of a
near-tangent pair lies inside the other cap by about `h²/2R > 0`, so with exact
comparisons it is always rejected, as it must be (it bounds the lens inside
both caps, not the accessible region). Once `2h < TAU_C` the two points merge
into a pinch instead.

### Step 6 — face walking with one dart array (`_walk`)

Arcs are stored with increasing `φ`, which runs counter-clockwise around the
cap axis with the cap on the left. The accessible region is therefore
traversed from `v_end` to `v_beg`, i.e. with decreasing `φ`. Each arc is one
*dart* leaving `v_end` and arriving at `v_beg`.

Every arc with vertices contributes two rows to one dart array: an out-dart at
`v_end` with departure tangent `t = −(n × u)/|n × u|` and a reversed in-dart at
`v_beg` with tangent `+(n × u)/|n × u|`, projected into the tangent plane of
the vertex `u` and measured as an angle in a frame there. Signed geodesic
curvature is `−cot α` for out-darts (cap on the right, curving right) and
`+cot α` for reversed in-darts, with `cot α = cos α / sin α` from the stored
values.

The array is sorted by `(vertex, angle)`. Within a vertex, darts whose angles
differ by less than `_TAU_DIR = 1e-9 rad` form a group (tangent circles,
pinches); a group straddling the `−π/π` seam is recognised by the wrap gap and
merged the same way. Every group shares its first angle, and a second sort by
`(vertex, snapped angle, curvature)` puts right-curving darts first, so the
wedge between grouped darts is exactly zero. In the resulting cyclic order per
vertex, reversed-in and out darts must strictly alternate; otherwise
`DegenerateGeometryError` is raised (nothing is merged or dropped silently).
A vertex with out-darts only fails the same check.

The successor of an in-dart is the previous dart in that order (the first
out-dart clockwise from its reversed tangent). The interior angle of the
region at that corner is `ι = angle(rev-in) − angle(out)` taken in `[0, 2π)`
and the turning angle is `π − ι` (left turn positive; a pinch has `ι = 0`,
turn `+π`). Successors default to the arc itself, so a full circle is its own
loop with no special case; loops are then traced by following successors until
an arc repeats.

### Step 7 — Gauss–Bonnet area

For a region with the boundary on its left, on a sphere of radius `R`:

```
A = R² · [ 2π χ − Σ_vertices turn + Σ_arcs dphi · cos α ]
```

The last term is `−∮ κ_g ds` for a small-circle arc traversed with its cap on
the right (`κ_g = −cot α / R`, `ds = R sin α dφ`); great circles (`cos α = 0`)
contribute nothing. `χ = 2·n_patches − n_loops`. Sanity check: a single cap
gives `2πR²(1 + cos α)`.

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

(a pinch visited twice by one walk counts as one loop). With no caps there are
no loops and no components, and the formula gives one patch of area `4πR²`.
Three or more caps can cover the sphere with no covering pair; then there are
no loops and one component, and the formula gives no patch and area 0. Loops
are counted as cycles of the successor permutation with the same
label-propagation routine that clusters vertices; their contents are never
needed.

### Tolerances (summary)

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
that square root is a NaN guard the margin makes unreachable, not a case the
kernels can survive (see NOTES.md, C++ port: a `DCHECK`).

## 3. `anal.py` — SAS and SES from arrangements

### Preparation (`prepare`)

1. **Overlaps.** KD-tree pairs `(i < j)` with `d < R_i + R_j − TAU_C`,
   decided once; a pair tangent to within `TAU_C` is not an overlap.
   Coincident centres (`d < 1e-3 Å`) raise. `rp ≤ 0` or a non-positive
   radius raises.
2. **Contained balls.** If `d ≤ |R_i − R_j| + TAU_C` the smaller ball is
   contained: it has no surface and generates no caps. These atoms are
   dropped from everything that follows.
3. **Order.** `active` is the caller's mask minus contained atoms; `need` is
   `active ∪ neighbours(active)`. Atoms are permuted to
   `[active | need \ active | occluders]` and the overlapping pairs (neither
   contained) are remapped and sorted by `(i, j)`. From here on state is read
   from index ranges: a sphere is solved iff its index is `< n_solve`, owns
   dots iff `< n_active`; a pair touches a solved sphere iff its smaller
   index is `< n_solve`. Vertex clusters are relabelled by their smallest
   atom, so probes are owner-sorted; active probes, active circles and active
   torus arcs are prefixes whose lengths (`n_active_probes`,
   `n_active_circles`, `n_active_arcs`) are counted once in `build` and read
   everywhere else. Dot owners are mapped back through the permutation at the
   very end.
4. **What the mask makes exact.** Every atom occludes, so the caps of a solved
   sphere are complete and convex patches of active atoms are exact. A torus
   arc is emitted for circles whose smaller atom is active; that sphere is
   solved, and its vertices come from triples whose smallest atom overlaps an
   active atom and is therefore solved too, so toroidal patches are exact. A
   concave face of a probe on active atom `a` is cut by competing probes
   within `2rp`; a competing probe hosted by atom `d` needs
   `|c_a − c_d| < R_a + R_d + 2rp`, so `d` may lie up to `2rp` beyond the
   overlap shell and outside `need`, in which case its probe is never
   enumerated and the cut is missing. This requires a crevice with probes on
   opposing walls. On the eleven oracle interfaces it does not occur: of
   495k dots produced under the mask, none lies inside any probe ball of the
   full-molecule SES (which has 25.6k vertex probes to the masked runs'
   11.8k). Should a structure ever need it, the remedy is a fourth atom class
   of probe hosts within `R_a + R_d + 2rp` of an active atom whose triples
   are enumerated and clustered without solving their arrangements.

### SAS (`SasGeometry.build`)

1. **Circles.** For each overlapping pair `(i, j)` with `i < n_solve` (a
   prefix of the sorted pairs), unit axis `u = (c_j − c_i)/d`,
   `a = (d² + R_i² − R_j²)/2d`, centre `t = c_i + a u`, radius
   `rl = √(R_i² − a²)`, frame `(e1, e2, u)`, and `a`, `d` themselves. The
   half-angles `θ_i = atan2(a, rl)`, `θ_j = atan2(d − a, rl)` are derived only
   where the saddle formulas need them.
2. **Caps from circle rows.** Circle `(i, j)` cuts sphere `i` with axis `u`,
   `cos α = a/R_i`, `sin α = rl/R_i`, frame `(e1, e2)`, and, if `j` is solved,
   sphere `j` with axis `−u`, `cos α = (d − a)/R_j`, `sin α = rl/R_j`, frame
   `(e1, −e2)` (right-handed about `−u`). All rows are sorted by sphere once;
   a sphere's caps are a slice, tagged `2·circle + side` (side 0 on the
   circle's first sphere), so the circle, the axis sign and the other atom
   are tag arithmetic wherever they are needed. No cap geometry is
   recomputed, and the smaller sphere's arc `φ` *is* the circle's `φ`.
3. **Triple vertices, computed once** (`_triple_candidates`). Candidates are
   `(i < j < k)` with `(i, j)` a circle and `k` a later neighbour of `i` that
   also pairs with `j` (all vectorised over the neighbour lists). The circle
   `(i, j)` is intersected with sphere `k`: with `w = t − c_k`,
   `g = (R_k² − |w|² − rl²)/2rl`, `A = w·e1`, `B = w·e2`, `amp² = A² + B²`,
   the points exist iff `h² = amp² − g² > 0` and are

   ```
   cos φ± = (A g ∓ B h)/amp²,   sin φ± = (B g ± A h)/amp²,
   x± = t + rl (cos φ± e1 + sin φ± e2)
   ```

   which is `φ_0 ± acos(g/amp)` with `φ_0 = atan2(B, A)` written without
   inverse trig. This is the **only** crossing decision on SAS spheres: a
   candidate with `h² > 0` whose two caps are present and distinct (after
   coincident-cap merging) on every solved sphere of the triple is a crossing
   pair on each of them. Those pairs form each sphere's crossing graph, which
   step 1 of §2 takes as given to split the other pairs into nested and apart
   and to hide nested caps. A triple one of whose caps is hidden on some
   sphere has no vertex (its points lie inside the hiding ball) but keeps its
   crossing edges on the spheres where both caps survive: dropping the edge
   there would let a cap whose remaining vertices are all inaccessible pass
   as a full circle tested against too few caps. Only then are the points
   computed, once per surviving triple, and handed to every sphere of it.
   Deciding `h² > 0` independently per sphere produced a vertex on one sphere
   and none on another at exact tangency, and hence a phantom concave face;
   deciding crossing by the trig comparisons of §2 while producing vertices
   from `h²` lost exactly one cap's area whenever rounding put an internally
   tangent pair on different sides of the two tests.
4. **Global clustering.** All raw points are clustered at `TAU_C`. A cluster's
   atoms are the union of its triples; k-fold coincidences give probes with
   four or more atoms. Clusters are relabelled by their smallest atom, the
   owner.
5. **Accessibility on the owner sphere.** Each cluster is decided once, on the
   sphere of its smallest atom, with the test of §2 step 4 against that
   sphere's caps, excusing the caps of the triples that generated it: the
   same incidence the per-sphere solve uses for its arcs, so a vertex is
   never accepted for a cap that then gets no dart. Every ball that can
   contain a point of the sphere overlaps it and so is one of its caps (or
   nested inside one), so this equals the test against all SAS balls; a
   single decision per cluster keeps every sphere sharing the cluster
   consistent. All clusters are decided in one vectorised pass, covered
   owner spheres included (two covering caps touching at one point host a
   legitimate probe). Accessible clusters are the **probes**.
6. **Per-sphere solve.** Each non-covered sphere receives its crossing edges,
   the (cluster, cap) incidence of its kept triples, the projected
   representatives, the accessibility flags and its frames, and runs steps
   5–7 of §2; a covered sphere gets an empty arrangement. The cap components
   of all spheres come from one label-propagation pass over a block-diagonal
   graph.
7. **Torus arcs.** The arcs of a sphere on circles it is the smaller sphere of
   (tag side bit 0) are the torus arcs as they stand (same frame, same `φ`);
   the larger sphere reports nothing. Vertex ids are mapped to probe ids through a map with a
   `-1` appended, so full-circle ends (`-1`) read back `-1` without a branch.

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

  which is below `rp²` for all `Δφ ≠ 0` exactly when `ρ < 0`. With `β0 = 0`
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
  `(−π/2, π/2)`), so no sine of a selected angle is evaluated, and the
  bracketed integral of each range is stored once per circle; the arc areas
  and the sampler's row totals are both gathers of it.

  The range is never empty once contained balls are gone: `θ_i + θ_j` is the
  angle at the probe in the triangle `(c_i, q, c_j)`, positive for any
  overlapping, non-contained pair.
- **Concave face** of probe `q` (only probes whose smallest atom is active).
  Caps on the probe sphere:
  1. one cap for every other probe `q'` strictly within `2rp` (a tangent probe
     cuts nothing): axis `(q' − q)/|q' − q|`, `cos α = |q' − q| / 2rp`,
     `sin α = √(1 − cos² α)`; every probe pair is measured once and read from
     both sides with opposite axes;
  2. one **hemisphere per accessible arc leaving `q`**, axis = the arc's
     departure tangent `±(u × radial)` at `q` (`+` when the arc leaves toward
     increasing `φ`). Rolling along that arc, the probe sweeps the half of its
     sphere facing the tangent, so that half is not SES. All departure
     tangents are computed in one pass from the cluster mean, normalised (the
     mean is off the circle by up to `TAU_C`) and grouped by probe.

  For an ordinary three-atom vertex, rule 2 gives exactly the three side planes
  of the contact triangle (the departure tangent of circle `(a, b)` is normal
  to the plane through `q`, `c_a`, `c_b`). The same rule handles merged k-fold
  vertices (one hemisphere per surviving arc, possibly more or fewer than
  three), a vertex that is the only vertex on a circle (two opposite
  hemispheres, zero face), and nearly coplanar contacts (zero face).

  Rolling probes on the arcs themselves need no extra cut. Every such cap is
  bounded by the bisector plane of `q` and `q(δ)`, and all these planes contain
  the torus axis; over an arc they form a pencil whose union is the union of
  its two extremes, the departure hemisphere and the cap of the arc's end
  vertex, which is already in the within-`2rp` list.

  The face arrangement is solved locally with `solve_caps` for the faces and
  caps that survive the filters below; its area is the concave area.

### Which probes can cut a face

Notation for a probe `x` on atoms `a` with contact directions
`ĉ_a = (c_a − x)/R_a`: the **face cone** is `C = {Σ λ_a ĉ_a : λ_a ≥ 0}` (for
an ordinary three-atom vertex, the complement of the three departure
hemispheres), the **contact triangle** is `T = conv{c_a}`, and `t(d)` is the
distance from `x` along the ray `d ∈ C` to `T`. Lemmas 1, 3 and 4 are
stated for three-atom probes; merged k-fold probes take the unfiltered path.
The face is cut by `y` where it enters the open ball `B°(y, rp)`.

**Lemma 1 (cuts lie beyond the contact plane).** Let `y = x + v` be any
point outside the interior of `U` — every accessible probe centre, rolling
or resting, and every other SAS point. Then no face point `x + rp d` with
`t(d) ≥ rp` lies in `B°(y, rp)`.

*Proof.* `y ∉ B°(c_a, R_a)` gives `|x + v − c_a|² ≥ R_a²`, i.e.
`v·ĉ_a ≤ |v|²/2R_a`. Write `d = Σ λ_a ĉ_a`; the ray meets the triangle at
`t(d) d = Σ μ_a (c_a − x)` with `Σ μ_a = 1`, so `λ_a = μ_a R_a / t(d)` and
`Σ λ_a / R_a = 1/t(d)`. Hence `v·d = Σ λ_a v·ĉ_a ≤ |v|²/2t(d)` and

```
|v − rp d|² = |v|² − 2 rp v·d + rp² ≥ |v|² (1 − rp/t(d)) + rp² ≥ rp².  ∎
```

**Corollary 1 (uncuttable faces).** `min_{d∈C} t(d) = dist(x, T)`, which is
the plane distance `h` when the foot of the perpendicular lies inside `T`
and larger otherwise. A probe with `dist(x, T) ≥ rp` — equivalently, whose
own ball does not reach its contact triangle — is **high**: nothing cuts its
face, which is the spherical triangle `{d : d·t_m ≤ 0}` bounded by the three
departure great circles. Its area is `rp²` times the spherical excess,

```
A = rp² (Σ_m interior angle_m − π) = rp² (2π − Σ_m ∠(t_m, t_{m+1})),
```

with `∠(t_m, t_{m+1}) = atan2(|t_m × t_{m+1}|, t_m · t_{m+1})` (an angle
consumed as an angle). Every other probe is **low** (k-fold probes count as
low). `_probe_heights` decides this once for every probe, vectorised; high
active faces never reach the neighbour query or `solve_caps`.

**Lemma 2 (cutting is symmetric).** If the cap of `y` leaves an arc on the
face of `x`, the cap of `x` leaves an arc on the face of `y`.

*Proof.* Points `z` of that arc are at distance exactly `rp` from `x` and
`y` and, being on the face boundary, at distance `≥ rp` from every other SAS
point. Near `z`, the eroded region is therefore the complement of
`B°(x, rp) ∪ B°(y, rp)`, and the SES is the outer boundary of that union:
it contains points of `S(y, rp)` next to `z` (the two spheres cross
transversally since `0 < |x − y| < 2rp`). Those points are SES points on
the sphere of `y`, hence in the face of `y`: inside its cone and outside
every other cap. The cap of `x` on `S(y, rp)` has `z` on its boundary and so
contains face points of `y` next to `z`. ∎ (Generic position: the arc has
positive length and no third probe ball passes through `z`; coincidences
below `TAU_C` are merged upstream.)

**Corollary 2 (both ends low).** A cutting pair has two low ends: if `y`
cuts `x` then `x` cuts `y`, so `y`'s face is cut and `y` is low by
Corollary 1. Because every filter below is a *necessary* condition for one
side to be cut, and the pair cuts on both sides or neither, a pair is
dropped **for both faces** as soon as either side fails any test.

**Lemma 3 (the cap must reach the beyond-plane cap).** By Lemma 1 the cut
part of the face lies inside `{d : d·n > cos β}`, `cos β = h/rp`, where `n`
is the contact-plane normal pointing from `x` toward the triangle. A cap
`{d : d·u > cos α}` meets it only if the angle between `u` and `n` is below
`α + β`:

```
u · n > cos(α + β) = cos α cos β − sin α sin β.
```

All four values are stored; no angle is recovered.

**Lemma 4 (the cap must meet the spherical triangle).** The cap meets the
closed triangle `S = {d : d·t_m ≤ 0}` iff `u ∈ S` or the angular distance
from `u` to the boundary of `S` is below `α`. Edge `m` is the arc of the
great circle `t_m · d = 0` between the corners `p_{m+1}` and `p_{m+2}`,
`p_m = ±(t_{m+1} × t_{m+2})/|·|` signed so that `t_m · p_m ≤ 0`. The foot of
`u` on that great circle is `f = u − (u·t_m) t_m`, and the cosine of the
angle from `u` to `f` is `|f| = √(1 − (u·t_m)²)`; `f` lies on the arc iff it
is on the arc's side of the plane through the origin and the corners'
bisector, `f · (p + q) ≥ |f| · p·(p + q)`. So the cap meets the boundary iff

```
(f on arc m  and  |f| > cos α)   for some m,   or   max_m u·p_m > cos α,
```

the corner test covering feet off their arcs. Comparisons are on cosines
only. The condition is necessary for the cap to cut anything and, for a
face with no other caps, sufficient.

**Order and exactness.** Per structure: heights of all probes (Corollary 1)
→ closed-form areas of the high active faces → probe pairs within `2rp`
→ drop pairs with a high end (Corollary 2) → drop pairs failing Lemma 3 on
either side → drop pairs failing Lemma 4 on either side → `solve_caps` on
the low active faces with the surviving caps. Dropped caps contain no face
point, so the arrangement's accessible region, its area and the dots are
unchanged; the pilot's brute-force test (`anal_test.py`) checks per-face
equality against solving every probe within `2rp`. On 1brs (4 374 faces,
`rp = 1.7`, united-atom radii, every atom active): 3 782 faces are high;
20 243 probe pairs lie within `2rp`, 1 243 have both ends low, 456 pass
Lemma 3, 390 pass Lemma 4 and 292 finally leave an arc on both faces. The
face stage of the 921-atom chain drops from 0.41 s to 0.06 s (0.16 s to
0.02 s with 266 active atoms; login node, preliminary).

### What is discontinuous

The SES area is genuinely discontinuous at a four-sphere coincidence: for a
fourth sphere passing `ε > 0` outside an existing vertex there are 8 probes
near it, for `ε < 0` there are 4, and the two limits differ. Below `TAU_C` the
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
- **Toroidal**, for every (arc, range) row of the active arcs at once:
  1. `k_β = max(round(rp (hi − lo) √density), 1)` rings of equal `β` width;
     ring `m` spans `[β_m, β_{m+1}]` and is sampled at its middle;
  2. exact ring area `a_m = rp · dphi · [rl Δβ − rp (sin β_{m+1} − sin β_m)]`;
  3. `k_φ,m = round(a_m · density)` dots on ring `m`, uniformly spaced in `φ`
     inside `[phi_beg, phi_beg + dphi]`, weight `a_m / k_φ,m`. Ring areas are
     rescaled so that the row's weights sum to its exact area; a row whose
     rings all round to zero is collapsed to one ring with
     `round(area · density)` dots, and only rows that still round to zero are
     dropped (recorded in `Dots.dropped_area`). Zero-width rows drop zero area
     and emit nothing through the same arithmetic.
  4. Dot position `p = q(φ) + rp (−cos β · radial(φ) + sin β · u)`, normal
     `(q − p)/rp`, owner the atom whose vdW surface is nearer. `cos β`,
     `sin β` and the owner are per ring: `|p − c_i|² = R_i² + rp² −
     2 rp (rl cos β − a sin β)` and `|p − c_j|² = R_j² + rp² − 2 rp (rl cos β +
     (d − a) sin β)` do not depend on `φ`. Ring edge sines are shared between
     neighbouring rings.

  This is the "reading 2" scheme: uniform spacing along the generating arc,
  φ count proportional to the ring's true area. Density is uniform, cusp
  rings get few or no dots instead of crowding, and there is no `ceil` bias.
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

Overlap enumeration uses one KD-tree; triple candidates come from sorted
neighbour lists and a `searchsorted` pair lookup. Each sphere's arrangement is
`O(m²)` in its cap count `m` (10–40 for proteins) and independent of all other
spheres. Global steps: one KD-tree clustering of the raw vertices, one
label-propagation pass for all cap graphs, one vectorised accessibility test
for all clusters, one probe-pair query for all faces. Per sphere and per face
there remain a KD-tree query for coincident caps and one for local vertex
clustering (≤ 40 points, so a pairwise test suffices in C++). Connected
components everywhere are minimum-label propagation with pointer jumping, not
sparse graphs. Sampling is linear in the number of dots. In the Python pilot
what remains is one interpreter iteration per solved sphere (twice: merge,
classify), per non-covered sphere (solve), per low active face (solve; high
faces are closed-form, see §3) and per active atom and face (sampling): about
0.78 s + 0.06 s + 0.05 s for the 921-atom chain with every atom active,
0.54 s + 0.02 s + 0.02 s with 266 active atoms (login node, preliminary).
