# Analytic molecular surfaces and shape complementarity {#analytic_surface}

[TOC]

How nuri::AnalyticSurface computes the solvent-accessible and
solvent-excluded surfaces of a set of spheres, how
nuri::SurfaceSampler places dots on them, and how nuri::ShapeComplementarity
turns two dot sets into the Lawrence-Colman statistic. The code is in
`src/desc/surface/analytic/` and `src/desc/surface/sc.cpp`.

## Preliminaries {#prelim}

### Sources {#prelim-sources}

The SES is the molecular surface of Richards @cite desc:surface:richards-1977
in the analytic patch decomposition of Connolly
@cite desc:surface:connolly-1983, whose Gauss-Bonnet patch areas are used,
as is Richmond's for the SAS @cite desc:surface:richmond-1984.
The characterization of the concave patches and the formulas used follow
Quan and Stamm @cite desc:surface:quan-stamm-2016 (their Theorem 5.1 and
formulas (4.19)/(5.25)/(5.28)); the derivations on this page are this
document's own. The shape complementarity statistic is that of Lawrence and
Colman @cite desc:sc:lawrence-colman-1993 with the defaults of Lawrence's
CCP4 `sc` @cite desc:sc:ccp4-sc, which Rosetta's `sc` application ports.
The united-atom radii are Table 5
of the Connolly MSP 3.9.2 manual @cite param:radii:connolly-msp-manual-2002,
see nuri::ConnollyRadii; no radii file is copied from any program.

### Design rules {#prelim-rules}

The rules are the Exact Geometric Computation principles of
@cite desc:surface:sharma-yap-2017 (every branching decision errorless,
numerical values approximate), with the consistency motivation of
@cite desc:surface:hoffmann-1989.

- **Filter first**: exceptional input (coincident atoms, contained balls,
  inactive atoms, a non-positive probe radius) is removed or partitioned in
  a preparation stage, and no kernel afterward carries a guard for it.
- **Trig values, not angles**: caps store \f$(\cos\alpha, \sin\alpha)\f$,
  predicates compare trig values (\f$\arccos x \le a\f$ is written
  \f$x \ge \cos a\f$), and `atan2` is used only where an angle is consumed
  as an angle.
- **One predicate per decision**: each geometric fact (overlap, containment,
  shared circle, crossing, accessibility, ring order, cap components, ...)
  is decided once, by an exact predicate on the numbers geogram triangulated
  (`BallExact` in `exact.cpp`) with geogram's perturbation for every tie
  that is perturbed, and stored; consumers read it and never re-derive it by
  another formula or default. Floating point decides only candidate
  prefilters that are supersets of the exact sets, necessary-condition
  filters whose failure mode is a sliver (@ref ses-pipeline "The filter
  pipeline"), and produces the outputs. There is no clustering tolerance:
  two roots are one vertex only when an exact predicate finds them one
  point (@ref sas-merge "Probe merge").
- **No accessibility slack**: a positive slack breaks consistency between
  vertex acceptance and arc tests; do not reintroduce one.
- **Roots are proven, not clamped**: a square root's argument is
  non-negative by a preparation filter or an exact predicate, so kernels
  assert rather than clamp, and a loosened tolerance is fixed by a filter in
  the preparation. The clamps that remain are real zero branches (the
  spindle root \f$\sqrt{r_p^2 - r_l^2}\f$ at \f$r_l = r_p\f$, the crossing half-chord) and one rounding guard
  on the circle radius of an exact overlap of sub-ulp depth (`sas.cpp`).

### Definitions {#prelim-defs}

- Atom \f$i\f$ has center \f$c_i\f$, van der Waals (vdW) radius \f$r_i\f$ and SAS
  radius \f$R_i = r_i + r_p\f$, where \f$r_p\f$ is the probe radius.
- \f$B(c, R)\f$ is the closed ball, \f$B^\circ(c, R)\f$ its interior,
  \f$S(c, R)\f$ the sphere. \f$U = \bigcup_i B(c_i, R_i)\f$ is closed. The
  **SAS** is the boundary of \f$U\f$; its points are the accessible probe
  centers. An SAS point \f$s\f$ **touches** the atoms \f$a\f$ with \f$|s - c_a|
  = R_a\f$.
- The **SES** is the boundary of \f$\{p : B(p, r_p) \subset U\}\f$: the
  points of \f$U\f$ at distance exactly \f$r_p\f$ from the SAS. A point closer
  than \f$r_p\f$ to some SAS point \f$s\f$ is **cut** by \f$s\f$.
- The SES decomposes into three patch families
  @cite desc:surface:connolly-1983 (nuri::SurfacePatch):
  - **convex**: the probe touches one atom; the patch is the atom's vdW
    sphere scaled from the accessible part of its SAS sphere;
  - **toroidal (saddle)**: the probe rolls on two atoms; its center traces an
    arc of the circle where the two SAS spheres meet;
  - **concave**: the probe rests on three (or more) atoms; its center is a
    **vertex** of the SAS and the patch is a region of the probe sphere. The
    atoms a vertex or probe touches are its **hosts**.
- A **cap** on a unit sphere is \f$\{x : n \cdot x > \cos\alpha\}\f$ for a
  unit axis \f$n\f$, stored as \f$(n, \cos\alpha, \sin\alpha)\f$. The cap
  that a ball \f$B(y, r_p)\f$ cuts on the probe sphere \f$S(x, r_p)\f$ has
  axis \f$(y - x)/|y - x|\f$ and \f$\cos\alpha = |x - y| / 2 r_p\f$; it is
  empty unless \f$|x - y| < 2 r_p\f$, and \f$\alpha \le \pi/2\f$.

Every patch family is derived from arrangements of caps: the accessible part of
a SAS sphere is the sphere minus the caps cut by neighboring SAS spheres, and
the concave face of a vertex is the probe sphere minus the caps cut by
neighboring vertex probes and by the arcs it rolls off along.

The region of a sphere outside a set of caps is measured by Gauss-Bonnet.
For a region with its boundary on the left, on a sphere of radius \f$R\f$,

\f[
A = R^2 \Bigl[ 2\pi\chi - \sum_{\text{vertices}} \text{turn}
  + \sum_{\text{arcs}} \Delta\varphi \cos\alpha \Bigr],
\f]

where the last term is \f$-\oint \kappa_g\,ds\f$ for a small-circle arc of
angular span \f$\Delta\varphi\f$ traversed with its cap on the right
(\f$\kappa_g = -\cot\alpha / R\f$, \f$ds = R \sin\alpha\, d\varphi\f$); great
circles contribute nothing. \f$\chi = 2 P - L\f$ with \f$P\f$ patches and
\f$L\f$ boundary loops, and \f$P\f$ is obtained without grouping loops into
patches: \f$P = 1 + L - C\f$ with \f$C\f$ the number of connected components
of the union of the caps (the sphere minus \f$L\f$ disjoint simple loops has
\f$L + 1\f$ faces alternating accessible/inaccessible, and each inaccessible
face is one cap component).

## Preparation {#prep}

`prepare` in `prep.cpp` turns centers, radii, a probe radius and an active
mask into a `SurfaceSpheres`. Its tolerance is
`kSurfaceLengthEps` \f$= \tau = 10^{-6}\,\AA\f$ (`stages.h`), used only for candidate
bands that are supersets of exactly decided sets; every decision is exact.

1. \anchor prep-step-1 **Candidates.**
   A non-positive probe radius, non-finite input, any coordinate beyond
   `kSurfaceMaxCoord` \f$= 10^4\,\AA\f$ in absolute value
   (where the bands below stop being supersets), and SAS radii with
   \f$R_{\min} < \sqrt{2 r_p^2 + 2\,\tau\,R_{\max}}\f$ (the hypothesis of
   @ref lemma-11 "Lemma 11" for positive radii, a vdW radius just above
   \f$(\sqrt 2 - 1) r_p\f$, \f$0.58\,\AA\f$ for water; the sign of the comparison is
   what rejects a non-positive atom radius) are rejected. Pairs \f$(i < j)\f$
   with \f$d_{ij} \le R_i + R_j + 2\tau\f$ are the **near** pairs, found by a
   nuri::VoxelGrid over the atoms within `kBandHops` \f$+ 1 = 5\f$ near-pair
   lengths of an active atom. Pairs below \f$R_i + R_j - 2\tau\f$ are overlaps
   outright; inside the band `BallExact::overlap` decides exactly on the lifted
   heights of the @ref sas-notation "SAS notation", and a pair tangent to within
   rounding is a tie that is *reported*, not perturbed: it carries no circle
   (the @ref prep-zero-measure "zero-measure rule"), but it still counts as a
   neighbor for the ranks of @ref prep-step-4 "step 4".
2. \anchor prep-step-2 **Contained balls.**
   The band \f$d_{ij} \le |R_i - R_j| + \tau\f$ selects candidates;
   `BallExact::contained` decides: with \f$K = \rho_i^2 - \rho_j^2 -
   d_{ij}^2\f$, \f$B_j \subset B_i\f$ iff \f$K \ge 0\f$ and \f$K^2 \ge 4
   \rho_j^2 d_{ij}^2\f$, internal tangency included (identical balls keep the
   active one, else the first). Coincident centers are contained pairs. A
   contained ball has no surface, generates no cap and is dropped from
   everything that follows, including the overlap graph.
3. \anchor prep-step-3 **Shared circles.**
   If a third sphere passes through the circle where two others meet, the sphere
   whose center lies between the other two on the
   axis lies inside their union: on each side of the circle plane the outer
   sphere bulges more, so each of the middle sphere's two caps sits inside the
   neighboring sphere's larger cap. It is dropped like a contained ball.
   Without this, one sphere would carry two caps on the same circle, and the
   middle sphere's two caps would be exact complements, a tie for the covered
   test; the kernel also evaluates `cuts` on triples that are not faces, and
   two caps on one circle would tie there. The triangles of the exact overlap
   graph are the candidates, a band on floating circle centers and axes a
   prefilter, and `BallExact::shared_circle` decides: circles \f$(a, b)\f$ and
   \f$(a, c)\f$ coincide iff \f$u = d_b \times d_c = 0\f$ and the discriminant
   \f$D\f$ of the @ref sas-notation "SAS notation" is zero; the middle is \f$a\f$ iff
   \f$d_b \cdot d_c < 0\f$, else the nearer of \f$b\f$, \f$c\f$.

   The three filters follow one \anchor prep-zero-measure **zero-measure rule**:
   a sphere, or a circle, whose surface or length is zero in the unperturbed
   configuration is removed, decided exactly, and an exact tie is that answer. A
   contained ball (internal tangency included), a shared-circle middle and an
   exactly tangent pair carry \f$O(\varepsilon)\f$ area or
   \f$O(\sqrt\varepsilon)\f$ length in the perturbed arrangement and nothing in
   its \f$\varepsilon \to 0\f$ limit. A sphere inside a band but not exactly
   contained or coincident is kept and its sliver computed like any other
   surface. The bands are supersets of the exact sets as long as the rounding of
   \f$d_{ij}\f$, circle centers and axes stays far below \f$\tau\f$; the lifted
   squared radius \f$\rho_i^2\f$ drifts from \f$R_i^2\f$ by about
   \f$\operatorname{ulp}(|c_i|^2 + W)\f$, which is what bounds the coordinates.
4. \anchor prep-step-4 **Order.**
   `active` is the caller's mask minus dropped atoms; `need` is `active` plus
   its near neighbors and `shell` the near neighbors of
   `need`. Occluders near no `shell`, `need` or `active` atom are dropped
   (@ref lemma-3 "Lemma 3"). Atoms are permuted to `[active | need | shell |
   occluders]`, ranks 0 to 3, and the overlap graph `g` over the kept atoms is
   remapped and sorted. From here on state is read from index ranges: a sphere
   owns dots iff its index is below `n_active`, has its arrangement solved iff
   below `n_solve`, has caps and hosts vertices iff below `n_enum`; a pair
   touches a sphere with caps iff its smaller index is below `n_enum`, and such
   a circle gets its ring of arcs. Probes are sorted by their smallest atom, the
   owner; active probes and active torus arcs are prefixes whose lengths are
   counted once in `build_sas` and read everywhere else, and the active circles,
   the only ones with saddle patches, are the pairs below `g.offset(n_active)`.
   Dot owners are mapped back through `order` at the very end. The predicates of
   steps 1-3 run only on pairs whose two atoms lie within four near-pair hops of
   an active atom: every kept atom is within three hops, its containing partners
   and shared-circle triangles within four, and ranks only propagate from rank
   \f$\le 2\f$ sources, so the prepared set, its order, `g` and `d` equal the
   all-pairs computation.
5. \anchor prep-step-5 **What the mask makes exact.**
   Every atom occludes, so the caps of any sphere below `n_enum` are complete
   and the convex patches of active atoms
   are exact. An active circle's first sphere is its active atom (active
   indices are the lowest), so active torus arcs come from solved spheres.
   The hosts of a probe on an active atom overlap pairwise, so they are
   in `need`, every circle between two of them is solved, and the probe's
   departure tangents are complete. By @ref lemma-11 "Lemma 11", a
   first-touch cutter (@ref lemma-8 "Lemma 8") of a face on an active atom,
   all a face needs by @ref lemma-6 "Lemma 6", has a host that overlaps
   one of that face's hosts, hence a host in `shell`: it is enumerated, its
   accessibility is decided like every other root, and its cap is present.
   Its own departure tangents may be incomplete (a circle between two of its
   occluder hosts gets no ring); it is then skipped by the triangle gate of
   @ref lemma-10 "Lemma 10" and is tested without the triangle filter, which is always
   allowed. Two shells are exactly what
   @ref corollary-3 "Corollary 3" needs, and one is not enough.

## Solvent-accessible surface (SAS) {#sas}

`triangulate` (`delaunay.cpp`, on the vendored geogram weighted Delaunay) and
`build_sas` (`sas.cpp`) construct the SAS on the **regular triangulation**
\f$RT\f$ of the weighted points \f$(c_i, \rho_i^2)\f$, the dual of the power
diagram of \f$\pi_i(x) = |x - c_i|^2 - \rho_i^2\f$. A point is strictly
inside ball \f$i\f$ iff \f$\pi_i(x) < 0\f$, on \f$S_i\f$ iff
\f$\pi_i(x) = 0\f$, and its power cell is
\f$V_i = \{x : \pi_i(x) \le \pi_m(x)\ \forall m\}\f$; a point of \f$S_i\f$
is accessible iff \f$x \in V_i\f$.

Every decision below is an exact predicate on these numbers
(@ref prelim-rules "design rules"); the bands of the preparation are the
only floating prefilters.

### Notation {#sas-notation}

\f$\rho_i^2 = W + |c_i|^2 - h_i\f$ is the lifted squared radius
(@ref sas-step-1 "step 1"). For a circle \f$(a, b)\f$: \f$d = c_b - c_a\f$, \f$G
= |d|^2\f$, \f$v = \tfrac12(h_b - h_a) - c_a \cdot d = \tfrac12(\rho_a^2 -
\rho_b^2 + G)\f$,
center \f$\mathrm{cntr} = c_a + (v/G)\, d\f$, squared radius
\f$r^2 = \rho_a^2 - v^2/G\f$; \f$\pi_b - \pi_a = 2(v - y \cdot d)\f$ for
\f$y = x - c_a\f$. For a face \f$(a, b, c)\f$ relative to \f$a\f$:
\f$d_b, d_c\f$, \f$u = d_b \times d_c\f$, \f$v_b, v_c\f$ as above,
\f$\lambda = |d_c|^2 v_b - (d_b \cdot d_c) v_c\f$,
\f$\mu = |d_b|^2 v_c - (d_b \cdot d_c) v_b\f$, the radical line
\f$\ell = \{\pi_a = \pi_b = \pi_c\} = \{c_a + y_\perp + s u\}\f$ with
\f$y_\perp = (\lambda d_b + \mu d_c)/|u|^2\f$, and the discriminant
\f$D = |u|^2 (\rho_a^2 - |y_\perp|^2)\f$, a polynomial in \f$(c, h)\f$.
\f$k\f$ is the number of spheres through a point, \f$L\f$ the number of
boundary loops on a sphere, \f$C\f$ the number of cap components.

### Construction {#sas-construction}

1. \anchor sas-step-1 **Weights and perturbation.**
   The kept spheres, in original index order so that the result is shared across
   active masks, are lifted to
   \f$(c_i, t_i)\f$, \f$t_i = \mathrm{fl}(\sqrt{W - R_i^2})\f$,
   \f$W = \max R^2\f$, and handed to geogram, which derives the height
   \f$h_i = t_i^2 + |c_i|^2\f$ (rounded, in a fixed evaluation order) and
   triangulates the weighted points with power
   \f$|x|^2 - 2 x \cdot c_i + h_i = \pi_i(x) + W\f$. Powers are compared only
   by differences, so the triangulation is the same for every common shift
   of the weights; the shift is fixed by \f$\rho_i^2 = W + |c_i|^2 - h_i\f$, a
   rational number within an ulp of \f$|c_i|^2 + W\f$ of \f$R_i^2\f$ but not
   equal to it. Working on \f$R_i^2\f$ instead is inconsistent at the
   \f$10^{-15}\f$ level, exactly where degenerate cells live.

   Exact ties are resolved by **Simulation of Simplicity** (SoS)
   @cite desc:surface:edelsbrunner-mucke-1990 on the heights alone, the
   scheme of @cite desc:surface:devillers-teillaud-2011 (centers fixed,
   radii perturbed by index, so every cell of the limit triangulation is a
   genuine cell of a nearby input): geogram sorts the five points of its
   predicate by address, which for one contiguous vertex array is index
   order, and its tie branch returns the sign of
   \f$\partial \mathcal{P} / \partial h_m\f$, \f$\mathcal{P}\f$ the predicate
   value, for \f$h_m \to h_m - e_m\f$ with \f$e_i \gg e_j\f$ for \f$i < j\f$:
   every ball grows, lower index first. Formally \f$\rho_i^2(\varepsilon) =
   \rho_i^2 + e_i\f$, \f$e_i = \varepsilon \eta^i\f$, with \f$0 < \varepsilon
   \ll \eta \ll 1\f$ and \f$\varepsilon\f$ below every power of \f$\eta\f$, so
   that at first order the term of the lowest index dominates every later one,
   every second-order term is negligible against every first-order one, and a
   \f$\sqrt\varepsilon\, \eta^k\f$ term of a vanished discriminant dominates
   every first-order \f$\varepsilon \eta^m\f$ term; the sign of a tied predicate
   is the sign of its first non-zero first-order coefficient in index order.
   Geogram's predicate is linear in the heights, the polynomial case the cited
   scheme covers. The perturbation is a limit along a curve in the sense of
   @cite desc:surface:seidel-1998, and predicates on circle intersections, which
   carry square roots, are perturbed by @cite desc:surface:qsp-2017; the
   predicates here are \f$C^1\f$ in the heights away from \f$D = 0\f$
   (discriminants, \f$A \pm B\sqrt D\f$), and the application of the rule to
   them is this document's: the first-order rule applies as long as one
   participant has a non-zero first-order coefficient, which holds for `cuts`
   (the three coefficients sum to \f$|u|^2\f$, non-zero unless the centers are
   collinear, which `cuts` meets only on a shared circle, removed in
   preparation), `side` (the coefficient of \f$\rho_c^2\f$ is \f$-|d_b|^2/2\f$),
   `accept` (the coefficient of the apex is \f$-|u|^2/2\f$, and the apex is
   never a face vertex) and `antipode` (the kernel is \f$\pi_c(Q)\f$ scaled by
   \f$G|u|^2/2\f$; for a fan sphere \f$c\f$ off the face \f$d\pi_c(Q)/d\rho_c^2
   = -1\f$, and against the face's own third sphere \f$-2\f$). Where \f$D(0) =
   0\f$ (a tangency perturbed to a cut) \f$\sqrt{D(\varepsilon)}\f$ is of order
   \f$\sqrt\varepsilon\f$ and dominates every first-order term unless its
   coefficient \f$B\f$ vanishes; `accept` and `antipode` carry that one
   \f$\sqrt\varepsilon\f$ rule. Not every tie is perturbed: `overlap`,
   `contained` and `shared_circle` report the tie and preparation takes the
   limit answer, and `ccw` and `half_plane` report zero for the caller to
   resolve structurally.

   Four **bounding points** of weight 0 at the corners of a regular
   tetrahedron of circumradius \f$4(\Delta + R_{\max}) + 1\f$ around the
   bounding box (\f$\Delta\f$ its half diagonal) are appended so the input is
   never coplanar, the initial simplex of Bowyer and Watson
   @cite desc:surface:bowyer-1981 @cite desc:surface:watson-1981; they are
   inert (@ref lemma-1 "Lemma 1"). A sphere that appears in no cell has an
   empty power cell, no accessible surface, and gets area 0 without a solve.
2. \anchor sas-step-2 **Caps.**
   The caps of sphere \f$s\f$ are its \f$RT\f$ neighbors \f$t\f$ with \f$(s,
   t)\f$ in `g` (@ref prep-step-1 "preparation step 1"); the overlap predicate
   `BallExact::overlap` is rational: with \f$T = \rho_s^2 + \rho_t^2 - |c_s -
   c_t|^2\f$ the balls overlap iff \f$T \ge 0\f$ or \f$T^2 < 4 \rho_s^2
   \rho_t^2\f$ (the cross term of \f$(\rho_s + \rho_t)^2\f$ is irrational); the
   second kernel is \f$4 G r^2\f$, so its exact zero is a tangency, a circle of
   zero length. The perturbation would resolve it to an overlap (the coefficient
   of \f$t\f$'s height is \f$4\rho_s^2 - 2T = 4\rho_s^2 + 4\rho_s\rho_t > 0\f$),
   but the tie is reported instead: by the
   @ref prep-zero-measure "zero-measure rule" the pair carries no circle. Three
   structural rules make this the \f$\varepsilon \to 0\f$ limit of the perturbed
   arrangement, in which the tangent pair would carry a circle of radius
   \f$O(\sqrt\varepsilon)\f$ cut by every sphere through the tangency point: the
   pair is not an edge of the overlap graph, so no face \f$(a, b, \cdot)\f$ is
   examined and no fan contains it; two caps of one sphere whose spheres do not
   overlap never intersect (@ref sas-step-5 "step 5"), so on a third sphere
   through the point the two tangent discs stay two components; and an apex that
   does not overlap some face sphere never rejects a root
   (@ref sas-step-3 "step 3"), since its power is non-negative on that sphere
   and zero only at the tangency point. On the tangent spheres themselves the
   boundary passes smoothly through the point (the perturbed detour, two corners
   of \f$\pi/2\f$ around a half circle with \f$\cos\alpha \to 1\f$, has net
   turning zero), and on the third sphere the lens of the two discs closes to
   two cusps whose corners sum to \f$2\pi\f$, the same area as two disjoint
   discs.
3. \anchor sas-step-3 **Vertices from faces.**
   Every face \f$(a, b, c)\f$ of a finite cell whose three spheres pairwise
   overlap and whose smallest index is below `n_enum`
   is handled once. Sphere \f$c\f$ cuts circle
   \f$(a, b)\f$ in two points, one, or none (`cuts`: the sign of \f$D\f$); a
   double root is a tie and is perturbed to a cut or a miss. The two roots
   \f$x_\pm = c_a + y_\perp \pm (\sqrt D / |u|^2)\, u\f$ are the candidate
   vertices; \f$x_\pm\f$ is accepted iff \f$\pi_l(x_\pm) \ge 0\f$ for both
   apexes \f$l\f$ of the two cells sharing the face (`accept`:
   \f$\pi_l - \pi_a\f$ is affine along \f$\ell\f$, so this is the sign of
   \f$A \pm B\sqrt D\f$, decided exactly; a tie is perturbed like the
   triangulation). An apex that is a bounding point (@ref lemma-1 "Lemma 1"(a))
   or misses one of the face spheres is not tested: its power on that sphere is
   non-negative, zero only at a tangency point, and the limit accepts there
   (@ref sas-step-2 "step 2"). Each accepted root is one **probe** with exactly
   the three atoms \f$a, b, c\f$, its position evaluated as \f$\mathrm{cntr} +
   \text{offset}\f$ (see @ref analysis-fp "Floating point"). @ref theorem-1
   "Theorem 1" states that this is exact, with no slack.
4. \anchor sas-step-4 **Rings from cell fans.**
   For every \f$RT\f$ edge \f$(a, b)\f$ that is a cap pair, the cells around the
   edge are walked in cyclic order (across the
   two faces of each cell that contain the edge; the cycle closes within the
   finite cells by @ref lemma-1 "Lemma 1"(b)). Each face \f$(a, b, c)\f$ of the
   fan that cuts the circle contributes its accepted roots in the order *exit
   from the facet, then entry into the facet* (entering ball \f$c\f$, then
   leaving it) along the circle's orientation about \f$a \to b\f$; the
   orientation of the fan comes from the sign of the cell (geogram cells are
   positively oriented), so no angle is ever compared. @ref lemma-2 "Lemma 2"
   says this is the cyclic order of the vertices on the circle. Consecutive ring
   entries alternate entry/exit (a structural `DCHECK`); the arc from an *entry
   to the next vertex is accessible*, from an exit it is not. An empty ring is a
   full circle, accessible iff *no fan face cuts the circle and none contains
   it* (`side`: the sign of \f$\mu\f$, which is the sign of \f$\pi_c\f$ on a
   circle \f$c\f$ does not cut; a face that cuts the circle with both roots
   rejected means the circle is covered).

   The arcs carry \f$\varphi\f$ and \f$\Delta\varphi\f$ in the circle frame
   for output and for the \f$\sum \Delta\varphi \cos\alpha\f$ term. The
   \f$+2\pi\f$ wrap is a decision, not a convention: Gauss-Bonnet sums only
   the accessible arcs, so placing the wrap on an accessible instead of an
   inaccessible arc changes the area by \f$2\pi R^2 \cos\alpha\f$, and at a
   ring whose vertices are all probes of one \f$k\f$-fold point every
   floating \f$\varphi\f$ difference is rounding noise. It is decided exactly
   against the rational reference ray \f$d \times \mathbf{e}_m\f$ in the
   circle's plane (\f$\mathbf{e}_m\f$ the coordinate axis with \f$m = \arg\min
   |d_m|\f$, `reference_ray`; the output frame's \f$e_1\f$ is its unit vector):
   each vertex is classified as on the ray, in \f$(0, \pi)\f$, at \f$\pi\f$, or
   in \f$(\pi, 2\pi)\f$ by the signs of two linear forms in \f$x -
   \mathrm{cntr}\f$ (`half_plane`, one square root each); two consecutive
   vertices in the same open half-plane are ordered by the sign of the oriented
   area \f$((x_i - \mathrm{cntr}) \times (x_j - \mathrm{cntr})) \cdot d\f$
   (`ccw`, an expression in two square roots reduced to one by squaring), and an
   exact zero within one half-plane means the two probes coincide (antipodal
   points fall into different classes). The numeric angle of each vertex is then
   made consistent with its exact class (0 on the ray, \f$\pi\f$ at \f$\pi\f$,
   `atan2` otherwise, folded into \f$[0, 2\pi]\f$), so the angles are monotone
   across classes and the wrap goes to the unique consecutive pair whose arc
   contains the ray; a coincident pair gets \f$\Delta\varphi = 0\f$, and a
   rounding-level negative \f$\Delta\varphi\f$ elsewhere is clamped at zero. If
   every consecutive pair coincides (the **single-point window**: all vertices
   are one point \f$x\f$ to the exact order), the circle away from the point is
   accessible iff its antipode \f$Q = 2\,\mathrm{cntr} - x\f$ has \f$\pi_c(Q)
   \ge 0\f$ for every fan sphere \f$c\f$ (`antipode`, exact, same form as
   `accept`), and the \f$2\pi\f$ goes to an arc of that accessibility. The
   window is justified because every perturbed vertex of the ring lies within
   \f$O(\varepsilon)\f$ of \f$x\f$ on a circle of radius bounded away from zero
   (@ref sas-step-2 "step 2" removed the zero-length ones), so the rest of the
   circle is one arc, and \f$Q\f$ lies on it. Each arc also carries its two end
   tangents, taken from the circle frame at each end vertex's own angle rather
   than from the probe positions: darts and arcs then describe one closed curve
   even on a circle too small for the positions to resolve, which is what
   Gauss-Bonnet needs.
5. \anchor sas-step-5 **Arrangement per sphere.** (`ArrangementSolver`,
   `arrangement.cpp`) On sphere \f$s\f$ a probe lies on exactly two circles, so
   it has one in-dart and one out-dart. Arcs are stored with increasing
   \f$\varphi\f$, which runs counter-clockwise around the cap axis with the cap
   on the left, so the accessible region is traversed against \f$\varphi\f$; the
   out-dart's departure tangent and the reversed in-dart's arrival tangent are
   projected into the tangent plane at the probe, and the corner is the signed
   angle \f$\iota\f$ between them, brought into \f$(-\pi/2, 3\pi/2]\f$, with
   turning \f$\pi - \iota\f$. Near any vertex the accessible region is an
   intersection of half-planes, a convex cone, so a genuine corner has \f$\iota
   \in [0, \pi]\f$: \f$\pi\f$ when a single cap bounds the region there (an
   internal tangency), 0 at a cusp. The successor of an arc follows the
   accessible arcs, a full circle is its own loop, and loops are the cycles of
   the successor permutation, counted by one visited-flag walk.

   \f$C\f$ is the number of components of the union of *all* caps of
   \f$s\f$, including caps with no accessible arc: the components of a finite
   union of open discs are the components of their intersection graph, and
   two open discs intersect iff their circles cross (`cuts`) or one circle
   lies inside the other's ball (`side`, either way); two discs touching at
   a point are not joined, as in @ref sas-step-2 "step 2". Collinear centers
   never tie in `cuts` here: coincident circles are removed in preparation, and
   otherwise
   \f$D = -|d_b|^2 (\kappa v_b - v_c)^2 < 0\f$ strictly for
   \f$d_c = \kappa d_b\f$. Caps sharing a probe are joined for free, caps whose
   spheres do not overlap are never joined (disjoint balls cut disjoint caps),
   and the remaining pairs need the predicates (`discs_intersect`), at most
   \f$m(m-1)/2\f$ per sphere. Counting only caps with arcs is wrong: a band of
   two rows of caps has a north loop from the top row and a south loop from the
   bottom row and no cap on both, so it would count two components instead of
   one and be off by \f$4\pi R^2\f$. Then \f$P = 1 + L - C\f$ and the
   Gauss-Bonnet formula give the area. The degenerate cases are structural: no
   caps, \f$4\pi R^2\f$; caps but no accessible arc, the accessible region has
   no boundary and is not the whole sphere, so it is empty, area 0. Nested caps
   carry no arc (their roots are rejected by @ref theorem-1 "Theorem 1") and
   join their container's component.

### Correctness {#sas-correctness}

\anchor lemma-1
**Lemma 1 (bounding points are inert).** In plain words: a bounding point never
rejects a vertex and never bounds a fan. Precisely:

- (a) Every bounding point has positive power on all of \f$U\f$, so it rejects
  no cut point of any face.

  *Proof.* Every point of \f$U\f$ lies within \f$\Delta + R_{\max}\f$ of the box
  center and a bounding point at distance \f$4(\Delta + R_{\max}) + 1\f$ from
  it, so its power, the squared distance for weight 0, is at least
  \f$(3(\Delta + R_{\max}) + 1)^2 > 0\f$ on \f$U\f$; a cut point lies on
  \f$U\f$.
- (b) Every hull face of \f$RT\f$ consists of bounding points only, so the cells
  around an edge between two spheres form a closed cycle of finite cells.

  *Proof.* The inradius of the tetrahedron is a third of its circumradius and
  exceeds \f$\Delta\f$, so every center is strictly inside it and the convex
  hull of the centers is the tetrahedron. Each bounding point \f$b\f$ is a
  vertex of \f$RT\f$: its own position lies in its power cell, since
  \f$\pi_b(b) = 0\f$ while \f$\pi_i(b) = |b - c_i|^2 - \rho_i^2 > 0\f$ for
  every sphere, \f$b\f$ being outside \f$U\f$ by the distance bound of (a). So
  the facets of the tetrahedron are the hull faces of
  \f$RT\f$, spanned by bounding points only, and no edge between two spheres
  lies on the hull.

\anchor theorem-1
**Theorem 1 (vertex recipe).** In plain words: a cut point of a face is on
the surface iff it is outside the two balls on either side of that face, and
every surface vertex is a cut point of some face. Fix \f$\varepsilon > 0\f$
small enough in the sense of @ref sas-step-1 "step 1"; in that configuration no
two triangulation cells share an orthocenter and no cell orthocenter has zero
power. Let \f$x\f$ be a cut point of circle \f$(a, b)\f$ with sphere
\f$c\f$, and let \f$abc\f$ be a face of \f$RT\f$ with apexes \f$l_1\f$,
\f$l_2\f$. Then \f$x\f$ is accessible iff \f$\pi_{l_1}(x) \ge 0\f$ and
\f$\pi_{l_2}(x) \ge 0\f$; and every accessible cut point is produced by some
face.

*Proof.* This is the duality of the power diagram and the regular
triangulation @cite desc:surface:aurenhammer-1987 applied to the boundary of
a union of balls @cite desc:surface:edelsbrunner-1995. The power cell
\f$V_m\f$ is the intersection of the half-spaces \f$\pi_m \le \pi_l\f$, and
the edge of the power diagram dual to the face \f$abc\f$ is the segment of
the radical line \f$\ell = \{\pi_a = \pi_b = \pi_c\}\f$ between the
orthocenters of the two cells sharing the face, where the apex constraints
are tight (a center in the face plane makes \f$\pi_m - \pi_a\f$ constant
along \f$\ell\f$ and never binds). A point of \f$S_a\f$ is accessible iff it
lies in \f$V_a\f$, so the accessible cut points of circle \f$(a, b)\f$ with
\f$c\f$ are the points of \f$S_a \cap \ell\f$ on that segment; at \f$x\f$,
\f$\pi_a = 0\f$, so membership is \f$\pi_{l_1} \ge 0 \wedge \pi_{l_2} \ge 0\f$.
Conversely an accessible cut point lies on exactly three spheres in this
configuration, so \f$V_a \cap V_b \cap V_c\f$ is one-dimensional, \f$abc\f$
is a face, and that face produces the point.

\anchor remark-1
**Remark 1 (degenerate cells).** At an unperturbed \f$k\f$-fold point the
recipe still holds pointwise, but
two cells can share an orthocenter, and then every face of the degenerate
cell accepts the same point, so the probe count and the ring alternation
match no arrangement. That configuration is never evaluated: ties in
`accept` are resolved by the perturbation, which separates the orthocenters
by \f$O(\varepsilon)\f$ in the direction the triangulation used. Four
spheres through one point form an ordinary cell whose orthocenter has zero
power: the tie sits at the end of the dual edge, and `accept` resolves it.
Five or more spheres on one orthosphere make a degenerate cell with a
zero-length dual edge; after the perturbation it is a chain of cells with
\f$O(\varepsilon)\f$ edges, every face of the chain produces its own cut
point at the same position to rounding, and each accepted one is a probe.

\anchor lemma-2
**Lemma 2 (fan order is ring order).** In plain words: walking the cells
around an edge visits the circle's vertices in their order along the circle.
Precisely: assume no face of the fan is tangent to the circle (a double root is
perturbed to a cut or a miss before this lemma applies) and no cell orthocenter
lies on the circle (a tie that `accept` perturbs away).

*Proof.* The circle \f$(a, b)\f$ and the power facet \f$P = V_a \cap V_b\f$ both lie in
the radical plane \f$\{\pi_a = \pi_b\}\f$; \f$P\f$ is a convex polygon
whose edges, in fan order, are the dual segments of the fan faces, and the
accessible arcs are the circle inside \f$P\f$. Since
\f$\mathrm{disc} \cap P\f$ is convex, its boundary alternates arcs of the
circle and edges of \f$P\f$ in one cyclic order, so the crossings occur
along the circle in the cyclic order of the edges, and on one edge the
circle first enters and then leaves the half-plane of ball \f$c\f$ (exit
from the facet, then entry) when the circle and \f$\partial P\f$ are both
run counter-clockwise about \f$a \to b\f$, the orientation the positively
oriented cells give the fan. Points not accepted by
@ref theorem-1 "Theorem 1" lie outside the segment and are skipped; a face
that cuts the circle with both roots rejected has its segment wholly inside
or wholly outside the disc, and the circle does not meet that edge.

\anchor theorem-2
**Theorem 2 (neighbor caps suffice).** In plain words: every buried point
of a sphere is buried by a triangulation neighbor, so the neighbor caps
alone give the right components.

*Proof.* The union of balls decomposes as
\f$U = \bigcup_s (B_s \cap V_s)\f$, and \f$x \in S_s\f$ is accessible iff
\f$x \in V_s\f$ @cite desc:surface:edelsbrunner-1995. \f$V_s\f$ is a
convex polyhedron whose facets are the \f$RT\f$ edges of \f$s\f$
(non-empty for a sphere that is an \f$RT\f$ vertex; one in no cell gets area 0
without a solve, @ref sas-step-1 "step 1"); if \f$x \notin V_s\f$ some facet
neighbor \f$n\f$ has \f$\pi_n(x) < 0\f$ strictly, and \f$n\f$ is a sphere since
a bounding point has \f$\pi \ge 0\f$ on \f$U\f$ (@ref lemma-1 "Lemma 1"(a)), so
a point of \f$S_s\f$ inside some open ball is inside the open ball of an
\f$RT\f$ neighbor. Hence the union of the open neighbor caps equals the union of
all open overlapping caps exactly (a tangent neighbor has an empty open cap,
consistent with @ref sas-step-2 "step 2"), the components of that union are the
components of @ref sas-step-5 "step 5", and every accessible arc on circle
\f$(s, t)\f$ lies in the facet \f$V_s \cap V_t\f$, so \f$(s, t)\f$ is an edge
and, by @ref theorem-1 "Theorem 1" and @ref lemma-2 "Lemma 2", its endpoints are
the ring of that edge.

\anchor lemma-3
**Lemma 3 (occluder crop).** In plain words: an occluder that is near no atom
with caps can be deleted before the triangulation without changing any surface
that is reported. Let \f$S\f$ be the kept spheres after preparation with their
ranks and
\f$O \subseteq \{o \in S : \operatorname{rank}(o) = 3,\ \text{no near pair } (o, a) \text{ with } \operatorname{rank}(a) \le 2\}\f$.
Let \f$S' = S \setminus O\f$ with the same \f$W\f$ and heights \f$t\f$, the
survivors in their relative index order, `g`, `d` and `order` restricted to
\f$S'\f$, and `n_active`, `n_solve`, `n_enum` unchanged (a restriction of the
prepared spheres, not a second `prepare`, whose \f$W\f$ and contained-ball drops
depend on the dropped atoms). Running `triangulate`, `build_sas`, `build_ses`
and the sampler on \f$S'\f$ instead of \f$S\f$ leaves unchanged, as sets:

- (a) the accepted roots, each named by its sorted sphere triple and root sign,
  with their positions, hence the probes, their atom lists;
- (b) on every sphere below `n_enum` the ring of every circle, the arcs with
  their \f$\varphi\f$, \f$\Delta\varphi\f$, ends and tangents, the union of its
  caps and its component count, and, where it is solved (below `n_solve`), its
  area to rounding;
- (c) the merged clusters, the SES patches and the dots.

The cap *list* of a sphere is not invariant (@ref remark-2 "Remark 2"); its
union is.

*Setting.* For a point \f$y \in S_a\f$ let
\f$X(y) = \{m \ne a : \pi_m(y) \le 0\}\f$ be the balls holding \f$y\f$,
split into \f$X^0(y)\f$ (\f$\pi_m(y) = 0\f$) and \f$X^-(y)\f$
(\f$\pi_m(y) < 0\f$); every other sphere has \f$\pi_m(y) > 0\f$ strictly,
which no perturbation changes.

*Proof.*

1. *Separation.* For \f$o \in O\f$ and \f$\operatorname{rank}(a) \le 2\f$,
   \f$|c_o - c_a| > \rho_o + \rho_a\f$: the pair is not near, so its floating
   distance exceeds \f$R_o + R_a + 2\tau\f$, and the near band is a superset
   of the exact overlaps and tangencies on \f$\rho\f$ within
   `kSurfaceMaxCoord`. Hence (a) \f$(o, a) \notin g\f$; (b) no enumerated
   face contains \f$o\f$, since a face needs pairwise `g`-edges and a vertex
   below `n_enum`; (c) \f$\pi_o > 0\f$ on \f$B_a\f$, so for every
   \f$y \in S_a\f$, \f$X(y) \subseteq S'\f$.
2. *Tie signs are order-local.* A predicate whose participants lie in
   \f$S'\f$ plus bounding points has the same sign in both runs. Untied, it
   is the same number. Tied, its sign is that of the first non-zero
   first-order coefficient in index order among its participants, and the
   perturbation of each point depends on its own index alone
   (@cite desc:surface:edelsbrunner-mucke-1990). `triangulate` lifts the
   kept spheres in original index order and appends the bounding points, so
   the relative order of any \f$S'\f$ subset is the same in both lifts, and
   geogram's tie branch sorts by address, the same order. The two runs
   differ in exponents, not in which term dominates.
3. *Root verdicts are local.* Let \f$f = (a, b, c)\f$ be a `g`-face with
   \f$\min f <\f$ `n_enum` and \f$x\f$ a root of it. Claim: the verdict on
   \f$(f, x)\f$ is a function of `g` on \f$f \cup X(x)\f$ and of signs of
   predicates on \f$f \cup X(x)\f$. With 1(c) and 2 this gives (a);
   positions are bitwise equal, since the root reads only the face's columns
   and \f$W\f$. Take the radical line \f$\ell\f$ of \f$f\f$; for every vertex
   \f$m \notin f\f$ the constraint \f$\pi_m \ge \pi_a\f$ cuts \f$\ell\f$ in
   a half-line \f$H_m\f$ with tight point \f$t_m\f$, or is all of \f$\ell\f$
   or none when \f$c_m\f$ lies in the face plane (@ref theorem-1 "Theorem 1"). The dual edge
   \f$F = \bigcap H_m\f$ is the segment between the two orthocenters when
   \f$f\f$ is a face of the triangulation and empty otherwise, and each
   endpoint is the tight point of its apex. \f$x\f$ is accepted iff \f$f\f$
   is a face and every tested apex (`g`-adjacent to all of \f$f\f$) has
   \f$x \in H_{\text{apex}}\f$.
   - *Far constraints.* For \f$m \notin X(x)\f$, \f$x\f$ is inside \f$H_m\f$
     at a margin from \f$t_m\f$ for every small \f$\varepsilon\f$: such an
     \f$m\f$ never rejects, tested or not. Bounding points are far by
     @ref lemma-1 "Lemma 1"(a), every \f$o \in O\f$ by 1(c).
   - *Strict containment rejects.* If \f$m \in X^-(x)\f$, \f$x\f$ is outside
     \f$H_m\f$ at a margin, and \f$F \subseteq H_m\f$; the apex on that side
     has its tight point at the margin from \f$x\f$, so it is not in
     \f$X^0(x)\f$ and not a bounding point: an untested apex is one of those
     two. So the apex is tested and rejects, or \f$f\f$ is no face; \f$x\f$
     is rejected in both runs.
   - *Otherwise.* With \f$X^-(x) = \emptyset\f$, on each side of \f$x\f$
     along \f$\ell\f$ the binder is the \f$X^0(x)\f$ member whose half-line
     excludes that side and whose perturbed tight point comes first, if such
     a member exists, else a far vertex. Which side a member excludes is
     unperturbed; which comes first, whether the two binders leave \f$F\f$
     non-empty, whether a binding member is tested and whether it accepts
     are signs of predicates on \f$f \cup X^0(x)\f$, and by 2 geogram makes
     the same choice in both runs. A far binder is untested or accepts.
4. *Rings, arcs, antipode.* Fix a circle \f$(i, j) \in g\f$ with
   \f$i <\f$ `n_enum` and the facet \f$P = V_i \cap V_j\f$ (@ref lemma-2 "Lemma 2"). Every
   accepted root of a face \f$(i, j, c)\f$ is in the fan of the edge, and
   the fan contributes exactly its accepted roots; a skipped fan face, whose
   third vertex is not `g`-adjacent to both, neither cuts nor contains the
   circle. By 3 the ring set is the same in both runs, as are each vertex's
   parity, class, angle and the `ccw` of each pair. Distinct vertices are
   ordered by position; coincident vertices at a \f$k\f$-fold point \f$x\f$
   are roots of faces \f$(i, j, c)\f$ with \f$c \in X^0(x)\f$, and near
   \f$x\f$ the edges of \f$P\f$ are the lines \f$\{\pi_c = \pi_i\}\f$,
   \f$c \in X^0(x)\f$, shifted by \f$O(\varepsilon)\f$, every other line at
   a margin, so their order around \f$P\f$ is a sign of predicates on
   \f$\{i, j\} \cup X^0(x)\f$, the same in both runs by 2. Hence the ring is
   the same cyclic sequence with the same wrap, arcs and tangents. A full
   circle is emitted iff the circle lies in \f$P\f$, iff every point of it
   has \f$\pi_m \ge 0\f$ for every \f$m\f$; spheres with \f$\pi_m > 0\f$ on
   the whole circle are irrelevant by compactness, and every other \f$m\f$
   holds a circle point \f$y \in S_i\f$, so \f$m \in X(y) \subseteq S'\f$.
   In the single-point window, \f$Q \notin P\f$ iff an edge constraint of
   \f$P\f$ fails at \f$Q\f$, and that edge's sphere holds \f$Q\f$ strictly,
   so it is in the fan; the answer is settled by far spheres strictly and by
   \f$X^0(Q) \subseteq S'\f$ through `antipode`.
5. *Caps, components, areas.* The two runs may list different neighbor caps
   for a sphere (@ref remark-2 "Remark 2"). By @ref theorem-2 "Theorem 2" in each run, the union of the listed
   open caps is the union of all `g`-caps of \f$s\f$, one set \f$U_s\f$ for
   both runs (1(a)). The solver joins two caps when a vertex links them or
   `discs_intersect` holds, i.e. when their open discs meet; the components
   of a finite union of open connected sets are those of their intersection
   graph, so the count is the number of components of \f$U_s\f$. A cap listed
   in one run only has an open disc inside \f$U_s\f$, connected and
   non-empty, so it meets caps of one component and changes no count; it
   links no vertex, since an accepted root on circle \f$(s, t)\f$ makes
   \f$(s, t, c)\f$ a face of both triangulations by 3. Loops follow the
   successor over the arc set (4); each corner comes from the same two arcs'
   tangents, bitwise. The turning sum and \f$\sum \Delta\varphi \cos\alpha\f$
   are summed in list order, which may differ, so the area agrees to
   rounding. A sphere with no edge gets 0 without a solve: its power cell is
   empty, so every point of \f$S_s\f$ is held by a ball of \f$S'\f$ (1(c)),
   the same in both runs; in the other run no root on \f$s\f$ is accepted,
   no circle of \f$s\f$ is full, and the solve with caps and no arc returns 0.
   So (b).
6. *Probes, SES, dots.* Probes are sorted by owner and `n_active` counts the
   same set; the merge joins exactly the coincident pairs (an exact
   predicate on the input, @ref sas-merge "Probe merge") and takes the
   sorted atom union. In
   `build_ses`, the convex area scales the SAS area; saddle ranges and
   integrals are per circle from `pts`, `sar`, `d` (bitwise) and saddle
   areas per arc from its \f$\Delta\varphi\f$; probe heights and contact
   triangles are per probe from its atoms, position and tangent set; the
   pair query takes every probe pair within \f$2 r_p\f$ (a set) and filters
   each pair by its own data, so the surviving pairs are one set. The face
   solver receives the same caps and drops the non-overlapping and duplicate
   ones by untied predicates; the area it reports is that of the arrangement
   of those caps, one value to rounding for every cap order. The sampler
   keeps the lattice directions outside every listed cap, i.e. outside
   \f$U_s\f$ (convex) or outside the live face caps (concave), the same
   sets; toroidal rows are per arc from bitwise data; owners are per dot
   from the same sorted atom lists. So (c).

*What may change.* Everything the triangulation numbers: cells, adjacency,
face and root ids, probe ids within one owner, the start of each ring, the
cap list of a sphere (a cap whose disc lies inside the union of the others
and carries no arc may appear or disappear), solver slots, the column order
of a probe's tangents and the cap order handed to the face solver, hence the
index order of its perturbation: an exact tie inside one face may resolve
differently, with the same area to rounding and the same live caps. The
first member of a merged cluster, hence which certified position of its
one exact point the probe carries, and whatever reads that position.
Floating sums in a data-dependent order, so areas agree to
about \f$10^{-14}\f$ relative, not bitwise; a patch's dot count
\f$\operatorname{lround}(\text{area} \cdot \text{density})\f$ flips only when
the product is within that rounding of a half-integer.

\anchor remark-2
**Remark 2 (the cap list is not invariant).** \f$s = (0, 0, 0)\f$,
\f$t = (1.2, 0, 0)\f$, \f$u = (0.6, 0, 0.5)\f$, \f$o = (0.6, 0, -1.92)\f$ with
\f$R = 1, 1, 1.4, 1\f$; \f$s\f$ active, \f$t\f$ and \f$u\f$ need, \f$o\f$
rank 3 and near nothing. The circle \f$(s, t)\f$ (plane \f$x = 0.6\f$,
radius 0.8) lies inside \f$B_u\f$, so it has no arc, and the cap of \f$t\f$
on \f$s\f$ lies inside the cap of \f$u\f$. On the radical plane
\f$x = 0.6\f$, \f$\pi_s \le \pi_u\f$ iff \f$z \le -1.07\f$ and
\f$\pi_s \le \pi_o\f$ iff \f$z \ge -0.87\f$: with \f$o\f$ the facet
\f$V_s \cap V_t\f$ is empty and \f$t\f$ is no neighbor of \f$s\f$; without
\f$o\f$ it is the half-plane \f$z \le -1.07\f$ and \f$t\f$ is one. Both runs
give \f$s\f$ the cap union \f$\mathrm{cap}(u)\f$, one component, the same
arcs and area.

\anchor remark-3
<b>Remark 3 (sharpness of @ref lemma-3 "Lemma 3").</b> The criterion is
the near band of rank \f$\le 2\f$, not less. A rank-3 atom \f$o\f$ near a rank-2 atom \f$a\f$ hosts faces
\f$(a, o, o')\f$ whose accepted roots are the probes with a host in `shell`
that @ref lemma-11 "Lemma 11" identifies as the first-touch cutters of faces on active
atoms; dropping \f$o\f$ deletes a cutter and its cap. Replacing the band by
`g` is not safe either: a sphere exactly tangent to \f$a\f$ is in
\f$X^0(x)\f$ at its tangency point \f$x\f$, and if \f$x\f$ is a root there
it takes part in the verdict of step 3 of the proof of @ref lemma-3 "Lemma 3"; the near band is the floating superset of
overlap and tangency that keeps \f$X(y) \subseteq S'\f$ for every \f$y\f$ on
a sphere with caps.

### Degenerate configurations {#sas-degenerate}

#### Corners at k-fold points {#sas-corners}

\anchor lemma-4
**Lemma 4 (corners at a k-fold point are limits).** In plain words: where
several circles of one sphere pass through one point, a perturbation spreads
that point into a short chain of vertices; the run evaluates every chain corner
at the point itself and gives the arcs between them length zero, and each of
those terms is exactly the limit of the perturbed term, so no merged corner is
ever needed. Precisely: let \f$x\f$ be a point of a solved sphere \f$s\f$ on
\f$k - 1 \ge 2\f$ of its circles, every other kept sphere at positive power at
\f$x\f$ or tangent to \f$s\f$ there (a partner carries no circle with
\f$s\f$, @ref theorem-3 "Theorem 3"), and let \f$v_1, \dots, v_m\f$ be the accepted roots at \f$x\f$ of
faces containing \f$s\f$. Let \f$\Sigma_\varepsilon\f$ be a family of
configurations with the run's \f$RT\f$, cells and fans, whose spheres converge
to the kept spheres as \f$\varepsilon \to 0\f$, in which for small
\f$\varepsilon\f$ the circles of \f$s\f$ near \f$x\f$ cross pairwise
transversally with no three through one point, and whose vertices
\f$v_j(\varepsilon)\f$ near \f$x\f$ correspond one to one to the \f$v_j\f$ and
are listed by the run's rings in the cyclic order of \f$\Sigma_\varepsilon\f$.
The configuration of @ref sas-step-1 "step 1" is one such family
(@ref theorem-1 "Theorem 1", @ref lemma-2 "Lemma 2"); \f$\Sigma\f$ of
@ref theorem-3 "Theorem 3" is another (steps 1-3 of its proof).

- (a) The corner \f$\iota_j\f$ the run computes at \f$v_j\f$ from the exact
  tangents of its two circles at \f$x\f$ is the limit of the perturbed corner at
  \f$v_j(\varepsilon)\f$ and lies in \f$[0, \pi]\f$.

  *Proof.* In \f$\Sigma_\varepsilon\f$ the two circles of
  \f$v_j\f$ cross transversally at \f$v_j(\varepsilon)\f$ and no other circle
  passes through it, so near the vertex the accessible region is the wedge
  outside both caps, the intersection of two half-planes of the tangent plane,
  and the perturbed corner is the wedge's opening, in \f$(0, \pi)\f$. As
  \f$\varepsilon \to 0\f$ the circles move continuously and
  \f$v_j(\varepsilon) \to x\f$, so the departing and arriving tangents converge
  to the tangents of the exact circles at \f$x\f$, the ones the run reads from
  each circle's frame at the vertex's own angle (@ref sas-step-4 "step 4"), and
  the corner converges to a value in \f$[0, \pi]\f$. The run folds the signed
  angle into \f$(-\pi/2, 3\pi/2]\f$ (@ref sas-step-5 "step 5"); that fold is
  continuous on the open window, which contains \f$[0, \pi]\f$ with a margin of
  \f$\pi/2\f$ on each side, so the folded value of the limit tangents is the
  limit of the perturbed corners. The endpoints occur: \f$\pi\f$ when the two
  circles are tangent at \f$x\f$ from the same side, 0 at a cusp.
- (b) An arc between two consecutive chain vertices on one circle carries
  \f$\Delta\varphi = 0\f$, the limit of its perturbed length, except the wrap
  arc of a single-point window, which is the rest of the circle in both.

  *Proof.* Two consecutive ring entries that are both roots at \f$x\f$ fall in
  one exact half-plane class with `ccw` exactly zero, and
  @ref sas-step-4 "step 4" assigns that pair \f$\Delta\varphi = 0\f$. The
  perturbed vertices lie within \f$o(1)\f$ of \f$x\f$ on a circle of
  radius bounded away from zero (@ref sas-step-2 "step 2"), so the perturbed arc
  between consecutive ones is either the short arc, of length
  \f$O(\varepsilon)\f$ and \f$\int\kappa_g \to 0\f$, or the rest of the circle;
  the wrap is decided against the exact reference ray, so the rest of the circle
  is a chain arc only in the single-point window, where @ref sas-step-4 "step 4"
  assigns it the \f$2\pi\f$.
- (c) When the circles cross pairwise transversally at \f$x\f$,
  \f$m \le k - 1\f$, and \f$m \le k - 2\f$ unless the chain closes into a loop
  of its own.

  *Proof.* Choose a neighborhood \f$N\f$ of \f$x\f$ in which the exact circles
  of \f$s\f$ through \f$x\f$ meet only at \f$x\f$; for \f$\varepsilon\f$ small
  every pair of \f$\Sigma_\varepsilon\f$ crosses exactly once in \f$N\f$, no three at one point,
  and the complement of the caps in \f$N\f$ is combinatorially the cell of an
  arrangement of \f$k - 1\f$ lines in general position: an intersection of
  \f$k - 1\f$ half-planes. A bounded cell has at most one vertex per edge,
  \f$m \le k - 1\f$, and an unbounded one, whose two ends leave \f$N\f$ as the
  arrival and departure arcs of the exact region, one fewer. Two circles of
  \f$s\f$ tangent at \f$x\f$ may cross twice near it, and then the chain can be
  longer; nothing reads \f$m\f$, so no bound is claimed.
- (d) In floating point the chain's turning sum deviates from the exact one by
  at most \f$m\f$ `kSurfaceAngleEps`.

  *Proof.* Each computed corner is within `kSurfaceAngleEps` of the exact one
  (@ref analysis-fp "Floating point"), and the sum of \f$m\f$ of them inherits
  \f$m\f$ times that.

At a pinch the tangent pair carries no circle and no root; on a third sphere
its two discs touch from opposite sides and stay two components
(@ref sas-step-2 "step 2"); the pair itself is @ref theorem-3 "Theorem 3".

#### Discontinuity at exact coincidences {#sas-discontinuity}

The output is the arrangement of the
perturbed kept spheres: preparation removes the zero-area spheres and the
zero-length circles, and the perturbation decides everything else. At a point
of \f$k \ge 4\f$ spheres it picks one resolution of the degeneracy (which
triples form cells and which of the coincident probes are accessible), and
rounding of the input under a rigid motion can pick another; areas differ
only at rounding level, but the probe set at such a point is not a
continuous function of the coordinates. The area is (@ref sas-merge "Probe merge").

#### The hybrid sweep {#sas-hybrid}

The kernel sweeps the SoS-perturbed arrangement of the kept spheres
restricted to the unperturbed overlap graph `g`. Objects of measure zero at
\f$\varepsilon = 0\f$ (contained balls, shared-circle middles, tangent-pair
circles and their cut points) are deleted before the sweep, whose vertices
are the accessible cut points of `g`-faces (@ref theorem-1 "Theorem 1");
two caps are joined iff a vertex links them or their open discs meet (a
touch is no join). This is correct iff every **pinch point**, a point where two kept spheres
touch, counts loops and
components one way: vertex present means joined (cusp pair, net area 0), absent
means not joined (two loops, two components). The alternative is to displace
the spheres numerically until every feature is separated and report the
displaced arrangement @cite desc:surface:halperin-shelton-1998; the limit is
taken here instead.

\anchor theorem-3
**Theorem 3 (the hybrid sweep is a limit).** In plain words: near a point
where two spheres touch, the sweep sees what it would see if every ball
shrank by one common power, just enough to part the touching pairs there;
the shrink leaves the power diagram alone, so nothing else near the point
changes side. Precisely: fix a pinch point \f$x_0\f$, let \f$K\f$ be the kept
spheres through it, \f$e_i\f$ the weight perturbations of @ref sas-step-1 "step
1", and for a tangent pair \f$(p, q)\f$ let \f$\gamma_{pq} = (\rho_q e_p +
\rho_p e_q)/(\rho_p + \rho_q)\f$, its **lens parameter**. The pair \f$(p^*,
q^*)\f$ at \f$x_0\f$ with the largest \f$\gamma\f$ is unique (the \f$\gamma\f$
are linear forms in \f$e\f$ with disjoint supports). Let \f$\gamma =
\gamma_{p^*q^*} + \delta\f$ with \f$\varepsilon^2 \ll \delta \ll
\varepsilon\eta^n\f$, \f$n\f$ the number of kept spheres, and let \f$\Sigma\f$
be the configuration with weights \f$\rho_i^2(\varepsilon) - \gamma\f$, a
genuine arrangement for all but finitely many \f$\delta\f$. Then in a ball
\f$N\f$ around \f$x_0\f$ that meets no sphere outside \f$K\f$, for every index
order and mask, the sweep's accepted roots, their ring order and the joins
between caps through \f$x_0\f$ are those of \f$\Sigma\f$; and on every solved
sphere the reported area is the area of the exact arrangement.

*Proof.* Powers are compared only by differences, so \f$\Sigma\f$ has the
\f$RT\f$, cells, orthocenters \f$o_T\f$ and fans of the run, and
@ref theorem-1 "Theorem 1" and @ref lemma-2 "Lemma 2" hold for it. Its ball
\f$i\f$ is \f$\{\pi_i < -\gamma\}\f$, so its cut points lie on the same
radical lines at power \f$-\gamma\f$ where the run's lie at 0. If a ball
contains \f$x_0\f$ in its interior, nothing near \f$x_0\f$ is accessible in
either configuration and that ball's cap meets every cap through \f$x_0\f$;
assume none does. Spheres outside \f$K\f$ have power bounded below in
\f$N\f$ and never reject there. Two spheres of \f$K\f$ with the same normal
at \f$x_0\f$ are nested and were dropped, so the tangent pairs at \f$x_0\f$
are disjoint pairs \f$(p, q)\f$ with opposite normals, the **partners**, and
every other pair of \f$K\f$ crosses at \f$x_0\f$ and is a `g` edge. Since
\f$x_0 = (\rho_q c_p + \rho_p c_q)/(\rho_p + \rho_q)\f$, expanding gives
exactly
\f$\Phi_{pq} := (\rho_q\pi_p + \rho_p\pi_q)/(\rho_p + \rho_q) = |x - x_0|^2 - \gamma_{pq}\f$,
so in \f$\Sigma\f$ no point has both \f$\pi_p\f$ and \f$\pi_q\f$ below
\f$-\gamma\f$: every pair at \f$x_0\f$ is apart.

1. *Orthocenters.* Let \f$T \subset K\f$ be a cell with power
   \f$\lambda_T\f$ at \f$o_T = x_0 + O(\varepsilon)\f$. For \f$m \notin T\f$,
   \f$\pi_m(o_T) - \lambda_T > 0\f$, and its first-order part has coefficient
   \f$-1\f$ on \f$e_m\f$, so it is at least \f$c\,\varepsilon\eta^n\f$.
   Trivially
   \f$\lambda_T = \Phi_{p^*q^*}(o_T) - [\rho_{q^*}(\pi_{p^*} - \lambda_T) + \rho_{p^*}(\pi_{q^*} - \lambda_T)](o_T)/(\rho_{p^*} + \rho_{q^*})\f$.
   If \f$T \supseteq \{p^*, q^*\}\f$ the bracket vanishes and
   \f$\lambda_T = |o_T - x_0|^2 - \gamma_{p^*q^*} \in (-\gamma, 0)\f$;
   otherwise the bracket is at least \f$c\,\varepsilon\eta^n \gg \delta\f$ and
   \f$\lambda_T < -\gamma - c'\varepsilon\eta^n\f$. The same identity for a
   second pair in \f$T\f$ would put \f$\lambda_T\f$ within \f$O(\varepsilon^2)\f$
   of both \f$-\gamma_{pq}\f$ and a value at or below \f$-\gamma_{p^*q^*}\f$,
   which differ at first order. So the only partners inside one cell are
   \f$p^*, q^*\f$, and the only apex in \f$K\f$ the run leaves untested is
   \f$q^*\f$ for a face containing \f$p^*\f$, or the reverse.
2. *Vertices.* A face containing a pair is not a `g` face and is skipped by
   the run; in \f$\Sigma\f$ its radical line has
   \f$\pi = \Phi_{pq} \ge -\gamma_{pq} > -\gamma\f$ and no root. Let
   \f$F \subset K\f$ be a `g` face, \f$P\f$ the common power of its spheres
   along \f$\ell_F\f$, and \f$l\f$ an apex, whose constraint is tight at
   \f$o_{F \cup l}\f$. (a) If \f$x_0\f$ is a simple root, \f$P\f$ is monotone
   in \f$N\f$, so the root at level \f$v \in \{0, -\gamma\}\f$ passes \f$l\f$
   according to the sign of \f$v - \lambda_{F \cup l}\f$. Both levels lie on
   the same side of every \f$\lambda\f$ of 1 except for the apex
   \f$q^*\f$ with \f$p^* \in F\f$, which the run does not test and which
   \f$\Sigma\f$ passes: with \f$\pi_{p^*} = P = -\gamma\f$ the identity gives
   \f$\pi_{q^*} - P = (1 + \rho_{q^*}/\rho_{p^*})\,\delta + O(\varepsilon^2) > 0\f$.
   (b) If \f$x_0\f$ is a double root, the centers of \f$F\f$ are coplanar
   with \f$x_0\f$, and a partner of a member of \f$F\f$ has its center on
   that plane, so it is never an apex (the cell would be flat): every apex
   in \f$K\f$ is tested. On \f$\ell_F\f$, \f$P = |u|^2 s^2 + \mu_F\f$ with the
   foot within \f$O(\varepsilon)\f$ of \f$x_0\f$. An apex in \f$K\f$ gives
   \f$\mu_F \le \lambda_{F \cup l}\f$; if both apexes are outside \f$K\f$,
   the foot lies on the dual edge, and the bracket of 1 at the foot is
   again at least \f$c\,\varepsilon\eta^n\f$. In both cases
   \f$\mu_F < -\gamma - c'\varepsilon\eta^n\f$, so both levels cut, at
   \f$|s| \gg \varepsilon\f$, beyond every \f$o_T\f$ of \f$K\f$, and a root is
   accepted at either level iff the dual edge leaves \f$N\f$ on its side.
   Faces with a sphere outside \f$K\f$ have no root in \f$N\f$. So the run's
   accepted roots in \f$N\f$ are the vertices of \f$\Sigma\f$ there.
3. *Rings.* A ring lists accepted roots in fan order, exit before entry, the
   fan of \f$\Sigma\f$; by @ref lemma-2 "Lemma 2" for \f$\Sigma\f$ that is
   the cyclic order of its vertices. A fan face cuts at level 0 iff it cuts
   at \f$-\gamma\f$ (2), a partner neither cuts nor contains, and in a
   single-point window the antipode lies away from \f$x_0\f$, where the
   decision is strict. So full-circle decisions, arcs and successors agree,
   and so does \f$L\f$.
4. *Components.* The run joins two caps iff they share an accepted root or
   their exact open discs meet; `discs_intersect` reads unperturbed signs
   and its tie falls to `side`, never tied after preparation. Discs that
   meet exactly meet in \f$\Sigma\f$, and a shared root is a crossing in
   \f$\Sigma\f$. Conversely let \f$D_j\f$ and \f$D_l\f$ meet in \f$\Sigma\f$
   but not exactly: their exact closures touch at one point \f$w \in N\f$
   with tangent circles. A cap crossing or containing \f$w\f$ meets both
   exactly. Otherwise the caps through \f$w\f$ form two nested chains, one
   on each side of the common tangent, each joined exactly; let \f$A\f$ and
   \f$B\f$ be their unions in \f$\Sigma\f$. Where the boundary of
   \f$A \cup B\f$ passes from \f$\partial A\f$ to \f$\partial B\f$ there is a
   point outside every cap: a vertex of \f$\Sigma\f$ on two non-partner
   caps (partners are apart), hence by 2 an accepted root that links
   the chains. So \f$C\f$ agrees; the partners' own discs are the case
   \f$A \cap B = \emptyset\f$.
5. *Area.* Outside the \f$O(\sqrt\varepsilon)\f$ neighborhoods of the
   degenerate points no decision is tied, and at a degenerate point without
   a tangent pair the run equals the \f$\varepsilon\f$-configuration, which
   serves as its \f$\Sigma\f$. On a sphere below `n_enum` every face
   through it is handled and its caps are complete. Let \f$R'_s\f$ be the
   region on it bounded by the comparison circles of each such point inside
   its own ball, joined by \f$C^1\f$-small interpolations across the annuli
   between, where every crossing is transversal. It is a genuine region with
   the run's vertices, arcs, \f$L\f$ and \f$C\f$; its corners and
   \f$\int\kappa_g\f$ converge to the run's exactly evaluated terms
   (@ref lemma-4 "Lemma 4", whose hypotheses \f$\Sigma\f$ satisfies by
   1-3), so
   Gauss-Bonnet for \f$R'_s\f$ converges to the run's sum, while the area of
   \f$R'_s\f$ converges to the exact area, as regions do.

\anchor remark-4
**Remark 4 (the shift is local, the tie is load-bearing).** The shift must be local to \f$x_0\f$: shifting by the largest lens parameter
over all pinch points would, at a pinch with a smaller one or at a
\f$k\f$-fold point decided by a higher index, outweigh the perturbation and
make rejected points accessible. The exact tie in `discs_intersect` is
load-bearing: with the coplanar centers \f$p = (-1, 0, 0)\f$,
\f$q = (1, 0, 0)\f$, \f$c, c' = (0.6, \pm 0.8, 0)\f$ and all radii 1, every
sphere passes through the origin, where \f$p\f$ and \f$q\f$ touch; on
\f$S_p\f$ the discs of \f$c\f$ and \f$c'\f$ touch there, the triangulation
takes the diagonal \f$pq\f$ (for the index order \f$p, q, c, c'\f$; with
\f$c\f$ first it takes \f$cc'\f$), and the perturbed circles would cross. Joining
the two discs would be off by \f$4\pi R^2\f$; the sweep keeps them apart
because `discs_intersect` reads the unperturbed sign and `side` reports a
touch.

#### Probe merge {#sas-merge}

Two roots are one probe iff they are one exact point. `build_sas` decides
it with `BallExact::coincide`: for roots \f$x\f$ of \f$f\f$ and \f$x'\f$
of \f$f'\f$ the difference
\f$|u|^2 |u'|^2 (x - x') = P_\delta + \sqrt D\, Q_\delta + \sqrt{D'}\,
Q'_\delta\f$ (\f$P_\delta, Q_\delta, Q'_\delta\f$ its own rational parts; the
kernel of `difference`, @ref analysis-fp "Floating point") has every component
zero, each decided as the unperturbed sign of an element of \f$\mathbb{Q}(\sqrt
D, \sqrt{D'})\f$ by repeated squaring, the comparison across quadratic
extensions of the CGAL spherical kernel @cite desc:surface:decastro-2009. A
tangency perturbed to a cut, whose two roots are one point, needs no branch
(\f$D = 0\f$). Candidates are the pairs within \f$\tau_m = 4 \cdot\f$
`kOffsetRelTol` \f$\cdot R_{\max} + 64 \cdot 2^{-53} \cdot\f$ `kSurfaceMaxCoord`
(`probe_merge_tol`), a certified superset of the coincident pairs: two certified
positions of one point differ by two offsets within `kOffsetRelTol` of a length
\f$\le R_{\max}\f$ (the factor 2 also covers the rounded model radii) and two
circle centers computed in double with their final sums, each within \f$8\sqrt 3
\cdot 2^{-53}\f$ `kSurfaceMaxCoord` of the exact one (@ref analysis-fp "Floating
point"). They are found by one pass over the probes bucketed on a lattice of
pitch \f$\tau_m\f$ and sorted by cell (a pair within \f$\tau_m\f$ shares a cell
or sits in adjacent cells, and a constant cell shift preserves the sorted order,
so each of the 13 forward neighbor shifts is one merge join) and joined by
union-find; a candidate already linked is skipped, since coincidence is
transitive. Roots that share no circle, as at a point where tangent pairs have
deleted the linking circles, are tested like any other. The clusters are the
event sites of @cite desc:surface:cazals-loriot-2009, which groups coincident
event points exactly and links them by union-find rather than perturbing the
spheres apart. A cluster's atoms are the union of its members' (a sphere through
the point none of whose roots is accepted is not listed: the perturbation holds
it inside by \f$\varepsilon\f$, and only the dot owners read the list), its
position that of the first member, and its tangents come from the arcs with one
end in the cluster plus its full circles (\f$\Delta\varphi \ge \pi\f$, the
grazing vertex gets \f$\pm t\f$); an internal arc, a diagonal of the
\f$k\f$-fold point with \f$\Delta\varphi = 0\f$ exactly, contributes nothing.
The merge has no error of its own: a cluster is one point, carried at one of its
certified positions; the SAS arcs, caps and areas are computed before the merge
and unchanged by it.

Distinct roots may be arbitrarily close, and a tolerance would not help:
an exact \f$k\f$-fold point of the input rounds, under a rigid motion, into
\f$k - 2\f$ distinct roots some \f$10^{-15}\f$ apart, and a tolerance that
snapped them back "simply activates rounding to zero"
@cite desc:surface:kettner-2008, with the same fractured boundary one scale
up. Such roots stay distinct probes, each with its three atoms; their
faces, cut by each other's near-hemisphere caps, partition the face of the
exact point with the same total area to rounding. The pair stage places
them by `difference`, whose direction is certified however close the roots
are (@ref analysis-fp "Floating point"), and the contact-plane and
triangle filters apply to single-root probes only (`members`); a cluster of
several roots is solved unfiltered, and at an exact tangency its antipodal
hemispheres end it empty (@ref corollary-4 "Corollary 4",
@ref remark-5 "Remark 5").

## Solvent-excluded surface (SES) {#ses}

`build_ses` (`ses.cpp`) derives the three patch families from the SAS.

### Patch areas {#ses-patches}

- **Convex area** of atom \f$i\f$: \f$A_{\text{sas}}(i) \cdot (r_i/R_i)^2\f$
  (radial scaling maps the SAS patch onto the vdW sphere, the SAS-to-contact
  scaling of @cite desc:surface:totrov-abagyan-1996).
- **Saddle** of an arc on the circle \f$(i, j)\f$ with span \f$\Delta\varphi\f$.
  The generating arc on the probe sphere is parametrized by \f$\beta\f$, the
  angle from the inward radial direction (toward the axis); \f$\beta < 0\f$
  leans toward atom \f$i\f$. The valid range is \f$[-\theta_i, \theta_j]\f$ with
  \f$\sin\theta_i = a/R_i\f$, \f$\sin\theta_j = (d - a)/R_j\f$, where \f$a\f$ is
  the circle plane's offset from \f$c_i\f$ and \f$d\f$ the center distance. The
  distance from the axis is \f$\varrho(\beta) = r_l - r_p \cos\beta\f$
  (\f$r_l\f$ the circle radius); for a **spindle torus** (\f$r_l < r_p\f$) it
  vanishes at \f$\beta_0\f$ with \f$\sin\beta_0 = \sqrt{r_p^2 - r_l^2}/r_p\f$,
  and the part \f$|\beta| < \beta_0\f$ lies beyond the axis. Those points are
  inside every other probe ball on the same circle, because for a probe at
  angular offset \f$\Delta\varphi\f$
  \f[
  |p - q(\varphi + \Delta\varphi)|^2 = r_p^2 + 2\varrho r_l (1 - \cos\Delta\varphi),
  \f]
  which is below \f$r_p^2\f$ for all \f$\Delta\varphi \ne 0\f$ exactly when
  \f$\varrho < 0\f$. Nothing else cuts a saddle (@ref corollary-1 "Corollary 1"(b)), so this cut is
  complete. With \f$\beta_0 = 0\f$ for \f$r_l \ge r_p\f$, the two ranges
  \f[
  [-\theta_i,\ \max(-\theta_i, \min(\theta_j, -\beta_0))]
  \quad\text{and}\quad
  [\min(\theta_j, \max(-\theta_i, \beta_0)),\ \theta_j]
  \f]
  cover every case: they tile the arc for an ordinary torus, cut out the
  spindle, and collapse to zero width where a part is absent (which also
  happens when \f$a < 0\f$ or \f$d - a < 0\f$). They are stored per active
  circle, and the area is
  \f[
  A = \Delta\varphi\, r_p \sum_{\text{ranges}}
      \bigl[ r_l (\beta_{\text{hi}} - \beta_{\text{lo}})
             - r_p (\sin\beta_{\text{hi}} - \sin\beta_{\text{lo}}) \bigr].
  \f]
  Endpoints are selected together with their sines (all \f$\beta\f$ lie in
  \f$(-\pi/2, \pi/2)\f$), and the bracketed integral of each range is stored
  once per circle; the arc areas and the sampler's row totals are both
  gathers of it. The range is never empty once contained balls are gone:
  \f$\theta_i + \theta_j\f$ is the angle at the probe in the triangle
  \f$(c_i, q, c_j)\f$.
- **Concave face** of probe \f$q\f$ (only probes whose smallest atom is
  active). Caps on the probe sphere:
  1. one cap for every other probe \f$q'\f$ strictly within \f$2 r_p\f$ (a
     tangent probe cuts nothing), as in the @ref prelim-defs "Definitions";
     every probe pair is measured once and read from both sides with opposite
     axes;
  2. one **departure hemisphere** per accessible arc leaving the probe, axis the
     arc's departure tangent at \f$q\f$. Rolling along that arc, the probe
     sweeps the half of its sphere facing the tangent, so that half is not SES.

  For an ordinary three-atom vertex, rule 2 gives exactly the three side
  planes of the contact triangle (the departure tangent of circle
  \f$(a, b)\f$ is normal to the plane through \f$q\f$, \f$c_a\f$,
  \f$c_b\f$). The same rule handles merged \f$k\f$-fold vertices (one
  hemisphere per surviving arc), a vertex that is the only vertex on a
  circle (two opposite hemispheres, zero face), and the two roots of a
  nearly tangent face (each cut to a sliver by the other's cap). These two
  kinds of caps are all that is needed: whatever SAS
  point cuts a face point, some vertex probe within \f$2 r_p\f$ cuts it too
  (@ref theorem-4 "Theorem 4").

  The face is solved by the same exact kernel as the SAS, on the probe
  sphere: each cap is lifted to a ball on the probe sphere (\f$(t, 0)\f$ for
  the hemisphere of tangent \f$t\f$, \f$(w, |w|^2)\f$ for the probe at
  \f$w\f$ relative to \f$q\f$), caps whose ball does not overlap the probe
  sphere (an exact tangency included) and the second of two identical caps
  are dropped, two complementary caps end the face empty, the roots of every
  cutting pair of live caps that lie outside every other live cap are the
  vertices, and each circle's ring is sorted by the exact half-plane class of
  the SAS rings, then by the numeric angle, with the perturbed `ccw` deciding
  only the pairs whose angles lie within `kRingSortEps` (= `kSurfaceAngleEps`)
  (the SAS itself never sorts: its ring order is the fan order of @ref lemma-2
  "Lemma 2"; the sort is sound by the certificate of
  @ref analysis-fp "Floating point"). Ties are perturbed in the order probe,
  then caps as given. The `ArrangementSolver` then walks the darts and applies
  Gauss-Bonnet as on a SAS sphere.

### Which probes can cut a face {#ses-cuts}

Notation for a vertex \f$x\f$ on atoms \f$a\f$ with contact directions
\f$\hat c_a = (c_a - x)/R_a\f$: the **face cone** is
\f$C_x = \{\sum \lambda_a \hat c_a : \lambda_a \ge 0\}\f$ (for an ordinary
three-atom vertex, the complement of the three departure hemispheres), the
**contact triangle** is \f$T_x = \operatorname{conv}\{c_a\}\f$, and
\f$t_x(d)\f$ is the distance from \f$x\f$ along the ray \f$d \in C_x\f$ to
\f$T_x\f$. The face is cut by a probe \f$y\f$ where it enters the open ball
\f$B^\circ(y, r_p)\f$. Merged probes take the unfiltered path (a cluster at
a coincident root has no face, @ref corollary-4 "Corollary 4"); the filters
are for single-root three-atom probes.

\anchor lemma-5
**Lemma 5 (contact hull).** In plain words: points that lie between an SAS point
and the atoms it touches are closer to that SAS point than to any other point
outside \f$U\f$. Precisely: let \f$s\f$ be an SAS point touching atoms
\f$A(s)\f$, \f$v\f$ any point not inside \f$U\f$ (every SAS point qualifies),
\f$q\f$ any point of the convex hull of \f$\{c_a : a \in A(s)\}\f$, and \f$p =
(1 - \omega) s + \omega q\f$ with \f$0 \le \omega \le 1\f$. Then \f$|p - v| \ge
|p - s|\f$, strictly for \f$\omega < 1\f$ and \f$v \ne s\f$. This is the
sufficiency half of Theorems 4.1, 4.2 and Corollary 4.1 of
@cite desc:surface:quan-stamm-2016 (the segment, triangle or polyhedron under an
SAS point has it as a closest SAS point), stated once over the convex hull so
that it covers \f$k\f$-fold points.

*Proof.* \f$v\f$ is outside the SAS ball of \f$a\f$ and \f$s\f$ is on it,
so \f$|v - c_a| \ge |s - c_a|\f$, which expands to
\f$(c_a - s) \cdot (v - s) \le |v - s|^2/2\f$. Averaging over the weights of
\f$q\f$ gives the same bound for \f$(q - s) \cdot (v - s)\f$, and then
\f$|p - v|^2 - |p - s|^2 = |v - s|^2 - 2\omega (q - s) \cdot (v - s) \ge (1 - \omega)|v - s|^2\f$.

\anchor corollary-1
**Corollary 1 (consequences of the contact hull).** In plain words: a probe's
face is cut only beyond its contact plane, a saddle only in its spindle, and a
probe whose ball misses its own contact triangle is cut by nothing. Precisely:
with \f$n\f$ the unit normal of the contact plane of vertex \f$x\f$ pointing
from \f$x\f$ toward it, \f$h\f$ the distance to the plane, and \f$x\f$ **low**
when \f$\operatorname{dist}(x, T_x) < r_p\f$ and **high** otherwise (the split
of (c)):

- (a) A face point \f$x + r_p d\f$ with \f$t_x(d) \ge r_p\f$ is cut by nothing,
  so the cut part of the face lies inside the **beyond-plane cap**
  \f$D_x = \{d : d \cdot n > \cos\beta\}\f$, \f$\cos\beta = h/r_p\f$, a cap
  whenever \f$h < r_p\f$, which holds on every low vertex; the bound is
  attained.

  *Proof.* Every \f$d \in C_x\f$ meets \f$T_x\f$: radial rescaling by \f$R_a\f$
  leaves the cone alone, so \f$C_x = \operatorname{cone}\{c_a - x\}\f$, exactly
  the cone over \f$\operatorname{conv}\{c_a\}\f$ from the apex \f$x\f$. Apply
  @ref lemma-5 "Lemma 5" with \f$s = x\f$, \f$q\f$ the point where the ray
  \f$d\f$ meets \f$T_x\f$ and \f$\omega = r_p / t_x(d)\f$. Since
  \f$t_x(d)\,(d \cdot n) = h\f$, the uncut directions \f$t_x(d) \ge r_p\f$ are
  exactly those outside \f$D_x\f$, and
  \f$h \le \operatorname{dist}(x, T_x) < r_p\f$ on a low vertex. \f$D_x\f$ is
  exactly the cap cut by the mirror triple point \f$x + 2hn\f$, the second point
  where the three SAS spheres meet (a vertex only if it is accessible); that
  pair cutting each other along the contact plane is the coincident-face case of
  @cite desc:surface:sanner-1996.
- (b) A saddle point with \f$\varrho(\beta) \ge 0\f$ is cut by nothing, so the
  spindle cut of the saddle ranges (@ref ses-patches "Patch areas") is the whole
  cut.

  *Proof.* The ray at angle \f$\beta\f$ from the inward radial direction meets
  the segment \f$[c_i, c_j]\f$ on the torus axis at distance
  \f$r_l / \cos\beta \ge r_p\f$, so @ref lemma-5 "Lemma 5" applies with that
  point as \f$q\f$. This is the toroidal case of Theorem 5.2 of
  @cite desc:surface:quan-stamm-2016 and the radial singularity of
  @cite desc:surface:sanner-1996.
- (c) \f$\min_{d \in C_x} t_x(d) = \operatorname{dist}(x, T_x)\f$, the plane
  distance \f$h\f$ when the foot of the perpendicular lies inside \f$T_x\f$ and
  the distance to the nearest edge otherwise. A vertex with
  \f$\operatorname{dist}(x, T_x) \ge r_p\f$, whose own ball does not reach its
  contact triangle, is high: nothing cuts its face, which is the spherical
  triangle \f$\{d : d \cdot t_m \le 0\}\f$ bounded by the three departure great
  circles, with area \f$r_p^2\f$ times the spherical excess,
  \f[
  A = r_p^2 \Bigl(2\pi - \sum_m \angle(t_m, t_{m+1})\Bigr),
  \qquad
  \angle(t_m, t_{m+1}) = \operatorname{atan2}(|t_m \times t_{m+1}|,\ t_m \cdot t_{m+1}).
  \f]
  Every other vertex is low (\f$k\f$-fold probes count as low). A high face
  whose three tangents are linearly dependent, or dependent within the tangent
  certificate (@ref lemma-10 "Lemma 10"), has no usable spherical triangle and
  goes to the face solver on its hemispheres alone; a high face is uncut, so the
  solver on the hemispheres alone returns the exact face.

  *Proof.* \f$t_x\f$ is the distance along the ray to \f$T_x\f$, and the rays of
  \f$C_x\f$ sweep all of \f$T_x\f$, so its minimum is the distance from \f$x\f$
  to the triangle; then (a) cuts nothing when it is at least \f$r_p\f$, and the
  uncut face is bounded by the departure great circles. That a probe whose ball
  misses its own contact triangle has an uncut face is stated in
  @cite desc:surface:sanner-1996; this is its proof.

\anchor lemma-6
**Lemma 6 (the first SAS point an inflating ball touches is a vertex).** Let
\f$x\f$ be a vertex whose contact directions span space (they fail to only when
\f$x\f$ lies in the plane of its atom centers; then every circle through \f$x\f$
is perpendicular to that plane and the reflection in it maps accessible arcs to
accessible arcs, so an arc leaving \f$x\f$ is two-sided and the face has no area
by @ref lemma-12 "Lemma 12"; @ref corollary-4 "Corollary 4" is the three-atom
case) and \f$d\f$ a unit direction strictly inside \f$C_x\f$ (\f$d = \sum
\lambda_a \hat c_a\f$ with every \f$\lambda_a > 0\f$, scaled to \f$|d| = 1\f$).
Inflate the balls \f$B_t = B(x + t d, t)\f$, \f$t > 0\f$, which are nested and
all pass through \f$x\f$, let \f$t_1\f$ be the largest \f$t\f$ for which the
open ball \f$B_t^\circ\f$ contains no SAS point, and suppose some SAS point lies
in \f$B^\circ(x + r_p d, r_p)\f$, so that \f$t_1\f$ is finite. Then \f$0 < t_1 <
r_p\f$, \f$B(x + t_1 d, t_1) \subset U\f$, and a second SAS point \f$s_1 \ne
x\f$ lies on its boundary; for all but a null set of directions \f$d\f$,
\f$s_1\f$ is a vertex, and the face point \f$x + r_p d\f$ is cut by it.

*Proof.*

1. *Small balls stay inside the union.* A point of \f$B_t^\circ\f$ is
   \f$x + t(d + e)\f$ with \f$|e| < 1\f$. It is inside the SAS ball of atom
   \f$a\f$ iff \f$\hat c_a \cdot (d + e) > t|d + e|^2/2R_a\f$. With
   \f$d = \sum \lambda_a \hat c_a\f$,
   \f$\sum \lambda_a \hat c_a \cdot (d + e) = 1 + d \cdot e > |d + e|^2/2\f$, so
   the largest \f$\hat c_a \cdot (d + e)\f$ exceeds
   \f$|d + e|^2/(2\sum\lambda_a)\f$, which is at least \f$t|d + e|^2/(2R_a)\f$
   for the maximizing \f$a\f$ once \f$t \le \min_b R_b / \sum_b \lambda_b\f$.

2. *SAS points near the vertex are never inside.* Near \f$x\f$ the SAS consists
   of the sphere patches leaving \f$x\f$; a tangent direction \f$w\f$ of the
   patch of atom \f$a\f$ satisfies \f$w \perp \hat c_a\f$ and
   \f$w \cdot \hat c_b \le 0\f$ for every other touched atom \f$b\f$. Hence
   \f$d \cdot w = \sum_{b \ne a} \lambda_b\, w \cdot \hat c_b \le 0\f$, with
   equality only for \f$w = 0\f$ because the contact directions span space. So
   \f$d \cdot (s - x) < 0\f$ for SAS points \f$s\f$ close to \f$x\f$, and
   \f$|s - (x + td)|^2 - t^2 = |s - x|^2 - 2t\, d \cdot (s - x) > 0\f$ for every
   \f$t\f$: no such \f$s\f$ is in any \f$B_t^\circ\f$.

3. *The first touch.* Hence \f$t_1 > 0\f$, and finite by hypothesis. The union
   of the open balls \f$B_t^\circ\f$, \f$t < t_1\f$, is the open ball
   \f$B^\circ(p_1, t_1)\f$, \f$p_1 = x + t_1 d\f$; it is connected, avoids the
   SAS and contains points inside \f$U\f$, so it lies inside \f$U\f$, and the
   closed ball lies in \f$U\f$. Slightly larger balls contain SAS points, which
   accumulate on the boundary of \f$B(p_1, t_1)\f$ at an SAS point \f$s_1\f$,
   and \f$s_1 \ne x\f$ by 2. So \f$x\f$ and \f$s_1\f$ are both nearest SAS
   points of \f$p_1\f$, at distance \f$t_1\f$, and \f$p_1\f$ lies on the
   bisector plane of \f$x\f$ and \f$s_1\f$.

4. *Not inside a sphere patch.* If \f$s_1\f$ were, on the patch of atom \f$a\f$,
   the ball \f$B(p_1, t_1) \subset U\f$ would touch the sphere \f$S(c_a, R_a)\f$
   from inside at \f$s_1\f$, so \f$p_1\f$ lies on the radius from \f$c_a\f$ to
   \f$s_1\f$ at distance \f$R_a - t_1\f$ from \f$c_a\f$. Then
   \f$|x - c_a| \le |x - p_1| + |p_1 - c_a| = R_a\f$, while \f$x\f$, an SAS
   point, has \f$|x - c_a| \ge R_a\f$; equality puts \f$p_1\f$ on the segment
   from \f$x\f$ to \f$c_a\f$, so the radius through \f$p_1\f$ is the radius
   through \f$x\f$ and \f$s_1 = x\f$. (If \f$p_1 = c_a\f$, then
   \f$d = \hat c_a\f$, a cone edge, excluded.)

5. *Not inside an arc.* If \f$s_1\f$ were, on the circle of atoms \f$a, b\f$,
   then near \f$s_1\f$ the region \f$U\f$ is the union of the two SAS balls, and
   a ball inside \f$U\f$ touching their common boundary at \f$s_1\f$ must have
   its center in the wedge spanned at \f$s_1\f$ by the directions to \f$c_a\f$
   and \f$c_b\f$ (a direction pointing away from both centers leaves both balls,
   hence \f$U\f$, while staying inside the ball). So \f$p_1\f$ lies in the plane
   through \f$s_1\f$ and the axis \f$c_a c_b\f$, within the angle at \f$s_1\f$
   of the triangle \f$(s_1, c_a, c_b)\f$. By @ref lemma-5 "Lemma 5" with
   \f$s = s_1\f$, \f$v = x\f$, no point strictly inside that triangle is
   equidistant from \f$x\f$ and \f$s_1\f$, so \f$p_1\f$ is on or beyond the
   segment \f$[c_a, c_b]\f$: on the axis or across it. Across the axis,
   \f$s_1\f$ is the *farthest* point of the circle from \f$p_1\f$: writing
   \f$p_1 = m + z \hat u + \xi \hat r\f$ with \f$m\f$ the circle's center,
   \f$\hat u\f$ its unit axis, \f$r_l\f$ its radius and \f$\hat r\f$ the radial
   direction of \f$s_1\f$, \f$|p_1 - y(\theta)|^2 = z^2 + \xi^2 + r_l^2 - 2\xi
   r_l \cos\theta\f$ (\f$y(\theta)\f$ the circle point at angle \f$\theta\f$
   from \f$s_1\f$), and across the axis \f$\xi < 0\f$, so the distance is
   largest at \f$\theta = 0\f$. The end vertices of the arc through \f$s_1\f$
   are therefore strictly closer than \f$t_1\f$, contradicting that no SAS point
   is closer than \f$t_1\f$. On the axis the whole circle is equidistant and an
   end vertex is also a nearest point; take it as \f$s_1\f$. (A vertex-free
   circle would need the ray of \f$d\f$ to meet a fixed line: a null set of
   directions.)

6. *The cut.* \f$p_1\f$ is on the bisector plane of \f$x\f$ and \f$s_1\f$, and
   \f$x + r_p d\f$ lies beyond \f$p_1\f$ on the ray from \f$x\f$ when
   \f$t_1 < r_p\f$, so it is closer to \f$s_1\f$ than to \f$x\f$:
   \f$|x + r_p d - s_1| < r_p\f$.

\anchor lemma-7
**Lemma 7 (caps from one circle are nested).** In plain words: of the probes
on one circle through \f$x\f$, the one farthest along the circle cuts
everything the others cut, and the probe rolling along the arc cuts nothing
beyond the departure hemisphere and the arc's end vertex. Let \f$x\f$ be a
vertex on the circle of atoms \f$a, b\f$ with an accessible arc leaving it,
and measure the angle \f$\psi\f$ about the axis \f$c_a c_b\f$ from the
plane through the axis and \f$x\f$, positive toward the arc. For a probe
\f$y\f$ on the circle at angle \f$\Delta\varphi \in (0, 2\pi)\f$ from
\f$x\f$ in that direction, the cap of \f$y\f$ on \f$S(x, r_p)\f$ meets the
face cone \f$C_x\f$ in \f$\{\pi < \psi < \pi + \Delta\varphi/2\}\f$. Hence
the caps of all probes on one circle are nested in the order of
\f$\Delta\varphi\f$, and the union of the caps of the probes rolling along the
arc is the departure hemisphere together with the cap of the arc's end
vertex.

*Proof.* Every point of the axis is equidistant from \f$x\f$ and \f$y\f$,
so the bisector plane of \f$x\f$ and \f$y\f$ contains the axis and makes the
angle \f$\Delta\varphi/2\f$ with the plane of \f$x\f$. The cap of \f$y\f$ is
the side of that plane containing \f$y\f$, which off the axis is
\f$\{\Delta\varphi/2 < \psi < \Delta\varphi/2 + \pi\}\f$; the axis itself
is a null set of directions. The departure tangent at \f$x\f$ is normal to
the plane of \f$x\f$, so the departure hemisphere is \f$\{0 < \psi < \pi\}\f$,
the limit \f$\Delta\varphi \to 0\f$, and \f$C_x\f$, which excludes it, lies
in \f$\{\pi \le \psi \le 2\pi\}\f$. Intersecting gives the claim, and the
union over \f$\Delta\varphi \in (0, \Delta\varphi_1]\f$ is
\f$\{0 < \psi < \pi + \Delta\varphi_1/2\}\f$.

\anchor theorem-4
**Theorem 4 (vertex caps suffice).** The concave face of a vertex \f$x\f$ is
\f$C_x\f$ minus the caps of the vertex probes within \f$2 r_p\f$, and
nothing else, up to a null set of directions, which affects neither area nor
dots. This is Theorem 5.1 of @cite desc:surface:quan-stamm-2016,
\f$P_- = P_0 \setminus \bigcup_{x \in K} B_{r_p}(x)\f$; the paper proves it
in the plane and states that space adds no essential difficulty, and
@ref lemma-6 "Lemma 6" is the proof in \f$\mathbb R^3\f$.

*Proof.* If any SAS point at all cuts a face point \f$x + r_p d\f$, that
point lies in \f$B_{r_p}^\circ\f$, so \f$t_1 < r_p\f$ and
@ref lemma-6 "Lemma 6" supplies a vertex within \f$r_p\f$ of the face
point. For the probes rolling along the arcs that leave \f$x\f$ this is
also @ref lemma-7 "Lemma 7".

\anchor lemma-8
**Lemma 8 (cutting is symmetric).** Let \f$x\f$, \f$y\f$ be vertices with
\f$|x - y| < 2 r_p\f$, and suppose the cap of \f$y\f$ removes a region of
positive area from the face of \f$x\f$ computed without \f$y\f$ (the cone
minus the caps of all other vertices within \f$2 r_p\f$). Then the ball of
\f$x\f$ reaches the face region of \f$y\f$ beyond \f$y\f$'s contact plane:
there is \f$\nu_y \in C_y\f$ with \f$|y + r_p \nu_y - x| < r_p\f$, and
\f$t_y(\nu_y) < r_p\f$.

*Proof.* Pick a removed face point \f$z = x + r_p d\f$ with \f$d\f$ strictly
inside the cone and outside the null set of @ref lemma-6 "Lemma 6". It is cut by \f$y\f$
and by no other vertex. Since \f$y \in B^\circ(z, r_p)\f$, the first-touch
radius of @ref lemma-6 "Lemma 6" satisfies \f$t_1 < r_p\f$, and its vertex \f$s_1\f$ is
within \f$r_p\f$ of \f$z\f$; the only such vertex is \f$y\f$. Call
\f$s_1\f$ the **first-touch cutter** of \f$z\f$; the rest of the proof uses
only that \f$y = s_1\f$, never that \f$y\f$ is the unique cutter. So
\f$q = p_1\f$ is equidistant from \f$x\f$ and \f$y\f$, at distance
\f$t_1 < r_p\f$, and \f$B(q, t_1) \subset U\f$. Both \f$x\f$
and \f$y\f$ lie on its boundary, so at each the direction to \f$q\f$ lies
in the face cone (a direction pointing away from every touched center leaves
\f$U\f$ while staying inside the ball):
\f$\nu_x = (q - x)/t_1 \in C_x\f$ and \f$\nu_y = (q - y)/t_1 \in C_y\f$. Then
\f$|y + r_p \nu_y - x| \le |y + r_p \nu_y - q| + |q - x| = (r_p - t_1) + t_1 = r_p\f$,
with equality only if \f$q - x\f$ and \f$q - y\f$ point the same way, which
with equal lengths means \f$x = y\f$. Finally @ref corollary-1 "Corollary 1"(a) at \f$y\f$, with
the SAS point \f$x\f$, gives \f$t_y(\nu_y) < r_p\f$.

\anchor corollary-2
**Corollary 2 (both ends low, symmetric drop).** If \f$y\f$ is the first-touch
cutter (@ref lemma-8 "Lemma 8") of a point of \f$x\f$'s face, then \f$y\f$ is
low and the cap of \f$x\f$ on \f$y\f$'s sphere meets both \f$y\f$'s beyond-plane
cap and \f$y\f$'s spherical triangle. Hence the two tests
(@ref lemma-9 "Lemma 9", @ref lemma-10 "Lemma 10") are necessary conditions for
cutting in either direction, and a failure on either side drops the pair *for
both faces*.

*Proof.* \f$x\f$ reaches \f$y\f$'s face beyond its plane by
@ref lemma-8 "Lemma 8", so \f$y\f$ is low (@ref corollary-1 "Corollary 1"(c)).
The x-side test is immediate from @ref corollary-1 "Corollary 1"(a) (the
removed direction lies in
\f$\mathrm{cap}(y) \cap D_x \cap \text{triangle}_x\f$), the y-side test is
@ref lemma-8 "Lemma 8"; by symmetry the same two conditions are necessary
for \f$x\f$ to cut \f$y\f$. The hypothesis of @ref lemma-8 "Lemma 8" that
the face is computed without \f$y\f$ cannot be dropped: a point of
\f$x\f$'s beyond-plane cap can lie inside \f$y\f$'s ball while \f$x\f$'s
ball misses \f$y\f$'s cone entirely; such a point is always inside some
other vertex ball, which is what @ref lemma-6 "Lemma 6" finds.

\anchor lemma-9
**Lemma 9 (cap reach).** The cap
\f$\{d : d \cdot \hat y > \cos\alpha\}\f$ of a probe \f$y\f$,
\f$\hat y = (y - x)/|y - x|\f$, cuts the face of \f$x\f$ only if
\f[
\hat y \cdot n > \cos(\alpha + \beta) = \cos\alpha \cos\beta - \sin\alpha \sin\beta,
\f]
with \f$n\f$, \f$\beta\f$ of the beyond-plane cap \f$D_x\f$. All four
values are stored; no angle is recovered. The side of the contact plane,
hence \f$n\f$, is read from the root: \f$x\f$ is root \f$\sigma\f$ of its
face, \f$x = c_a + y_\perp + \sigma \sqrt D\, u / |u|^2\f$ with
\f$u = (c_b - c_a) \times (c_c - c_a)\f$, so \f$n = -\sigma\, u/|u|\f$
whatever the rounded height; only \f$h = |(x - c_a) \cdot u| / |u|\f$ is
floating, and the test is a necessary condition for every \f$h > 0\f$. The
computed \f$h\f$ is within the position certificate
\f$\delta h\f$ (@ref analysis-fp "Floating point") of the exact one, and a
test that flips on that error drops only a cap whose overlap with
\f$D_x\f$ is shallower than the error in \f$\beta\f$, a region inside
the band of that angular width along the boundary of \f$D_x\f$, of area at
most \f$2\pi r_p^2\, \delta(\cos\beta) = 2\pi r_p\, \delta h\f$, about
\f$2\pi r_p \rho_{\max} 2^{-26}\f$. A side read from the floating height would be noise once \f$h\f$ is below the
position certificate, and a wrong side drops the mirror root's cap, the one
that cuts the face to \f$D_x\f$.

*Proof.* By @ref corollary-1 "Corollary 1"(a) the cut part of the face lies
inside \f$D_x = \{d : d \cdot n > \cos\beta\}\f$, and a cap meets it only
if the angle between \f$\hat y\f$ and \f$n\f$ is below \f$\alpha + \beta\f$.
The step
\f$\angle(\hat y, n) < \alpha + \beta \iff \hat y \cdot n > \cos(\alpha + \beta)\f$ needs
\f$\alpha + \beta \le \pi\f$, which holds throughout: probe-on-probe caps
have \f$\cos\alpha = |x - y|/2r_p \ge 0\f$ and the beyond-plane cap has
\f$\cos\beta = h/r_p \ge 0\f$, so both are at most \f$\pi/2\f$. Under that
bound the test is not merely necessary but exact for
\f$\mathrm{cap} \cap D_x \ne \emptyset\f$.

\anchor lemma-10
**Lemma 10 (triangle reach).** The cap meets the
closed triangle \f$S = \{d : d \cdot t_m \le 0\}\f$ iff \f$\hat y \in S\f$ or the
angular distance from \f$\hat y\f$ to the boundary of \f$S\f$ is below
\f$\alpha\f$, which three tests decide without a square root.

*Proof.* Edge \f$m\f$ is the arc of the great circle
\f$t_m \cdot d = 0\f$ inside the other two half-spaces,
\f$\{d : t_m \cdot d = 0,\ t_k \cdot d \le 0 \text{ for } k \ne m\}\f$,
between the corners \f$p_{m+1}\f$ and \f$p_{m+2}\f$,
\f$p_m = \pm(t_{m+1} \times t_{m+2})/|\cdot|\f$ signed so that
\f$t_m \cdot p_m \le 0\f$. The foot of \f$\hat y\f$ on that great circle is
\f$f = \hat y - (\hat y \cdot t_m) t_m\f$ up to a positive factor, the squared cosine
of the angle from \f$\hat y\f$ to \f$f\f$ is \f$1 - (\hat y \cdot t_m)^2\f$, and
\f$f\f$ lies on the arc iff
\f[
\hat y \cdot t_k - (\hat y \cdot t_m)(t_k \cdot t_m) \le 0
\quad\text{for both } k \ne m,
\f]
a test whose floating error is of the order of one rounding (the bisector
form \f$p \cdot (p + q) = 1 + p \cdot q\f$ is pure rounding for an edge
near \f$\pi\f$). So the cap meets the triangle iff
\f$\hat y \cdot t_m \le 0\f$ for all \f$m\f$ (axis inside \f$S\f$), or
\f$f\f$ on arc \f$m\f$ and \f$1 - (\hat y \cdot t_m)^2 > \cos^2\alpha\f$ for some
\f$m\f$ (edge reached), or \f$\max_m \hat y \cdot p_m > \cos\alpha\f$ (corner
reached), the corner test covering feet off their arcs. The first disjunct
is not redundant: a neighbor probe whose cap lies entirely in the interior
of the triangle, a hole punched in the middle of a face, meets \f$S\f$
without meeting \f$\partial S\f$, and dropping it would delete the hole.

No square root is taken; all comparisons are on cosines or their squares.

The tangents are computed, each within
\f$\eta_t =\f$ `kTangentEps` \f$=\f$ `kSurfaceAngleEps` \f$+ 4 \cdot 2^{-53}\f$
of the exact one (@ref analysis-fp "Floating point"), so the test runs on
a computed triangle \f$\tilde S\f$ with corners \f$\tilde p_m\f$. Let
\f$\Delta = \tilde t_1 \cdot (\tilde t_2 \times \tilde t_3)\f$. Every
direction of the exact triangle \f$S\f$ is within
\f$H = \arcsin\bigl(3\eta_t / (|\Delta| - 8 \cdot 2^{-53})\bigr)\f$ of \f$\tilde
S\f$ (the \f$8 \cdot 2^{-53}\f$ is the rounding of \f$\Delta\f$ itself): the
corners satisfy \f$\tilde t_k \cdot \tilde p_m = 0\f$ for \f$k \ne m\f$ and
\f$\tilde t_m \cdot \tilde p_m = -|\Delta| / |\tilde t_{m+1} \times \tilde
t_{m+2}| \le -|\Delta|\f$, so \f$v = \sum_m \tilde p_m / |\sum_m \tilde p_m|\f$
has \f$\tilde t_k \cdot v \le -|\Delta|/3\f$ for every \f$k\f$; a direction \f$d
\in S\f$ has \f$\tilde t_k \cdot d \le \eta_t\f$, hence \f$w = d + (3\eta_t /
(|\Delta| - 8 \cdot 2^{-53}))\, v\f$ lies in \f$\tilde S\f$ at angle at most
\f$H\f$ from \f$d\f$. A cap that meets \f$S\f$ therefore meets \f$\tilde S\f$
with its radius widened by \f$H\f$, and the filter tests \f$\alpha + H\f$
(\f$\cos(\alpha + H)\f$ from the stored \f$\cos\alpha, \sin\alpha\f$; a cap with
\f$\alpha + H \ge \pi/2\f$ passes, as the squared comparisons need \f$\alpha + H
< \pi/2\f$). The triangle is used only when \f$|\Delta| \ge\f$ `kTriangleDetMin`
\f$= 6\eta_t + 8 \cdot 2^{-53}\f$, which keeps \f$\sin H \le 1/2\f$ and the corner
signs \f$\tilde t_m \cdot \tilde p_m \le -|\Delta|\f$ clear of their rounding;
below it, and for a probe with other than three tangents, the probe passes the
triangle test; a cluster of several roots passes both filters and is solved
unfiltered, which is sound since the filters only ever drop. The condition is
necessary for the cap to cut anything and, for a face with no other caps,
sufficient up to \f$H\f$.

\anchor lemma-11
**Lemma 11 (host overlap).** In plain words: whenever one probe is the
first-touch cutter of a point of another's face, some atom under the first
touches some atom under the second, provided no atom is smaller than about 0.41
probe radii. Precisely: let \f$y\f$ be the first-touch cutter (@ref lemma-8
"Lemma 8") of a face point of \f$x\f$, \f$R_x\f$ and \f$R_y\f$ the smallest SAS
radii among the hosts of \f$x\f$ and \f$y\f$, \f$R_{\max}\f$ the largest SAS
radius of the structure, and call two atoms overlapping when \f$|c_a - c_d| <
R_a + R_d - \tau\f$ (exact non-overlap on \f$\rho\f$, as the preparation decides
it, implies non-overlap in this sense, since \f$|\rho - R| \ll \tau\f$). If no
host of \f$y\f$ overlaps any host of \f$x\f$, then \f$|x - y|^2 > 2 R_x R_y -
4\tau R_{\max}\f$. Since \f$|x - y| < 2 r_p\f$, some host pair overlaps whenever
\f$R_x R_y \ge 2 r_p^2 + 2\tau R_{\max}\f$, which the preparation enforces for
the smallest SAS radius; without the tolerance the bound is \f$|x - y|^2 > 2 R_x
R_y\f$ and the threshold is \f$R \ge \sqrt 2\, r_p\f$, i.e. a vdW radius of
\f$(\sqrt 2 - 1) r_p\f$.

*Proof.*

1. *Notation.* Take the point \f$q\f$ from the proof of @ref lemma-8 "Lemma 8":
   \f$x = q - t_1 \nu_x\f$, \f$y = q - t_1 \nu_y\f$ with \f$\nu_x \in C_x\f$,
   \f$\nu_y \in C_y\f$, \f$t_1 < r_p\f$. Write \f$w = x - y = t_1(\nu_y - \nu_x)\f$,
   \f$\zeta = \nu_x \cdot \nu_y < 1\f$, so
   \f$|w|^2 = 2t_1^2(1 - \zeta)\f$,
   \f$w \cdot \nu_x = -t_1(1 - \zeta)\f$,
   \f$w \cdot \nu_y = t_1(1 - \zeta)\f$. Let
   \f$\nu_x = \sum_a \lambda_a \hat c_a\f$ and \f$\nu_y = \sum_d \mu_d \hat c_d\f$ with
   non-negative weights over the hosts, and abbreviate
   \f$\Lambda = \sum\lambda_a\f$, \f$\Lambda' = \sum\lambda_a/R_a\f$,
   \f$M = \sum\mu_d\f$, \f$M' = \sum\mu_d/R_d\f$, \f$P = t_1\Lambda'\f$,
   \f$Q = t_1 M'\f$.

2. *Both probes are outside the other's balls.* \f$|y - c_a| \ge R_a\f$ with
   \f$y - c_a = -w - R_a \hat c_a\f$ gives
   \f$w \cdot \hat c_a \ge -|w|^2/2R_a\f$; weighting by \f$\lambda_a\f$ and
   summing,
   \f$-t_1(1 - \zeta) \ge -(|w|^2/2)\Lambda' = -t_1^2(1 - \zeta)\Lambda'\f$,
   i.e. \f$P \ge 1\f$. Likewise \f$|x - c_d| \ge R_d\f$ gives \f$Q \ge 1\f$.

3. *A non-overlapping host pair.*
   \f$c_a - c_d = w + R_a \hat c_a - R_d \hat c_d\f$, so
   \f$|c_a - c_d| \ge R_a + R_d - \tau\f$ expands to
   \f$|w|^2 + 2R_a\, w \cdot \hat c_a - 2R_d\, w \cdot \hat c_d \ge 2R_a R_d (1 + \hat c_a \cdot \hat c_d) - 2\tau(R_a + R_d)\f$
   (the \f$\tau^2\f$ term only helps and is dropped).

4. *Average over both cones.* Multiply each pair's inequality by
   \f$\lambda_a \mu_d/(R_a R_d)\f$ and sum. The left side becomes
   \f$|w|^2 \Lambda' M' + 2M'(w \cdot \nu_x) - 2\Lambda'(w \cdot \nu_y) = 2(1 - \zeta)(PQ - P - Q)\f$,
   the right side \f$2(\Lambda M + \zeta) - 2\tau(\Lambda M' + \Lambda' M)\f$.
   Since \f$\Lambda \le R_{\max}\Lambda'\f$ and \f$M \le R_{\max} M'\f$, the
   tolerance term is at most \f$4\tau R_{\max} \Lambda' M'\f$; hence
   \f$(1 - \zeta)(PQ - P - Q) \ge (\Lambda M - 2\tau R_{\max}\Lambda' M') + \zeta\f$.

5. *Conclude.* \f$\Lambda \ge R_x \Lambda'\f$ and \f$M \ge R_y M'\f$ give
   \f$\Lambda M - 2\tau R_{\max}\Lambda' M' \ge (R_x R_y - 2\tau R_{\max})\, PQ/t_1^2\f$,
   and \f$P + Q \ge 2\f$ gives
   \f$(1 - \zeta)(PQ - 2) \ge (1 - \zeta)(PQ - P - Q)\f$. Together,
   \f$PQ\,[(1 - \zeta) - (R_x R_y - 2\tau R_{\max})/t_1^2] \ge 2 - \zeta > 0\f$,
   so \f$R_x R_y - 2\tau R_{\max} < t_1^2(1 - \zeta) = |w|^2/2\f$.

\anchor corollary-3
**Corollary 3 (two neighbor shells suffice).** Enumerating the vertices
that have a host in `active`, its near neighbors or their near neighbors
captures every first-touch cutter of every face on an active atom
(@ref prep-step-5 "preparation step 5"), which by @ref lemma-6 "Lemma 6"
leaves every such face exact. One hop does not suffice.

*Proof.* The hosts of a probe are pairwise near: two hosts from one triple
share a point, and two hosts from different triples of a merged cluster
share the cluster's one exact point (@ref sas-merge "Probe merge"), so
\f$|c_a - c_d| \le R_a + R_d\f$ in either case. An
overlap is a near pair, so by @ref lemma-11 "Lemma 11" a host of a first-touch
cutter of
\f$x\f$ is within two near hops of every host of \f$x\f$. One hop is not
enough: the cutter's hosts need only touch a *neighbor* of the active atom,
and a probe wedged between two nearly opposite atoms and touched by a third
from above can be cut by a shallow probe on the other side whose hosts
overlap neither the third atom nor anything active. The radius threshold is
not vacuous: with vdW radii of \f$0.03\,\AA\f$ at \f$r_p \approx 1\f$
a cutter exists none of whose hosts overlaps any host of the cut face.

\anchor lemma-12
**Lemma 12 (a two-sided arc through a vertex leaves it no face).** In plain
words: if the probe can roll away from a vertex along one circle in both
directions, the positions it rolls through cover the whole probe sphere
except the great circle perpendicular to the rolling direction, so the
vertex's face has no area, whatever else cuts it. Precisely: let \f$x\f$ be
a vertex on the circle \f$(a, b)\f$ of center \f$m\f$ and radius \f$r > 0\f$,
\f$\gamma(s)\f$ its arclength parametrization with \f$\gamma(0) = x\f$ and
unit tangent \f$t = \gamma'(0)\f$, and let \f$\gamma(s)\f$ be an SAS point
for every \f$0 < |s| < \delta\f$. Then for every unit \f$d\f$ with
\f$d \cdot t \ne 0\f$ the face point \f$x + r_p d\f$ lies strictly inside
\f$B(\gamma(s), r_p)\f$ for every \f$s\f$ of the sign of \f$d \cdot t\f$ with
\f$0 < |s| < \min(\delta, s_0)\f$,
\f$s_0 = 2r \arctan\bigl(r_p |d \cdot t| / (r + r_p)\bigr)\f$; the face of
\f$x\f$ lies in the great circle \f$\{d : d \cdot t = 0\}\f$ and has area 0.

*Proof.* With \f$\rho' = (x - m)/r\f$ and \f$\theta = s/r\f$,
\f$\gamma(s) - x = r\sin\theta\, t - r(1 - \cos\theta)\rho'\f$ and
\f$|\gamma(s) - x|^2 = 2r^2(1 - \cos\theta)\f$, so
\f$|x + r_p d - \gamma(s)|^2 < r_p^2\f$ iff
\f$2 r_p [\, r \sin\theta\, (d \cdot t) - r(1 - \cos\theta)(d \cdot \rho')\,]
> 2 r^2 (1 - \cos\theta)\f$; dividing by \f$4 r \sin^2(\theta/2) > 0\f$ gives
\f$r_p \cot(\theta/2)\, (d \cdot t) > r + r_p\, (d \cdot \rho')\f$, implied by
\f$r_p \cot(\theta/2)\, (d \cdot t) > r + r_p\f$ since \f$d \cdot \rho' \le 1\f$,
i.e. by \f$\operatorname{sign}\theta = \operatorname{sign}(d \cdot t)\f$ and
\f$|\tan(\theta/2)| < r_p |d \cdot t| / (r + r_p)\f$. Any such \f$s\f$ below
\f$\delta\f$ is an SAS point whose open probe ball holds \f$x + r_p d\f$.

\anchor lemma-13
**Lemma 13 (coincident roots).** In plain words: the two roots of a face
coincide exactly when the point lies in the plane of the three centers; then
every circle through it is tangent to the plane's normal there, and a circle
of two of the atoms is accessible or covered by the third according to the
sign of one barycentric coordinate. Precisely: for a face \f$(a, b, c)\f$ with
\f$u \ne 0\f$ and \f$\Pi\f$ the plane of the centers:

- (a) \f$D = 0\f$ iff the two roots coincide iff the root \f$x\f$ lies in
  \f$\Pi\f$; then \f$h = 0\f$, the mirror point \f$x + 2hn\f$ is \f$x\f$, and
  the three spheres meet only at \f$x\f$.

  *Proof.* The roots \f$c_a + y_\perp \pm (\sqrt D/|u|^2) u\f$ lie on the line
  through \f$c_a + y_\perp \in \Pi\f$ along \f$u \perp \Pi\f$, which meets
  \f$\Pi\f$ once, and \f$\pi_a(c_a + y_\perp + s u) = s^2 |u|^2 - D/|u|^2\f$.
- (b) Each of the circles \f$(a, b)\f$, \f$(a, c)\f$, \f$(b, c)\f$ has tangent
  \f$\pm n\f$ at \f$x\f$, \f$n\f$ the unit normal of \f$\Pi\f$.

  *Proof.* The tangent is perpendicular to the axis \f$c_j - c_i\f$ and to the
  radius \f$x - m_{ij}\f$, two independent vectors of \f$\Pi\f$.
- (c) With \f$x = w_a c_a + w_b c_b + w_c c_c\f$, \f$\sum w = 1\f$, every point
  \f$y\f$ satisfies \f$w_a \pi_a(y) + w_b \pi_b(y) + w_c \pi_c(y) = |y -
  x|^2\f$. Hence on the circle \f$(i, j)\f$, \f$\pi_k(y) = |y - x|^2 / w_k\f$
  (\f$\{i, j, k\} = \{a, b, c\}\f$; no \f$w\f$ vanishes), so the circle minus
  \f$x\f$ is strictly outside \f$B_k\f$ iff \f$w_k > 0\f$ and strictly inside
  iff \f$w_k < 0\f$.

  *Proof.* \f$\sum w_m \pi_m(y) = |y|^2 - 2 y \cdot x + K\f$ with \f$K =
  |x|^2\f$ because every \f$\pi_m(x) = 0\f$; on the circle \f$\pi_i = \pi_j =
  0\f$, and \f$w_k = 0\f$ would put the whole circle at \f$x\f$.
- (d) \f$w_k > 0\f$ iff \f$x\f$ and \f$c_k\f$ lie on the same side of the line
  \f$c_i c_j\f$. Since the \f$w\f$ sum to 1, at least one is positive: at least
  one circle through \f$x\f$ is accessible near \f$x\f$ on both sides, at most
  two are covered (all accessible inside the triangle of centers, one covered
  across an edge, two beyond a corner).

  *Proof.* \f$w_k\f$ is the ratio of the signed distances of \f$x\f$ and
  \f$c_k\f$ from the line \f$c_i c_j\f$, and \f$x\f$'s is \f$r_{ij} > 0\f$ on
  its own side; three negatives cannot sum to 1.

Two degeneracies of @cite desc:surface:halperin-shelton-1998 (p. 186), three
spheres through one point and a circle tangent to a third sphere, are by (a)
and (b) one configuration. It is the probe height \f$h = 0\f$ that
@cite desc:surface:connolly-1983 (pp. 550-551) defines and skips.

\anchor corollary-4
**Corollary 4 (a coincident-root vertex has no face).** Let \f$x\f$ lie on
exactly three kept spheres \f$a, b, c\f$ with \f$D = 0\f$ for the face and
\f$\pi_l(x) > 0\f$ for every other kept sphere. Then \f$x\f$ is an SAS vertex
(\f$\pi_m(x + sn) = s^2\f$ for \f$m \in \{a, b, c\}\f$, since \f$n \perp x -
c_m\f$ by @ref lemma-13 "Lemma 13"(a)), at least one circle through it is
accessible on both sides (@ref lemma-13 "Lemma 13"(d), continuity for the other
spheres), and by @ref lemma-12 "Lemma 12" with \f$t = n\f$ its face lies in the
great circle \f$\Pi \cap S(x, r_p)\f$ and has area 0, whatever the other probes
do. In particular the two coincident roots do not carry complementary
hemispheres: @ref corollary-1 "Corollary 1"(a) at \f$h = 0\f$ constrains only
the planar cone \f$C_x\f$, a null set of directions. This is the limit \f$h \to
0\f$ of the coincident reduced-surface faces that @cite desc:surface:sanner-1996
(p. 312) deletes when their probes cut each other through the face plane, stated
exactly.

\anchor remark-5
**Remark 5 (what the run does at a coincident root).** Both roots are at
\f$x\f$, accepted, and merged into one cluster (`members == 2`, three atoms).
On each circle through \f$x\f$ the ring is a single-point window: its
antipode has \f$\pi_k = 4 r_{ij}^2 / w_k\f$ (@ref lemma-13 "Lemma 13"(c)),
so the \f$2\pi\f$ arc is accessible iff \f$w_k > 0\f$, and the cluster
receives the tangent pair \f$\pm n\f$ from every such full circle (or from the
two arcs at \f$x\f$ when a fourth sphere interrupts the loop elsewhere) and
nothing from a covered circle, whose only arc at the cluster is the zero-length
diagonal. Coincident ring vertices share one angle, so the pair is bitwise
antipodal. The contact-plane and triangle filters skip the cluster, every low
neighbor within \f$2 r_p\f$ is kept, and the face solver returns 0 at the first
live complementary pair, before any cutter is examined. When the perturbation
resolves the tie \f$D = 0\f$ to a miss instead, no probe exists and `side`
covers or frees each circle by the same sign; both branches give the exact area
0.

### The filter pipeline {#ses-pipeline}

The pipeline, per structure: heights (@ref lemma-9 "Lemma 9") and the high/low
split (@ref corollary-1 "Corollary 1"(c)) of all probes, then probe pairs within
\f$2 r_p\f$ (a nuri::VoxelGrid query), then drop pairs with a high end (both
ends low), so that a high active face is never solved against any cap, then drop
pairs failing @ref lemma-9 "Lemma 9" on either side, then drop pairs failing
@ref lemma-10 "Lemma 10" on either side, tested on the survivors only, then
closed-form areas of the high active faces with a triangle, and the face solver
on every other active face with the surviving caps. By
@ref theorem-5 "Theorem 5" the dropped caps change nothing. The filters are
necessary conditions evaluated in floating point: a rounding flip can only drop
a cap that reaches the face by a rounding-scale sliver, an area effect far below
what the brute-force test (every face solved against every probe within \f$2
r_p\f$) enforces, never a topological one.

\anchor theorem-5
**Theorem 5 (the filters are exact).** Dropping every probe pair with a high
end, or failing @ref lemma-9 "Lemma 9" or @ref lemma-10 "Lemma 10" on either
side, leaves the accessible region, the area and the dots of every face
unchanged.

*Proof.* For almost every cut face point \f$z\f$ (outside the null set of
@ref lemma-6 "Lemma 6") its first-touch cutter \f$s_1\f$ is a vertex
within \f$r_p\f$ of \f$z\f$ that satisfies the conclusion of
@ref lemma-8 "Lemma 8", so by @ref corollary-2 "Corollary 2" the pair
\f$(x, s_1)\f$ is low at both ends and passes both filters on both sides. A
cap that fails a filter is therefore never \f$s_1\f$ for any point, every cut
face point keeps its first-touch cutter, and the face computed from the
survivors is the same.

## Dot sampling {#dots}

nuri::SurfaceSampler (`sample.cpp`) places dots *inside the exact patch
domains* above. Their weights are normalized so that the sum over a patch
equals the patch's analytic area; dot counts follow the requested density.
Only the count is a sampled quantity. Normals point from the surface into the
solvent; for toroidal and concave dots that is toward the probe center, so
nuri::SurfaceDots::probes() recovers the probe positions.

- **Convex** (atom \f$i\f$ below `n_active`). A Fibonacci lattice
  @cite core:geom:fiblat-2010 @cite core:geom:fiblat-2020 with
  \f$\max(\operatorname{lround}(4\pi r_i^2 \cdot \text{density}), 1)\f$ directions is
  generated on the unit sphere (one lattice per distinct count, cached); a
  direction is kept iff it is outside every cap of the atom. Dot
  \f$c_i + r_i d\f$, normal \f$d\f$, weight \f$A_{\text{convex}}(i)/n_{\text{kept}}\f$,
  owner \f$i\f$. The SAS is sampled the same way on the SAS spheres.
- **Toroidal** (row by row, one row per active arc and cusp side). When no cusp
  falls inside the valid range the two ranges are merged into one row: rounding
  the ring count on each half separately would double its error on narrow
  saddles and leave a seam at \f$\beta = 0\f$.
  1. \f$k_\beta = \max(\operatorname{lround}(r_p (\beta_{\text{hi}} - \beta_{\text{lo}}) \sqrt{\text{density}}), 1)\f$
     rings of equal \f$\beta\f$ width; ring \f$m\f$ spans
     \f$[\beta_m, \beta_{m+1}]\f$ and is sampled at its middle;
  2. exact ring area
     \f$a_m = r_p\, \Delta\varphi\, [r_l \Delta\beta - r_p (\sin\beta_{m+1} - \sin\beta_m)]\f$;
  3. \f$k_{\varphi, m} = \operatorname{lround}(a_m \cdot \text{density})\f$
     dots on ring \f$m\f$, uniformly spaced in \f$\varphi\f$ inside the arc
     at \f$\varphi = \varphi_{\text{beg}} + (n + \tfrac14 \text{or} \tfrac34)\,\Delta\varphi/k_{\varphi,m}\f$,
     the quarter alternating with \f$m\f$ so neighboring rings interleave;
     weight \f$a_m / k_{\varphi, m}\f$. Ring areas are rescaled so that the
     row's weights sum to its exact area; a row whose rings all round to zero
     is collapsed to one ring with
     \f$\operatorname{lround}(\text{area} \cdot \text{density})\f$ dots, and
     only rows that still round to zero are dropped (recorded in
     nuri::SurfaceDots::dropped_area()). An absent spindle side is a
     zero-width row: it drops zero area and emits nothing.
  4. Dot position \f$p = q(\varphi) + r_p(-\cos\beta \cdot \text{radial}(\varphi) + \sin\beta \cdot u)\f$,
     normal \f$(q - p)/r_p\f$, owner the atom whose vdW surface is nearer.
     \f$\cos\beta\f$, \f$\sin\beta\f$ and the owner are per ring:
     \f$|p - c_i|^2 = R_i^2 + r_p^2 - 2r_p(r_l \cos\beta - a \sin\beta)\f$ and
     \f$|p - c_j|^2 = R_j^2 + r_p^2 - 2r_p(r_l \cos\beta + (d - a)\sin\beta)\f$
     do not depend on \f$\varphi\f$.
- **Concave** (face of probe \f$q\f$). One probe-sphere Fibonacci lattice
  (\f$\max(\operatorname{lround}(4\pi r_p^2 \cdot \text{density}), 1)\f$ directions,
  not rotated per face: weights are \f$A_{\text{face}}/n_{\text{kept}}\f$, so
  orientation is unbiased) is tested against the face arrangement; a
  direction is kept iff it is outside every live cap, which encodes both the
  polygon of departure hemispheres and the neighbor-probe cuts. Dot
  \f$q + r_p d\f$, normal \f$-d\f$, weight \f$A_{\text{face}}/n_{\text{kept}}\f$,
  owner the atom whose vdW surface is nearest, from
  \f$|p - c_a|^2 = R_a^2 + r_p^2 - 2 r_p R_a (d \cdot \hat c_a)\f$.

Dots are stored as one block per patch kind (nuri::SurfaceDots::offset(),
nuri::SurfaceDots::count()) and `atom` carries the original index. Counts
round half away from zero.

## Shape complementarity {#sc}

nuri::ShapeComplementarity (`sc.cpp`) implements the statistic of
@cite desc:sc:lawrence-colman-1993 with the parameters of nuri::ScParams,
whose defaults are those of Lawrence's CCP4 `sc` @cite desc:sc:ccp4-sc,
ported by Rosetta's `sc` application
(\f$r_p = 1.7\,\AA\f$, density \f$15\,\AA^{-2}\f$, weight 0.5, band \f$1.5\,\AA\f$, separation \f$8\,\AA\f$,
clamp 0.999). For each side:

1. **Active atoms.** Those with an atom of the other side within `sep`
   (Rosetta's "buried atoms"), narrowed by the caller's mask; every other atom
   of the side occludes (Rosetta's "blocked atoms"). The SES of the side is
   built with that mask and sampled at the density.
2. **Burial.** A dot is buried when its probe center lies strictly inside a
   solvent-accessible sphere of the other side (\f$|p_{\text{probe}} - c| < R + r_p\f$).
3. **Peripheral trim.** A buried dot within `band` of a non-buried dot of its
   own side is trimmed; the survivors are the interface dots.
4. **Scores.** Each surviving dot is paired with the nearest surviving dot of
   the other side (nuri::VoxelGrid::nearest(); equal distances take the
   smallest index) at distance \f$d\f$ with normals \f$n, n'\f$, and scores
   \f$-(n \cdot n') \exp(-\text{weight} \cdot d^2)\f$ clamped to
   \f$[-\text{clamp}, \text{clamp}]\f$.
5. **Medians.** The medians of the distances and of the scores are exact (the
   two middle values averaged for an even count); a 0.02-bin interpolated
   median, Rosetta's estimator, differs by less than 0.001.

The statistic is the mean of the two sides' median scores, the reported
distance the mean of their median separations, and the area the sum of the
trimmed areas. Dots are equal-weight samples within a convex or concave
patch and within a toroidal ring, and the medians ignore areas (an
area-weighted median differs by about 0.001). Every neighbor query (active
atoms, burial, trim, nearest dot) is a
nuri::VoxelGrid query on squared distances with the comparisons above.

## Analysis {#analysis}

### Complexity {#analysis-complexity}

Let \f$n\f$ be the kept atoms, \f$m\f$ the largest number of overlapping
neighbors of one atom (the cap count; about 15 for proteins), \f$p\f$ the
probes (at most the faces of the triangulation, \f$\sim 12n\f$, so
\f$O(n)\f$), \f$k_p\f$ the largest number of probes within \f$2 r_p\f$ of
one probe (bounded by packing), and \f$N_d\f$ the dots,
\f$N_d \approx \text{density} \cdot \text{area}\f$. In short, a shape
complementarity run costs \f$O(n \log n + n m^2 + N_d\,m)\f$ time, which
for bounded \f$m\f$ is \f$O(n \log n + N_d)\f$, and \f$O(n m + N_d)\f$
memory; the stages:

- **Preparation**: two nuri::VoxelGrid queries, \f$O(n m)\f$ for the
  near pairs and the exact predicates on them, plus \f$O(n + nm)\f$ for
  the ranks and the sort of `g`.
- **SAS**: one regular triangulation, \f$O(n \log n)\f$ expected
  (linear expected size for well-distributed points
  @cite desc:surface:dwyer-1991, inserted in the biased random order of
  @cite desc:surface:amenta-choi-rote-2003) with about \f$6n\f$ cells
  (measured on dense random sets: \f$6.1n\f$ cells, \f$12.4n\f$ faces,
  mean degree 14.5); every face is examined once, each
  behind a floating-point filter that falls back to exact arithmetic only
  near degeneracy, \f$O(n)\f$ predicates; each sphere's arrangement is
  \f$O(m^2)\f$ in its cap count for the disc-intersection components and
  \f$O(m)\f$ for the walk, independent of all other spheres, so
  \f$O(n m^2)\f$ in all; the probe merge is one sort, \f$O(p \log p)\f$,
  plus one exact coincidence predicate per candidate pair not already
  linked: one per root at a \f$k\f$-fold point, none elsewhere.
- **SES**: heights and contact triangles \f$O(p)\f$; the probe-pair query
  \f$O(p\,k_p)\f$ and the two filters \f$O(1)\f$ per pair; the face solver
  \f$O(k_p^2)\f$ per low face (its caps are the surviving neighbors plus
  three hemispheres), \f$O(p\,k_p^2)\f$ in all; saddles \f$O(n m)\f$ per
  arc count. Since \f$k_p \le m\f$ up to a constant, the SES is
  \f$O(n m^2)\f$.
- **Sampling**: each convex or concave dot is tested against the caps of its
  sphere or face, \f$O(N_d\,m)\f$; toroidal rows are closed-form,
  \f$O(N_d + n m)\f$. Lattices are cached per dot count.
- **Shape complementarity**: the active-atom, burial and trim queries are
  nuri::VoxelGrid queries at a fixed cutoff, \f$O(n m)\f$ for the atoms
  and \f$O(N_d\,m)\f$ for the dots; the nearest-dot query is
  nuri::VoxelGrid::nearest(), \f$O(1)\f$ expected per dot when the cell
  size matches the dot spacing; the medians use `nth_element`, \f$O(N_d)\f$.
  Two surfaces, two samplings, and these passes give the total above.

### Floating point {#analysis-fp}

The predicates live in one translation unit compiled
without fast-math (like the vendored geogram sources), because the
floating-point filter's running error bound and the exact fallback both
assume round-to-nearest without reassociation or contraction. Each predicate
is evaluated first in double with a running error bound
(\f$|v - \text{exact}| \le e\f$, certified iff \f$|v| > 2e\f$; a literal
zero is certified exactly) and falls back to exact arithmetic only when the
bound does not certify the sign, the filtered floating point of
@cite desc:surface:fortune-vanwyk-1996 with the runtime bound of
@cite desc:surface:burnikel-funke-seel-2001, as geogram's predicate kit
does @cite desc:surface:levy-2016. The exact type is a dyadic
\f$m \cdot 2^e\f$ with an arbitrary-precision integer \f$m\f$: the
adaptive expansions of @cite desc:surface:shewchuk-1997 cannot serve here,
since they assume no underflow and a high-degree polynomial of rounded-zero
coordinates (\f$10^{-16}\f$) has terms below the double exponent range,
where underflow silently breaks their exactness. Perturbation
ties are resolved by the same kernel evaluated on dual numbers, which yields
the first-order coefficients automatically. The lower rungs of the fallback
(the deeper perturbation orders, an exact zero under `sqrt_floor`, an
uncertifiable quotient) are reached by the bit pattern of an input, not by
its geometry; no test fixture targets them and their coverage is incidental.

Positions and vertex angles come from the **offset**
\f$x - \mathrm{cntr} = (P \pm \sqrt D\, Q)/(G|u|^2)\f$ of a root from its
circle's center, with
\f$P = G(|u|^2 (c_f - c_a) + \lambda d_b + \mu d_c) - |u|^2 v\, d\f$
(\f$\lambda, \mu, d_b, d_c\f$ of the face relative to its first center
\f$c_f\f$; \f$v, d, G\f$ of the circle \f$(a, b)\f$) and \f$Q = G u\f$
rational and \f$P \perp Q\f$: it is taken from the filtered double when the
running bound certifies it to `kOffsetRelTol` \f$= 2^{-26}\f$ of its own
length, and otherwise from the exact dyadic (\f$P\f$, \f$Q\f$, \f$D\f$
rounded to double and combined in a few more roundings), so the direction
from the center is right to \f$2^{-26}\f$ on a circle of any radius. The
certificate is relative to the offset's own length; coordinate-scale
roundings in \f$v\f$ only enlarge the certified error (measured
\f$10^{-14}\f$ at the origin, \f$10^{-9}\f$ at \f$1000\,\AA\f$). This matters for a
circle tiny only by rounding (a tangent pair under a rigid motion has radius
about \f$10^{-8}\,\AA\f$, and a non-tie can be far smaller): the floating
discriminant and squared radius of such a circle are smaller than their own
rounding error, and the vertex angles of a position taken from them are
noise. The probe position is \f$\mathrm{cntr} + \text{offset}\f$ with the
center \f$\mathrm{cntr}\f$ (@ref sas-notation "Notation") in plain double, so it
is within \f$1.01 \cdot 2^{-26} \rho_{\max} + 8\sqrt 3 \cdot 2^{-53}\f$
`kSurfaceMaxCoord` \f$+ 15 \cdot 2^{-53} \rho_{\max}\f$ of the exact root (the
last \f$\sqrt 3 \cdot 2^{-53}\f$ of each is the rounding of the sum itself)
(\f$\rho_{\max}\f$ the largest model radius, \f$\le R_{\max} + \sqrt{12 \cdot
2^{-53}}\f$ `kSurfaceMaxCoord`), which sets the candidate band of the
@ref sas-merge "probe merge"; \f$\varphi\f$, dart angles and areas inherit at
most the \f$2^{-26}\f$ relative error. The difference of two roots is never
taken from their positions: `difference` evaluates \f$P_\delta + \sqrt D\,
Q_\delta + \sqrt{D'}\, Q'_\delta\f$ under the same certificate, and otherwise
from the exact dyadics with the square roots taken to \f$k\f$ extra bits,
doubled until the sum's bound is below \f$2^{-52}\f$ of its length, read at the
sum's own scale. It returns a unit direction and a length separately, because
distinct roots have no separation that fits a double: a separation bound of the
kind of @cite desc:surface:bfmss-2009 on one component, a polynomial of degree 9
in inputs within `kSurfaceMaxCoord`, is about \f$2^{-1800}\f$. The direction is
right however close the roots are; a length that underflows lifts the neighbor's
cap as the hemisphere limit (the lift takes the unit direction once the squared
length is 0). The floating comparisons that remain are output-only: the fold of
each vertex's `atan2` into its exact half-plane class (the class carries the
side of the ray, `atan2` only the magnitude, so a vertex at \f$\pi \mp \delta\f$
folds to \f$\pi\f$, never to 0 or \f$2\pi\f$), the normalization of the dart
angle \f$\iota\f$ and the \f$\max(\cdot, 0)\f$ clamps. A genuine corner is
\f$\pi - \angle(h_a, h_b)\f$ for the two cap normals in the tangent plane at the
vertex (the accessible region there is the wedge outside both caps,
@ref sas-step-5 "SAS step 5"); the computed corner deviates by at most one
offset certificate per dart plus roundoff, \f$2 \cdot 2^{-26} + 2^{-40}\f$
(`kSurfaceAngleEps`), the vertex normal's error entering only quadratically, so
the reflex check \f$\iota \le \pi +\f$ `kSurfaceAngleEps` is a `DCHECK` on an
invariant, not a decision. A departure tangent carries the same bound plus its
normalization (`kTangentEps`), which gates the contact triangle (@ref lemma-10
"Lemma 10").

The face solver's ring sort (@ref ses-patches "Patch areas") is sound for
the same reason: each vertex's offset is certified to `kOffsetRelTol` of its
length, the frame shared by the ring is an orientation-preserving linear map
and so only reparametrizes the angles monotonically, and the dots, `atan2`,
the class fold and the subtraction add roundoff only; a pair further apart
than `kRingSortEps` is in the exact order, and an exact tie is always within
it.

## Related work {#related}

### Related constructions {#related-constructions}

- The alpha complex, the dual complex of @cite desc:surface:edelsbrunner-1995
  that @ref theorem-2 "Theorem 2" uses as the triangulation restricted to
  overlapping pairs, underlies the alpha-shape area of
  @cite desc:surface:liang-1998, which sums inclusion-exclusion terms, one
  per simplex, instead of solving each sphere's arrangement.
- The decomposition \f$U = \bigcup_s (B_s \cap V_s)\f$ of
  @ref theorem-2 "Theorem 2" is the restriction decomposition on which
  @cite desc:surface:cazals-kanhere-loriot-2011 build a certified volume of
  a union of balls, with interval arithmetic and a per-construction analysis
  of the predicates; the area here is certified by exact predicates instead.
- The contour-buildup algorithm @cite desc:surface:totrov-abagyan-1996
  assembles each atom's boundary by inserting its neighbors' circles one at
  a time into a circular arc list, "instead of calculating all possible arcs
  and then filtering" (p. 139); here the regular triangulation names the
  neighbors (@ref theorem-2 "Theorem 2") and the fan gives the ring order
  (@ref lemma-2 "Lemma 2"), with no sequential insertion.
- The exact arrangement of the circles on one sphere
  (@ref sas-step-5 "SAS step 5") was previously computed by a sweep over all
  its intersecting neighbors @cite desc:surface:cazals-loriot-2009.
- The reduced-surface algorithm of @cite desc:surface:sanner-1996 computes
  the probe-probe cuts of @ref theorem-4 "Theorem 4" without proof.
- SESA @cite desc:surface:wang-2024 computes SES areas analytically from an
  exhaustive probe-probe intersection search and identifies the exterior
  surface as a component of a region graph; the port keeps every component
  (the cSES of @cite desc:surface:quan-stamm-2016) and leaves the exterior
  split to later work.

### MolSurfComp {#related-molsurfcomp}

The authors' Matlab implementation of the surface, MolSurfComp
(<https://github.com/quanchaoyu/MolSurfComp>, LGPL-3.0), is distributed under
terms incompatible with this project's Apache-2.0 license. No code, formula,
constant, tolerance, table, identifier or file structure from it appears in
this source tree; the implementation was written from the paper and from the
derivations on this page, and MolSurfComp was read once, to audit its concave
cut set against this page; nothing was taken from it. That audit found four
prunings of the concave cut set that the paper does not state (its Theorem
5.1 removes \f$B_{r_p}(x)\f$ for every SAS intersection point \f$x\f$ within
\f$2 r_p\f$ and imposes no filter):

| Pruning | MolSurfComp (asserted in code) | Here |
|---|---|---|
| high probe implies an uncut face, by height over the plane of the three atom centers | asserted | not adopted. Sanner's criterion, proved as @ref corollary-1 "Corollary 1"(c) on the distance to the contact *triangle* |
| only low probes may cut | asserted | not adopted. @ref corollary-2 "Corollary 2" (cut symmetry), proved |
| first-atom neighborhood restriction | asserted | not adopted. @ref lemma-11 "Lemma 11" and @ref corollary-3 "Corollary 3" give the provable version: a first-touch cutter's hosts reach the cut face's hosts within two near hops, one hop is not enough |
| keep only the largest departure angle among probes sharing two atoms | asserted | @ref lemma-7 "Lemma 7", proved, not a code path |

Since \f$\operatorname{dist}(x, T) \ge h\f$, strictly more faces are uncut
here, and the two implementations disagree observably (on 1BRS, 3782 high
faces here against 3520 there; on 2PTC, 1686 against 1555).

### Rosetta {#related-rosetta}

Rosetta's `sc` application was consulted for its defaults and deviations
and run as a numeric oracle for the statistic; it is neither linked nor
redistributed and no code was taken. Intentional deviations from it, as
observed:

- Rosetta builds missing C-terminal `OXT` atoms; the input is scored as given.
- At torus cusps (probe-circle radius below \f$r_p\f$) Rosetta emits only one
  of the two arc halves; both valid halves are kept here (paper-correct, a
  tiny area effect).
- Rosetta uses `float`, this code `double`, and the sampling patterns differ
  (Fibonacci lattice and equal-area rings against Rosetta's grid), so
  per-interface values differ by about \f$\pm 0.01\f$.
- Burial and the peripheral band are identical in definition.
