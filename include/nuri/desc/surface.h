//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_DESC_SURFACE_H_
#define NURI_DESC_SURFACE_H_

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include <absl/functional/function_ref.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  enum class SrSasaMethod {
    kAuto = 0,
    kDirect,
    kOctree,
  };

  extern ArrayXd sr_sasa_impl(const Matrix3Xd &pts, const ArrayXd &radii,
                              int nprobe, SrSasaMethod method);
}  // namespace internal

/**
 * @brief Calculate the Solvent-Accessible Surface Area (SASA) of a molecule
 *        conformation using the Shrake-Rupley algorithm.
 *
 * @param mol The input molecule.
 * @param conf The conformation matrix.
 * @param nprobe The number of probe spheres. Default is 92.
 * @param rprobe The radius of the probe spheres. Default is 1.4 angstroms.
 * @param method Whether prefer direct or octree method. Default is auto.
 *               This is mainly for testing purpose; in most cases, the auto
 *               method will choose the optimal method.
 * @return The calculated SASA values per atom (in angstroms squared).
 */
extern ArrayXd shrake_rupley_sasa(
    const Molecule &mol, const Matrix3Xd &conf, int nprobe = 92,
    double rprobe = 1.4,
    internal::SrSasaMethod method = internal::SrSasaMethod::kAuto);

namespace internal {
  constexpr double kSurfaceLengthEps = 1e-6;
  /**
   * The near, contained and shared-circle bands are supersets of the exact
   * sets only while `ulp(|c|^2)` stays far below `kSurfaceLengthEps`.
   */
  constexpr double kSurfaceMaxCoord = 1e4;
  /**
   * A vertex offset from its circle centre is taken from the floating filter
   * when certified to this fraction of its length (`BallExact::offset`).
   */
  constexpr double kOffsetRelTol = 0x1p-26;
  /**
   * Two probes at one point differ by two offsets, each certified to
   * `kOffsetRelTol` of a length `≤ rmax`, taken twice for margin; the centre
   * rounding (`≤ 2^-52 · kSurfaceMaxCoord`) is far below. `build_sas` merges
   * probes within this distance.
   */
  constexpr double probe_merge_tol(double rmax) {
    return 4 * kOffsetRelTol * rmax;
  }
  /**
   * Error bound on a corner of `ArrangementSolver::walk`: one offset
   * certificate per dart (the vertex normal enters only quadratically) plus
   * the roundoff of frames, `atan2`, `sin`/`cos` and the wrap.
   */
  constexpr double kSurfaceAngleEps = 2 * kOffsetRelTol + 0x1p-40;

  class OffsetTable {
  public:
    OffsetTable(): off_(ArrayXi::Zero(1)) { }

    explicit OffsetTable(int n): off_(n + 1) { }

    explicit OffsetTable(ArrayXi &&off) noexcept: off_(std::move(off)) { }

    int operator[](int i) const { return offset(i); }

    int offset(int i) const { return off_[i]; }

    int degree(int i) const { return off_[i + 1] - off_[i]; }

    int max_deg() const {
      return (off_.tail(size()) - off_.head(size())).maxCoeff();
    }

    int size() const { return static_cast<int>(off_.size()) - 1; }

    ArrayXi &off() { return off_; }

    const ArrayXi &off() const { return off_; }

  private:
    ArrayXi off_;
  };

  /**
   * Union-find with path halving; the smaller root absorbs, so every root is
   * the smallest member of its set.
   */
  class UnionFind {
  public:
    explicit UnionFind(int n): parent_(ArrayXi::LinSpaced(n, 0, n - 1)) { }

    int find(int x) {
      while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];
        x = parent_[x];
      }
      return x;
    }

    void merge(int a, int b) {
      auto [lo, hi] = nuri::minmax(find(a), find(b));
      parent_[hi] = lo;
    }

    int n_sets() const {
      int n = 0;
      for (int x = 0; x < parent_.size(); ++x)
        n += static_cast<int>(parent_[x] == x);
      return n;
    }

    /**
     * Relabel every element with its set id, numbered by smallest member;
     * returns the number of sets.
     */
    int relabel() {
      const int n = static_cast<int>(parent_.size());
      for (int x = 0; x < n; ++x)
        parent_[x] = find(x);

      int k = 0;
      for (int x = 0; x < n; ++x)
        parent_[x] = parent_[x] == x ? k++ : parent_[parent_[x]];
      return k;
    }

    ArrayXi &labels() { return parent_; }

  private:
    ArrayXi parent_;
  };

  class CSR {
  public:
    using const_iterator = ArrayXi::const_iterator;

    CSR() = default;

    CSR(ArrayXi &&adj, OffsetTable &&off) noexcept
        : adj_(std::move(adj)), tbl_(std::move(off)) { }

    const_iterator begin(int i) const { return adj_.begin() + tbl_[i]; }

    const_iterator end(int i) const { return adj_.begin() + tbl_[i + 1]; }

    int offset(int i) const { return tbl_[i]; }

    int degree(int i) const { return tbl_.degree(i); }

    int max_deg() const { return tbl_.max_deg(); }

    auto nbrs(int i) const { return adj_.segment(tbl_[i], degree(i)); }

    int eid(const_iterator it) const {
      return static_cast<int>(it - adj_.begin());
    }

    int m() const { return static_cast<int>(adj_.size()); }

    int n() const { return tbl_.size(); }

    template <class F, class B>
    void for_each_triangle(int n_rows, const F &on_match,
                           const B &before_row) const {
      for (int i = 0; i < n_rows; ++i) {
        if (degree(i) < 2)
          continue;

        before_row(i);
        const auto ei = end(i);
        for (auto pij = begin(i); pij < ei; ++pij) {
          const int j = *pij;
          const auto ej = end(j);
          for (auto pik = pij + 1, pjk = begin(j); pik < ei && pjk < ej;) {
            const int ki = *pik, kj = *pjk;
            if (ki == kj)
              on_match(i, j, ki, pij, pik, pjk);
            pik += value_if(ki <= kj);
            pjk += value_if(kj <= ki);
          }
        }
      }
    }

    ArrayXi &adj() { return adj_; }

    const ArrayXi &adj() const { return adj_; }

    ArrayXi &off() { return tbl_.off(); }

    const ArrayXi &off() const { return tbl_.off(); }

  private:
    ArrayXi adj_;
    OffsetTable tbl_;
  };

  /**
   * Atoms ordered `[active | need | shell | occluders]` with contained spheres
   * dropped: `order` maps new to old indices, spheres `< n_active` own
   * surface, `< n_solve` get arrangements, `< n_enum` get caps. `g` is the
   * forward overlap graph on new indices (`row(i)` = sorted `j > i`) over all
   * kept spheres, decided exactly for the lifted heights `t = √(wmax − sar²)`
   * (the near band only ranks atoms); pair `q` carries a circle iff
   * `i < n_enum`, and `d[q]` is its centre distance.
   */
  struct SaPrep {
    Matrix3Xd pts;
    ArrayXd sar;
    ArrayXd t;
    double wmax;
    ArrayXi order;
    CSR g;
    ArrayXd d;
    int n_active;
    int n_solve;
    int n_enum;
  };

  extern std::optional<SaPrep> prepare(const Matrix3Xd &pts, const ArrayXd &sar,
                                       const ArrayXb &active, double rp);

  enum class Sgn : int { kNeg = -1, kZero = 0, kPos = 1 };

  /**
   * Three triangulation vertices; root labels refer to this order:
   * `x± = c_a + y_⊥ ± (√D / |u|²) u` with `u = (c_b − c_a) × (c_c − c_a)`.
   */
  struct BallTriple {
    int a, b, c;
  };

  /**
   * Exact predicates on a set of balls: the power arrangement of weighted
   * points as geogram triangulates them. The balls are the SAS spheres, and
   * later the caps of an SES face lifted to balls. Each ball has centre `c_i`,
   * height `h_i = ((t_i² + x_i²) + y_i²) + z_i²` as geogram evaluates it (or
   * given directly), and squared radius `ρ_i² = W + |c_i|² − h_i`. Every
   * exact tie is resolved by
   * geogram's perturbation: every squared radius grows, lower index first.
   * Compiled without fast-math (`NURI_STRICT_FP_SRCS` in src/CMakeLists.txt);
   * the header carries no arithmetic. `kForceExact` skips the floating
   * filter (tests only).
   */
  template <bool kForceExact>
  class BallExactImpl {
  public:
    BallExactImpl() = default;

    static BallExactImpl make(const Matrix4Xd &lifted, double wmax);
    static BallExactImpl make(const Matrix3Xd &centers, const ArrayXd &heights,
                              double wmax);

    int n() const { return static_cast<int>(h_.size()); }
    const Matrix3Xd &centers() const { return c_; }
    const ArrayXd &h() const { return h_; }
    double wmax() const { return w_; }
    double rho2(int i) const;

    /**
     * Whether balls `a`, `b` overlap; an exact tangency is zero, a circle of
     * zero length that carries no cap.
     */
    Sgn overlap(int a, int b) const;
    /**
     * Which of balls `a`, `b` (0 or 1) lies inside the other, -1 if neither,
     * 2 if they are identical; internal tangency counts.
     */
    int contained(int a, int b) const;
    /**
     * Which sphere of `f` (0, 1, 2) lies between the other two on their
     * common axis when the three share one circle, -1 otherwise.
     */
    int shared_circle(BallTriple f) const;
    /**
     * Whether sphere `c` cuts circle `(a, b)` in two points (symmetric in the
     * triple; a tangency is perturbed to a cut or a miss).
     */
    Sgn cuts(BallTriple f) const;
    /**
     * Sign of `π_c` on circle `(a, b)`, valid when `c` does not cut it:
     * positive outside ball `c`, negative inside.
     */
    Sgn side(int a, int b, int c) const;
    /**
     * Whether the discs of caps `j`, `l` on sphere `s` intersect: their
     * circles cross, or one circle lies inside the other's ball; two
     * circles touching from outside are not joined.
     */
    bool discs_intersect(int s, int j, int l) const;
    /**
     * `π_l(x) ≥ 0` for root `x` of face `f`: the cut point is not inside ball
     * `l`. Requires `cuts(f)` positive.
     */
    Sgn accept(BallTriple f, bool plus, int l) const;
    /**
     * Both roots of `f` in double, `first` the `plus` root; requires
     * `cuts(f)` positive.
     */
    std::pair<Vector3d, Vector3d> roots(BallTriple f) const;
    Vector3d root(BallTriple f, bool plus) const;
    /**
     * `x − cntr` for root `x` of `f` on circle `(a, b)`, in double but to
     * rounding of its own length: the direction is right on a circle of any
     * radius. Requires `cuts(f)` non-negative.
     */
    Vector3d offset(int a, int b, BallTriple f, bool plus) const;

    /**
     * Position of root `x` of `f` on circle `(a, b)` relative to the
     * reference ray `r = (c_b − c_a) × e_k`, `k = reference_axis(c_b − c_a)`:
     * 0 on the ray, 1 in `(0, π)`, 2 at `π`, 3 in `(π, 2π)`, counter-clockwise
     * about `c_b − c_a`.
     */
    int half_plane(int a, int b, BallTriple f, bool plus) const;
    /**
     * Sign of the oriented angle from root `x_i` to root `x_j` about
     * `c_b − c_a` on circle `(a, b)`; zero iff the points coincide or are
     * antipodal (no perturbation: the caller separates the two cases by
     * `half_plane`).
     */
    Sgn ccw(int a, int b, BallTriple fi, bool plus_i, BallTriple fj,
            bool plus_j) const;
    /**
     * `ccw` with an exact zero (coincident or antipodal roots) resolved by
     * the perturbation. Requires `cuts` positive for both faces.
     */
    Sgn ccw_perturbed(int a, int b, BallTriple fi, bool plus_i, BallTriple fj,
                      bool plus_j) const;
    /**
     * `π_c(Q) ≥ 0` for the antipode `Q = 2 cntr − x` of root `x` of `f` on
     * circle `(a, b)`.
     */
    Sgn antipode(int a, int b, BallTriple f, bool plus, int c) const;

    static int reference_axis(const Vector3d &d) {
      int k;
      d.cwiseAbs().minCoeff(&k);
      return k;
    }

    static bool selftest();

  private:
    Matrix3Xd c_;
    ArrayXd h_;
    double w_ = 0;
  };

  extern template class BallExactImpl<false>;
  extern template class BallExactImpl<true>;

  using BallExact = BallExactImpl<false>;

  /**
   * Regular (weighted Delaunay) triangulation of every kept sphere. Vertices
   * are the kept spheres in original index order, then four far bounding
   * points, so the result is shared across active masks: `vertex` maps an
   * original atom index to its vertex (-1 if dropped), `tets` holds the finite
   * cells, `adj(lf, c)` the cell across the face opposite local vertex `lf`
   * (-1 past the hull), and `nbrs(v)` the edges of sphere vertex `v`. A sphere
   * with no edges has an empty power cell, i.e. no accessible surface.
   * `face(lf, c)` numbers the faces (shared ids across the two cells),
   * `edge_cell` gives one cell containing each `nbrs` edge, and `ex` holds the
   * exact predicates on the lifted vertices (spheres, then corners).
   */
  struct SasDelaunay {
    Array4Xi tets;
    Array4Xi adj;
    Array4Xi face;
    int n_faces = 0;
    CSR nbrs;
    ArrayXi edge_cell;
    ArrayXi vertex;
    BallExact ex;
  };

  extern SasDelaunay triangulate(const SaPrep &sa);

  struct SasCircle {
    Vector3d axis, cntr;
    double a, rl;
    int i, j;
  };

  /**
   * `phi` is measured in the circle frame `e1 = normalize(d × e_k)`,
   * `d = c_j − c_i`, `k = BallExact::reference_axis(d)`, `e2 = axis × e1`.
   * `beg`, `end` are probes, both -1 for a full circle.
   */
  struct SasArc {
    double phi, dphi;
    int circ, beg, end;
  };

  struct SasCaps {
    CSR h;
    Matrix3Xd axis;
    ArrayXd cosa, sina;
  };

  struct SasProbes {
    CSR atoms;
    Matrix3Xd pos, tan;
    OffsetTable tan_off;
    int n_active;
  };

  struct SasGeometry {
    std::vector<SasCircle> circles;

    SasCaps caps;
    SasProbes probes;

    std::vector<SasArc> arcs;
    int n_active_arcs;

    ArrayXd area;
  };

  extern SasGeometry build_sas(const SaPrep &sa, const SasDelaunay &del);

  /**
   * SES patches of the active atoms. `convex_area` is per active atom
   * `i < sa.n_active`; `saddle_beta` (`lo0, hi0, lo1, hi1`, the valid ranges
   * of the generating arc) and `saddle_integral` (one integral per range)
   * are per active circle `q < sa.g.offset(sa.n_active)`; `saddle_area` is
   * per active arc (prefix `n_active_arcs` of `geo.arcs`). `face_area` and
   * the caps `face_off`, `face_axis`, `face_cosa`, `face_sina` (hemispheres
   * of the departure tangents first, then the neighbour probes that may
   * cut, less the caps dropped by the face solver's hygiene) are per active
   * probe `p < probes.n_active`.
   */
  struct SesGeometry {
    ArrayXd convex_area;
    Matrix4Xd saddle_beta;
    Matrix2Xd saddle_integral;
    ArrayXd saddle_area;
    ArrayXd face_area;
    OffsetTable face_off;
    Matrix3Xd face_axis;
    ArrayXd face_cosa, face_sina;
  };

  extern SesGeometry build_ses(const SaPrep &sa, const SasGeometry &geo,
                               double rp);

  /**
   * Gauss–Bonnet area of one sphere from its caps and the accessible arcs on
   * their circles. Fill with `begin`, `add_cap`, `add_vertex` and `add_arc`
   * (ids are assigned in call order; an arc runs from `beg` to `end`, both -1
   * for a full circle), then `solve`
   * with the exact test whether two caps' discs intersect; buffers persist
   * across problems.
   */
  class ArrangementSolver {
  public:
    /**
     * Sizes every buffer once for at most `mcap` caps, `kcap` vertices and
     * `acap` arcs per problem.
     */
    ArrangementSolver(int mcap, int kcap, int acap);

    void begin(double radius);

    int add_cap(double cosa);

    int add_vertex(const Vector3d &dir);

    /**
     * `tbeg` is the departing tangent at `beg`, `tend` the arriving tangent
     * at `end`, both unit and consistent with the arc's own angles so that
     * corners and arcs describe one closed curve even where the probe
     * positions cannot resolve the circle.
     */
    void add_arc(int cap, int beg, int end, double dphi, const Vector3d &tbeg,
                 const Vector3d &tend) {
      arcs_.push_back({ cap, beg, end, dphi, tbeg, tend });
    }

    double solve(absl::FunctionRef<bool(int, int)> intersects);

  private:
    struct Arc {
      int cap, beg, end;
      double dphi;
      Vector3d tbeg, tend;
    };

    struct Darts {
      int in = -1, out = -1;
      double ain = 0, aout = 0;
    };

    std::pair<int, double> walk(UnionFind &uf);

    double radius_ = 0;
    int m_ = 0, k_ = 0;

    ArrayXd cosa_;

    Matrix3Xd dirs_, ea_, eb_;

    std::vector<Arc> arcs_;

    ArrayXi succ_;
    ArrayXb seen_;
    std::vector<Darts> darts_;
  };

  /**
   * Exact Gauss–Bonnet area of one SES concave face: the probe sphere of
   * radius `rp` at the origin outside every cap, cap `k` lifted to the ball
   * `(n_k, h_k)` with `π_k(y) = |y − n_k|² − (rp² + |n_k|² − h_k)`, so that
   * on the sphere `π_k < 0` is the cap: `(t, 0)` for the hemisphere of
   * tangent `t`, `(w, |w|²)` for the probe at `w`; `cosa_k` is the cosine
   * of its angular radius. Ties are resolved by `BallExact`'s perturbation
   * in the order probe, then caps as given. On return `live` marks the caps
   * that took part: a cap whose ball does not overlap the probe sphere (an
   * exact tangency included) is dropped, and of two identical caps the
   * higher index; two complementary caps end the face empty with `live` as
   * it stands at that pair. This is the face solver of `build_ses` for one
   * face (tests only).
   */
  extern double solve_ses_face(const Matrix3Xd &n, const ArrayXd &h,
                               const ArrayXd &cosa, double rp, ArrayXb &live);
}  // namespace internal

template <class Key, class Map>
void argsort_bucket(ArrayXi &idxs, internal::OffsetTable &off, const Key &key,
                    const Map &map) {
  return argsort_bucket(idxs, off.off(), key, map);
}
}  // namespace nuri

#endif /* NURI_DESC_SURFACE_H_ */
