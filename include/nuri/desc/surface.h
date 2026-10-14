//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_DESC_SURFACE_H_
#define NURI_DESC_SURFACE_H_

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include <absl/base/attributes.h>
#include <absl/container/flat_hash_map.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"

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
  class OffsetTable {
  public:
    OffsetTable(): off_(ArrayXi::Zero(1)) { }

    explicit OffsetTable(int n): off_(n + 1) { }

    explicit OffsetTable(ArrayXi &&off) noexcept: off_(std::move(off)) { }

    int operator[](int i) const { return offset(i); }

    int offset(int i) const { return off_[i]; }

    int degree(int i) const { return off_[i + 1] - off_[i]; }

    int max_deg() const {
      return size() == 0 ? 0
                         : (off_.tail(size()) - off_.head(size())).maxCoeff();
    }

    int size() const { return static_cast<int>(off_.size()) - 1; }

    ArrayXi &off() { return off_; }

    const ArrayXi &off() const { return off_; }

  private:
    ArrayXi off_;
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

    /**
     * Edge id of `j` in row `i`, -1 if absent.
     */
    int find(int i, int j) const {
      const auto e = end(i), it = std::lower_bound(begin(i), e, j);
      return it != e && *it == j ? eid(it) : -1;
    }

    int m() const { return static_cast<int>(adj_.size()); }

    int n() const { return tbl_.size(); }

    ArrayXi &adj() { return adj_; }

    const ArrayXi &adj() const { return adj_; }

    ArrayXi &off() { return tbl_.off(); }

    const ArrayXi &off() const { return tbl_.off(); }

  private:
    ArrayXi adj_;
    OffsetTable tbl_;
  };
}  // namespace internal

template <class Key, class Map>
void argsort_bucket(ArrayXi &idxs, internal::OffsetTable &off, const Key &key,
                    const Map &map) {
  return argsort_bucket(idxs, off.off(), key, map);
}

enum class SurfaceKind : int { kSas = 0, kSes = 1 };

enum class SurfacePatch : int { kConvex = 0, kToroidal, kConcave };

namespace internal {
  /**
   * Caps on one sphere per row; cap `k` of row `s` covers the directions `d`
   * with `axis(k) · d > cosa[k]`.
   */
  struct CapTable {
    OffsetTable off;
    Matrix3Xd axis;
    ArrayXd cosa;
  };

  /**
   * Per active sphere; row `i` is sphere `i`.
   */
  struct ConvexPatches {
    ArrayXd area;
    CapTable caps;
  };

  /**
   * Circle of spheres `i < j`, with its frame `e1 = normalize(d × e_k)`,
   * `d = c_j − c_i`, `k = reference_axis(d)`, `e2 = axis × e1`. `aij` is
   * the plane's offset along `axis` from `c_i` and from `c_j`.
   */
  struct SasCircle {
    Vector3d axis, cntr;
    Vector3d e1, e2;
    double aij[2], rl;
    int i, j;
  };

  /**
   * `phi` is measured in the frame `e1`, `e2` of `SasCircle`. `beg`, `end`
   * are probes, both -1 for a full circle.
   */
  struct RingArc {
    double phi, dphi;
    int circ, beg, end;
  };

  /**
   * Per active circle (`circles`, `beta`, `integral`) and per active arc
   * (`arcs`, `area`).
   */
  struct ToroidalPatches {
    std::vector<SasCircle> circles;
    Matrix4Xd beta;
    Matrix2Xd integral;
    std::vector<RingArc> arcs;
    ArrayXd area;
  };

  /**
   * Per active probe: its centre, contact atoms, face area and face caps.
   */
  struct ConcavePatches {
    Matrix3Xd pos;
    CSR atoms;
    ArrayXd area;
    CapTable caps;
  };

  /**
   * Analytic patches of one surface over the kept spheres, in prepared
   * order: the active spheres first, then the spheres that shape their
   * patches. `order()` maps a kept sphere to its original atom. A SAS has
   * convex patches only.
   */
  class SurfaceGeometry {
  public:
    SurfaceGeometry(SurfaceKind kind, double rp, Matrix3Xd &&pts, ArrayXd &&sar,
                    ArrayXd &&r, ArrayXi &&order, ConvexPatches &&convex,
                    ToroidalPatches &&toroidal, ConcavePatches &&concave);

    SurfaceKind kind() const { return kind_; }

    double rp() const { return rp_; }

    int n_atoms() const { return static_cast<int>(order_.size()); }

    const ArrayXi &order() const { return order_; }

    const Matrix3Xd &pts() const { return pts_; }

    const ArrayXd &radii() const {
      return kind_ == SurfaceKind::kSas ? sar_ : r_;
    }

    const ArrayXd &sar() const { return sar_; }

    const ArrayXd &r() const { return r_; }

    const ConvexPatches &convex() const { return convex_; }

    const ToroidalPatches &toroidal() const { return toroidal_; }

    const ConcavePatches &concave() const { return concave_; }

    const ArrayXd &area(SurfacePatch patch) const;

    double total_area(SurfacePatch patch) const;

    double total_area() const;

  private:
    SurfaceKind kind_;
    double rp_;

    Matrix3Xd pts_;
    ArrayXd sar_, r_;
    ArrayXi order_;

    ConvexPatches convex_;
    ToroidalPatches toroidal_;
    ConcavePatches concave_;
  };
}  // namespace internal

/**
 * Dots on a surface with their outward normals (into the solvent); `probes()`
 * gives the centre of the probe touching the surface at each dot. `atom()`
 * is the ORIGINAL atom index owning each dot; the dots come in the blocks of
 * `SurfacePatch`, block `p` being `[offset(p), offset(p) + count(p))`.
 * Weights are equal within a patch and sum to its analytic area; patches
 * that received no dot add their area to `dropped_area()`.
 */
class SurfaceDots {
public:
  SurfaceDots() = default;

  SurfaceDots(SurfaceKind kind, double rp, Matrix3Xd &&pts, Matrix3Xd &&nrm,
              ArrayXd &&area, ArrayXi &&atom, internal::OffsetTable &&patch,
              double dropped_area);

  SurfaceKind kind() const { return kind_; }

  double rp() const { return rp_; }

  int n() const { return static_cast<int>(area_.size()); }

  int offset(SurfacePatch patch) const {
    return patch_[static_cast<int>(patch)];
  }

  int count(SurfacePatch patch) const {
    return patch_.degree(static_cast<int>(patch));
  }

  const Matrix3Xd &pts() const { return pts_; }

  const Matrix3Xd &nrm() const { return nrm_; }

  const ArrayXd &area() const { return area_; }

  const ArrayXi &atom() const { return atom_; }

  double dropped_area() const { return dropped_area_; }

  Matrix3Xd probes() const;

  /**
   * Drop every dot not in `mask`, in place.
   */
  void keep(const ArrayXb &mask);

  SurfaceDots subset(const ArrayXb &mask) const;

private:
  Matrix3Xd pts_, nrm_;
  ArrayXd area_;
  ArrayXi atom_;
  internal::OffsetTable patch_;
  SurfaceKind kind_;
  double rp_;
  double dropped_area_ = 0;
};

/**
 * @brief Parameters of the shape complementarity statistic.
 *
 * `rp` is the probe radius (angstroms) the solvent-excluded surfaces are
 * built with, `density` the number of surface dots per square angstrom. An
 * atom takes part in the interface when its centre lies within `sep` of an
 * atom of the other side; a dot is buried when its probe centre lies inside a
 * solvent-accessible sphere of the other side, and a buried dot within `band`
 * of an exposed dot of its own side is trimmed as peripheral. Each remaining
 * dot scores `-(n · n') exp(-weight d²)` against the nearest remaining dot of
 * the other side, clamped to `[-clamp, clamp]`.
 */
struct ScParams {
  double rp = 1.7;
  double density = 15;
  double weight = 0.5;
  double band = 1.5;
  double sep = 8;
  double clamp = 0.999;
};

/**
 * @brief Per-side summary of a shape complementarity calculation.
 *
 * `n_atoms` atoms, of which `n_active` take part in the interface; `n_dots`
 * sampled dots, `n_buried` of them buried and `n_trimmed` kept after the
 * peripheral trim, covering `trimmed_area` square angstroms. `d_median` and
 * `s_median` are the medians of the nearest-dot distance and of the score
 * over the kept dots; equal-distance ties take the smallest dot index.
 */
struct ScSide {
  int n_atoms, n_active, n_dots, n_buried, n_trimmed;
  double trimmed_area, d_median, s_median;
  SurfaceDots dots;
  ArrayXd distance, score;
};

/**
 * @brief Shape complementarity of an interface.
 *
 * `sc` is the statistic of Lawrence and Colman (1993): the mean of the two
 * sides' median scores. `distance` is the mean of their median nearest-dot
 * distances, `area` the total surface area of the kept dots.
 */
struct ScResult {
  ScSide sides[2];
  double sc, distance, area;
};

namespace internal {

  /**
   * Atoms ordered `[active | need | shell | occluders]`, the occluders being
   * the kept atoms near a sphere with caps, with contained spheres and the
   * other occluders dropped: `order` maps new to old indices, spheres
   * `< n_active` own
   * surface, `< n_solve` get arrangements, `< n_enum` get caps. `g` is the
   * forward overlap graph on new indices (`row(i)` = sorted `j > i`) over all
   * kept spheres, decided exactly for the lifted heights `t = √(wmax − sar²)`
   * (the near band only ranks atoms); pair `q` carries a circle iff
   * `i < n_enum`, and `d[q]` is its centre distance. `r` are the atom radii
   * as given, `sar = r + rp` the accessible ones for the probe radius `rp`,
   * `rmax` the largest `sar` of the kept atoms.
   */
  struct SurfaceSpheres {
    Matrix3Xd pts;
    ArrayXd r;
    ArrayXd sar, sar2;
    ArrayXd t;
    double wmax;
    double rp = -1;
    double rmax;
    ArrayXi order;
    CSR g;
    ArrayXd d;
    int n_active;
    int n_solve;
    int n_enum;
  };

  extern std::optional<SurfaceSpheres> prepare(ConstRef<Matrix3Xd> pts,
                                               ConstRef<ArrayXd> r,
                                               ConstRef<ArrayXb> active,
                                               double rp);

  /**
   * Buffers of one `SurfaceSampler::sample` call, sized once for the largest
   * lattice, cap list, probe contact list and saddle row; `lattices` caches
   * the Fibonacci lattice per dot count.
   */
  struct SampleScratch {
    absl::flat_hash_map<int, Matrix3Xd> lattices;
    ArrayXb outside;
    MatrixX3d axt;
    Matrix3Xd kept, contacts;
    ArrayXd big, r;
    MatrixXd depth;
    ArrayXd sin_edge, beta, area;
    ArrayXi k_phi;
  };

  /**
   * Buffers of one `ShapeComplementarity::compute` call: the neighbour query
   * outputs and the per-dot statistics before their medians.
   */
  struct ScScratch {
    std::vector<int> idxs;
    std::vector<double> distsq;
    VoxelGrid::Scratch grid;
    std::vector<double> d, s;
  };
}  // namespace internal

/**
 * Analytic SAS and SES of a set of spheres. Active atoms own surface, the
 * rest only occlude; the inputs must outlive the object. `build_sas` stops
 * after the SAS, `build_ses` builds both and keeps the SES.
 */
class AnalyticSurface {
public:
  AnalyticSurface(ConstRef<Matrix3Xd> pts, ConstRef<ArrayXd> radii);

  /**
   * False when the inputs are rejected (see `internal::prepare`). May be
   * called again with another mask; storage only grows.
   */
  ABSL_MUST_USE_RESULT bool initialize(double rp);
  ABSL_MUST_USE_RESULT bool initialize(double rp, ConstRef<ArrayXb> active);

  bool initialized() const { return rp() > 0; }

  internal::SurfaceGeometry build_sas() const;

  internal::SurfaceGeometry build_ses() const;

  std::pair<internal::SurfaceGeometry, internal::SurfaceGeometry>
  build_both() const;

  double rp() const { return ss_.rp; }

  /**
   * Kept spheres: those within reach of an active one.
   */
  int n_atoms() const { return static_cast<int>(ss_.order.size()); }

  int n_active() const { return ss_.n_active; }

  const ArrayXi &order() const { return ss_.order; }

private:
  Eigen::Ref<const Matrix3Xd> pts_;
  Eigen::Ref<const ArrayXd> radii_;
  internal::SurfaceSpheres ss_;
};

/**
 * Dots at a fixed density; lattices and scratch persist across calls.
 */
class SurfaceSampler {
public:
  explicit SurfaceSampler(double density);

  double density() const { return density_; }

  SurfaceDots sample(const internal::SurfaceGeometry &geo);

private:
  double density_;
  internal::SampleScratch scratch_;
};

/**
 * Shape complementarity of Lawrence and Colman (1993). The atoms within `sep`
 * of the other side own the interface surface, the rest only occlude; a mask
 * narrows the owners further. Scratch persists across calls.
 */
class ShapeComplementarity {
public:
  explicit ShapeComplementarity(ScParams params = {});

  const ScParams &params() const { return params_; }

  std::optional<ScResult> compute(ConstRef<Matrix3Xd> pts_a,
                                  ConstRef<ArrayXd> radii_a,
                                  ConstRef<Matrix3Xd> pts_b,
                                  ConstRef<ArrayXd> radii_b);

  std::optional<ScResult>
  compute(ConstRef<Matrix3Xd> pts_a, ConstRef<ArrayXd> radii_a,
          ConstRef<ArrayXb> active_a, ConstRef<Matrix3Xd> pts_b,
          ConstRef<ArrayXd> radii_b, ConstRef<ArrayXb> active_b);

private:
  ScParams params_;
  SurfaceSampler sampler_;
  internal::ScScratch scratch_;
};

/**
 * @brief Shape complementarity statistic of Lawrence and Colman (1993)
 *        between two sets of spheres.
 *
 * @param pts_a Atom centres of the first side.
 * @param radii_a Van der Waals radii of the first side; the probe radius is
 *        added internally.
 * @param pts_b Atom centres of the second side.
 * @param radii_b Van der Waals radii of the second side.
 * @param params Probe radius, dot density and the statistic's parameters.
 * @return The statistic with its per-side summaries, or nullopt when the
 *         surface of either side could not be prepared or no interface dot of
 *         either side survives trimming.
 */
extern std::optional<ScResult>
shape_complementarity(ConstRef<Matrix3Xd> pts_a, ConstRef<ArrayXd> radii_a,
                      ConstRef<Matrix3Xd> pts_b, ConstRef<ArrayXd> radii_b,
                      const ScParams &params = {});

/**
 * @brief Shape complementarity statistic of Lawrence and Colman (1993)
 *        between two sets of spheres, with the interface owners restricted.
 *
 * @param pts_a Atom centres of the first side.
 * @param radii_a Van der Waals radii of the first side; the probe radius is
 *        added internally.
 * @param active_a Atoms of the first side allowed to own interface surface;
 *        the others only occlude. An owner must still lie within `sep` of the
 *        other side.
 * @param pts_b Atom centres of the second side.
 * @param radii_b Van der Waals radii of the second side.
 * @param active_b Atoms of the second side allowed to own interface surface.
 * @param params Probe radius, dot density and the statistic's parameters.
 * @return The statistic with its per-side summaries, or nullopt when the
 *         surface of either side could not be prepared or no interface dot of
 *         either side survives trimming.
 */
extern std::optional<ScResult>
shape_complementarity(ConstRef<Matrix3Xd> pts_a, ConstRef<ArrayXd> radii_a,
                      ConstRef<ArrayXb> active_a, ConstRef<Matrix3Xd> pts_b,
                      ConstRef<ArrayXd> radii_b, ConstRef<ArrayXb> active_b,
                      const ScParams &params = {});
}  // namespace nuri

#endif /* NURI_DESC_SURFACE_H_ */
