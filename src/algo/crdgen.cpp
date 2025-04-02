//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/crdgen.h"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/algo/optim.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  using constants::kCos100;
  using constants::kCos102;
  using constants::kCos112;
  using constants::kCos115;
  using constants::kCos125;
  using constants::kCos175;
  using constants::kCos75;

  constexpr double kVdwRadDownscale = 0.85;
  constexpr double kMaxInterAtomDist = 5.0;

  class RandomSource {
  public:
    RandomSource(int seed): seed_(seed) { }

    void ensure_seeded() {
      if (seeded_)
        return;

      rng.seed(seed_);
      seeded_ = true;
    }

    double uniform_real(double lo, double hi) const {
      ABSL_DCHECK(seeded_);
      return std::uniform_real_distribution<double>(lo, hi)(rng);
    }

  private:
    inline static thread_local std::mt19937 rng;

    int seed_;
    bool seeded_ = false;
  };

  class DistanceBounds {
  public:
    DistanceBounds(const int n): bounds_(n, n) {
      bounds_.matrix().diagonal().setZero();
    }

    Array2Xd bsq_inv() const;

    void fill_trial_distances(MatrixXd &dists, RandomSource &rng) {
      if (init_) {
        fill_trial_distances_impl<true>(dists, rng);
        init_ = false;
      } else {
        rng.ensure_seeded();
        fill_trial_distances_impl<false>(dists, rng);
      }

      ABSL_DVLOG(1) << "trial distances:\n" << dists;
    }

    int n() const { return static_cast<int>(bounds_.cols()); }

    double &ub(int i, int j) {
      ABSL_DCHECK(i <= j);
      return bounds_(j, i);
    }

    double ub(int i, int j) const {
      ABSL_DCHECK(i <= j);
      return bounds_(j, i);
    }

    double &lb(int i, int j) {
      ABSL_DCHECK(i <= j);
      return bounds_(i, j);
    }

    double lb(int i, int j) const {
      ABSL_DCHECK(i <= j);
      return bounds_(i, j);
    }

    template <int offset = 0>
    auto lb_head(int i) {
      static_assert(offset <= 1);
      return bounds_.col(i).head(i + offset);
    }

    template <int offset = 0>
    auto lb_tail(int i) {
      static_assert(offset <= 1);
      return bounds_.row(i).tail(n() - i - 1 + offset).transpose();
    }

    template <int offset = 0>
    auto ub_head(int i) {
      static_assert(offset <= 1);
      return bounds_.row(i).head(i + offset).transpose();
    }

    template <int offset = 0>
    auto ub_tail(int i) {
      static_assert(offset <= 1);
      return bounds_.col(i).tail(n() - i - 1 + offset);
    }

    template <int offset = 0>
    auto lb_head(int i) const {
      static_assert(offset <= 1);
      return bounds_.col(i).head(i + offset);
    }

    template <int offset = 0>
    auto ub_head(int i) const {
      static_assert(offset <= 1);
      return bounds_.row(i).head(i + offset).transpose();
    }

    const ArrayXXd &data() const { return bounds_; }

    // NOLINTNEXTLINE(clang-diagnostic-unused-function)
    friend std::ostream &operator<<(std::ostream &os,
                                    const DistanceBounds &db) {
      return os << db.bounds_.transpose();
    }

  private:
    // Havel's distribution function:
    // Eq. 43, Distance Geometry: Theory, Algorithms, and Chemical Applications.
    // In Encycl. Comput. Chem., 1998, p. 731.
    template <bool first>
    void fill_trial_distances_impl(MatrixXd &dists,
                                   const RandomSource &rng) const {
      dists.diagonal().setZero();

      for (int i = 0; i < n() - 1; ++i) {
        for (int j = i + 1; j < n(); ++j) {
          double luq = lb(i, j) / ub(i, j);
          luq *= luq;
          luq *= luq;

          double dq;
          if constexpr (first) {
            dq = (1 + luq) * 0.5;
          } else {
            dq = rng.uniform_real(luq, 1.0);
          }
          dists(i, j) = dists(j, i) = ub(i, j) * std::sqrt(std::sqrt(dq));
        }
      }
    }

    // lower triangle is upper bound, upper triangle is lower bound
    ArrayXXd bounds_;
    bool init_ = true;
  };

  Array2Xd DistanceBounds::bsq_inv() const {
    const auto nc2 = n() * (n() - 1) / 2;
    Array2Xd bsq(2, nc2);

    for (int i = 1, k = 0; i < n(); k += i, ++i) {
      bsq.row(0).segment(k, i) = lb_head(i).transpose().square().inverse();
      bsq.row(1).segment(k, i) = ub_head(i).transpose().square().inverse();
    }

    return bsq;
  }

  struct PQEntry {
    int curr;
    int prev;
    double length;
  };

  // NOLINTNEXTLINE(clang-diagnostic-unused-function)
  bool operator<(PQEntry lhs, PQEntry rhs) {
    return lhs.length < rhs.length;
  }

  void set_constraints(const Molecule &mol, DistanceBounds &bounds,
                       ArrayXd &radii) {
    const int n = mol.num_atoms();

    const double max_upper = n * kMaxInterAtomDist;
    for (int i = 0; i < n; ++i) {
      bounds.lb_head(i).setZero();
      bounds.ub_tail(i) = max_upper;
    }

    for (int i = 0; i < n; ++i)
      radii[i] = mol.atom(i).data().element().covalent_radius();

    ArrayXd bds(mol.num_bonds());
    for (auto bond: mol.bonds()) {
      int src = bond.src().id(), dst = bond.dst().id();
      double pred_bond_len = radii[src] + radii[dst];

      switch (bond.data().order()) {
      case constants::kOtherBond:
        ABSL_LOG(INFO) << "unknown bond order, assuming single bond";
        ABSL_FALLTHROUGH_INTENDED;
      case constants::kSingleBond:
        if (bond.data().is_conjugated())
          pred_bond_len *= 0.9;
        break;
      case constants::kDoubleBond:
        pred_bond_len *= 0.87;
        break;
      case constants::kTripleBond:
        pred_bond_len *= 0.78;
        break;
      case constants::kQuadrupleBond:
        pred_bond_len *= 0.7;
        break;
      case constants::kAromaticBond:
        pred_bond_len *= 0.9;
        break;
      }

      bounds.lb(src, dst) = bounds.ub(src, dst) = bds[bond.id()] =
          pred_bond_len;
    }

    ABSL_DVLOG(2) << "after bond length constraints:\n" << bounds;

    ArrayXd bdsq = bds.square();
    for (auto atom: mol) {
      if (atom.degree() < 2)
        continue;

      double cos_upper, cos_lower;
      switch (atom.data().hybridization()) {
      case constants::kUnbound:
      case constants::kTerminal:
        ABSL_LOG(WARNING) << "inconsistent hybridization state for atom "
                          << atom.id() << ": " << atom.data().hybridization();
        ABSL_FALLTHROUGH_INTENDED;
      case constants::kSP:
        cos_upper = -1;  // kCos180
        cos_lower = kCos175;
        break;
      case constants::kSP2:
        cos_upper = kCos125;
        cos_lower = kCos115;
        break;
      case constants::kOtherHyb:
        ABSL_LOG(WARNING) << "unknown hybridization state for atom "
                          << atom.id() << "; assuming sp3";
        ABSL_FALLTHROUGH_INTENDED;
      case constants::kSP3:
        cos_upper = kCos112;
        cos_lower = kCos102;  // cyclopentane ~104.5 degrees
        break;
      case constants::kSP3D:
        cos_upper = kCos125;
        cos_lower = kCos75;
        break;
      case constants::kSP3D2:
        cos_upper = kCos100;
        cos_lower = kCos75;
        break;
      }

      for (int i = 0; i < atom.degree() - 1; ++i) {
        for (int j = i + 1; j < atom.degree(); ++j) {
          const int bi = atom[i].eid(), bj = atom[j].eid();
          const double bdsqsum = bdsq[bi] + bdsq[bj];
          const double bdmul = 2 * bds[bi] * bds[bj];

          auto [ni, nj] = nuri::minmax(atom[i].dst().id(), atom[j].dst().id());
          const double ub = bounds.ub(ni, nj) = nuri::min(
              bounds.ub(ni, nj), std::sqrt(bdsqsum - bdmul * cos_upper));
          bounds.lb(ni, nj) = nuri::clamp(
              std::sqrt(bdsqsum - bdmul * cos_lower), bounds.lb(ni, nj), ub);
        }
      }
    }

    ABSL_DVLOG(2) << "after bond angle constraints:\n" << bounds;

    for (int i = 0; i < n; ++i)
      radii[i] = mol.atom(i).data().element().vdw_radius();
    radii *= kVdwRadDownscale;

    for (int i = 0; i < n; ++i) {
      bounds.lb_head(i) =
          (bounds.lb_head(i) > 0)
              .select(bounds.lb_head(i), radii[i] + radii.head(i));
    }
  }

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  std::vector<int> update_upper_bounds(DistanceBounds &bounds,
                                       const Molecule &mol, ArrayXd &urev) {
    const int n = mol.num_atoms();

    std::vector<int> roots;
    roots.reserve(2ULL * mol.num_fragments());

    internal::ClearablePQ<PQEntry> pq;
    ArrayXb visited = ArrayXb::Zero(n);

    for (int root = 0; root < n; ++root) {
      if (visited[root])
        continue;

      roots.push_back(root);
      visited[root] = true;

      for (auto nei: mol.atom(root)) {
        int next = nei.dst().id();
        pq.push({ next, root, bounds.ub(root, next) });
      }

      while (!pq.empty()) {
        auto [curr, prev, curr_len] = pq.pop_get();
        if (curr_len > bounds.ub(root, curr))
          continue;

        visited[curr] = true;

        const double prev_len = bounds.ub(root, prev);
        for (auto nei: mol.atom(curr)) {
          int next = nei.dst().id();
          if (visited[next])
            continue;

          auto [ni, nk1] = nuri::minmax(prev, next);
          auto [nj, nk2] = nuri::minmax(curr, next);
          double new_ub = nuri::min(prev_len + bounds.ub(ni, nk1),
                                    curr_len + bounds.ub(nj, nk2));
          double &ref_ub = bounds.ub(root, next);
          if (new_ub <= ref_ub) {
            ref_ub = new_ub;
            pq.push({ next, curr, new_ub });
          }
        }
      }
    }

    // i < j < k
    for (int i: roots) {
      for (int j = i + 1; j < n - 1; ++j) {
        bounds.ub_tail(j) = bounds.ub_tail(j).min(
            bounds.ub(i, j) + bounds.ub_tail(i).tail(n - j - 1));
      }
    }

    ABSL_DVLOG(2) << "after forward upper bound update:\n" << bounds;

    const int mid = static_cast<int>(roots.size());
    visited.setZero();

    for (int root = n - 1; root >= 0; --root) {
      if (visited[root])
        continue;

      roots.push_back(root);
      visited[root] = true;

      auto u_r = urev.head(root + 1);
      u_r.setConstant(n * kMaxInterAtomDist);
      u_r[root] = 0;

      for (auto nei: mol.atom(root)) {
        int next = nei.dst().id();
        u_r[next] = bounds.ub(next, root);
        pq.push({ next, root, bounds.ub(next, root) });
      }

      while (!pq.empty()) {
        auto [curr, prev, curr_len] = pq.pop_get();
        if (curr_len > u_r[curr])
          continue;

        visited[curr] = true;

        const double prev_len = u_r[prev];
        for (auto nei: mol.atom(curr)) {
          int next = nei.dst().id();
          if (visited[next])
            continue;

          auto [ni, nk1] = nuri::minmax(prev, next);
          auto [nj, nk2] = nuri::minmax(curr, next);
          double new_ub = nuri::min(prev_len + bounds.ub(ni, nk1),
                                    curr_len + bounds.ub(nj, nk2));
          double &ref_ub = u_r[next];
          if (new_ub < ref_ub) {
            ref_ub = new_ub;
            pq.push({ next, curr, new_ub });
          }
        }
      }

      bounds.ub_head(root) = bounds.ub_head(root).min(u_r.head(root));
    }

    // j < k < i
    for (int r = mid; r < roots.size(); ++r) {
      int i = roots[r];
      for (int j = 0; j < i - 1; ++j) {
        bounds.ub_tail(j).head(i - j - 1) =
            bounds.ub_tail(j).head(i - j - 1).min(
                bounds.ub(j, i) + bounds.ub_head(i).tail(i - j - 1));
      }
    }

    return roots;
  }

  void update_lower_bounds(DistanceBounds &bounds, const Molecule &mol,
                           const std::vector<int> &roots, ArrayXd &jb_buf) {
    const int n = mol.num_atoms();
    const int mid = mol.num_fragments();
    ABSL_DCHECK(2ULL * mid == roots.size());

    ArrayXi j_buf(n);

    for (int r = 0; r < mid; ++r) {
      int i = roots[r];

      auto js = j_buf.head(n - i - 1);
      absl::c_iota(js, i + 1);
      absl::c_sort(js, [&](int lhs, int rhs) {
        return bounds.ub(i, lhs) > bounds.ub(i, rhs);
      });

      // i < j, i < k, j != k
      auto u_j = jb_buf.head(n - i - 1);
      for (int j: js) {
        u_j.head(j - i - 1) = bounds.ub_head(j).tail(j - i - 1);
        u_j.tail(n - j) = bounds.ub_tail<1>(j);

        // u_j[j - i - 1] == 0, lb(i, j) - ub(j, j) == lb(i, j)
        // so, the following effectivecly calculates:
        //    clamp(lb(i, j), lb(i, k) - ub(j, k), ub(i, j))
        bounds.lb(i, j) =
            nuri::min((bounds.lb_tail(i) - u_j).maxCoeff(), bounds.ub(i, j));
      }

      // i < j < k
      for (int k = n - 1; k > i + 1; --k) {
        for (int j = k - 1; j > i; --j) {
          bounds.lb(j, k) =
              nuri::min(bounds.ub(j, k), std::max({
                                             bounds.lb(j, k),
                                             bounds.lb(i, j) - bounds.ub(i, k),
                                             bounds.lb(i, k) - bounds.ub(i, j),
                                         }));
        }
      }
    }

    ABSL_DVLOG(2) << "after forward lower bound update:\n" << bounds;

    for (int r = mid; r < roots.size(); ++r) {
      int i = roots[r];

      auto js = j_buf.head(i);
      absl::c_iota(js, 0);
      absl::c_sort(js, [&](int lhs, int rhs) {
        return bounds.ub(lhs, i) > bounds.ub(rhs, i);
      });

      // j < i, k < i, j != k
      auto u_j = jb_buf.head(i);
      for (int j: js) {
        u_j.head(j) = bounds.ub_head(j);
        u_j.tail(i - j) = bounds.ub_tail<1>(j).head(i - j);

        // Same here.
        bounds.lb(j, i) =
            nuri::min((bounds.lb_head(i) - u_j).maxCoeff(), bounds.ub(j, i));
      }

      // j < k < i
      for (int k = i - 1; k >= 1; --k) {
        for (int j = k - 1; j >= 0; --j) {
          bounds.lb(j, k) =
              nuri::min(bounds.ub(j, k), std::max({
                                             bounds.lb(j, k),
                                             bounds.lb(j, i) - bounds.ub(k, i),
                                             bounds.lb(k, i) - bounds.ub(j, i),
                                         }));
        }
      }
    }
  }

  DistanceBounds init_bounds(const Molecule &mol) {
    const int n = mol.num_atoms();

    DistanceBounds bounds(n);
    ArrayXd temp(n);

    set_constraints(mol, bounds, temp);
    ABSL_DVLOG(1) << "initial bounds matrix:\n" << bounds;

    std::vector roots = update_upper_bounds(bounds, mol, temp);
    ABSL_DVLOG(1) << "after upper bound update:\n" << bounds;

    update_lower_bounds(bounds, mol, roots, temp);
    ABSL_DVLOG(1) << "after lower bound update:\n" << bounds;

    return bounds;
  }

  // Havel's distance error function:
  // E3 in Distance Geometry in Molecular Modeling, 1994, Ch.6, p. 311.
  // NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
  double distance_error(MutRef<Array4Xd> &g, ConstRef<Array4Xd> x,
                        const Array2Xd &bsq_inv) {
    double ub_err = 0, lb_err = 0;

    for (int i = 1, k = 0; i < x.cols(); ++i) {
      for (int j = 0; j < i; ++j, ++k) {
        const double lbsq_inv = bsq_inv(0, k), ubsq_inv = bsq_inv(1, k);

        Array4d diff = x.col(i) - x.col(j);
        const double dsq = diff.matrix().squaredNorm();

        const double ub_term = nonnegative(dsq * ubsq_inv - 1);

        const double lb_inner_inv = 2 / (1 + dsq * lbsq_inv);
        const double lb_term = nonnegative(lb_inner_inv - 1);

        diff *= 4 * ubsq_inv * ub_term
                - 2 * (lb_inner_inv * lb_inner_inv) * lbsq_inv * lb_term;
        g.col(i) += diff;
        g.col(j) -= diff;

        ub_err += ub_term * ub_term;
        lb_err += lb_term * lb_term;
      }
    }

    return ub_err + lb_err;
  }

  // NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
  double extra_dimension_error(MutRef<Array4Xd> &g, ConstRef<Array4Xd> x) {
    g.row(3) += 2 * x.row(3);
    return x.row(3).square().sum();
  }

  struct RefTetrad {
    Array4i idxs;
    double v_lb;
    double v_ub;
  };

  double tetrad_volume_loss(MutRef<Array4Xd> &g, ConstRef<Array4Xd> x,
                            const RefTetrad &tetrad) {
    auto x3d = x.topRows(3);

    Matrix<double, 3, 4> zs = x3d(Eigen::all, tetrad.idxs);
    zs.leftCols<3>().colwise() -= zs.col(3);

    Matrix<double, 3, 4> grad;
    grad.col(0) = zs.col(1).cross(zs.col(2));

    double vol = zs.col(0).dot(grad.col(0));
    double vol_err;
    if (vol < tetrad.v_lb) {
      vol_err = vol - tetrad.v_lb;
    } else if (vol > tetrad.v_ub) {
      vol_err = vol - tetrad.v_ub;
    } else {
      return 0;
    }

    grad.col(1) = zs.col(2).cross(zs.col(0));
    grad.col(2) = zs.col(0).cross(zs.col(1));
    grad.leftCols<3>() *= 2 * vol_err;

    grad.col(3) = -grad.leftCols<3>().rowwise().sum();

    g.topRows(3)(Eigen::all, tetrad.idxs) += grad.array();

    return vol_err * vol_err;
  }

  // NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
  double volumetric_error(MutRef<Array4Xd> &g, ConstRef<Array4Xd> x,
                          const std::vector<RefTetrad> &tetrads) {
    double loss = 0;
    for (const auto &tetrad: tetrads)
      loss += tetrad_volume_loss(g, x, tetrad);
    return loss;
  }

  template <bool MinimizeFourth>
  double
  // NOLINTNEXTLINE(clang-diagnostic-unused-template)
  error_funcgrad(ArrayXd &ga, ConstRef<ArrayXd> xa, const Array2Xd &bsq_inv,
                 const std::vector<RefTetrad> &tetrads, const Eigen::Index n) {
    ga.setZero();

    MutRef<Array4Xd> g = ga.reshaped(4, n);
    ConstRef<Array4Xd> x = xa.reshaped(4, n);

    ABSL_DVLOG(3) << "current coordinates:\n" << x.transpose();

    const double e1 = distance_error(g, x, bsq_inv);
    const double e2 = volumetric_error(g, x, tetrads);
    if constexpr (!MinimizeFourth)
      return e1 + e2;

    const double e3 = extra_dimension_error(g, x);
    return e1 + e2 + e3;
  }

  Array4i chiral_tetrad_ids(Molecule::Atom atom) {
    ABSL_DCHECK(atom.degree() == 3 || atom.degree() == 4);

    Array4i idxs;
    for (int i = 0; i < atom.degree(); ++i)
      idxs[i] = atom[i].dst().id();
    if (atom.degree() == 3)
      idxs[3] = atom.id();
    return idxs;
  }

  constexpr double k6OverSqrt288 =
      0.3535533905932737622004221810524245196424179688442370182941699344;

  RefTetrad chiral_ref_tetrad(Molecule::Atom atom,
                              const DistanceBounds &bounds) {
    Array4i idxs = chiral_tetrad_ids(atom);

    Matrix4d bdsq = bounds.data()(idxs, idxs).cwiseAbs2();
    bdsq.diagonal().setZero();

    // Tetrad volume with Cayley-Menger determinant
    // Divide by sqrt(288) / 6 to be consistent with the error function which
    // omits the 1/6 factor of the tetrahedron volume
    Matrix<double, 5, 5> cm;
    cm.col(0).setOnes();
    cm.row(0).setOnes();
    cm.diagonal().setConstant(0);

    cm.bottomRightCorner<4, 4>() = bdsq.selfadjointView<Eigen::Upper>();
    double v_lb = std::sqrt(nonnegative(cm.determinant())) * k6OverSqrt288;

    cm.bottomRightCorner<4, 4>() = bdsq.selfadjointView<Eigen::Lower>();
    double v_ub = std::sqrt(nonnegative(cm.determinant())) * k6OverSqrt288;

    // Might be inconsistent due to incomplete triangle inequality smoothing
    std::tie(v_lb, v_ub) = nuri::minmax(v_lb, v_ub);

    // CW -> negative, CCW -> positive
    if (atom.data().is_clockwise()) {
      v_lb = -v_lb;
      v_ub = -v_ub;
      std::swap(v_lb, v_ub);
    }

    return { idxs, v_lb, v_ub };
  }

  RefTetrad planar_ref_tetrad(Molecule::Atom atom) {
    ABSL_DCHECK(atom.degree() == 3);

    Array4i idxs { atom[0].dst().id(), atom[1].dst().id(), atom[2].dst().id(),
                   atom.id() };
    return { idxs, -0.1, +0.1 };
  }

  template <bool init_random>
  bool generate_coords_impl(const Molecule &mol, Matrix3Xd &conf, int max_trial,
                            int seed) {
    const Eigen::Index n = mol.num_atoms();

    DistanceBounds bounds = init_bounds(mol);
    const auto bsq_inv = bounds.bsq_inv();

    std::vector<RefTetrad> tetrads;
    for (auto atom: mol) {
      if (atom.data().hybridization() == constants::kSP2
          && atom.degree() == 3) {
        ABSL_LOG_IF(INFO, atom.data().is_chiral())
            << "atom " << atom.id()
            << " has planar geometry but is marked chiral";
        tetrads.push_back(planar_ref_tetrad(atom));
      } else if (atom.data().is_chiral()) {
        if (atom.degree() < 3) {
          ABSL_LOG(INFO) << "chiral atom " << atom.id() << " has only "
                         << atom.degree() << " neighbors; skipping";
          continue;
        }
        tetrads.push_back(chiral_ref_tetrad(atom, bounds));
      }
    }

    // TODO(jnooree): handle coplanar groups

    Matrix4Xd trial(4, n);
    MatrixXd dists(n, n);

    LBfgs<internal::LBfgsImpl> optim(trial.reshaped().array(), {});

    auto first_fg = [&](ArrayXd &ga, const auto &xa) {
      return error_funcgrad<false>(ga, xa, bsq_inv, tetrads, n);
    };

    auto second_fg = [&](ArrayXd &ga, const auto &xa) {
      return error_funcgrad<true>(ga, xa, bsq_inv, tetrads, n);
    };

    RandomSource rng(seed);
    bool success = false;
    for (int iter = 0; iter < max_trial; ++iter) {
      if constexpr (init_random) {
        rng.ensure_seeded();
        for (int j = 0; j < n; ++j) {
          for (int i = 0; i < 4; ++i) {
            trial(i, j) = rng.uniform_real(static_cast<double>(-3 * n),
                                           static_cast<double>(3 * n));
          }
        }
      } else {
        bounds.fill_trial_distances(dists, rng);
        dists = dists.cwiseAbs2();
        if (!embed_distances_4d(trial, dists))
          continue;
      }

      ABSL_DVLOG(1) << "initial trial coordinates:\n" << trial.transpose();

      LbfgsResult res = optim.minimize(first_fg, 1e-3, 1e-6);
      if (res.code != LbfgsResultCode::kSuccess)
        continue;

      ABSL_DVLOG(1) << "after 4D minimization:\n" << trial.transpose();

      res = optim.minimize(second_fg);
      if (res.code != LbfgsResultCode::kSuccess)
        continue;

      ABSL_DVLOG(1) << "after 3D projection:\n" << trial.transpose();

      success = true;
      break;
    }
    if (!success)
      return false;

    conf = trial.topRows(3);
    return true;
  }
}  // namespace

bool generate_coords(const Molecule &mol, Matrix3Xd &conf, int max_trial,
                     int seed) {
  const Eigen::Index n = mol.num_atoms();

  if (n != conf.cols()) {
    ABSL_LOG(ERROR) << "size mismatch: " << n << " atoms in the molecule, but "
                    << conf.cols() << " columns in the matrix";
    return false;
  }

  if (n <= 4) {
    ABSL_LOG(INFO) << "too few atoms; randomly initializing trial coordinates";
    return generate_coords_impl<true>(mol, conf, max_trial, seed);
  }

  return generate_coords_impl<false>(mol, conf, max_trial, seed);
}
}  // namespace nuri
