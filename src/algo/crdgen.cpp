//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/crdgen.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/random/random.h>

#include "nuri/eigen_config.h"
#include "nuri/algo/optim.h"
#include "nuri/core/geometry.h"
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

  class DistanceBounds {
  public:
    DistanceBounds(const int n): bounds_(n, n) {
      bounds_.matrix().diagonal().setZero();
    }

    std::pair<ArrayXd, ArrayXd> lbsq_ubsq() const;

    // Havel's distribution function:
    // Eq. 43, Distance Geometry: Theory, Algorithms, and Chemical Applications.
    // In Encycl. Comput. Chem., 1998, p. 731.
    void fill_trial_distances(MatrixXd &dists, const double ubeta) const {
      dists.diagonal().setZero();

      const double lbeta = 1 - ubeta;
      ABSL_DCHECK(ubeta >= 0);
      ABSL_DCHECK(lbeta >= 0);

      for (int i = 0; i < n() - 1; ++i) {
        for (int j = i + 1; j < n(); ++j) {
          double ub_tet = ub(i, j);
          ub_tet *= ub_tet;
          ub_tet *= ub_tet;

          double lb_tet = lb(i, j);
          lb_tet *= lb_tet;
          lb_tet *= lb_tet;

          dists(i, j) = dists(j, i) =
              std::sqrt(std::sqrt(lbeta * lb_tet + ubeta * ub_tet));
        }
      }
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

  private:
    // lower triangle is upper bound, upper triangle is lower bound
    ArrayXXd bounds_;
  };

  std::pair<ArrayXd, ArrayXd> DistanceBounds::lbsq_ubsq() const {
    const auto nc2 = n() * (n() - 1) / 2;
    ArrayXd lbsq(nc2), ubsq(nc2);

    for (int i = 1, k = 0; i < n(); k += i, ++i) {
      lbsq.segment(k, i) = lb_head(i).square();
      ubsq.segment(k, i) = ub_head(i).square();
    }

    return { lbsq, ubsq };
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

    for (int i = 0; i < n; ++i)
      radii[i] = mol.atom(i).data().element().vdw_radius() * kVdwRadDownscale;

    double max_upper = n * kMaxInterAtomDist;
    for (int i = 0; i < n; ++i) {
      bounds.lb_head(i) = radii[i] + radii.head(i);
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

      do {
        auto [curr, prev, curr_len] = pq.pop_get();
        if (curr_len > bounds.ub(root, curr))
          continue;

        visited[curr] = true;

        const double prev_len = bounds.ub(root, prev);
        for (auto nei: mol.atom(curr)) {
          int next = nei.dst().id();
          if (visited[next])
            continue;

          auto [ni, nj] = nuri::minmax(prev, next);
          double new_ub = prev_len + bounds.ub(ni, nj);
          double &ref_ub = bounds.ub(root, next);
          if (new_ub <= ref_ub) {
            ref_ub = new_ub;
            pq.push({ next, curr, new_ub });
          }
        }
      } while (!pq.empty());
    }

    // i < j < k
    for (int i: roots) {
      for (int j = i + 1; j < n - 1; ++j) {
        bounds.ub_tail(j) = bounds.ub_tail(j).min(
            bounds.ub(i, j) + bounds.ub_tail(i).tail(n - j - 1));
      }
    }

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

      do {
        auto [curr, prev, curr_len] = pq.pop_get();
        if (curr_len > u_r[curr])
          continue;

        visited[curr] = true;

        const double prev_len = u_r[prev];
        for (auto nei: mol.atom(curr)) {
          int next = nei.dst().id();
          if (visited[next])
            continue;

          auto [ni, nj] = nuri::minmax(prev, next);
          double new_ub = prev_len + bounds.ub(ni, nj);
          double &ref_ub = u_r[next];
          if (new_ub < ref_ub) {
            ref_ub = new_ub;
            pq.push({ next, curr, new_ub });
          }
        }
      } while (!pq.empty());

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

    std::vector roots = update_upper_bounds(bounds, mol, temp);
    update_lower_bounds(bounds, mol, roots, temp);

    return bounds;
  }

  // Havel's distance error function:
  // E3 in Distance Geometry in Molecular Modeling, 1994, Ch.6, p. 311.
  // NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
  double distance_error(MutRef<Array4Xd> &g, ConstRef<Array4Xd> x,
                        Array4Xd &diffs, ArrayXd &t1, ArrayXd &t2,
                        const ArrayXd &lbsq, const ArrayXd &ubsq) {
    const auto n = x.cols();

    for (int i = 1, k = 0; i < n; ++i)
      for (int j = 0; j < i; ++j, ++k)
        diffs.col(k) = x.col(i) - x.col(j);

    t1 = diffs.matrix().colwise().squaredNorm().transpose();

    t2 = (t1 / ubsq - 1).max(0);
    for (int i = 1, k = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j, ++k) {
        Array4d grad = 4 / ubsq[k] * t2[k] * diffs.col(k);
        g.col(i) += grad;
        g.col(j) -= grad;
      }
    }
    const double ub_err = t2.square().sum();

    t2 = 1 + t1 / lbsq;
    t1 = (2 / t2 - 1).max(0);

    t2 = t2.square();
    for (int i = 1, k = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j, ++k) {
        Array4d grad = -8 * t1[k] / t2[k] / lbsq[k] * diffs.col(k);
        g.col(i) += grad;
        g.col(j) -= grad;
      }
    }
    const double lb_err = t1.square().sum();

    return ub_err + lb_err;
  }

  // NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
  double extra_dimension_error(MutRef<Array4Xd> &g, ConstRef<Array4Xd> x) {
    g.row(3) += 2 * x.row(3);
    return x.row(3).square().sum();
  }

  template <bool MinimizeFourth>
  // NOLINTNEXTLINE(clang-diagnostic-unused-template)
  double error_funcgrad(ArrayXd &ga, ConstRef<ArrayXd> xa, Array4Xd &diffs,
                        ArrayXd &t1, ArrayXd &t2,
                        const std::pair<ArrayXd, ArrayXd> &bounds_squared,
                        const int n) {
    ga.setZero();

    MutRef<Array4Xd> g = ga.reshaped(4, n);
    ConstRef<Array4Xd> x = xa.reshaped(4, n);

    const double e1 = distance_error(g, x, diffs, t1, t2, bounds_squared.first,
                                     bounds_squared.second);
    if constexpr (!MinimizeFourth)
      return e1;

    const double e2 = extra_dimension_error(g, x);
    return e1 + e2;
  }

  // NOLINTNEXTLINE(*-non-const-global-variables)
  thread_local absl::InsecureBitGen rng;
}  // namespace

bool generate_coords(const Molecule &mol, Matrix3Xd &conf, int max_trial) {
  const Eigen::Index n = mol.num_atoms();

  if (n != conf.cols()) {
    ABSL_LOG(ERROR) << "size mismatch: " << n << " atoms in the molecule, but "
                    << conf.cols() << " columns in the matrix";
    return false;
  }

  DistanceBounds bounds = init_bounds(mol);
  const std::pair bounds_squared = bounds.lbsq_ubsq();
  ABSL_DLOG(INFO) << "initial bounds:\n" << bounds.data() << "\n";

  Matrix4Xd trial(4, n);
  MatrixXd dists(mol.size(), mol.size());

  ArrayXi nbd = ArrayXi::Zero(4 * n);
  Array2Xd dummy_bds(2, 4 * n);
  LBfgsB optim(trial.reshaped().array(), { nbd, dummy_bds });

  const auto nc2 = bounds_squared.first.size();
  Array4Xd diffs(4, nc2);
  ArrayXd t1(nc2), t2(nc2);

  auto first_fg = [&](ArrayXd &ga, const auto &xa) {
    return error_funcgrad<false>(ga, xa, diffs, t1, t2, bounds_squared, n);
  };

  auto second_fg = [&](ArrayXd &ga, const auto &xa) {
    return error_funcgrad<true>(ga, xa, diffs, t1, t2, bounds_squared, n);
  };

  double beta = 0.5;
  bool success = false;

  for (int iter = 0; iter < max_trial;
       beta = absl::Uniform(absl::IntervalClosed, rng, 0.0, 1.0), ++iter) {
    bounds.fill_trial_distances(dists, beta);
    dists.cwiseAbs2();

    if (!embed_distances_4d(trial, dists))
      continue;

    LbfgsbResult res = optim.minimize(first_fg, 1e+10, 1e-3);
    if (res.code != LbfgsbResultCode::kSuccess)
      continue;

    res = optim.minimize(second_fg);
    if (res.code != LbfgsbResultCode::kSuccess)
      continue;

    success = true;
    break;
  }
  if (!success)
    return false;

  conf = trial.topRows(3);
  return true;
}
}  // namespace nuri
