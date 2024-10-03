//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/optim.h"

#include <cstdlib>

#include "nuri/utils.h"

namespace nuri {
namespace internal {
  LbfgsbActiveOut lbfgsb_active(ArrayXd &x, const LbgfsbBounds &bounds) {
    LbfgsbActiveOut ret { ArrayXi::Zero(x.size()) };

    for (int i = 0; i < x.size(); ++i) {
      if (bounds.has_lb(i) && x[i] < bounds.lb(i)) {
        x[i] = bounds.lb(i);
        ret.projected = true;
      }

      if (bounds.has_ub(i) && x[i] > bounds.ub(i)) {
        x[i] = bounds.ub(i);
        ret.projected = true;
      }

      if (!bounds.is_boxed(i))
        ret.boxed = false;

      if (!bounds.has_bound(i)) {
        ret.iwhere[i] = -1;
        continue;
      }

      ret.constrained = true;
      if (bounds.is_boxed(i) && bounds.ub(i) - bounds.lb(i) <= 0.0)
        ret.iwhere[i] = 3;
    }

    return ret;
  }

  double lbfgsb_projgr(const ArrayXd &x, const ArrayXd &gx,
                       const LbgfsbBounds &bounds) {
    double norm = 0.0;

    for (int i = 0; i < gx.size(); ++i) {
      double gxi = gx[i];

      if (gxi < 0) {
        if (bounds.has_ub(i)) {
          gxi = nuri::max(x[i] - bounds.ub(i), gxi);
        }
      } else if (bounds.has_lb(i)) {
        gxi = nuri::min(x[i] - bounds.lb(i), gxi);
      }

      norm = nuri::max(norm, std::abs(gxi));
    }

    return norm;
  }

  bool lbfgsb_bmv(VectorXd &p, const VectorXd &v, const MatrixXd &sy,
                  const MatrixXd &wt) {
    const auto col = sy.cols();
    ABSL_DCHECK(col > 0);

    if ((sy.diagonal().array() == 0).any()
        || (wt.diagonal().array() == 0).any())
      return false;

    ArrayXd smul =
        v.head(col - 1).array() / sy.diagonal().head(col - 1).array();
    for (int i = 0; i < col; ++i) {
      double ssum =
          (sy.row(i).head(i).array() * smul.head(i).transpose()).sum();
      p[col + i] = v[col + i] + ssum;
    }
    wt.triangularView<Eigen::Upper>().transpose().solveInPlace(p.tail(col));
    wt.triangularView<Eigen::Upper>().solveInPlace(p.tail(col));

    p.head(col) = -v.head(col).array() / sy.diagonal().array();
    for (int i = 0; i < col; ++i) {
      double ssum =
          sy.col(i).tail(col - i - 1).dot(p.tail(col - i - 1)) / sy(i, i);
      p[i] += ssum;
    }

    return true;
  }

  namespace {
    struct CauchyFindBreaksOut {
      std::vector<std::pair<int, double>> breaks;
      bool any_free = false;
      bool bounded = true;
    };

    CauchyFindBreaksOut cauchy_find_breaks(ArrayXi &iwhere, const ArrayXd &x,
                                           const ArrayXd &gx,
                                           const LbgfsbBounds &bounds) {
      CauchyFindBreaksOut ret;

      for (int i = 0; i < x.size(); ++i) {
        const double neg_gxi = -gx[i];

        if (iwhere[i] != 3 && iwhere[i] != -1) {
          iwhere[i] = 0;

          if (bounds.has_lb(i)) {
            if (x[i] <= bounds.lb(i) && neg_gxi <= 0)
              iwhere[i] = 1;
          }

          if (bounds.has_ub(i)) {
            if (x[i] >= bounds.ub(i) && neg_gxi >= 0)
              iwhere[i] = 2;
          }

          if (neg_gxi == 0)  // NOLINT(clang-diagnostic-float-equal)
            iwhere[i] = -3;
        }

        if (iwhere[i] != 0 && iwhere[i] != -1)
          continue;

        if (bounds.has_lb(i) && neg_gxi < 0) {
          ret.breaks.push_back({ i, (x[i] - bounds.lb(i)) / -neg_gxi });
        } else if (bounds.has_ub(i) && neg_gxi > 0) {
          ret.breaks.push_back({ i, (bounds.ub(i) - x[i]) / neg_gxi });
        } else {
          ret.any_free = true;
          if (neg_gxi != 0)  // NOLINT(clang-diagnostic-float-equal)
            ret.bounded = false;
        }
      }

      return ret;
    }

    std::pair<bool, bool> cauchy_handle_breaks(
        ArrayXd &xcp, ArrayXi &iwhere, VectorXd &p, VectorXd &v, VectorXd &c,
        VectorXd &d, double &f1, double &f2, double &dtm, double &tsum,
        std::vector<std::pair<int, double>> &&breaks, const ArrayXd &x,
        const LbgfsbBounds &bounds, const MatrixXd &ws, const MatrixXd &wy,
        const MatrixXd &sy, const MatrixXd &wt, const double theta,
        const double f2_org, const bool bounded) {
      if (breaks.empty())
        return { true, false };

      const auto nbreaks = breaks.size();
      const auto col = wy.cols();

      auto cmp = [](std::pair<int, double> lhs, std::pair<int, double> rhs) {
        return lhs.second > rhs.second;
      };
      internal::ClearablePQ<std::pair<int, double>, decltype(cmp)> pq(
          cmp, std::move(breaks));
      VectorXd wbp(2 * col);

      double tj = 0;
      while (!pq.empty()) {
        int ibp;

        double tj0 = tj;
        std::tie(ibp, tj) = pq.pop_get();

        double dt = tj - tj0;
        if (dtm < dt)
          return { true, false };

        tsum += dt;
        double dibp = d[ibp];
        d[ibp] = 0;
        double zibp;
        if (dibp > 0) {
          zibp = bounds.ub(ibp) - x[ibp];
          xcp[ibp] = bounds.ub(ibp);
          iwhere[ibp] = 2;
        } else {
          zibp = bounds.lb(ibp) - x[ibp];
          xcp[ibp] = bounds.lb(ibp);
          iwhere[ibp] = 1;
        }

        if (pq.empty() && nbreaks == x.size()) {
          dtm = dt;
          return { true, true };
        }

        double dibp2 = dibp * dibp;
        f1 += dt * f2 + dibp2 - theta * dibp * zibp;
        f2 -= theta * dibp2;
        if (col > 0) {
          wbp.head(col) = wy.row(ibp);
          wbp.tail(col) = theta * ws.row(ibp);
          if (!lbfgsb_bmv(v, wbp, sy, wt))
            return { false, false };

          c += dt * p;
          double wmc = c.dot(v), wmp = p.dot(v), wmw = wbp.dot(v);
          p -= dibp * wbp;
          f1 += dibp * wmc;
          f2 += dibp * 2 * wmp - dibp2 * wmw;
        }

        if (bounded) {
          f1 = f2 = dtm = 0;
          return { true, false };
        }

        f2 = nuri::max(kEpsMach * f2_org, f2);
        dtm = -f1 / f2;
      }

      return { true, false };
    }
  }  // namespace

  bool lbfgsb_cauchy(ArrayXd &xcp, ArrayXi &iwhere, VectorXd &p, VectorXd &v,
                     VectorXd &c, const ArrayXd &x, const ArrayXd &gx,
                     const LbgfsbBounds &bounds, const MatrixXd &ws,
                     const MatrixXd &wy, const MatrixXd &sy, const MatrixXd &wt,
                     const double sbgnrm, const double theta) {
    const auto col = wy.cols();

    xcp = x;
    if (sbgnrm <= 0)
      return true;

    auto [breaks, any_free, bounded] =
        cauchy_find_breaks(iwhere, x, gx, bounds);
    if (breaks.empty() && !any_free)
      return true;

    VectorXd d = (iwhere != 0 && iwhere != -1).select(0, -gx);
    p.head(col).noalias() = wy.transpose() * d;
    p.tail(col).noalias() = theta * ws.transpose() * d;

    double f1 = -d.squaredNorm();
    double f2 = -theta * f1;
    const double f2_org = f2;
    if (col > 0) {
      if (!lbfgsb_bmv(v, p, sy, wt))
        return false;
      f2 -= v.dot(p);
    }

    c.setZero();

    double dtm = -f1 / f2, tsum = 0;
    auto [success, z_is_gcp] = cauchy_handle_breaks(
        xcp, iwhere, p, v, c, d, f1, f2, dtm, tsum, std::move(breaks), x,
        bounds, ws, wy, sy, wt, theta, f2_org, bounded);
    if (!success)
      return false;

    if (!z_is_gcp) {
      dtm = nonnegative(dtm);
      tsum += dtm;
      xcp += tsum * d.array();
    }

    if (col > 0)
      c += dtm * p;

    return true;
  }
}  // namespace internal
}  // namespace nuri

// #include "nuri/algo/optim.h"
