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

      if (!bounds.has_both(i))
        ret.boxed = false;

      if (!bounds.has_bound(i)) {
        ret.iwhere[i] = -1;
        continue;
      }

      ret.constrained = true;
      if (bounds.has_both(i) && bounds.ub(i) - bounds.lb(i) <= 0.0)
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

  namespace {
    void formk_shift_wn1(MatrixXd &wn1) {
      const auto m = wn1.cols() / 2, mm1 = m - 1;

      //  Shift old part of WN1.
      wn1.topLeftCorner(mm1, mm1).triangularView<Eigen::Lower>() =
          wn1.block(1, 1, mm1, mm1).triangularView<Eigen::Lower>();
      wn1.block(m, m, mm1, mm1).triangularView<Eigen::Lower>() =
          wn1.block(m + 1, m + 1, mm1, mm1).triangularView<Eigen::Lower>();
      wn1.block(m, 0, mm1, mm1) = wn1.block(m + 1, 1, mm1, mm1);
    }

    void formk_add_new_wn1(MatrixXd &wn1, const Eigen::Ref<const ArrayXi> &free,
                           const Eigen::Ref<const ArrayXi> &bound,
                           const MatrixXd &ws, const MatrixXd &wy) {
      const auto m = wn1.cols() / 2, col = ws.cols(), cm1 = col - 1;

      if (m == col)
        formk_shift_wn1(wn1);

      // Put new rows in blocks (1,1), (2,1) and (2,2).
      wn1.row(cm1).head(col).noalias() =
          wy(free, cm1).transpose() * wy(free, Eigen::all);
      wn1.row(m + cm1).head(col).noalias() =
          ws(bound, cm1).transpose() * wy(bound, Eigen::all);
      wn1.row(m + cm1).segment(m, col).noalias() =
          ws(bound, cm1).transpose() * ws(bound, Eigen::all);

      // Put new column in block (2,1)
      wn1.col(cm1).segment(m, col).noalias() =
          ws(free, Eigen::all).transpose() * wy(free, cm1);
    }

    void formk_update_wn1(MatrixXd &wn1, const std::vector<int> &enter,
                          const std::vector<int> &leave, const MatrixXd &ws,
                          const MatrixXd &wy) {
      const auto m = wn1.cols() / 2, col = ws.cols();
      const auto upcl = nuri::min(m, col) - 1;

      // modify the old parts in blocks (1,1) and (2,2) due to changes
      // in the set of free variables.
      for (int j = 0; j < upcl; ++j) {
        wn1.col(j).segment(j, upcl - j).noalias() +=
            wy(enter, Eigen::all).transpose().middleRows(j, upcl - j)
                * wy(enter, j)
            - wy(leave, Eigen::all).transpose().middleRows(j, upcl - j)
                  * wy(leave, j);
      }
      for (int j = 0; j < upcl; ++j) {
        wn1.col(m + j).segment(m + j, upcl - j).noalias() -=
            ws(enter, Eigen::all).transpose().middleRows(j, upcl - j)
                * ws(enter, j)
            - ws(leave, Eigen::all).transpose().middleRows(j, upcl - j)
                  * ws(leave, j);
      }

      // Modify the old parts in block (2,1)
      for (int j = 0; j < upcl; ++j) {
        wn1.col(j).segment(m, j + 1).noalias() +=
            ws(enter, Eigen::all).transpose().topRows(j + 1) * wy(enter, j)
            - ws(leave, Eigen::all).transpose().topRows(j + 1) * wy(leave, j);

        wn1.col(j).segment(m + j + 1, upcl - j - 1).noalias() -=
            ws(enter, Eigen::all).transpose().middleRows(j + 1, upcl - j - 1)
                * wy(enter, j)
            - ws(leave, Eigen::all).transpose().middleRows(j + 1, upcl - j - 1)
                  * wy(leave, j);
      }
    }

    void formk_prepare_wn(MatrixXd &wnt, const MatrixXd &wn1,
                          const MatrixXd &sy, const double theta) {
      const auto m = wn1.cols() / 2, col = sy.cols();

      for (int j = 0; j < col; ++j) {
        wnt.col(j).segment(j, col - j) = wn1.col(j).segment(j, col - j) / theta;

        wnt.col(j).segment(col, j + 1) = wn1.col(j).segment(m, j + 1);
        wnt.col(j).segment(col + j + 1, col - j - 1) =
            -wn1.col(j).segment(m + j + 1, col - j - 1);
      }

      wnt.diagonal().head(col) += sy.diagonal();

      for (int j = 0; j < col; ++j) {
        wnt.col(col + j).segment(col + j, col - j) =
            wn1.col(m + j).segment(m + j, col - j) * theta;
      }
    }

    bool formk_factorize_wn(MatrixXd &wnt, Eigen::LLT<MatrixXd> &llt) {
      const auto col = wnt.cols() / 2;

      //    first Cholesky factor (1,1) block of wn to get LL'
      //                      with L stored in the lower triangle of wn_t.
      llt.compute(wnt.topLeftCorner(col, col));
      if (llt.info() != Eigen::Success)
        return false;
      wnt.topLeftCorner(col, col).triangularView<Eigen::Lower>() =
          llt.matrixL();

      // Then form L^-1(-L_a'+R_z') in the (2,1) block.
      wnt.topLeftCorner(col, col).triangularView<Eigen::Lower>().solveInPlace(
          wnt.bottomLeftCorner(col, col).transpose());

      // Form S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z') in the lower
      //  triangle of (2,2) block of wn_t.
      for (int i = 0; i < col; ++i) {
        wnt.col(col + i).tail(col - i).noalias() +=
            wnt.row(col + i).head(col)
            * wnt.bottomLeftCorner(col - i, col).transpose();
      }

      //     Cholesky factorization of (2,2) block of wn_t.
      llt.compute(wnt.bottomRightCorner(col, col));
      if (llt.info() != Eigen::Success)
        return false;
      wnt.bottomRightCorner(col, col).triangularView<Eigen::Lower>() =
          llt.matrixL();

      return true;
    }
  }  // namespace

  bool lbfgsb_formk(MatrixXd &wnt, MatrixXd &wn1, Eigen::LLT<MatrixXd> &llt,
                    const Eigen::Ref<const ArrayXi> &free,
                    const Eigen::Ref<const ArrayXi> &bound,
                    const std::vector<int> &enter,
                    const std::vector<int> &leave, const MatrixXd &ws,
                    const MatrixXd &wy, const MatrixXd &sy, const double theta,
                    const bool updated) {
    // Form the lower triangular part of
    //           WN1 = [Y' ZZ'Y   L_a'+R_z']
    //                 [L_a+R_z   S'AA'S   ]
    //    where L_a is the strictly lower triangular part of S'AA'Y
    //          R_z is the upper triangular part of S'ZZ'Y.
    if (updated)
      formk_add_new_wn1(wn1, free, bound, ws, wy);

    formk_update_wn1(wn1, enter, leave, ws, wy);

    // Form the upper triangle of WN = [D+Y' ZZ'Y/theta   -L_a'+R_z' ]
    //                                 [-L_a +R_z        S'AA'S*theta]
    formk_prepare_wn(wnt, wn1, sy, theta);

    // Form the upper triangle of WN= [  LL'            L^-1(-L_a'+R_z')]
    //                                [(-L_a +R_z)L'^-1   S'AA'S*theta  ]
    return formk_factorize_wn(wnt, llt);
  }
}  // namespace internal
}  // namespace nuri

// #include "nuri/algo/optim.h"
