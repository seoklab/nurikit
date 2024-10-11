//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/optim.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  bool lbfgsb_errclb(const ArrayXd &x, const ArrayXi &nbd,
                     const Array2Xd &bounds, const int m, const double factr) {
    if (x.size() <= 0)
      return false;
    if (m <= 0)
      return false;
    if (factr < 0.0)
      return false;

    if (x.size() != nbd.size() || ((nbd < 0) || (nbd > 3)).any())
      return false;
    if (x.size() != bounds.cols()
        || ((nbd == 3).transpose() && bounds.row(0) > bounds.row(1)).any())
      return false;

    return true;
  }

  std::pair<bool, bool> lbfgsb_active(MutRef<ArrayXd> &x, ArrayXi &iwhere,
                                      const LbfgsbBounds &bounds) {
    bool constrained = false, boxed = true;

    for (int i = 0; i < x.size(); ++i) {
      if (bounds.has_lb(i) && x[i] < bounds.lb(i))
        x[i] = bounds.lb(i);

      if (bounds.has_ub(i) && x[i] > bounds.ub(i))
        x[i] = bounds.ub(i);

      if (!bounds.has_both(i))
        boxed = false;

      if (!bounds.has_bound(i)) {
        iwhere[i] = -1;
        continue;
      }

      constrained = true;
      if (bounds.has_both(i) && bounds.ub(i) - bounds.lb(i) <= 0.0)
        iwhere[i] = 3;
    }

    return { constrained, boxed };
  }

  double lbfgsb_projgr(ConstRef<ArrayXd> x, const ArrayXd &gx,
                       const LbfgsbBounds &bounds) {
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

  bool lbfgsb_bmv(MutRef<VectorXd> &p, MutRef<ArrayXd> &smul,
                  ConstRef<VectorXd> v, ConstRef<MatrixXd> sy,
                  ConstRef<MatrixXd> wtt) {
    const auto col = sy.cols();
    ABSL_DCHECK(col > 0);

    if ((sy.diagonal().array() == 0).any()
        || (wtt.diagonal().array() == 0).any())
      return false;

    smul = v.head(col - 1).array() / sy.diagonal().head(col - 1).array();
    for (int i = 0; i < col; ++i) {
      double ssum =
          (sy.row(i).head(i).array() * smul.head(i).transpose()).sum();
      p[col + i] = v[col + i] + ssum;
    }
    wtt.triangularView<Eigen::Lower>().solveInPlace(p.tail(col));
    wtt.triangularView<Eigen::Lower>().transpose().solveInPlace(p.tail(col));

    p.head(col) = -v.head(col).array() / sy.diagonal().array();
    for (int i = 0; i < col; ++i) {
      double ssum =
          sy.col(i).tail(col - i - 1).dot(p.tail(col - i - 1)) / sy(i, i);
      p[i] += ssum;
    }

    return true;
  }

  namespace {
    std::pair<bool, bool> cauchy_find_breaks(ArrayXi &iwhere,
                                             std::vector<CauchyBrkpt> &brks,
                                             ConstRef<ArrayXd> x,
                                             const ArrayXd &gx,
                                             const LbfgsbBounds &bounds) {
      bool any_free = false, bounded = true;

      brks.clear();
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

        const auto size = brks.size(), capacity = brks.capacity();
        ABSL_ASSUME(size < capacity);
        if (bounds.has_lb(i) && neg_gxi < 0) {
          brks.push_back({ i, (x[i] - bounds.lb(i)) / -neg_gxi });
        } else if (bounds.has_ub(i) && neg_gxi > 0) {
          brks.push_back({ i, (bounds.ub(i) - x[i]) / neg_gxi });
        } else {
          any_free = true;
          if (neg_gxi != 0)  // NOLINT(clang-diagnostic-float-equal)
            bounded = false;
        }
      }

      return { any_free, bounded };
    }

    struct CauchyHandleBreaksResult {
      double dtm;
      double tsum = 0;
      bool success = true;
      bool z_is_gcp = false;
    };

    CauchyHandleBreaksResult cauchy_handle_breaks(
        ArrayXd &xcp, ArrayXi &iwhere, MutRef<VectorXd> &p, MutRef<VectorXd> &v,
        MutRef<VectorXd> &c, ArrayXd &d, MutRef<VectorXd> &wbp,
        MutRef<ArrayXd> &smul, ClearablePQ<CauchyBrkpt, std::greater<>> &pq,
        double f1, double f2, ConstRef<ArrayXd> x, const LbfgsbBounds &bounds,
        ConstRef<MatrixXd> ws, ConstRef<MatrixXd> wy, ConstRef<MatrixXd> sy,
        ConstRef<MatrixXd> wtt, const double theta, const double f2_org,
        const bool bounded) {
      const auto nbreaks = pq.size();
      const auto col = wy.cols();

      CauchyHandleBreaksResult ret { -f1 / f2 };

      pq.rebuild();
      c.setZero();

      double tj0 = 0;
      while (!pq.empty()) {
        auto [ibp, tj] = pq.pop_get();
        double dt = tj - tj0;
        if (ret.dtm < dt)
          return ret;

        tj0 = tj;

        ret.tsum += dt;
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
          ret.dtm = dt;
          ret.z_is_gcp = true;
          return ret;
        }

        double dibp2 = dibp * dibp;
        f1 += dt * f2 + dibp2 - theta * dibp * zibp;
        f2 -= theta * dibp2;
        if (col > 0) {
          wbp.head(col) = wy.row(ibp);
          wbp.tail(col) = theta * ws.row(ibp);
          if (!lbfgsb_bmv(v, smul, wbp, sy, wtt)) {
            ret.success = false;
            return ret;
          }

          c += dt * p;
          double wmc = c.dot(v), wmp = p.dot(v), wmw = wbp.dot(v);
          p -= dibp * wbp;
          f1 += dibp * wmc;
          f2 += dibp * 2 * wmp - dibp2 * wmw;
        }

        f2 = nuri::max(kEpsMach * f2_org, f2);
        ret.dtm = -f1 / f2;
      }

      if (bounded)
        ret.dtm = 0;

      return ret;
    }
  }  // namespace

  bool lbfgsb_cauchy(ArrayXd &xcp, ArrayXi &iwhere, MutRef<VectorXd> &p,
                     MutRef<VectorXd> &v, MutRef<VectorXd> &c, ArrayXd &d,
                     MutRef<VectorXd> &wbp, MutRef<ArrayXd> &smul,
                     ClearablePQ<CauchyBrkpt, std::greater<>> &brks,
                     ConstRef<ArrayXd> x, const ArrayXd &gx,
                     const LbfgsbBounds &bounds, ConstRef<MatrixXd> ws,
                     ConstRef<MatrixXd> wy, ConstRef<MatrixXd> sy,
                     ConstRef<MatrixXd> wtt, const double sbgnrm,
                     const double theta) {
    const auto col = wy.cols();

    xcp = x;
    if (sbgnrm <= 0)
      return true;

    auto [any_free, bounded] =
        cauchy_find_breaks(iwhere, brks.data(), x, gx, bounds);
    if (brks.empty() && !any_free)
      return true;

    d = (iwhere != 0 && iwhere != -1).select(0, -gx);
    p.head(col).noalias() = wy.transpose() * d.matrix();
    p.tail(col).noalias() = ws.transpose() * d.matrix() * theta;

    double f1 = -d.matrix().squaredNorm();
    double f2 = -theta * f1;
    const double f2_org = f2;
    if (col > 0) {
      if (!lbfgsb_bmv(v, smul, p, sy, wtt))
        return false;
      f2 -= v.dot(p);
    }

    auto [dtm, tsum, success, z_is_gcp] = cauchy_handle_breaks(
        xcp, iwhere, p, v, c, d, wbp, smul, brks, f1, f2, x, bounds, ws, wy, sy,
        wtt, theta, f2_org, bounded);
    if (!success)
      return false;

    if (!z_is_gcp) {
      dtm = nonnegative(dtm);
      tsum += dtm;
      xcp += tsum * d.array();
    }

    c += dtm * p;

    return true;
  }

  namespace {
    bool lbfgsb_freev(ArrayXi &free_bound, int &nfree, ArrayXi &enter_leave,
                      int &nenter, int &nleave, const ArrayXi &iwhere,
                      const int iter, const bool constrained,
                      const bool updated) {
      const auto n = free_bound.size();

      if (iter > 0 && constrained) {
        nenter = nleave = 0;

        for (int i = 0; i < nfree; ++i) {
          int k = free_bound[i];
          if (iwhere[k] > 0)
            enter_leave[n - ++nleave] = k;
        }

        for (int i = nfree; i < free_bound.size(); ++i) {
          int k = free_bound[i];
          if (iwhere[k] <= 0)
            enter_leave[nenter++] = k;
        }
      }

      nfree = 0;
      auto iact = free_bound.size();
      for (int i = 0; i < free_bound.size(); ++i) {
        if (iwhere[i] <= 0) {
          free_bound[nfree] = i;
          ++nfree;
        } else {
          --iact;
          free_bound[iact] = i;
        }
      }

      return nenter + nleave > 0 || updated;
    }

    void formk_shift_wn1(MatrixXd &wn1) {
      const auto m = wn1.cols() / 2, mm1 = m - 1;

      //  Shift old part of WN1.
      wn1.topLeftCorner(mm1, mm1).triangularView<Eigen::Lower>() =
          wn1.block(1, 1, mm1, mm1).triangularView<Eigen::Lower>();
      wn1.block(m, m, mm1, mm1).triangularView<Eigen::Lower>() =
          wn1.block(m + 1, m + 1, mm1, mm1).triangularView<Eigen::Lower>();
      wn1.block(m, 0, mm1, mm1) = wn1.block(m + 1, 1, mm1, mm1);
    }

    void formk_add_new_wn1(MatrixXd &wn1, ConstRef<ArrayXi> free,
                           ConstRef<ArrayXi> bound, ConstRef<MatrixXd> ws,
                           ConstRef<MatrixXd> wy, const int prev_col) {
      const auto m = wn1.cols() / 2, col = ws.cols(), cm1 = col - 1;

      if (prev_col == m)
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

    void formk_update_wn1(MatrixXd &wn1, ConstRef<ArrayXi> enter,
                          ConstRef<ArrayXi> leave, ConstRef<MatrixXd> ws,
                          ConstRef<MatrixXd> wy, const Eigen::Index upcl) {
      const auto m = wn1.cols() / 2;

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

    void formk_prepare_wn(MutRef<MatrixXd> &wnt, const MatrixXd &wn1,
                          ConstRef<MatrixXd> sy, const double theta) {
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

    bool formk_factorize_wn(MutRef<MatrixXd> &wnt, Eigen::LLT<MatrixXd> &llt) {
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

  bool lbfgsb_formk(MutRef<MatrixXd> &wnt, MatrixXd &wn1,
                    Eigen::LLT<MatrixXd> &llt, ConstRef<ArrayXi> free,
                    ConstRef<ArrayXi> bound, ConstRef<ArrayXi> enter,
                    ConstRef<ArrayXi> leave, ConstRef<MatrixXd> ws,
                    ConstRef<MatrixXd> wy, ConstRef<MatrixXd> sy,
                    const double theta, const int prev_col,
                    const bool updated) {
    // Form the lower triangular part of
    //           WN1 = [Y' ZZ'Y   L_a'+R_z']
    //                 [L_a+R_z   S'AA'S   ]
    //    where L_a is the strictly lower triangular part of S'AA'Y
    //          R_z is the upper triangular part of S'ZZ'Y.
    if (updated)
      formk_add_new_wn1(wn1, free, bound, ws, wy, prev_col);

    formk_update_wn1(wn1, enter, leave, ws, wy, ws.cols() - value_if(updated));

    // Form the upper triangle of WN = [D+Y' ZZ'Y/theta   -L_a'+R_z' ]
    //                                 [-L_a +R_z        S'AA'S*theta]
    formk_prepare_wn(wnt, wn1, sy, theta);

    // Form the upper triangle of WN= [  LL'            L^-1(-L_a'+R_z')]
    //                                [(-L_a +R_z)L'^-1   S'AA'S*theta  ]
    return formk_factorize_wn(wnt, llt);
  }

  bool lbfgsb_cmprlb(MutRef<VectorXd> &r, MutRef<VectorXd> &p,
                     MutRef<ArrayXd> &smul, ConstRef<VectorXd> c,
                     ConstRef<ArrayXi> free, ConstRef<ArrayXd> x,
                     const ArrayXd &z, const ArrayXd &gx, ConstRef<MatrixXd> ws,
                     ConstRef<MatrixXd> wy, ConstRef<MatrixXd> sy,
                     ConstRef<MatrixXd> wtt, const double theta,
                     const bool constrained) {
    const auto col = ws.cols();

    if (!constrained && col > 0) {
      r = -gx;
      return true;
    }

    r = theta * (x(free) - z(free)) - gx(free);

    if (!lbfgsb_bmv(p, smul, c, sy, wtt))
      return false;

    r.noalias() += wy(free, Eigen::all) * p.head(col)
                   + ws(free, Eigen::all) * p.tail(col) * theta;

    return true;
  }

  bool lbfgsb_subsm(ArrayXd &x, ArrayXd &xp, MutRef<VectorXd> &d,
                    MutRef<VectorXd> &wv, ConstRef<MatrixXd> wnt,
                    ConstRef<ArrayXi> free, ConstRef<ArrayXd> xx,
                    const ArrayXd &gg, ConstRef<MatrixXd> ws,
                    ConstRef<MatrixXd> wy, const LbfgsbBounds &bounds,
                    const double theta) {
    const auto col = ws.cols(), nsub = free.size();
    if (nsub <= 0)
      return true;

    // Compute wv = W'Zd.
    wv.head(col).noalias() = wy(free, Eigen::all).transpose() * d;
    wv.tail(col).noalias() = ws(free, Eigen::all).transpose() * d * theta;

    // Compute wv:=K^(-1)wv.
    if ((wnt.diagonal().array() == 0).any())
      return false;
    wnt.triangularView<Eigen::Lower>().solveInPlace(wv);
    wv.head(col) *= -1;
    wnt.triangularView<Eigen::Lower>().transpose().solveInPlace(wv);

    // Compute d = (1/theta)d + (1/theta**2)Z'W wv.
    d.noalias() += wy(free, Eigen::all) * wv.head(col) / theta
                   + ws(free, Eigen::all) * wv.tail(col);
    d /= theta;

    // -----------------------------------------------------
    // Let us try the projection, d is the Newton direction.
    xp = x;

    bool bounded = false;
    for (int i = 0; i < nsub; ++i) {
      int k = free[i];
      const double xk_new = xp[k] + d[i];
      switch (bounds.raw_nbd(k)) {
      case 0:
        x[k] = xk_new;
        break;
      case 1:
        x[k] = nuri::max(bounds.lb(k), xk_new);
        if (x[k] <= bounds.lb(k))
          bounded = true;
        break;
      case 2:
        x[k] = nuri::min(bounds.ub(k), xk_new);
        if (x[k] >= bounds.ub(k))
          bounded = true;
        break;
      case 3:
        x[k] = nuri::clamp(xk_new, bounds.lb(k), bounds.ub(k));
        if (x[k] <= bounds.lb(k) || x[k] >= bounds.ub(k))
          bounded = true;
        break;
      default:
        ABSL_UNREACHABLE();
        break;
      }
    }
    if (!bounded)
      return true;

    // Check sign of the directional derivative
    double dd_p = (x - xx).matrix().dot(gg.matrix());
    if (dd_p <= 0)
      return true;

    x = xp;
    double alpha = 1, curr_alpha = alpha;
    int ibd = -1;
    for (int i = 0; i < nsub; ++i) {
      const int k = free[i];
      const double dk = d[i];
      if (!bounds.has_bound(k))
        continue;

      if (dk < 0 && bounds.has_lb(k)) {
        double diff = bounds.lb(k) - x[k];
        if (diff >= 0) {
          curr_alpha = 0;
        } else if (dk * alpha < diff) {
          curr_alpha = diff / dk;
        }
      } else if (dk > 0 && bounds.has_ub(k)) {
        double diff = bounds.ub(k) - x[k];
        if (diff <= 0) {
          curr_alpha = 0;
        } else if (dk * alpha > diff) {
          curr_alpha = diff / dk;
        }
      }

      if (curr_alpha < alpha) {
        alpha = curr_alpha;
        ibd = i;
      }
    }

    if (alpha < 1) {
      ABSL_ASSUME(ibd >= 0);

      const int k = free[ibd];
      const double dk = d[ibd];
      if (dk > 0) {
        x[k] = bounds.ub(k);
      } else if (dk < 0) {
        x[k] = bounds.lb(k);
      }

      d[ibd] = 0;
    }

    x(free) += alpha * d.array();

    return true;
  }

  namespace {
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    void dcstep(double &stx, double &fx, double &dx, double &sty, double &fy,
                double &dy, double &stp, bool &brackt, const double fp,
                const double dp, const double stpmin, const double stpmax) {
      const double sgnd = dp * std::copysign(1.0, dx);

      auto theta_gamma_common = [&]() {
        double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
        double s = Array3d { theta, dx, dp }.abs().maxCoeff();
        double theta_scaled = theta / s;
        double gamma = s
                       * std::sqrt(nonnegative(theta_scaled * theta_scaled
                                               - dx / s * dp / s));
        return std::make_pair(theta, gamma);
      };

      auto update_intervals = [&](double new_step) {
        if (fp > fx) {
          sty = stp;
          fy = fp;
          dy = dp;
        } else {
          if (sgnd < 0) {
            sty = stx;
            fy = fx;
            dy = dx;
          }
          stx = stp;
          fx = fp;
          dx = dp;
        }

        stp = new_step;
      };

      // First case: A higher function value. The minimum is bracketed.
      // If the cubic step is closer to stx than the quadratic step, the
      // cubic step is taken, otherwise the average of the cubic and
      // quadratic steps is taken.
      if (fp > fx) {
        auto [theta, gamma] = theta_gamma_common();
        if (stp < stx)
          gamma = -gamma;

        double p = (gamma - dx) + theta;
        double q = ((gamma - dx) + gamma) + dp;
        double r = p / q;

        double stpc = stx + r * (stp - stx);
        double stpq =
            stx + dx / ((fx - fp) / (stp - stx) + dx) / 2 * (stp - stx);

        brackt = true;
        if (std::abs(stpc - stx) < std::abs(stpq - stx)) {
          update_intervals(stpc);
        } else {
          update_intervals(stpc + (stpq - stpc) / 2);
        }

        return;
      }

      // Second case: A lower function value and derivatives of opposite
      // sign. The minimum is bracketed. If the cubic step is farther from
      // stp than the secant step, the cubic step is taken, otherwise the
      // secant step is taken.
      if (sgnd < 0.0) {
        auto [theta, gamma] = theta_gamma_common();
        if (stp > stx)
          gamma = -gamma;

        double p = (gamma - dp) + theta;
        double q = ((gamma - dp) + gamma) + dx;
        double r = p / q;

        double stpc = stp + r * (stx - stp);
        double stpq = stp + dp / (dp - dx) * (stx - stp);

        brackt = true;
        if (std::abs(stpc - stp) > std::abs(stpq - stp)) {
          update_intervals(stpc);
        } else {
          update_intervals(stpq);
        }

        return;
      }

      // Third case: A lower function value, derivatives of the same sign,
      // and the magnitude of the derivative decreases.

      // The cubic step is computed only if the cubic tends to infinity
      // in the direction of the step or if the minimum of the cubic
      // is beyond stp. Otherwise the cubic step is defined to be the
      // secant step.
      if (std::abs(dp) < std::abs(dx)) {
        auto [theta, gamma] = theta_gamma_common();
        if (stp > stx)
          gamma = -gamma;

        double p = (gamma - dp) + theta;
        double q = (gamma + (dx - dp)) + gamma;
        double r = p / q;

        double stpc;
        if (r < 0 && gamma != 0) {  // NOLINT(clang-diagnostic-float-equal)
          stpc = stp + r * (stx - stp);
        } else if (stp > stx) {
          stpc = stpmax;
        } else {
          stpc = stpmin;
        }
        double stpq = stp + dp / (dp - dx) * (stx - stp);

        double stpf;
        if (brackt) {
          // A minimizer has been bracketed. If the cubic step is
          // closer to stp than the secant step, the cubic step is
          // taken, otherwise the secant step is taken.
          if (std::abs(stpc - stp) < std::abs(stpq - stp)) {
            stpf = stpc;
          } else {
            stpf = stpq;
          }
          if (stp > stx) {
            stpf = nuri::min(stp + 0.66 * (sty - stp), stpf);
          } else {
            stpf = nuri::max(stp + 0.66 * (sty - stp), stpf);
          }
        } else {
          // A minimizer has not been bracketed. If the cubic step is
          // farther from stp than the secant step, the cubic step is
          // taken, otherwise the secant step is taken.
          if (std::abs(stpc - stp) > std::abs(stpq - stp)) {
            stpf = stpc;
          } else {
            stpf = stpq;
          }
          stpf = nuri::clamp(stpf, stpmin, stpmax);
        }
        update_intervals(stpf);

        return;
      }

      // Fourth case: A lower function value, derivatives of the same sign,
      // and the magnitude of the derivative does not decrease. If the
      // minimum is not bracketed, the step is either stpmin or stpmax,
      // otherwise the cubic step is taken.
      if (brackt) {
        double theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
        double s = Array3d { theta, dy, dp }.abs().maxCoeff();
        double theta_scaled = theta / s;
        double gamma = s
                       * std::sqrt(nonnegative(theta_scaled * theta_scaled
                                               - dy / s * dp / s));
        if (stp > sty)
          gamma = -gamma;

        double p = (gamma - dp) + theta;
        double q = ((gamma - dp) + gamma) + dy;
        double r = p / q;
        double stpc = stp + r * (sty - stp);
        update_intervals(stpc);
      } else if (stp > stx) {
        update_intervals(stpmax);
      } else {
        update_intervals(stpmin);
      }
    }
  }  // namespace

  enum class LbfgsbLnsrch::DcsrchStatus : std::uint8_t {
    kFound,
    kConverged,
    kContinue,
  };

  LbfgsbLnsrch::LbfgsbLnsrch(MutRef<ArrayXd> &x, const ArrayXd &t,
                             const ArrayXd &z, const ArrayXd &d,
                             const LbfgsbBounds &bounds, const double f0,
                             const double g0, const int iter,
                             const bool constrained, const bool boxed,
                             const double ftol, const double gtol,
                             const double xtol) noexcept
      : x_(&x), t_(&t), z_(&z), d_(&d), bounds_(&bounds),
        dtd_(d.matrix().squaredNorm()), dnorm_(std::sqrt(dtd_)), finit_(f0),
        ginit_(g0), gtest_(ftol * ginit_), gtol_(gtol), xtol_(xtol), fx_(f0),
        gx_(g0), fy_(f0), gy_(g0) {
    const auto n = x.size();

    if (constrained && iter == 0) {
      stepmax_ = 1;
    } else if (constrained) {
      for (int i = 0; i < n; ++i) {
        if (!bounds.has_bound(i))
          continue;

        double a1 = d[i];
        if (a1 < 0 && bounds.has_lb(i)) {
          double a2 = bounds.lb(i) - x[i];
          if (a2 >= 0) {
            stepmax_ = 0;
            break;
          }
          if (a1 * stepmax_ < a2)
            stepmax_ = a2 / a1;
        } else if (a1 > 0 && bounds.has_ub(i)) {
          double a2 = bounds.ub(i) - x[i];
          if (a2 <= 0) {
            stepmax_ = 0;
            break;
          }
          if (a1 * stepmax_ > a2)
            stepmax_ = a2 / a1;
        }
      }
    }

    if (iter == 0 && !boxed) {
      step_ = nuri::min(1 / dnorm_, stepmax_);
    } else {
      step_ = 1;
    }

    // init for dcsrch
    width_ = stepmax_ - stepmin_;
    width1_ = 2 * width_;
    stmax_ = step_ + xtrapu_ * step_;

    step_x();
  }

  bool LbfgsbLnsrch::search(const double f, const double g) {
    auto status = dcsrch(f, g);
    if (status == DcsrchStatus::kConverged)
      return true;

    step_x();
    return status == DcsrchStatus::kFound;
  }

  LbfgsbLnsrch::DcsrchStatus LbfgsbLnsrch::dcsrch(const double f,
                                                  const double g) {
    double ftest = finit() + step_ * gtest_;

    // Termination check conditions
    if ((step_ >= stepmax_ && f <= ftest && g <= gtest_)
        || (step_ <= stepmin_ && (f > ftest || g >= gtest_)))
      return DcsrchStatus::kConverged;
    if (f <= ftest && std::abs(g) <= -gtol_ * ginit())
      return DcsrchStatus::kConverged;

    // If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the algorithm
    // enters the second stage.
    if (stage1_ && f <= ftest && g >= 0)
      stage1_ = false;

    // A modified function is used to predict the step during the first stage
    // if a lower function value has been obtained but the decrease is not
    // sufficient.

    if (stage1_ && f <= fx_ && f > ftest) {
      double fm = f - step_ * gtest_;
      double fxm = fx_ - stx_ * gtest_;
      double fym = fy_ - sty_ * gtest_;
      double gm = g - gtest_;
      double gxm = gx_ - gtest_;
      double gym = gy_ - gtest_;

      dcstep(stx_, fxm, gxm, sty_, fym, gym, step_, brackt_, fm, gm, stmin_,
             stmax_);

      fx_ = fxm + stx_ * gtest_;
      fy_ = fym + sty_ * gtest_;
      gx_ = gxm + gtest_;
      gy_ = gym + gtest_;
    } else {
      dcstep(stx_, fx_, gx_, sty_, fy_, gy_, step_, brackt_, f, g, stmin_,
             stmax_);
    }

    // Decide if a bisection step is needed.
    if (brackt_) {
      if (std::abs(sty_ - stx_) >= 0.66 * width1_)
        step_ = stx_ + 0.5 * (sty_ - stx_);
      width1_ = width_;
      width_ = std::abs(sty_ - stx_);
    }

    // Set the minimum and maximum steps allowed for stp.
    if (brackt_) {
      std::tie(stmin_, stmax_) = nuri::minmax(stx_, sty_);
    } else {
      stmin_ = step_ * xtrapl_ * (step_ - stx_);
      stmax_ = step_ + xtrapu_ * (step_ - stx_);
    }

    step_ = nuri::clamp(step_, stepmin_, stepmax_);

    // If further progress is not possible, let stp be the best point obtained
    // during the search.

    if (brackt_
        && (step_ <= stmin_ || step_ >= stmax_
            || stmax_ - stmin_ <= xtol_ * stmax_)) {
      step_ = stx_;
      return DcsrchStatus::kFound;
    }

    return DcsrchStatus::kContinue;
  }

  void LbfgsbLnsrch::step_x() {
    if (step_ == 1) {  // NOLINT(clang-diagnostic-float-equal)
      x() = z();
      return;
    }

    x() = step_ * d() + t();
    for (int i = 0; i < x().size(); ++i) {
      if (bounds().has_lb(i)) {
        x()[i] = nuri::max(bounds().lb(i), x()[i]);
      }
      if (bounds().has_ub(i)) {
        x()[i] = nuri::min(bounds().ub(i), x()[i]);
      }
    }
  }

  bool lbfgsb_prepare_lnsrch(
      MutRef<ArrayXd> x, ArrayXd &z, ArrayXd &xp, ArrayXd &r, ArrayXd &d,
      ArrayXi &iwhere, MatrixXd &wnt_d, MatrixXd &wn1, MatrixXd &ws_d,
      MatrixXd &wy_d, MatrixXd &sy_d, MatrixXd &wtt_d, VectorXd &p_d,
      VectorXd &v_d, VectorXd &c_d, VectorXd &wbp_d, ArrayXi &free_bound,
      int &nfree, ArrayXi &enter_leave, int &nenter, int &nleave,
      ClearablePQ<CauchyBrkpt, std::greater<>> &brks, Eigen::LLT<MatrixXd> &llt,
      const ArrayXd &gx, const LbfgsbBounds &bounds, const double sbgnrm,
      const double theta, const bool updated, const bool constrained,
      const int iter, const int col, const int prev_col) {
    const auto n = x.size();

    MutRef<MatrixXd> wnt(wnt_d.topLeftCorner(2 * col, 2 * col));
    MutRef<MatrixXd> ws(ws_d.leftCols(col)), wy(wy_d.leftCols(col));
    MutRef<MatrixXd> sy(sy_d.topLeftCorner(col, col)),
        wtt(wtt_d.topLeftCorner(col, col));
    MutRef<VectorXd> p(p_d.head(2 * col)), v(v_d.head(2 * col)),
        c(c_d.head(2 * col)), wbp = wbp_d.head(2 * col);

    MutRef<ArrayXd> smul = wbp_d.tail(nonnegative(col - 1)).array();

    bool need_k;
    if (!constrained && col > 0) {
      z = x;
      need_k = updated;
    } else {
      if (!lbfgsb_cauchy(z, iwhere, p, v, c, d, wbp, smul, brks, x, gx, bounds,
                         ws, wy, sy, wtt, sbgnrm, theta))
        return false;

      need_k = lbfgsb_freev(free_bound, nfree, enter_leave, nenter, nleave,
                            iwhere, iter, constrained, updated);
    }

    if (nfree > 0 && col > 0) {
      if (need_k) {
        if (!lbfgsb_formk(wnt, wn1, llt, free_bound.head(nfree),
                          free_bound.tail(n - nfree), enter_leave.head(nenter),
                          enter_leave.tail(nleave), ws, wy, sy, theta, prev_col,
                          updated))
          return false;
      }

      MutRef<VectorXd> rfree(r.head(nfree).matrix());
      if (!lbfgsb_cmprlb(rfree, p, smul, c, free_bound.head(nfree), x, z, gx,
                         ws, wy, sy, wtt, theta, constrained))
        return false;
      if (!lbfgsb_subsm(z, xp, rfree, p, wnt, free_bound.head(nfree), x, gx, ws,
                        wy, bounds, theta))
        return false;
    }

    return true;
  }

  namespace {
    int lbfgsb_matupd(MatrixXd &ws, MatrixXd &wy, MatrixXd &ss, MatrixXd &sy,
                      const ArrayXd &d, const ArrayXd &r, const double dr,
                      const double step, const double dtd, const int pcol) {
      const int m = static_cast<int>(ws.cols());
      ABSL_ASSUME(pcol <= m);

      int col, ci;
      if (pcol < m) {
        col = pcol + 1;
        ci = pcol;
      } else {
        col = m;
        ci = m - 1;
        ws.leftCols(ci) = ws.rightCols(ci);
        wy.leftCols(ci) = wy.rightCols(ci);
      }

      ws.col(ci) = d;
      wy.col(ci) = r;

      // Form the middle matrix in B.
      // update the upper triangle of SS (m,m), and the lower triangle of SY
      // (m,m):
      if (pcol == m) {
        ABSL_ASSUME(ci == m - 1);
        ss.topLeftCorner(ci, ci).triangularView<Eigen::Upper>() =
            ss.bottomRightCorner(ci, ci).triangularView<Eigen::Upper>();
        sy.topLeftCorner(ci, ci).triangularView<Eigen::Lower>() =
            sy.bottomRightCorner(ci, ci).triangularView<Eigen::Lower>();
      }

      ss.col(ci).head(col).noalias() =
          ws.leftCols(col).transpose() * d.matrix();
      sy.row(ci).head(col).noalias() =
          d.transpose().matrix() * wy.leftCols(col);

      ss(ci, ci) = step * step * dtd;
      sy(ci, ci) = dr;

      return col;
    }

    bool lbfgsb_formt(MutRef<MatrixXd> wtt, Eigen::LLT<MatrixXd> &llt,
                      MutRef<MatrixXd> sy_scaled, ConstRef<MatrixXd> sy,
                      ConstRef<MatrixXd> ss, const double theta) {
      const auto col = ss.cols();

      // Form the upper half of  T = theta*SS + L*D^(-1)*L', store T in the
      // upper triangle of the array wt.
      wtt.triangularView<Eigen::Lower>() =
          (theta * ss.transpose()).triangularView<Eigen::Lower>();

      sy_scaled.array() =
          sy.array().rowwise() / sy.diagonal().array().transpose();
      for (int i = 1; i < col; ++i) {
        wtt.col(i).tail(col - i).noalias() +=
            sy.bottomLeftCorner(col - i, i)
            * sy_scaled.row(i).head(i).transpose();
      }

      llt.compute(wtt);
      if (llt.info() != Eigen::Success)
        return false;

      wtt.triangularView<Eigen::Lower>() = llt.matrixL();
      return true;
    }
  }  // namespace

  bool lbfgsb_prepare_next_iter(ArrayXd &r, ArrayXd &d, MatrixXd &ws_d,
                                MatrixXd &wy_d, MatrixXd &ss_d, MatrixXd &sy_d,
                                MatrixXd &wtt_d, MatrixXd &wnt_d, double &theta,
                                int &col, bool &updated,
                                Eigen::LLT<MatrixXd> &llt, const double gd,
                                const double ginit, const double step,
                                const double dtd) {
    double rr = r.matrix().squaredNorm(), dr, ddum;
    if (step == 1) {  // NOLINT(clang-diagnostic-float-equal)
      dr = gd - ginit;
      ddum = -ginit;
    } else {
      dr = (gd - ginit) * step;
      d *= step;
      ddum = -ginit * step;
    }

    if (dr <= kEpsMach * ddum) {
      updated = false;
      return true;
    }

    updated = true;
    theta = rr / dr;
    col = lbfgsb_matupd(ws_d, wy_d, ss_d, sy_d, d, r, dr, step, dtd, col);
    // wn will be recalculated in the next iteration, use for temporary
    // storage
    return lbfgsb_formt(wtt_d.topLeftCorner(col, col), llt,
                        wnt_d.topLeftCorner(col, col),
                        sy_d.topLeftCorner(col, col),
                        ss_d.topLeftCorner(col, col), theta);
  }
}  // namespace internal
}  // namespace nuri
