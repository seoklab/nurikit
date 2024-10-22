//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/optim.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
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

  namespace {
    template <class VT, class SYT, class WTT>
    bool lbfgs_bmv_impl(MutVecBlock<VectorXd> p, MutVecBlock<ArrayXd> smul,
                        const VT &v, const SYT &sy, const WTT &wtt) {
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
      wtt.template triangularView<Eigen::Lower>().solveInPlace(p.tail(col));
      wtt.template triangularView<Eigen::Lower>().transpose().solveInPlace(
          p.tail(col));

      p.head(col).noalias() =
          sy.transpose().template triangularView<Eigen::StrictlyUpper>()
          * p.tail(col);
      p.head(col) -= v.head(col);
      p.head(col).array() /= sy.diagonal().array();

      return true;
    }
  }  // namespace

  bool lbfgs_bmv(MutVecBlock<VectorXd> p, MutVecBlock<ArrayXd> smul,
                 ConstRef<VectorXd> v, ConstRef<MatrixXd> sy,
                 ConstRef<MatrixXd> wtt) {
    return lbfgs_bmv_impl(p, smul, v, sy, wtt);
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

    CauchyHandleBreaksResult cauchy_handle_breaks(LBfgsB &L, double f1,
                                                  double f2,
                                                  const double f2_org,
                                                  const bool bounded) {
      const auto &x = L.x();
      const auto &bounds = L.bounds();

      auto ws = L.ws(), wy = L.wy();
      auto sy = L.sy(), wtt = L.wtt();
      auto p = L.p(), v = L.v(), c = L.c(), wbp = L.wbp();
      auto &xcp = L.z(), &d = L.d();
      auto smul = L.smul();
      auto &iwhere = L.iwhere();
      auto &pq = L.brks();
      const double theta = L.theta();
      const int col = L.col();

      const auto nbreaks = pq.size();
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
          if (!lbfgs_bmv(v, smul, wbp, sy, wtt)) {
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

  bool lbfgsb_cauchy(LBfgsB &lbfgsb, const ArrayXd &gx, const double sbgnrm) {
    // NOLINTNEXTLINE(readability-identifier-naming)
    auto &L = lbfgsb;

    const auto &x = L.x();
    const auto &bounds = L.bounds();

    auto ws = L.ws(), wy = L.wy();
    auto sy = L.sy(), wtt = L.wtt();
    auto p = L.p(), v = L.v(), c = L.c();
    auto &xcp = L.xcp(), &d = L.d();
    auto smul = L.smul();
    auto &iwhere = L.iwhere();
    auto &brks = L.brks();
    const double theta = L.theta();
    const int col = L.col();

    xcp = x;
    if (sbgnrm <= 0)
      return true;

    auto [any_free, bounded] =
        cauchy_find_breaks(iwhere, brks.data(), x, gx, bounds);
    if (brks.empty() && !any_free)
      return true;

    d = (iwhere != 0 && iwhere != -1).select(0, -gx);
    p.head(col).noalias() = wy.transpose() * d.matrix();
    p.tail(col).noalias() = theta * (ws.transpose() * d.matrix());

    double f1 = -d.matrix().squaredNorm();
    double f2 = -theta * f1;
    const double f2_org = f2;
    if (col > 0) {
      if (!lbfgs_bmv(v, smul, p, sy, wtt))
        return false;
      f2 -= v.dot(p);
    }

    auto [dtm, tsum, success, z_is_gcp] =
        cauchy_handle_breaks(L, f1, f2, f2_org, bounded);
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

  bool lbfgsb_subsm(LBfgsB &lbfgsb, const ArrayXd &gg) {
    // NOLINTNEXTLINE(readability-identifier-naming)
    auto &L = lbfgsb;

    auto &xx = L.x();
    const auto &bounds = L.bounds();
    auto wnt = L.wnt();
    auto ws = L.ws(), wy = L.wy();
    auto wv = L.p();
    auto &x = L.z(), &xp = L.xp();
    auto d = L.rfree();
    auto free = L.free();
    const double theta = L.theta();

    const int col = L.col();
    const auto nsub = L.free().size();
    if (nsub <= 0)
      return true;

    // Compute wv = W'Zd.
    wv.head(col).noalias() = wy(free, Eigen::all).transpose() * d;
    wv.tail(col).noalias() = theta * (ws(free, Eigen::all).transpose() * d);

    // Compute wv:=K^(-1)wv.
    if ((wnt.diagonal().array() == 0).any())
      return false;
    wnt.triangularView<Eigen::Lower>().solveInPlace(wv);
    wv.head(col) = -wv.head(col);
    wnt.triangularView<Eigen::Lower>().transpose().solveInPlace(wv);

    // Compute d = (1/theta)d + (1/theta**2)Z'W wv.
    d.noalias() += (1 / theta) * (wy(free, Eigen::all) * wv.head(col));
    d.noalias() += ws(free, Eigen::all) * wv.tail(col);
    d /= theta;

    // -----------------------------------------------------
    // Let us try the projection, d is the Newton direction.
    if (!L.constrained()) {
      x(free) += d.array();
      return true;
    }

    xp = x;
    x(free) += d.array();

    bool bounded = false;
    for (int i = 0; i < nsub; ++i) {
      const int k = free[i];
      switch (bounds.raw_nbd(k)) {
      case 0:
        break;
      case 1:
        x[k] = nuri::max(bounds.lb(k), x[k]);
        if (x[k] <= bounds.lb(k))
          bounded = true;
        break;
      case 2:
        x[k] = nuri::min(bounds.ub(k), x[k]);
        if (x[k] >= bounds.ub(k))
          bounded = true;
        break;
      case 3:
        x[k] = nuri::clamp(x[k], bounds.lb(k), bounds.ub(k));
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
}  // namespace internal

LBfgsB::LBfgsB(MutRef<ArrayXd> x, internal::LbfgsbBounds bounds, const int m)
    : x_(x), bounds_(bounds), wnt_(2 * m, 2 * m), wn1_(2 * m, 2 * m),
      ws_(n(), m), wy_(n(), m), ss_(m, m), sy_(m, m), wtt_(m, m), p_(2 * m),
      v_(2 * m), c_(2 * m), wbp_(2 * m), z_(n()), r_(n()), t_(n()), d_(n()),
      smul_(m - 1), iwhere_(ArrayXi::Zero(n())), free_bound_(n()),
      enter_leave_(n()), constrained_(false), boxed_(true) {
  brks().data().reserve(n());

  for (int i = 0; i < n(); ++i) {
    if (bounds_.has_lb(i) && x_[i] < bounds_.lb(i))
      x_[i] = bounds_.lb(i);

    if (bounds_.has_ub(i) && x_[i] > bounds_.ub(i))
      x_[i] = bounds_.ub(i);

    if (!bounds_.has_both(i))
      boxed_ = false;

    if (!bounds_.has_bound(i)) {
      iwhere_[i] = -1;
      continue;
    }

    constrained_ = true;
    if (bounds_.has_both(i) && bounds_.ub(i) - bounds_.lb(i) <= 0.0)
      iwhere_[i] = 3;
  }
}

bool LBfgsB::freev(const int iter) {
  const auto n = free_bound_.size();

  if (iter > 0 && constrained()) {
    nenter_ = nleave_ = 0;

    for (int i = 0; i < nfree_; ++i) {
      int k = free_bound_[i];
      if (iwhere()[k] > 0)
        enter_leave_[n - ++nleave_] = k;
    }

    for (int i = nfree_; i < free_bound_.size(); ++i) {
      int k = free_bound_[i];
      if (iwhere()[k] <= 0)
        enter_leave_[nenter_++] = k;
    }
  }

  nfree_ = 0;
  auto iact = free_bound_.size();
  for (int i = 0; i < free_bound_.size(); ++i) {
    if (iwhere()[i] <= 0) {
      free_bound_[nfree_] = i;
      ++nfree_;
    } else {
      --iact;
      free_bound_[iact] = i;
    }
  }

  return nenter_ + nleave_ > 0 || updated();
}

namespace {
  void formk_shift_wn1(MatrixXd &wn1, const int m) {
    const auto mm1 = m - 1;

    //  Shift old part of WN1.
    wn1.topLeftCorner(mm1, mm1).triangularView<Eigen::Lower>() =
        wn1.block(1, 1, mm1, mm1);
    wn1.block(m, m, mm1, mm1).triangularView<Eigen::Lower>() =
        wn1.block(m + 1, m + 1, mm1, mm1);
    wn1.block(m, 0, mm1, mm1) = wn1.block(m + 1, 1, mm1, mm1);
  }

  void formk_add_new_wn1(LBfgsB &L) {
    auto &wn1 = L.wn1();
    auto ws = L.ws(), wy = L.wy();
    auto free = L.free(), bound = L.bound();
    const auto prev_col = L.prev_col();

    const auto m = L.m(), col = L.col(), cm1 = col - 1;

    if (prev_col == m)
      formk_shift_wn1(wn1, m);

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

  void formk_update_wn1(LBfgsB &L) {
    auto &wn1 = L.wn1();
    auto enter = L.enter(), leave = L.leave();

    const auto m = L.m(), upcl = L.col() - value_if(L.updated());
    auto ws = L.ws().leftCols(upcl), wy = L.wy().leftCols(upcl);

    // modify the old parts in blocks (1,1) and (2,2) due to changes
    // in the set of free variables.
    for (int j = 0; j < upcl; ++j) {
      wn1.col(j).segment(j, upcl - j).noalias() +=
          wy(enter, Eigen::all).transpose().bottomRows(upcl - j) * wy(enter, j);
      wn1.col(j).segment(j, upcl - j).noalias() -=
          wy(leave, Eigen::all).transpose().bottomRows(upcl - j) * wy(leave, j);
    }
    for (int j = 0; j < upcl; ++j) {
      wn1.col(m + j).segment(m + j, upcl - j).noalias() -=
          ws(enter, Eigen::all).transpose().bottomRows(upcl - j) * ws(enter, j);
      wn1.col(m + j).segment(m + j, upcl - j).noalias() +=
          ws(leave, Eigen::all).transpose().bottomRows(upcl - j) * ws(leave, j);
    }

    // Modify the old parts in block (2,1)
    for (int j = 0; j < upcl; ++j) {
      wn1.col(j).segment(m, j + 1).noalias() +=
          ws(enter, Eigen::all).transpose().topRows(j + 1) * wy(enter, j);
      wn1.col(j).segment(m + j + 1, upcl - j - 1).noalias() -=
          ws(enter, Eigen::all).transpose().bottomRows(upcl - j - 1)
          * wy(enter, j);

      wn1.col(j).segment(m, j + 1).noalias() -=
          ws(leave, Eigen::all).transpose().topRows(j + 1) * wy(leave, j);
      wn1.col(j).segment(m + j + 1, upcl - j - 1).noalias() +=
          ws(leave, Eigen::all).transpose().bottomRows(upcl - j - 1)
          * wy(leave, j);
    }
  }

  template <class SYT>
  void formk_prepare_wn(MutBlock<MatrixXd> wnt, const MatrixXd &wn1, SYT sy,
                        const double theta, const int m, const int col) {
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

  bool formk_factorize_wn(MutBlock<MatrixXd> wnt, const int col) {
    //    first Cholesky factor (1,1) block of wn to get LL'
    //                      with L stored in the lower triangle of wn_t.

    // This is required because inplace llt ctor does not accept rvalue.
    auto wnt_11 = wnt.topLeftCorner(col, col);
    Eigen::LLT<Eigen::Ref<MatrixXd>> llt_11(wnt_11);
    if (llt_11.info() != Eigen::Success)
      return false;

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

    // Same here.
    auto wnt_22 = wnt.bottomRightCorner(col, col);
    Eigen::LLT<Eigen::Ref<MatrixXd>> llt_22(wnt_22);
    return llt_22.info() == Eigen::Success;
  }

  bool lbfgsb_formk(LBfgsB &L) {
    // Form the lower triangular part of
    //           WN1 = [Y' ZZ'Y   L_a'+R_z']
    //                 [L_a+R_z   S'AA'S   ]
    //    where L_a is the strictly lower triangular part of S'AA'Y
    //          R_z is the upper triangular part of S'ZZ'Y.
    if (L.updated())
      formk_add_new_wn1(L);

    formk_update_wn1(L);

    // Form the upper triangle of WN = [D+Y' ZZ'Y/theta   -L_a'+R_z' ]
    //                                 [-L_a +R_z        S'AA'S*theta]
    formk_prepare_wn(L.wnt(), L.wn1(), L.sy(), L.theta(), L.m(), L.col());

    // Form the upper triangle of WN= [  LL'            L^-1(-L_a'+R_z')]
    //                                [(-L_a +R_z)L'^-1   S'AA'S*theta  ]
    return formk_factorize_wn(L.wnt(), L.col());
  }

  bool lbfgsb_cmprlb(LBfgsB &L, const ArrayXd &gx) {
    const auto &x = L.x();
    auto ws = L.ws(), wy = L.wy();
    auto sy = L.sy(), wtt = L.wtt();
    auto p = L.p(), c = L.c();
    const auto &z = L.z();
    auto r = L.rfree();
    auto smul = L.smul();
    auto free = L.free();
    const double theta = L.theta();
    const auto col = L.col();
    const bool constrained = L.constrained();

    if (!constrained && col > 0) {
      r = -gx;
      return true;
    }

    r = (theta * (x - z) - gx)(free);

    if (!internal::lbfgs_bmv(p, smul, c, sy, wtt))
      return false;

    r.noalias() += wy(free, Eigen::all) * p.head(col);
    r.noalias() += theta * (ws(free, Eigen::all) * p.tail(col));

    return true;
  }
}  // namespace

bool LBfgsB::prepare_lnsrch(const ArrayXd &gx, const double sbgnrm,
                            const int iter) {
  bool need_k;
  if (!constrained() && col() > 0) {
    z() = x();
    need_k = updated();
  } else {
    if (!internal::lbfgsb_cauchy(*this, gx, sbgnrm))
      return false;

    need_k = freev(iter);
  }

  if (nfree_ > 0 && col() > 0) {
    if (need_k) {
      if (!lbfgsb_formk(*this))
        return false;
    }

    if (!lbfgsb_cmprlb(*this, gx))
      return false;
    if (!internal::lbfgsb_subsm(*this, gx))
      return false;
  }

  return true;
}

namespace {
  int lbfgs_matupd(MatrixXd &ws, MatrixXd &wy, MatrixXd &ss, MatrixXd &sy,
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
          ss.bottomRightCorner(ci, ci);
      sy.topLeftCorner(ci, ci).triangularView<Eigen::Lower>() =
          sy.bottomRightCorner(ci, ci);
    }

    ss.col(ci).head(col).noalias() = ws.leftCols(col).transpose() * d.matrix();
    sy.row(ci).head(col).noalias() = d.transpose().matrix() * wy.leftCols(col);

    ss(ci, ci) = step * step * dtd;
    sy(ci, ci) = dr;

    return col;
  }

  template <class WTT, class SYST, class SYT, class SST>
  bool lbfgs_formt(WTT wtt, SYST sy_scaled, SYT sy, SST ss,
                   const double theta) {
    const auto col = ss.cols();

    // Form the upper half of  T = theta*SS + L*D^(-1)*L', store T in the
    // upper triangle of the array wt.
    wtt.template triangularView<Eigen::Lower>() = theta * ss.transpose();
    for (int i = 1; i < col; ++i) {
      sy_scaled.head(i).array() =
          sy.row(i).head(i).transpose().array() / sy.diagonal().head(i).array();
      wtt.col(i).tail(col - i).noalias() +=
          sy.bottomLeftCorner(col - i, i) * sy_scaled.head(i);
    }

    Eigen::LLT<Eigen::Ref<MatrixXd>> llt(wtt);
    return llt.info() == Eigen::Success;
  }
}  // namespace

bool LBfgsB::prepare_next_iter(const double gd, const double ginit,
                               const double step, const double dtd) {
  double rr = r().matrix().squaredNorm(), dr, ddum;

  if (step == 1) {  // NOLINT(clang-diagnostic-float-equal)
    dr = gd - ginit;
    ddum = -ginit;
  } else {
    dr = (gd - ginit) * step;
    d() *= step;
    ddum = -ginit * step;
  }

  if (dr <= internal::kEpsMach * ddum) {
    updated_ = false;
    return true;
  }

  updated_ = true;
  theta_ = rr / dr;
  prev_col_ = col();
  col_ = lbfgs_matupd(ws_, wy_, ss_, sy_, d(), r(), dr, step, dtd, col());
  // wn will be recalculated in the next iteration, use for temporary storage
  return lbfgs_formt(wtt_.topLeftCorner(col(), col()), wnt_.col(0),
                     sy_.topLeftCorner(col(), col()),
                     ss_.topLeftCorner(col(), col()), theta());
}
}  // namespace nuri
