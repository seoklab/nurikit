//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <tuple>
#include <utility>

#include <absl/log/absl_check.h>

#include "nuri/algo/optim.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  Dcsrch::Dcsrch(const double f0, const double g0, double step0,
                 const double stepmin, const double stepmax, const double ftol,
                 const double gtol, const double xtol) noexcept
      : finit_(f0), ginit_(g0), stepmin_(stepmin), stepmax_(stepmax),
        gtest_(ftol * ginit_), gtol_(gtol), xtol_(xtol), step_(step0), fx_(f0),
        gx_(g0), fy_(f0), gy_(g0), stmax_(step_ + kXTrapU * step_),
        width_(stepmax_ - stepmin_), width1_(2 * width_) {
    ABSL_DCHECK_GE(step0, stepmin);
    ABSL_DCHECK_LE(step0, stepmax);
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

  DcsrchStatus Dcsrch::operator()(const double f, const double g) {
    double ftest = finit() + step_ * gtest_;

    // Termination check conditions
    if ((step_ >= stepmax_ && f <= ftest && g <= gtest_)
        || (step_ <= stepmin_ && (f > ftest || g >= gtest_)))
      return DcsrchStatus::kConverged;
    if (f <= ftest && std::abs(g) <= -gtol_ * ginit())
      return DcsrchStatus::kConverged;

    // If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the algorithm
    // enters the second stage.
    if (!bisect_ && f <= ftest && g >= 0)
      bisect_ = true;

    // A modified function is used to predict the step during the first stage
    // if a lower function value has been obtained but the decrease is not
    // sufficient.

    if (!bisect_ && f <= fx_ && f > ftest) {
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
      stmin_ = step_ * kXTrapL * (step_ - stx_);
      stmax_ = step_ + kXTrapU * (step_ - stx_);
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
}  // namespace internal
}  // namespace nuri
