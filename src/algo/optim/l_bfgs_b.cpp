//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "optim_internal.h"
#include "nuri/algo/optim.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    Dcsrch lbfgsb_dcsrch_init(MutRef<ArrayXd> &x, const ArrayXd &d,
                              const LbfgsbBounds &bounds, const double dnorm,
                              const int iter, const bool constrained,
                              const bool boxed, const double f0,
                              const double g0, const double ftol,
                              const double gtol, const double xtol) {
      const auto n = x.size();

      double stepmax = Dcsrch::kStepMax;
      if (constrained && iter == 0) {
        stepmax = 1;
      } else if (constrained) {
        for (int i = 0; i < n; ++i) {
          if (!bounds.has_bound(i))
            continue;

          double a1 = d[i];
          if (a1 < 0 && bounds.has_lb(i)) {
            double a2 = bounds.lb(i) - x[i];
            if (a2 >= 0) {
              stepmax = 0;
              break;
            }
            if (a1 * stepmax < a2)
              stepmax = a2 / a1;
          } else if (a1 > 0 && bounds.has_ub(i)) {
            double a2 = bounds.ub(i) - x[i];
            if (a2 <= 0) {
              stepmax = 0;
              break;
            }
            if (a1 * stepmax > a2)
              stepmax = a2 / a1;
          }
        }
      }

      double step0;
      if (iter == 0 && !boxed) {
        step0 = nuri::min(1 / dnorm, stepmax);
      } else {
        step0 = 1;
      }

      return Dcsrch(f0, g0, step0, Dcsrch::kStepMin, stepmax, ftol, gtol, xtol);
    }
  }  // namespace

  LbfgsbLnsrch::LbfgsbLnsrch(MutRef<ArrayXd> &x, const ArrayXd &t,
                             const ArrayXd &z, const ArrayXd &d,
                             const LbfgsbBounds &bounds, const double f0,
                             const double g0, const int iter,
                             const bool constrained, const bool boxed,
                             const double ftol, const double gtol,
                             const double xtol) noexcept
      : x_(&x), t_(&t), z_(&z), d_(&d), bounds_(&bounds),
        dtd_(d.matrix().squaredNorm()), dnorm_(std::sqrt(dtd_)),
        dcsrch_(lbfgsb_dcsrch_init(x, d, bounds, dnorm_, iter, constrained,
                                   boxed, f0, g0, ftol, gtol, xtol)) {
    step_x();
  }

  bool LbfgsbLnsrch::search(const double f, const double g) {
    auto status = dcsrch_(f, g);
    if (status == DcsrchStatus::kConverged)
      return true;

    step_x();
    return status == DcsrchStatus::kFound;
  }

  void LbfgsbLnsrch::step_x() {
    if (step() == 1) {  // NOLINT(clang-diagnostic-float-equal)
      x() = z();
      return;
    }

    x() = step() * d() + t();
    for (int i = 0; i < x().size(); ++i) {
      if (bounds().has_lb(i)) {
        x()[i] = nuri::max(bounds().lb(i), x()[i]);
      }
      if (bounds().has_ub(i)) {
        x()[i] = nuri::min(bounds().ub(i), x()[i]);
      }
    }
  }

  LBfgsBImpl::LBfgsBImpl(MutRef<ArrayXd> x, LbfgsbBounds bounds, int m)
      : wn1_(2 * m, 2 * m), bounds_(bounds), iwhere_(ArrayXi::Zero(x.size())),
        free_bound_(n()), enter_leave_(n()), constrained_(false), boxed_(true) {
    brks_.data().reserve(n());

    for (int i = 0; i < n(); ++i) {
      if (bounds_.has_lb(i) && x[i] < bounds_.lb(i))
        x[i] = bounds_.lb(i);

      if (bounds_.has_ub(i) && x[i] > bounds_.ub(i))
        x[i] = bounds_.ub(i);

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

  LbfgsbLnsrch LBfgsBImpl::lnsrch(MutRef<ArrayXd> &x, const ArrayXd &t,
                                  const ArrayXd &z, const ArrayXd &d, double f0,
                                  double g0, int iter, double ftol, double gtol,
                                  double xtol) const noexcept {
    return LbfgsbLnsrch(x, t, z, d, bounds_, f0, g0, iter, constrained_, boxed_,
                        ftol, gtol, xtol);
  }

  void LBfgsBImpl::reset() {
    nfree_ = n();
    nenter_ = nleave_ = 0;
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

    void formk_add_new_wn1(LBfgsBase &L, LBfgsBImpl &impl) {
      MatrixXd &wn1 = impl.wn1();
      auto ws = L.ws(), wy = L.wy();
      auto free = impl.free(), bound = impl.bound();
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

    void formk_update_wn1(LBfgsBase &L, LBfgsBImpl &impl) {
      auto &wn1 = impl.wn1();
      auto enter = impl.enter(), leave = impl.leave();

      const auto m = L.m(), upcl = L.col() - value_if(L.updated());
      auto ws = L.ws().leftCols(upcl), wy = L.wy().leftCols(upcl);

      // modify the old parts in blocks (1,1) and (2,2) due to changes
      // in the set of free variables.
      for (int j = 0; j < upcl; ++j) {
        wn1.col(j).segment(j, upcl - j).noalias() +=
            wy(enter, Eigen::all).transpose().bottomRows(upcl - j)
            * wy(enter, j);
        wn1.col(j).segment(j, upcl - j).noalias() -=
            wy(leave, Eigen::all).transpose().bottomRows(upcl - j)
            * wy(leave, j);
      }
      for (int j = 0; j < upcl; ++j) {
        wn1.col(m + j).segment(m + j, upcl - j).noalias() -=
            ws(enter, Eigen::all).transpose().bottomRows(upcl - j)
            * ws(enter, j);
        wn1.col(m + j).segment(m + j, upcl - j).noalias() +=
            ws(leave, Eigen::all).transpose().bottomRows(upcl - j)
            * ws(leave, j);
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

    void formk_prepare_wn(MutBlock<MatrixXd> wnt, const MatrixXd &wn1,
                          Eigen::Block<MatrixXd> sy, const double theta,
                          const int m, const int col) {
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

    bool lbfgsb_formk(LBfgsBase &L, LBfgsBImpl &impl) {
      // Form the lower triangular part of
      //           WN1 = [Y' ZZ'Y   L_a'+R_z']
      //                 [L_a+R_z   S'AA'S   ]
      //    where L_a is the strictly lower triangular part of S'AA'Y
      //          R_z is the upper triangular part of S'ZZ'Y.
      if (L.updated())
        formk_add_new_wn1(L, impl);

      formk_update_wn1(L, impl);

      // Form the upper triangle of WN = [D+Y' ZZ'Y/theta   -L_a'+R_z' ]
      //                                 [-L_a +R_z        S'AA'S*theta]
      formk_prepare_wn(L.wnt(), impl.wn1(), L.sy(), L.theta(), L.m(), L.col());

      // Form the upper triangle of WN= [  LL'            L^-1(-L_a'+R_z')]
      //                                [(-L_a +R_z)L'^-1   S'AA'S*theta  ]
      return formk_factorize_wn(L.wnt(), L.col());
    }

    bool lbfgsb_cmprlb(LBfgsBase &L, LBfgsBImpl &impl, const ArrayXd &gx) {
      const auto &x = L.x();
      auto ws = L.ws(), wy = L.wy();
      auto sy = L.sy(), wtt = L.wtt();
      auto p = L.p(), c = L.c();
      const auto &z = L.z();
      auto smul = L.smul();
      auto free = impl.free();
      const double theta = L.theta();
      const auto col = L.col();
      const bool constrained = impl.constrained();

      if (!constrained && col > 0) {
        L.r() = -gx;
        return true;
      }

      auto r = L.r().head(impl.nfree()).matrix();
      r = (theta * (x - z) - gx)(free);

      if (!lbfgs_bmv_impl(p, smul, c, sy, wtt))
        return false;

      r.noalias() += wy(free, Eigen::all) * p.head(col);
      r.noalias() += theta * (ws(free, Eigen::all) * p.tail(col));

      return true;
    }
  }  // namespace

  bool LBfgsBImpl::prepare_lnsrch(LBfgsBase &lbfgs, const ArrayXd &gx,
                                  double sbgnrm, int iter) {
    bool need_k;
    if (!constrained_ && lbfgs.col() > 0) {
      lbfgs.z() = lbfgs.x();
      need_k = lbfgs.updated();
    } else {
      if (!lbfgsb_cauchy(lbfgs, *this, gx, sbgnrm))
        return false;

      need_k = freev(iter) || lbfgs.updated();
    }

    if (nfree_ > 0 && lbfgs.col() > 0) {
      if (need_k) {
        if (!lbfgsb_formk(lbfgs, *this))
          return false;
      }

      if (!lbfgsb_cmprlb(lbfgs, *this, gx))
        return false;
      if (!lbfgsb_subsm(lbfgs, *this, gx))
        return false;
    }

    return true;
  }

  double LBfgsBImpl::projgr(ConstRef<ArrayXd> x, const ArrayXd &gx) {
    double norm = 0.0;

    for (int i = 0; i < gx.size(); ++i) {
      double gxi = gx[i];

      if (gxi < 0) {
        if (bounds_.has_ub(i)) {
          gxi = nuri::max(x[i] - bounds_.ub(i), gxi);
        }
      } else if (bounds_.has_lb(i)) {
        gxi = nuri::min(x[i] - bounds_.lb(i), gxi);
      }

      norm = nuri::max(norm, std::abs(gxi));
    }

    return norm;
  }

  bool LBfgsBImpl::freev(const int iter) {
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

    return nenter_ + nleave_ > 0;
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

    CauchyHandleBreaksResult
    cauchy_handle_breaks(LBfgsBase &L, LBfgsBImpl &impl, double f1, double f2,
                         const double f2_org, const bool bounded) {
      const auto &x = L.x();
      const auto &bounds = impl.bounds();

      auto ws = L.ws(), wy = L.wy();
      auto sy = L.sy(), wtt = L.wtt();
      auto p = L.p(), v = L.v(), c = L.c(), wbp = L.wbp();
      auto &xcp = L.z(), &d = L.d();
      auto smul = L.smul();
      auto &iwhere = impl.iwhere();
      auto &pq = impl.brks();
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
          if (!lbfgs_bmv_impl(v, smul, wbp, sy, wtt)) {
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

  bool lbfgsb_cauchy(LBfgsBase &lbfgsb, LBfgsBImpl &impl, const ArrayXd &gx,
                     double sbgnrm) {
    // NOLINTNEXTLINE(readability-identifier-naming)
    auto &L = lbfgsb;

    const auto &x = L.x();
    const auto &bounds = impl.bounds();

    auto ws = L.ws(), wy = L.wy();
    auto sy = L.sy(), wtt = L.wtt();
    auto p = L.p(), v = L.v(), c = L.c();
    auto &xcp = L.xcp(), &d = L.d();
    auto smul = L.smul();
    auto &iwhere = impl.iwhere();
    auto &brks = impl.brks();
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
      if (!lbfgs_bmv_impl(v, smul, p, sy, wtt))
        return false;
      f2 -= v.dot(p);
    }

    auto [dtm, tsum, success, z_is_gcp] =
        cauchy_handle_breaks(L, impl, f1, f2, f2_org, bounded);
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

  bool lbfgsb_subsm(LBfgsBase &lbfgsb, LBfgsBImpl &impl, const ArrayXd &gg) {
    // NOLINTNEXTLINE(readability-identifier-naming)
    auto &L = lbfgsb;

    auto &xx = L.x();
    const auto &bounds = impl.bounds();
    auto wnt = L.wnt();
    auto ws = L.ws(), wy = L.wy();
    auto wv = L.p();
    auto &x = L.z(), &xp = L.xp();
    auto d = L.r().head(impl.nfree()).matrix();
    auto free = impl.free();
    const double theta = L.theta();

    const int col = L.col();
    const auto nsub = impl.nfree();
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
    if (!impl.constrained()) {
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
}  // namespace internal
}  // namespace nuri
