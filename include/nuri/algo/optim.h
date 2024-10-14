//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_OPTIM_H_
#define NURI_ALGO_OPTIM_H_

/// @cond
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>

#include <Eigen/Dense>

#include <absl/log/absl_check.h>
/// @endcond

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
enum class LbfgsbResultCode {
  kSuccess,
  kMaxIterReached,
  kInvalidInput,
  kAbnormalTerm,
};

struct LbfgsbResult {
  LbfgsbResultCode code;
  int niter;
  double fx;
  ArrayXd gx;
};

class LBfgsB;

namespace internal {
  constexpr double kEpsMach = 2.220446049250313e-16;

  /**
   * nbd == 0x1 if has lower bound,
   *        0x2 if has upper bound,
   *        0x1 | 0x2 if both
   */
  class LbfgsbBounds {
  public:
    LbfgsbBounds(const ArrayXi &nbd, const Array2Xd &bounds)
        : nbd_(&nbd), bds_(&bounds) {
      check_sizes();
    }

    LbfgsbBounds(ArrayXi &&nbd, Array2Xd &&bounds) = delete;

    bool has_bound(int i) const { return nbd()[i] != 0; }

    bool has_lb(int i) const { return (nbd()[i] & 0x1) != 0; }

    bool has_ub(int i) const { return (nbd()[i] & 0x2) != 0; }

    bool has_both(int i) const { return nbd()[i] == (0x1 | 0x2); }

    int raw_nbd(int i) const { return nbd()[i]; }

    double lb(int i) const {
      ABSL_DCHECK(has_lb(i));
      return bds()(0, i);
    }

    double ub(int i) const {
      ABSL_DCHECK(has_ub(i));
      return bds()(1, i);
    }

  private:
    const ArrayXi &nbd() const { return *nbd_; }

    const Array2Xd &bds() const { return *bds_; }

    void check_sizes() const {
      ABSL_DCHECK(nbd().size() == bds().cols());
      ABSL_DCHECK(((nbd() >= 0) && (nbd() <= 3)).all());
      ABSL_DCHECK(
          ((nbd() != 3).transpose() || bds().row(0) <= bds().row(1)).all());
    }

    const ArrayXi *nbd_;
    const Array2Xd *bds_;
  };

  class LbfgsbLnsrch {
  public:
    LbfgsbLnsrch(MutRef<ArrayXd> &x, const ArrayXd &t, const ArrayXd &z,
                 const ArrayXd &d, const LbfgsbBounds &bounds, double f0,
                 double g0, int iter, bool constrained, bool boxed,
                 double ftol = 1e-3, double gtol = 0.9,
                 double xtol = 0.1) noexcept;

    bool search(double f, double g);

    double dtd() const { return dtd_; }

    double step() const { return step_; }

    double xstep() const { return step_ * dnorm_; }

    double finit() const { return finit_; }

    double ginit() const { return ginit_; }

  private:
    MutRef<ArrayXd> &x() { return *x_; }

    const ArrayXd &t() const { return *t_; }

    const ArrayXd &z() const { return *z_; }

    const ArrayXd &d() const { return *d_; }

    const LbfgsbBounds &bounds() const { return *bounds_; }

    enum class DcsrchStatus : std::uint8_t;
    DcsrchStatus dcsrch(double f, double g);

    void step_x();

    // NOLINTNEXTLINE(readability-identifier-naming)
    constexpr static double xtrapl_ = 1.1, xtrapu_ = 4, stepmin_ = 0;

    MutRef<ArrayXd> *x_;
    const ArrayXd *t_, *z_, *d_;
    const LbfgsbBounds *bounds_;
    double dtd_, dnorm_;

    double stepmax_ = 1e+10;
    double finit_, ginit_, gtest_;
    double gtol_, xtol_;

    double step_;

    // The variables stx, fx, gx contain the values of the step, function,
    // and derivative at the best step.
    // The variables sty, fy, gy contain the value of the step, function,
    // and derivative at sty.
    // The variables stp, f, g contain the values of the step, function,
    // and derivative at stp.
    double stx_ = 0, fx_, gx_;
    double sty_ = 0, fy_, gy_;
    double stmin_ = 0, stmax_;
    double width_, width1_;

    bool brackt_ = false, stage1_ = true;
  };

  struct CauchyBrkpt {
    int ibp;
    double tj;
  };

  inline bool operator>(CauchyBrkpt lhs, CauchyBrkpt rhs) {
    return lhs.tj > rhs.tj;
  }

  extern bool lbfgsb_errclb(const ArrayXd &x, const ArrayXi &nbd,
                            const Array2Xd &bounds, int m, double factr);

  extern double lbfgsb_projgr(ConstRef<ArrayXd> x, const ArrayXd &gx,
                              const LbfgsbBounds &bounds);

  extern bool lbfgs_bmv(MutVecBlock<VectorXd> p, MutVecBlock<ArrayXd> smul,
                        ConstRef<VectorXd> v, ConstRef<MatrixXd> sy,
                        ConstRef<MatrixXd> wtt);

  extern bool lbfgsb_cauchy(LBfgsB &lbfgsb, const ArrayXd &gx, double sbgnrm);

  extern bool lbfgsb_subsm(LBfgsB &lbfgsb, const ArrayXd &gg);
}  // namespace internal

class LBfgsB {
public:
  LBfgsB(MutRef<ArrayXd> x, internal::LbfgsbBounds bounds, int m);

  template <class FuncGrad>
  LbfgsbResult minimize(FuncGrad fg, double factr, int maxiter, int maxls,
                        double pgtol);

  /* State modifiers, only for implementations */

  int n() const { return static_cast<int>(x_.size()); }

  int m() const { return static_cast<int>(ws_.cols()); }

  auto &x() { return x_; }

  const auto &x() const { return x_; }

  const internal::LbfgsbBounds &bounds() const { return bounds_; }

  bool constrained() const { return constrained_; }

  bool boxed() const { return boxed_; }

  auto wnt() { return wnt_.topLeftCorner(2 * col_, 2 * col_); }
  auto &wnt_raw() { return wnt_; }

  auto &wn1() { return wn1_; }

  auto ws() { return ws_.leftCols(col_); }

  auto wy() { return wy_.leftCols(col_); }

  auto ss() { return ss_.topLeftCorner(col_, col_); }

  auto sy() { return sy_.topLeftCorner(col_, col_); }

  auto wtt() { return wtt_.topLeftCorner(col_, col_); }

  auto p() { return p_.head(2 * col_); }

  auto v() { return v_.head(2 * col_); }

  auto c() { return c_.head(2 * col_); }

  auto wbp() { return wbp_.head(2 * col_); }

  auto smul() { return smul_.head(nonnegative(col_ - 1)); }

  auto &z() { return z_; }
  auto &xcp() { return z_; }

  auto &r() { return r_; }
  auto rfree() { return r_.head(nfree_).matrix(); }

  auto &t() { return t_; }
  auto &xp() { return t_; }

  auto &d() { return d_; }

  auto &iwhere() { return iwhere_; }

  auto free() const { return free_bound_.head(nfree_); }

  auto bound() const { return free_bound_.tail(n() - nfree_); }

  auto enter() const { return enter_leave_.head(nenter_); }

  auto leave() const { return enter_leave_.tail(nleave_); }

  auto &brks() { return brks_; }

  double theta() const { return theta_; }

  int col() const { return col_; }

  int prev_col() const { return prev_col_; }

  bool updated() const { return updated_; }

  /* Test utils; normally not used */

  void update_col(int col) { col_ = col; }

  void update_theta(double theta) { theta_ = theta; }

  auto &free_bound() { return free_bound_; }

  void update_nfree(int nfree) { nfree_ = nfree; }

private:
  bool freev(int iter);

  bool formk();

  bool prepare_lnsrch(const ArrayXd &gx, double sbgnrm, int iter);

  bool prepare_next_iter(double gd, double ginit, double step, double dtd);

  double d_dot(const ArrayXd &gx) const { return gx.matrix().dot(d_.matrix()); }

  void reset_memory() {
    theta_ = 1.0;
    col_ = prev_col_ = 0;
    updated_ = false;
  }

  /*  System, size n */
  MutRef<ArrayXd> x_;
  /* Bound constraints */
  internal::LbfgsbBounds bounds_;

  /* (2m, 2m) */
  MatrixXd wnt_, wn1_;
  /* (n, m) */
  MatrixXd ws_, wy_;
  /* (m, m) */
  MatrixXd ss_, sy_, wtt_;
  /* 2*m */
  VectorXd p_, v_, c_, wbp_;
  /* n */
  ArrayXd z_, r_, t_, d_;
  /* m-1 */
  ArrayXd smul_;

  /* n */
  ArrayXi iwhere_, free_bound_, enter_leave_;

  internal::ClearablePQ<internal::CauchyBrkpt, std::greater<>> brks_;

  double theta_;
  int col_, prev_col_;
  bool constrained_, boxed_, updated_;

  int nfree_, nenter_, nleave_;
};

template <class FuncGrad>
LbfgsbResult LBfgsB::minimize(FuncGrad fg, const double factr,
                              const int maxiter, const int maxls,
                              const double pgtol) {
  const double tol = factr * internal::kEpsMach;

  ArrayXd gx(n());
  double fx = fg(gx, x());

  internal::AllowEigenMallocScoped<false> ems;

  double sbgnrm = internal::lbfgsb_projgr(x(), gx, bounds());
  if (sbgnrm <= pgtol)
    return { LbfgsbResultCode::kSuccess, 0, fx, std::move(gx) };

  reset_memory();
  nfree_ = n();
  nenter_ = nleave_ = 0;

  int iter = 0;
  for (; iter < maxiter; ++iter) {
    if (!prepare_lnsrch(gx, sbgnrm, iter)) {
      reset_memory();
      continue;
    }

    d() = z() - x();

    double gd = d_dot(gx);
    if (gd >= 0) {
      if (col() == 0)
        return { LbfgsbResultCode::kAbnormalTerm, iter + 1, fx, std::move(gx) };

      reset_memory();
      continue;
    }

    t() = x();
    r() = gx;

    internal::LbfgsbLnsrch lnsrlb(x(), t(), z(), d(), bounds(), fx, gd, iter,
                                  constrained(), boxed());
    bool converged = false;
    for (int i = 0; !converged && i < maxls; ++i) {
      NURI_EIGEN_ALLOW_MALLOC(true);
      fx = fg(gx, x());
      NURI_EIGEN_ALLOW_MALLOC(false);

      gd = d_dot(gx);
      converged = lnsrlb.search(fx, gd);
    }

    if (!converged) {
      x() = t();
      fx = lnsrlb.finit();
      gx = r();

      if (col() == 0)
        return { LbfgsbResultCode::kAbnormalTerm, iter + 1, fx, std::move(gx) };

      reset_memory();
      continue;
    }

    sbgnrm = internal::lbfgsb_projgr(x(), gx, bounds());
    if (sbgnrm <= pgtol)
      return { LbfgsbResultCode::kSuccess, iter + 1, fx, std::move(gx) };

    double ddum = std::max({ std::abs(lnsrlb.finit()), std::abs(fx), 1.0 });
    if (lnsrlb.finit() - fx <= tol * ddum)
      return { LbfgsbResultCode::kSuccess, iter + 1, fx, std::move(gx) };

    r() = gx - r();
    if (!prepare_next_iter(gd, lnsrlb.ginit(), lnsrlb.step(), lnsrlb.dtd()))
      reset_memory();
  }

  return { LbfgsbResultCode::kMaxIterReached, maxiter, fx, std::move(gx) };
}

template <class FuncGrad>
LbfgsbResult l_bfgs_b(FuncGrad &&fg, MutRef<ArrayXd> x, const ArrayXi &nbd,
                      const Array2Xd &bounds, const int m = 10,
                      const double factr = 1e+7, const double pgtol = 1e-5,
                      const int maxiter = 15000, const int maxls = 20) {
  bool args_ok = internal::lbfgsb_errclb(x, nbd, bounds, m, factr);
  if (!args_ok)
    return { LbfgsbResultCode::kInvalidInput, 0, 0, {} };

  LBfgsB lbfgsb(x, { nbd, bounds }, m);
  return lbfgsb.minimize(std::forward<FuncGrad>(fg), factr, maxiter, maxls,
                         pgtol);
}
}  // namespace nuri

#endif /* NURI_ALGO_OPTIM_H_ */
