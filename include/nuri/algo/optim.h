//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_OPTIM_H_
#define NURI_ALGO_OPTIM_H_

//! @cond
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>
//! @endcond

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
  constexpr double kSqrtEpsMach =
      1.4901161193847655978721599999999997530663236602513281125266e-8;

  enum class DcsrchStatus : std::uint8_t {
    kFound,
    kConverged,
    kContinue,
  };

  class Dcsrch {
  public:
    Dcsrch(double f0, double g0, double step0, double stepmin = 0,
           double stepmax = 1e+10, double ftol = 1e-3, double gtol = 0.9,
           double xtol = 0.1) noexcept;

    DcsrchStatus operator()(double f, double g);

    double step() const { return step_; }

    double finit() const { return finit_; }

    double ginit() const { return ginit_; }

  private:
    constexpr static double kXTrapL = 1.1, kXTrapU = 4;

    double finit_, ginit_;
    double stepmin_, stepmax_;
    double gtest_, gtol_, xtol_;

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

    bool brackt_ = false, bisect_ = false;
  };

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

    double step() const { return dcsrch_.step(); }

    double xstep() const { return step() * dnorm_; }

    double finit() const { return dcsrch_.finit(); }

    double ginit() const { return dcsrch_.ginit(); }

    constexpr static double kStepMin = 0, kStepMax = 1e+10;

  private:
    MutRef<ArrayXd> &x() { return *x_; }

    const ArrayXd &t() const { return *t_; }

    const ArrayXd &z() const { return *z_; }

    const ArrayXd &d() const { return *d_; }

    const LbfgsbBounds &bounds() const { return *bounds_; }

    void step_x();

    MutRef<ArrayXd> *x_;
    const ArrayXd *t_, *z_, *d_;
    const LbfgsbBounds *bounds_;
    double dtd_, dnorm_;

    Dcsrch dcsrch_;
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

/**
 * @brief L-BFGS-B minimizer
 * @sa l_bfgs_b
 *
 * References:
 * - RH Byrd, P Lu, J Nocedal and C Zhu. *SIAM J. Sci. Stat. Comput.* **1995**,
 *   *16* (5), 1190-1208. DOI:[10.1137/0916069](https://doi.org/10.1137/0916069)
 * - C Zhu, RH Byrd, J Nocedal. *ACM Trans. Math. Softw.* **1997**, *23* (4),
 *   550-560. DOI:[10.1145/279232.279236](https://doi.org/10.1145/279232.279236)
 * - JL Morales, J Nocedal. *ACM Trans. Math. Softw.* **2011**, *38* (1), 7.
 *   DOI:[10.1145/2049662.2049669](https://doi.org/10.1145/2049662.2049669)
 *
 * This implementation is based on the C implementation of L-BFGS-B in the SciPy
 * library, which is a translation of the original Fortran code by Ciyou Zhu,
 * Richard Byrd, and Jorge Noceda. Both are released under the BSD 3-Clause
 * License and the original license is included below.
 *
 * \code{.unparsed}
 * Copyright (c) 2011 Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis
 * Morales.
 * Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endcode
 */
class LBfgsB {
public:
  /**
   * @brief Prepare L-BFGS-B minimization algorithm.
   *
   * @param x Initial guess. Will be modified in-place.
   * @param bounds Bounds for each variable.
   * @param m The maximum number of variable metric corrections used to define
   *        the limited memory matrix.
   */
  LBfgsB(MutRef<ArrayXd> x, internal::LbfgsbBounds bounds, int m = 10);

  /**
   * @brief Minimize a function using L-BFGS-B algorithm.
   *
   * @tparam FuncGrad Function object that computes the function value and
   *         gradient. Function value should be returned and gradient should be
   *         updated in the input gradient vector.
   * @param fg Function object.
   * @param factr Stop when function value changes by less than this factor
   *        times the machine precision.
   * @param pgtol Stop when the projected gradient is less than this value.
   * @param maxiter Maximum number of iterations.
   * @param maxls Maximum number of line search steps.
   * @return A struct with the result code, number of iterations, final function
   *         value, and final gradient.
   */
  template <class FuncGrad>
  LbfgsbResult minimize(FuncGrad fg, double factr = 1e+7, double pgtol = 1e-5,
                        int maxiter = 15000, int maxls = 20);

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
                              const double pgtol, const int maxiter,
                              const int maxls) {
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

/**
 * @brief Minimize a function using L-BFGS-B algorithm.
 *
 * @tparam FuncGrad Function object that computes the function value and
 *         gradient. Function value should be returned and gradient should be
 *         updated in the input gradient vector.
 * @param fg Function object.
 * @param x Initial guess. Will be modified in-place.
 * @param nbd Bound type for each variable. 0x1 if has lower bound, 0x2 if has
 *        upper bound, 0x1 | 0x2 if both.
 * @param bounds Bounds for each variable. First row is lower bound and second
 *        row is upper bound.
 * @param m The maximum number of variable metric corrections used to define the
 *          limited memory matrix.
 * @param factr Stop when function value changes by less than this factor times
 *        the machine precision.
 * @param pgtol Stop when the projected gradient is less than this value.
 * @param maxiter Maximum number of iterations.
 * @param maxls Maximum number of line search steps.
 * @return A struct with the result code, number of iterations, final function
 *         value, and final gradient.
 *
 * @note The input `x` will be modified in-place.
 * @sa LBfgsB
 *
 * References:
 * - RH Byrd, P Lu, J Nocedal and C Zhu. *SIAM J. Sci. Stat. Comput.* **1995**,
 *   *16* (5), 1190-1208. DOI:[10.1137/0916069](https://doi.org/10.1137/0916069)
 * - C Zhu, RH Byrd, J Nocedal. *ACM Trans. Math. Softw.* **1997**, *23* (4),
 *   550-560. DOI:[10.1145/279232.279236](https://doi.org/10.1145/279232.279236)
 * - JL Morales, J Nocedal. *ACM Trans. Math. Softw.* **2011**, *38* (1), 7.
 *   DOI:[10.1145/2049662.2049669](https://doi.org/10.1145/2049662.2049669)
 *
 * This implementation is based on the C implementation of L-BFGS-B in the SciPy
 * library, which is a translation of the original Fortran code by Ciyou Zhu,
 * Richard Byrd, and Jorge Noceda. Both are released under the BSD 3-Clause
 * License and the original license is included below.
 *
 * \code{.unparsed}
 * Copyright (c) 2011 Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis
 * Morales.
 * Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endcode
 */
template <class FuncGrad>
LbfgsbResult l_bfgs_b(FuncGrad &&fg, MutRef<ArrayXd> x, const ArrayXi &nbd,
                      const Array2Xd &bounds, const int m = 10,
                      const double factr = 1e+7, const double pgtol = 1e-5,
                      const int maxiter = 15000, const int maxls = 20) {
  bool args_ok = internal::lbfgsb_errclb(x, nbd, bounds, m, factr);
  if (!args_ok)
    return { LbfgsbResultCode::kInvalidInput, 0, 0, {} };

  LBfgsB lbfgsb(x, { nbd, bounds }, m);
  return lbfgsb.minimize(std::forward<FuncGrad>(fg), factr, pgtol, maxiter,
                         maxls);
}

enum class BfgsResultCode {
  kSuccess,
  kMaxIterReached,
  kInvalidInput,
  kAbnormalTerm,
};

struct BfgsResult {
  BfgsResultCode code;
  int niter;
  double fx;
  ArrayXd gx;
};

/**
 * @brief BFGS minimizer
 * @sa bfgs
 *
 * References:
 *   - "Broyden-Fletcher-Goldfarb-Shanno algorithm",
 *     [Wikipedia](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
 *     (Accessed 2024-10-25).
 *
 * This implementation is based on the Python implementation of BFGS in the
 * SciPy library, with optimized Hessian update step suggested by the linked
 * Wikipedia page. The original implementation is released under the BSD
 * 3-Clause License (included below).
 *
 * \code{.unparsed}
 * Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endcode
 */
class Bfgs {
public:
  /**
   * @brief Prepare BFGS minimization algorithm.
   *
   * @param x Initial guess. Will be modified in-place.
   */
  Bfgs(MutRef<ArrayXd> x);

  /**
   * @brief Minimize a function using BFGS algorithm.
   *
   * @tparam FuncGrad Function object that computes the function value and
   *         gradient. Function value should be returned and gradient should be
   *         updated in the input gradient vector.
   * @param fg Function object.
   * @param pgtol Stop when the projected gradient is less than this value.
   * @param xrtol Stop when the relative change in x is less than this value.
   * @param maxiter Maximum number of iterations. If negative, it will be set to
   *        200 times the number of variables.
   * @param maxls Maximum number of line search steps.
   * @param ftol Tolerance for the relative change in the function value.
   * @param gtol Gradient tolerance for the line search step.
   * @param xtol Tolerance for the relative change in x.
   * @return A struct with the result code, number of iterations, final function
   *         value, and final gradient.
   */
  template <class FuncGrad>
  BfgsResult minimize(FuncGrad fg, const double pgtol = 1e-5,
                      const double xrtol = 0, int maxiter = -1,
                      const int maxls = 100, const double ftol = 1e-4,
                      const double gtol = 0.9, const double xtol = 1e-14) {
    using internal::Dcsrch;
    using internal::DcsrchStatus;

    if (maxiter < 0)
      maxiter = 200 * static_cast<int>(x().size());

    Hk_.setIdentity();

    ArrayXd gfk(x().size());

    const double f0 = fg(gfk, x());
    double gnorm = gfk.abs().maxCoeff();
    if (gnorm <= pgtol)
      return { BfgsResultCode::kSuccess, 0, f0, std::move(gfk) };

    int k = 0;
    double fk = f0, fkm1 = fk + gfk.matrix().norm() * 0.5;
    for (; k < maxiter; ++k) {
      Dcsrch dcsrch = prepare_lnsrch(gfk, fk, fkm1, ftol, gtol, xtol);
      fkm1 = fk;

      bool success = false;
      for (int iter = 0; iter < maxls; ++iter) {
        xk() = x() + dcsrch.step() * pk().array();
        fk = fg(gfkp1(), xk());

        auto status = dcsrch(fk, gfkp1().matrix().dot(pk()));
        if (status == DcsrchStatus::kContinue)
          continue;

        success = true;
        break;
      }
      if (!success)
        return { BfgsResultCode::kAbnormalTerm, k + 1, fk, std::move(gfk) };

      bool converged = prepare_next_iter(gfk, dcsrch.step(), pgtol, xrtol);
      if (converged)
        return { BfgsResultCode::kSuccess, k + 1, fk, std::move(gfk) };
    }

    return { BfgsResultCode::kMaxIterReached, maxiter, fk, std::move(gfk) };
  }

private:
  internal::Dcsrch prepare_lnsrch(const ArrayXd &gfk, double fk, double fkm1,
                                  double ftol, double gtol, double xtol);

  bool prepare_next_iter(ArrayXd &gfk, double step, double pgtol, double xrtol);

  MutRef<ArrayXd> &x() { return x_; }

  // NOLINTNEXTLINE(readability-identifier-naming)
  Eigen::SelfAdjointView<MatrixXd, Eigen::Upper> Hk() {
    return Hk_.selfadjointView<Eigen::Upper>();
  }

  VectorXd &pk() { return pk_; }

  ArrayXd &xk() { return xk_; }
  Eigen::MatrixWrapper<ArrayXd> sk() { return xk_.matrix(); }

  VectorXd &yk() { return yk_; }

  ArrayXd &gfkp1() { return gfkp1_; }
  // NOLINTNEXTLINE(readability-identifier-naming)
  Eigen::MatrixWrapper<ArrayXd> Hk_yk() { return gfkp1_.matrix(); }

  MutRef<ArrayXd> x_;
  ArrayXd xk_, gfkp1_;

  MatrixXd Hk_;
  VectorXd pk_, yk_;
};

/**
 * @brief Minimize a function using BFGS algorithm.
 *
 * @tparam FuncGrad Function object that computes the function value and
 *         gradient. Function value should be returned and gradient should be
 *         updated in the input gradient vector.
 * @param fg Function object.
 * @param x Initial guess. Will be modified in-place.
 * @param pgtol Stop when the projected gradient is less than this value.
 * @param xrtol Stop when the relative change in x is less than this value.
 * @param maxiter Maximum number of iterations. If negative, it will be set to
 *        200 times the number of variables.
 * @param maxls Maximum number of line search steps.
 * @param ftol Tolerance for the relative change in the function value.
 * @param gtol Gradient tolerance for the line search step.
 * @param xtol Tolerance for the relative change in x.
 * @return A struct with the result code, number of iterations, final function
 *         value, and final gradient.
 *
 * @note The input `x` will be modified in-place.
 * @sa Bfgs
 *
 * References:
 *   - "Broyden-Fletcher-Goldfarb-Shanno algorithm",
 *     [Wikipedia](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
 *     (Accessed 2024-10-25).
 *
 * This implementation is based on the Python implementation of BFGS in the
 * SciPy library, with optimized Hessian update step suggested by the linked
 * Wikipedia page. The original implementation is released under the BSD
 * 3-Clause License (included below).
 *
 * \code{.unparsed}
 * Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endcode
 */
template <class FuncGrad>
inline BfgsResult bfgs(FuncGrad &&fg, MutRef<ArrayXd> x,
                       const double pgtol = 1e-5, const double xrtol = 0,
                       int maxiter = -1, const int maxls = 100,
                       const double ftol = 1e-4, const double gtol = 0.9,
                       const double xtol = 1e-14) {
  Bfgs bfgs(x);
  return bfgs.minimize(std::forward<FuncGrad>(fg), pgtol, xrtol, maxiter, maxls,
                       ftol, gtol, xtol);
}
}  // namespace nuri

#endif /* NURI_ALGO_OPTIM_H_ */
