//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_OPTIM_H_
#define NURI_ALGO_OPTIM_H_

#include <cmath>
#include <cstdint>
#include <functional>

#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  constexpr double kEpsMach = 2.220446049250313e-16;

  extern bool lbfgsb_errclb(const ArrayXd &x, const ArrayXi &nbd,
                            const Array2Xd &bounds, int m, double factr);

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
      ABSL_DCHECK((bds().row(0) <= bds().row(1)).all());
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

    // NOLINTNEXTLINE(readability-identifier-naming)
    constexpr static double xtrapl_ = 1.1, xtrapu_ = 4, stepmin_ = 0;

    MutRef<ArrayXd> *x_;
    const ArrayXd *t_, *z_, *d_;
    const LbfgsbBounds *bounds_;
    double dtd_, dnorm_;

    double stepmax_;
    double finit_, ginit_, gtest_;
    double ftol_, gtol_, xtol_;

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

  extern std::pair<bool, bool> lbfgsb_active(MutRef<ArrayXd> &x,
                                             ArrayXi &iwhere,
                                             const LbfgsbBounds &bounds);

  extern double lbfgsb_projgr(ConstRef<ArrayXd> x, const ArrayXd &gx,
                              const LbfgsbBounds &bounds);

  extern bool lbfgsb_bmv(MutRef<VectorXd> &p, MutRef<ArrayXd> &smul,
                         ConstRef<VectorXd> v, ConstRef<MatrixXd> sy,
                         ConstRef<MatrixXd> wtt);

  struct CauchyBrkpt {
    int ibp;
    double tj;
  };

  inline bool operator>(CauchyBrkpt lhs, CauchyBrkpt rhs) {
    return lhs.tj > rhs.tj;
  }

  extern bool lbfgsb_cauchy(
      ArrayXd &xcp, ArrayXi &iwhere, MutRef<VectorXd> &p, MutRef<VectorXd> &v,
      MutRef<VectorXd> &c, ArrayXd &d, MutRef<VectorXd> &wbp,
      MutRef<ArrayXd> &smul, ClearablePQ<CauchyBrkpt, std::greater<>> &brks,
      ConstRef<ArrayXd> x, const ArrayXd &gx, const LbfgsbBounds &bounds,
      ConstRef<MatrixXd> ws, ConstRef<MatrixXd> wy, ConstRef<MatrixXd> sy,
      ConstRef<MatrixXd> wtt, double sbgnrm, double theta);

  extern bool lbfgsb_formk(MutRef<MatrixXd> &wnt, MatrixXd &wn1,
                           Eigen::LLT<MatrixXd> &llt, ConstRef<ArrayXi> free,
                           ConstRef<ArrayXi> bound, ConstRef<ArrayXi> enter,
                           ConstRef<ArrayXi> leave, ConstRef<MatrixXd> ws,
                           ConstRef<MatrixXd> wy, ConstRef<MatrixXd> sy,
                           double theta, bool updated);

  extern bool lbfgsb_cmprlb(MutRef<VectorXd> &r, MutRef<VectorXd> &p,
                            MutRef<ArrayXd> &smul, ConstRef<VectorXd> c,
                            ConstRef<ArrayXi> free, ConstRef<ArrayXd> x,
                            const ArrayXd &z, const ArrayXd &gx,
                            ConstRef<MatrixXd> ws, ConstRef<MatrixXd> wy,
                            ConstRef<MatrixXd> sy, ConstRef<MatrixXd> wtt,
                            double theta, bool constrained);

  extern bool lbfgsb_subsm(ArrayXd &x, ArrayXd &xp, MutRef<VectorXd> &d,
                           MutRef<VectorXd> &wv, ConstRef<MatrixXd> wnt,
                           ConstRef<ArrayXi> free, ConstRef<ArrayXd> xx,
                           const ArrayXd &gg, ConstRef<MatrixXd> ws,
                           ConstRef<MatrixXd> wy, const LbfgsbBounds &bounds,
                           double theta);

  extern bool lbfgsb_prepare_lnsrch(
      MutRef<ArrayXd> x, ArrayXd &z, ArrayXd &xp, ArrayXd &r, ArrayXd &d,
      ArrayXi &iwhere, MatrixXd &wnt_d, MatrixXd &wn1, MatrixXd &ws_d,
      MatrixXd &wy_d, MatrixXd &sy_d, MatrixXd &wtt_d, VectorXd &p_d,
      VectorXd &v_d, VectorXd &c_d, VectorXd &wbp_d, ArrayXi &free_bound,
      int &nfree, ArrayXi &enter_leave, int &nenter, int &nleave,
      ClearablePQ<CauchyBrkpt, std::greater<>> &brks, Eigen::LLT<MatrixXd> &llt,
      const ArrayXd &gx, const LbfgsbBounds &bounds, double sbgnrm,
      double theta, bool updated, bool constrained, int iter, int col);

  extern bool lbfgsb_prepare_next_iter(ArrayXd &r, ArrayXd &d, MatrixXd &ws_d,
                                       MatrixXd &wy_d, MatrixXd &ss_d,
                                       MatrixXd &sy_d, MatrixXd &wtt_d,
                                       MatrixXd &wnt_d, double &theta, int &col,
                                       bool &updated, Eigen::LLT<MatrixXd> &llt,
                                       double gd, double ginit, double step,
                                       double dtd);

  template <class FuncGrad>
  bool lbfgsb_main(FuncGrad fg, MutRef<ArrayXd> x, const LbfgsbBounds &bounds,
                   const int m, const double factr, const int maxiter,
                   const int maxls, const double pgtol) {
    const auto n = x.size();
    const double tol = factr * kEpsMach;

    ArrayXi iwhere = ArrayXi::Zero(n);
    auto [constrained, boxed] = lbfgsb_active(x, iwhere, bounds);

    ArrayXd gx(n);
    double fx = fg(gx, x);
    double sbgnrm = lbfgsb_projgr(x, gx, bounds);
    if (sbgnrm <= pgtol)
      return true;

    MatrixXd wnt(2 * m, 2 * m), wn1(2 * m, 2 * m);
    MatrixXd ws(n, m), wy(n, m);
    MatrixXd ss(m, m), sy(m, m), wtt(m, m);
    VectorXd p(2 * m), v(2 * m), c(2 * m), wbp(3 * m - 1);
    ArrayXd z(n), r(n), t(n), d(n);
    Eigen::LLT<MatrixXd> llt(1);

    ArrayXi free_bound(n), enter_leave(n);
    ClearablePQ<CauchyBrkpt, std::greater<>> brks;
    brks.data().reserve(n);

    int nfree = static_cast<int>(n), nenter = 0, nleave = 0, col = 0;
    double theta = 1;
    bool updated = false;

    auto reset_memory = [&]() {
      col = 0;
      theta = 1.0;
      updated = false;
    };

    for (int iter = 0; iter < maxiter; ++iter) {
      if (!lbfgsb_prepare_lnsrch(
              x, z, t, r, d, iwhere, wnt, wn1, ws, wy, sy, wtt, p, v, c, wbp,
              free_bound, nfree, enter_leave, nenter, nleave, brks, llt, gx,
              bounds, sbgnrm, theta, updated, constrained, iter, col)) {
        reset_memory();
        continue;
      }

      d = z - x;
      t = x;
      r = gx;

      double gd = gx.matrix().dot(d.matrix());
      LbfgsbLnsrch lnsrlb(x, t, z, d, bounds, fx, gd, iter, constrained, boxed);
      bool converged = false;
      for (int i = 0; i < maxls; ++i) {
        if (gd >= 0)
          break;

        converged = lnsrlb.search(fx, gd);
        if (converged)
          break;

        fx = fg(gx, x);
        gd = gx.matrix().dot(d.matrix());
      }

      if (!converged) {
        x = t;
        fx = lnsrlb.finit();
        gx = r;

        // Abnormal termination
        if (col == 0)
          return false;

        reset_memory();
        continue;
      }

      sbgnrm = lbfgsb_projgr(x, gx, bounds);
      if (sbgnrm <= pgtol)
        return true;

      double ddum = std::max({ std::abs(lnsrlb.finit()), std::abs(fx), 1.0 });
      if (lnsrlb.finit() - fx <= tol * ddum)
        return true;

      r = gx - r;
      if (!lbfgsb_prepare_next_iter(r, d, ws, wy, ss, sy, wtt, wnt, theta, col,
                                    updated, llt, gd, lnsrlb.ginit(),
                                    lnsrlb.step(), lnsrlb.dtd())) {
        reset_memory();
      }
    }

    return false;
  }
}  // namespace internal

template <class FuncGrad>
bool l_bfgs_b(FuncGrad &&fg, MutRef<ArrayXd> x, const ArrayXi &nbd,
              const Array2Xd &bounds, const int m, const double factr,
              const int maxiter, const int maxls, const double pgtol) {
  bool args_ok = internal::lbfgsb_errclb(x, nbd, bounds, m, factr);
  if (!args_ok)
    return false;

  internal::LbfgsbBounds bds(nbd, bounds);
  return internal::lbfgsb_main(std::forward<FuncGrad>(fg), x, bds, m, factr,
                               maxiter, maxls, pgtol);
}
}  // namespace nuri

#endif /* NURI_ALGO_OPTIM_H_ */
