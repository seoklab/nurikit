//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdlib>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "optim_internal.h"
#include "nuri/algo/optim.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  LbfgsLnsrch::LbfgsLnsrch(MutRef<ArrayXd> &x, const ArrayXd &t,
                           const ArrayXd &z, const ArrayXd &d, const double f0,
                           const double g0, const int iter, const double ftol,
                           const double gtol, const double xtol) noexcept
      : x_(&x), t_(&t), z_(&z), d_(&d), dtd_(d.matrix().squaredNorm()),
        dnorm_(std::sqrt(dtd_)),
        dcsrch_(f0, g0, iter == 0 ? nuri::min(1 / dnorm_, Dcsrch::kStepMax) : 1,
                Dcsrch::kStepMin, Dcsrch::kStepMax, ftol, gtol, xtol) {
    step_x();
  }

  bool LbfgsLnsrch::search(const double f, const double g) {
    auto status = dcsrch_(f, g);
    if (status == DcsrchStatus::kConverged)
      return true;

    step_x();
    return status == DcsrchStatus::kFound;
  }

  void LbfgsLnsrch::step_x() {
    if (step() == 1) {  // NOLINT(clang-diagnostic-float-equal)
      x() = z();
      return;
    }

    x() = step() * d() + t();
  }

  LBfgsImpl::LBfgsImpl(int m): wn1_(m + 1, m) { }

  LbfgsLnsrch LBfgsImpl::lnsrch(MutRef<ArrayXd> &x, const ArrayXd &t,
                                const ArrayXd &z, const ArrayXd &d, double f0,
                                double g0, int iter, double ftol, double gtol,
                                double xtol) noexcept {
    return LbfgsLnsrch(x, t, z, d, f0, g0, iter, ftol, gtol, xtol);
  }

  namespace {
    bool lbfgs_cauchy(LBfgsBase &lbfgs, const ArrayXd &gx, double sbgnrm) {
      // NOLINTNEXTLINE(readability-identifier-naming)
      auto &L = lbfgs;

      const auto &x = L.x();

      auto ws = L.ws(), wy = L.wy();
      auto p = L.p(), c = L.c();
      auto &xcp = L.xcp(), &d = L.d();
      const double theta = L.theta();
      const int col = L.col();

      xcp = x;
      if (sbgnrm <= 0)
        return true;

      d = -gx;
      p.head(col).noalias() = wy.transpose() * d.matrix();
      p.tail(col).noalias() = theta * (ws.transpose() * d.matrix());

      ABSL_DCHECK_GT(theta, 0);

      const double dtm = 1 / theta;
      xcp += dtm * d.array();
      c += dtm * p;

      return true;
    }

    void formk_shift_wn1(MatrixXd &wn1, const int m) {
      const auto mm1 = m - 1;

      //  Shift old part of WN1.
      wn1.topLeftCorner(m, mm1) = wn1.bottomRightCorner(m, mm1);
    }

    void formk_add_new_wn1(LBfgsBase &L, LBfgsImpl &impl) {
      auto &wn1 = impl.wn1();
      auto ws = L.ws(), wy = L.wy();
      const auto prev_col = L.prev_col();

      const auto m = L.m(), col = L.col(), cm1 = col - 1;

      if (prev_col == m)
        formk_shift_wn1(wn1, m);

      // Put new row in block (1,1)
      wn1.row(col).head(col).noalias() = wy.col(cm1).transpose() * wy;
      // Put new column in block (2,1)
      wn1.col(cm1).head(col).noalias() = ws.transpose() * wy.col(cm1);
    }

    void formk_prepare_wn(MutBlock<MatrixXd> wnt, const MatrixXd &wn1,
                          Eigen::Block<MatrixXd> sy, const double theta,
                          const int col) {
      wnt.topLeftCorner(col, col).triangularView<Eigen::Lower>() =
          wn1.block(1, 0, col, col) / theta;
      wnt.diagonal().head(col) += sy.diagonal();

      wnt.bottomLeftCorner(col, col) =
          wn1.topLeftCorner(col, col).triangularView<Eigen::Upper>();
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
      wnt_11.triangularView<Eigen::Lower>().solveInPlace(
          wnt.bottomLeftCorner(col, col).transpose());

      // Form S'AA'S*theta + (L^-1(-L_a'+R_z'))'L^-1(-L_a'+R_z') in the lower
      //  triangle of (2,2) block of wn_t.
      for (int i = 0; i < col; ++i) {
        // NB: unlike bounded version, overwrite (2,2) block as it should be all
        // zero before this. Worth it? Not sure, but it is an optimization.
        wnt.col(col + i).tail(col - i).noalias() =
            wnt.row(col + i).head(col)
            * wnt.bottomLeftCorner(col - i, col).transpose();
      }

      //     Cholesky factorization of (2,2) block of wn_t.

      // Same here.
      auto wnt_22 = wnt.bottomRightCorner(col, col);
      Eigen::LLT<Eigen::Ref<MatrixXd>> llt_22(wnt_22);
      return llt_22.info() == Eigen::Success;
    }

    bool lbfgs_formk(LBfgsBase &L, LBfgsImpl &impl) {
      // Form the lower triangular part of
      //           WN1 = [Y' ZZ'Y   L_a'+R_z']
      //                 [L_a+R_z   S'AA'S   ]
      //    where L_a is the strictly lower triangular part of S'AA'Y
      //          R_z is the upper triangular part of S'ZZ'Y.
      if (L.updated())
        formk_add_new_wn1(L, impl);

      // Form the upper triangle of WN = [D+Y' ZZ'Y/theta   -L_a'+R_z' ]
      //                                 [-L_a +R_z        S'AA'S*theta]
      formk_prepare_wn(L.wnt(), impl.wn1(), L.sy(), L.theta(), L.col());

      // Form the upper triangle of WN= [  LL'            L^-1(-L_a'+R_z')]
      //                                [(-L_a +R_z)L'^-1   S'AA'S*theta  ]
      return formk_factorize_wn(L.wnt(), L.col());
    }

    void lbfgs_cmprlb(LBfgsBase &L, const ArrayXd &gx) {
      ABSL_DCHECK_GT(L.col(), 0);
      L.r() = -gx;
    }

    bool lbfgs_subsm(LBfgsBase &lbfgs) {
      // NOLINTNEXTLINE(readability-identifier-naming)
      auto &L = lbfgs;

      auto wnt = L.wnt();
      auto ws = L.ws(), wy = L.wy();
      auto wv = L.p();
      auto &x = L.z();
      auto d = L.r().matrix();
      const double theta = L.theta();

      const int col = L.col();

      // Compute wv = W'Zd.
      wv.head(col).noalias() = wy.transpose() * d;
      wv.tail(col).noalias() = theta * (ws.transpose() * d);

      // Compute wv:=K^(-1)wv.
      if ((wnt.diagonal().array() == 0).any())
        return false;
      wnt.triangularView<Eigen::Lower>().solveInPlace(wv);
      wv.head(col) = -wv.head(col);
      wnt.triangularView<Eigen::Lower>().transpose().solveInPlace(wv);

      // Compute d = (1/theta)d + (1/theta**2)Z'W wv.
      d.noalias() += (1 / theta) * (wy * wv.head(col));
      d.noalias() += ws * wv.tail(col);
      d /= theta;

      // -----------------------------------------------------
      // Let us try the projection, d is the Newton direction.
      x += d.array();
      return true;
    }
  }  // namespace

  bool LBfgsImpl::prepare_lnsrch(LBfgsBase &lbfgs, const ArrayXd &gx,
                                 double sbgnrm, int /* iter */) {
    if (lbfgs.col() == 0)
      return lbfgs_cauchy(lbfgs, gx, sbgnrm);

    lbfgs.z() = lbfgs.x();

    if (lbfgs.updated() && !lbfgs_formk(lbfgs, *this))
      return false;

    lbfgs_cmprlb(lbfgs, gx);

    return lbfgs_subsm(lbfgs);
  }

  double LBfgsImpl::projgr(ConstRef<ArrayXd> /* x */, const ArrayXd &gx) {
    return gx.abs().maxCoeff();
  }

  bool lbfgs_errclb(const ArrayXd &x, const int m, const double factr) {
    return x.size() > 0 && m > 0 && factr >= 0.0;
  }
}  // namespace internal
}  // namespace nuri
