//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdlib>

#include <absl/base/optimization.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "optim_internal.h"
#include "nuri/algo/optim.h"

namespace nuri {
namespace internal {
  LBfgsBase::LBfgsBase(MutRef<ArrayXd> x, int m)
      : x_(x), wnt_(2 * m, 2 * m), ws_(n(), m), wy_(n(), m), ss_(m, m),
        sy_(m, m), wtt_(m, m), p_(2 * m), v_(2 * m), c_(2 * m), wbp_(2 * m),
        z_(n()), r_(n()), t_(n()), d_(n()), smul_(m - 1) { }

  bool lbfgs_bmv(MutVecBlock<VectorXd> p, MutVecBlock<ArrayXd> smul,
                 ConstRef<VectorXd> v, ConstRef<MatrixXd> sy,
                 ConstRef<MatrixXd> wtt) {
    return lbfgs_bmv_impl(p, smul, v, sy, wtt);
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

      ss.col(ci).head(col).noalias() =
          ws.leftCols(col).transpose() * d.matrix();
      sy.row(ci).head(col).noalias() =
          d.transpose().matrix() * wy.leftCols(col);

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
        sy_scaled.head(i).array() = sy.row(i).head(i).transpose().array()
                                    / sy.diagonal().head(i).array();
        wtt.col(i).tail(col - i).noalias() +=
            sy.bottomLeftCorner(col - i, i) * sy_scaled.head(i);
      }

      Eigen::LLT<Eigen::Ref<MatrixXd>> llt(wtt);
      return llt.info() == Eigen::Success;
    }
  }  // namespace

  bool LBfgsBase::prepare_next_iter(const double gd, const double ginit,
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
    // wn will be recalculated in the next iteration, use for temporary
    // storage
    return lbfgs_formt(wtt_.topLeftCorner(col(), col()), wnt_.col(0),
                       sy_.topLeftCorner(col(), col()),
                       ss_.topLeftCorner(col(), col()), theta());
  }
}  // namespace internal
}  // namespace nuri
