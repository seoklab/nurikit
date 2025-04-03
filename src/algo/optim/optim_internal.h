//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"

namespace nuri {
namespace internal {
  template <class VT, class SYT, class WTT>
  inline static bool lbfgs_bmv_impl(MutVecBlock<VectorXd> p,
                                    MutVecBlock<ArrayXd> smul, const VT &v,
                                    const SYT &sy, const WTT &wtt) {
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
}  // namespace internal
}  // namespace nuri
