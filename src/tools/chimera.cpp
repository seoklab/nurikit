//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/tools/chimera.h"

#include <cmath>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/utils.h"

namespace nuri {
extern MmResult match_maker(ConstRef<Matrix3Xd> query,
                            ConstRef<Matrix3Xd> templ, const double cutoff,
                            const double global_ratio,
                            const double viol_ratio) {
  const double cutoffsq = cutoff * cutoff;
  const int n = static_cast<int>(query.cols());

  Matrix3Xd qbuf = query, tbuf = templ;
  ArrayXd dsqs(n);
  ArrayXi order(n);
  absl::c_iota(order, 0);

  Isometry3d xform;
  double msd;
  std::tie(xform, msd) = qcp_inplace(qbuf, tbuf);

  int nali = n;
  while (true) {
    if (ABSL_PREDICT_FALSE(msd < 0)) {
      ABSL_LOG(WARNING) << "Alignment failed during Match-Maker iteration";
      nali = 0;
      break;
    }

    auto o = order.head(nali);

    int nviol = 0;
    for (int i: o) {
      double dsq = dsqs[i] =
          (xform * query.col(i) - templ.col(i)).squaredNorm();
      nviol += value_if(dsq > cutoffsq);
    }

    int ndel = static_cast<int>(
        std::ceil(nuri::min(nali * global_ratio, nviol * viol_ratio)));
    if (ndel <= 0)
      break;

    nali -= ndel;
    if (nali <= 0) {
      nali = 0;
      break;
    }

    absl::c_nth_element(o, o.end() - ndel,
                        [&](int i, int j) { return dsqs[i] < dsqs[j]; });

    auto q = qbuf.leftCols(nali);
    auto t = tbuf.leftCols(nali);
    auto p = order.head(nali);

    q = query(E::all, p);
    t = templ(E::all, p);
    std::tie(xform, msd) = qcp_inplace(q, t);
  }

  order.conservativeResize(nali);
  return MmResult { xform, order, msd };
}
}  // namespace nuri
