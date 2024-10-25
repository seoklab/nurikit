//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <Eigen/Dense>

#include <absl/log/absl_log.h>

#include "nuri/eigen_config.h"
#include "nuri/algo/optim.h"

namespace nuri {
using internal::Dcsrch;

Bfgs::Bfgs(MutRef<ArrayXd> x)
    : x_(x), xk_(x.size()), gfkp1_(x.size()), Hk_(x.size(), x.size()),
      pk_(x.size()), yk_(x.size()) { }

Dcsrch Bfgs::prepare_lnsrch(const ArrayXd &gfk, const double fk,
                            const double fkm1, const double ftol,
                            const double gtol, const double xtol) {
  pk().noalias() = -1 * (Hk() * gfk.matrix());

  const double gk = gfk.matrix().dot(pk());
  double step0;
  if (std::abs(gk) > internal::kSqrtEpsMach) {
    step0 = nuri::min(1.0, 1.01 * 2 * (fk - fkm1) / gk);
    if (step0 <= 0)
      step0 = 1;
  } else {
    step0 = 1;
  }

  return { fk, gk, step0, 1e-100, 1e+100, ftol, gtol, xtol };
}

namespace {
  void swap_gfks(ArrayXd &gfk, ArrayXd &gfkp1) noexcept {
    ABSL_ASSUME(gfk.size() == gfkp1.size());
    ABSL_ASSUME(gfk.rows() == gfkp1.rows());
    ABSL_ASSUME(gfk.cols() == gfkp1.cols());

    gfk.swap(gfkp1);
  }
}  // namespace

bool Bfgs::prepare_next_iter(ArrayXd &gfk, const double step,
                             const double pgtol, const double xrtol) {
  sk() = step * pk();
  yk() = gfkp1() - gfk;
  swap_gfks(gfk, gfkp1_);

  x().matrix() += sk();

  const double gnorm = gfk.abs().maxCoeff();
  if (gnorm <= pgtol)
    return true;

  if (step * pk().norm() <= xrtol * (x().matrix().norm() + xrtol))
    return true;

  double rhok_inv = yk().dot(sk()), rhok;
  if (ABSL_PREDICT_FALSE(rhok_inv <= internal::kEpsMach)) {
    ABSL_LOG(WARNING) << "divide-by-zero encountered: rhok assumed large";
    rhok = 1e+3;
  } else {
    rhok = 1 / rhok_inv;
  }

  Hk_yk().noalias() = Hk() * yk();

  Hk().rankUpdate(sk(), rhok * (1 + rhok * yk().dot(Hk_yk())))
      .rankUpdate(Hk_yk(), sk(), -rhok);

  return false;
}
}  // namespace nuri
