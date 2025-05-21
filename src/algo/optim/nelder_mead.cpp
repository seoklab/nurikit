//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <utility>

#include <absl/algorithm/container.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include "nuri/algo/optim.h"

namespace nuri {
NelderMead::NelderMead(MutRef<ArrayXXd> data)
    : data_(data), c_(data.rows()), r_(data.rows()), ets_(data.rows()),
      idxs_(data.cols()) {
  ABSL_DCHECK_EQ(data_.rows(), data_.cols());
  ABSL_DCHECK_GE(n(), 1);

  absl::c_iota(idxs_, 0);
}

void NelderMead::argpartiton_min1_max2() {
  auto fx_cmp = [this](int i, int j) { return data_(n(), i) < data_(n(), j); };

  std::nth_element(idxs_.begin(), idxs_.end() - 2, idxs_.end(), fx_cmp);

  auto it = std::min_element(idxs_.begin(), idxs_.end() - 2, fx_cmp);
  std::swap(*it, idxs_[0]);
}

void NelderMead::centroid() {
  c_.head(n()) = data_(Eigen::all, idxs_.head(n())).rowwise().mean().head(n());
}

void NelderMead::reflection(double alpha) {
  r_.head(n()) = (c_ + alpha * (c_ - max())).head(n());
}

void NelderMead::expansion(ArrayXd &e, double gamma) const {
  e.head(n()) = (c_ + gamma * (r_ - c_)).head(n());
}

void NelderMead::contraction(ArrayXd &t, double rho) const {
  t.head(n()) = (c_ + rho * (r_ - c_)).head(n());
}

void NelderMead::shrink(double sigma) {
  auto simplex = data_.topRows(n());

  ets_.head(n()) = (1 - sigma) * min().head(n());
  simplex(Eigen::all, idxs_.tail(n())) =
      (sigma * simplex(Eigen::all, idxs_.tail(n()))).colwise() + ets_.head(n());
}

namespace internal {
  bool nm_check_input(ConstRef<ArrayXXd> data, const double alpha,
                      const double gamma, const double rho,
                      const double sigma) {
    const auto n = data.rows() - 1;

    if (data.rows() != data.cols()) {
      ABSL_LOG(WARNING) << "System size " << data.cols()
                        << " is inconsistent with dimension " << n;
      return false;
    }

    if (n < 1) {
      ABSL_LOG(WARNING) << "Nelder-Mead got <= 0D simplex";
      return false;
    }

    if (alpha <= 0) {
      ABSL_LOG(WARNING) << "Nelder-Mead reflection coefficient must be > 0";
      return false;
    }

    if (gamma <= 1) {
      ABSL_LOG(WARNING) << "Nelder-Mead expansion coefficient must be > 1";
      return false;
    }

    if (rho <= 0 || rho >= 1) {
      ABSL_LOG(WARNING)
          << "Nelder-Mead contraction coefficient must be in range (0, 1)";
      return false;
    }

    if (sigma <= 0 || sigma >= 1) {
      ABSL_LOG(WARNING)
          << "Nelder-Mead shrink coefficient must be in range (0, 1)";
      return false;
    }

    return true;
  }
}  // namespace internal

ArrayXXd nm_prepare_simplex(ConstRef<ArrayXd> x0, double eps) {
  const auto n = x0.size();

  ArrayXXd data(n + 1, n + 1);
  data.topRows(n) = x0.replicate(1, n + 1);
  data.topRightCorner(n, n).matrix().diagonal() +=
      (x0.abs() < eps).select(ArrayXd::Constant(n, 0.00025), 0.05).matrix();
  return data;
}
}  // namespace nuri
