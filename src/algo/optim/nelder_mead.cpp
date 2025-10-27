//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <utility>

#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/algo/optim.h"

namespace nuri {
NelderMead::NelderMead(MutRef<ArrayXXd> data)
    : data_(data), c_(data.rows()), r_(data.rows()), ets_(data.rows()) {
  ABSL_DCHECK_EQ(data_.rows(), data_.cols());
  ABSL_DCHECK_GE(n(), 1);
}

namespace {
  void indirect_argsort3(Array3d &v, Array3i &i) {
    if (v[0] > v[1]) {
      std::swap(v[0], v[1]);
      std::swap(i[0], i[1]);
    }
    if (v[0] > v[2]) {
      std::swap(v[0], v[2]);
      std::swap(i[0], i[2]);
    }
    if (v[1] > v[2]) {
      std::swap(v[1], v[2]);
      std::swap(i[1], i[2]);
    }
  }
}  // namespace

void NelderMead::partiton_min1_max2() {
  auto fs = data_.row(n());

  if (n() == 1) {
    if (fs[0] > fs[1])
      data_.col(0).swap(data_.col(1));
    return;
  }

  Array3d vals = fs.tail<3>();
  Array3i idxs = { argmax() - 2, argmax() - 1, argmax() };
  indirect_argsort3(vals, idxs);

  for (int i = argmax() - 3; i >= 0; --i) {
    double v = fs[i];
    if (v > vals[2]) {
      vals[1] = vals[2];
      idxs[1] = idxs[2];
      vals[2] = v;
      idxs[2] = i;
    } else if (v > vals[1]) {
      vals[1] = v;
      idxs[1] = i;
    } else if (v <= vals[0]) {
      vals[0] = v;
      idxs[0] = i;
    }
  }

  Array3i out = { 0, argmax() - 1, argmax() };
  for (int i = 0; i < 3; ++i) {
    if (out[i] == idxs[i])
      continue;

    data_.col(out[i]).swap(data_.col(idxs[i]));

    for (int j = i + 1; j < 3; ++j) {
      if (out[i] == idxs[j])
        idxs[j] = idxs[i];
    }
  }
}

void NelderMead::centroid() {
  c_.head(n()) = data_.leftCols(n()).rowwise().mean().head(n());
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
  simplex.rightCols(n()) =
      (sigma * simplex.rightCols(n())).colwise() + ets_.head(n());
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
