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
template <class DT>
NelderMead<DT>::NelderMead(MutRef<ArrayXX<DT>> data)
    : data_(data), c_(data.rows()), r_(data.rows()), ets_(data.rows()) {
  ABSL_DCHECK_EQ(data_.rows(), data_.cols());
  ABSL_DCHECK_GE(n(), 1);
}

namespace {
  template <class DT>
  using Array3 = Eigen::Array<DT, 3, 1>;

  template <class DT>
  void indirect_argsort3(Array3<DT> &v, Array3i &i) {
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

template <class DT>
void NelderMead<DT>::partiton_min1_max2() {
  auto fs = data_.row(n());

  if (n() == 1) {
    if (fs[0] > fs[1])
      data_.col(0).swap(data_.col(1));
    return;
  }

  Array3<DT> vals = fs.template tail<3>();
  Array3i idxs = { argmax() - 2, argmax() - 1, argmax() };
  indirect_argsort3(vals, idxs);

  for (int i = argmax() - 3; i >= 0; --i) {
    DT v = fs[i];
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

template <class DT>
void NelderMead<DT>::centroid() {
  c_.head(n()) = data_.leftCols(n()).rowwise().mean().head(n());
}

template <class DT>
void NelderMead<DT>::reflection(DT alpha) {
  r_.head(n()) = (c_ + alpha * (c_ - max())).head(n());
}

template <class DT>
void NelderMead<DT>::expansion(ArrayX<DT> &e, DT gamma) const {
  e.head(n()) = (c_ + gamma * (r_ - c_)).head(n());
}

template <class DT>
void NelderMead<DT>::contraction(ArrayX<DT> &t, DT rho) const {
  t.head(n()) = (c_ + rho * (r_ - c_)).head(n());
}

template <class DT>
void NelderMead<DT>::shrink(DT sigma) {
  auto simplex = data_.topRows(n());

  ets_.head(n()) = (1 - sigma) * min().head(n());
  simplex.rightCols(n()) =
      (sigma * simplex.rightCols(n())).colwise() + ets_.head(n());
}

template class NelderMead<double>;
template class NelderMead<float>;

namespace internal {
  template <class DT>
  bool nm_check_input(ConstRef<ArrayXX<DT>> data, const DT alpha,
                      const DT gamma, DT rho, DT sigma) {
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

    if (rho <= static_cast<DT>(0) || rho >= static_cast<DT>(1)) {
      ABSL_LOG(WARNING)
          << "Nelder-Mead contraction coefficient must be in range (0, 1)";
      return false;
    }

    if (sigma <= static_cast<DT>(0) || sigma >= static_cast<DT>(1)) {
      ABSL_LOG(WARNING)
          << "Nelder-Mead shrink coefficient must be in range (0, 1)";
      return false;
    }

    return true;
  }

  template bool nm_check_input<double>(ConstRef<ArrayXX<double>> data,
                                       double alpha, double gamma, double rho,
                                       double sigma);
  template bool nm_check_input<float>(ConstRef<ArrayXX<float>> data,
                                      float alpha, float gamma, float rho,
                                      float sigma);
}  // namespace internal

template <class DT>
ArrayXX<DT> nm_prepare_simplex(ConstRef<ArrayX<DT>> x0, DT eps) {
  const auto n = x0.size();

  ArrayXX<DT> data(n + 1, n + 1);
  data.topRows(n) = x0.replicate(1, n + 1);
  data.topRightCorner(n, n).matrix().diagonal() +=
      (x0.abs() < eps)
          .select(ArrayX<DT>::Constant(n, static_cast<DT>(0.00025)),
                  static_cast<DT>(0.05))
          .matrix();
  return data;
}

template ArrayXX<double> nm_prepare_simplex(ConstRef<ArrayX<double>> x0,
                                            double eps);
template ArrayXX<float> nm_prepare_simplex(ConstRef<ArrayX<float>> x0,
                                           float eps);
}  // namespace nuri
