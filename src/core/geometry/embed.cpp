//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <exception>

#include <absl/log/absl_log.h>
#include <Eigen/Dense>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/Util/SelectionRule.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"

namespace nuri {
namespace {
  template <class PtsLike>
  bool embed_distances_impl(PtsLike pts, MatrixXd &dsqs) {
    using Spectra::CompInfo;
    using Spectra::SortRule;
    constexpr Eigen::Index ndim = PtsLike::RowsAtCompileTime;

    if (dsqs.cols() != dsqs.rows() || dsqs.cols() != pts.cols()) {
      ABSL_LOG(WARNING) << "size mismatch; cannot embed distances";
      return false;
    }

    const int n = static_cast<int>(dsqs.cols());

    double norm_dist = 0;
    for (int i = 1; i < dsqs.cols(); ++i)
      norm_dist += dsqs.col(i).head(i).sum();
    norm_dist /= n * n;

    VectorXd d0sq = dsqs.colwise().mean().array() - norm_dist;

    dsqs *= -1;
    dsqs.colwise() += d0sq;
    dsqs.rowwise() += d0sq.transpose();
    dsqs /= 2;

    ABSL_DVLOG(1) << "metric matrix:\n" << dsqs;

    try {
      Spectra::DenseSymMatProd<double> op(dsqs);
      // This constructor might throw
      Spectra::SymEigsSolver<decltype(op)> eigs(
          op, ndim, std::min(dsqs.cols(), ndim * 2));
      eigs.init();
      auto nconv = eigs.compute(SortRule::LargestAlge);
      if (eigs.info() != CompInfo::Successful || nconv < ndim) {
        ABSL_LOG(WARNING) << "solver failed";
        return false;
      }

      Array<double, ndim, 1> evals_sqrt = eigs.eigenvalues().head<ndim>();
      if ((evals_sqrt < 0).any())
        return false;
      evals_sqrt = evals_sqrt.sqrt();

      pts = (eigs.eigenvectors(ndim).transpose().array().colwise() * evals_sqrt)
                .matrix();
    } catch (const std::exception &e) {
      ABSL_LOG(WARNING) << "solver failed: " << e.what();
      return false;
    }

    return true;
  }
}  // namespace

bool embed_distances_3d(Eigen::Ref<Matrix3Xd> pts, MatrixXd &dsqs) {
  return embed_distances_impl(pts, dsqs);
}

extern bool embed_distances_4d(Eigen::Ref<Matrix4Xd> pts, MatrixXd &dsqs) {
  return embed_distances_impl(pts, dsqs);
}
}  // namespace nuri
