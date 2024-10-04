//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_OPTIM_H_
#define NURI_ALGO_OPTIM_H_

#include <Eigen/Dense>

#include "nuri/eigen_config.h"

namespace nuri {
namespace internal {
  constexpr double kEpsMach = 2.220446049250313e-16;

  class LbgfsbBounds {
  public:
    LbgfsbBounds(const ArrayXi &nbd, const Array2Xd &bounds)
        : nbd_(nbd), bds_(bounds) {
      check_sizes();
    }

    LbgfsbBounds(ArrayXi &&nbd, Array2Xd &&bounds) noexcept
        : nbd_(std::move(nbd)), bds_(std::move(bounds)) {
      check_sizes();
    }

    bool has_bound(int i) const { return nbd_[i] != 0; }

    bool has_lb(int i) const { return (nbd_[i] & 0x1) != 0; }

    bool has_ub(int i) const { return (nbd_[i] & 0x2) != 0; }

    bool is_boxed(int i) const { return nbd_[i] == (0x1 | 0x2); }

    double lb(int i) const {
      ABSL_DCHECK(has_lb(i));
      return bds_(0, i);
    }

    double ub(int i) const {
      ABSL_DCHECK(has_ub(i));
      return bds_(1, i);
    }

  private:
    void check_sizes() const {
      ABSL_DCHECK(nbd_.size() == bds_.cols());
      ABSL_DCHECK(((nbd_ >= 0) && (nbd_ <= 3)).all());
      ABSL_DCHECK((bds_.row(0) <= bds_.row(1)).all());
    }

    ArrayXi nbd_;
    Array2Xd bds_;
  };

  /**
   * nbd == 0x1 if has lower bound,
   *        0x2 if has upper bound,
   *        0x1 | 0x2 if both
   */

  struct LbfgsbActiveOut {
    ArrayXi iwhere;
    bool projected = false;
    bool constrained = false;
    bool boxed = true;
  };

  extern LbfgsbActiveOut lbfgsb_active(ArrayXd &x, const LbgfsbBounds &bounds);

  extern double lbfgsb_projgr(const ArrayXd &x, const ArrayXd &gx,
                              const LbgfsbBounds &bounds);

  extern bool lbfgsb_bmv(VectorXd &p, const VectorXd &v, const MatrixXd &sy,
                         const MatrixXd &wt);

  /**
   * @param xcp shape (n,)
   * @param iwhere shape (n,)
   * @param p shape (2m,)
   * @param v shape (2m,)
   * @param c shape (2m,)
   * @param x shape (n,)
   * @param gx shape (n,)
   * @param bounds shape (n,)
   * @param ws shape (n, m)
   * @param wy shape (n, m)
   * @param sy shape (m, m)
   * @param wt shape (m, m)
   */
  extern bool lbfgsb_cauchy(ArrayXd &xcp, ArrayXi &iwhere, VectorXd &p,
                            VectorXd &v, VectorXd &c, const ArrayXd &x,
                            const ArrayXd &gx, const LbgfsbBounds &bounds,
                            const MatrixXd &ws, const MatrixXd &wy,
                            const MatrixXd &sy, const MatrixXd &wt,
                            double sbgnrm, double theta);

  extern bool lbfgsb_formk(MatrixXd &wnt, MatrixXd &wn1,
                           Eigen::LLT<MatrixXd> &llt,
                           const Eigen::Ref<const ArrayXi> &free,
                           const Eigen::Ref<const ArrayXi> &bound,
                           const std::vector<int> &enter,
                           const std::vector<int> &leave, const MatrixXd &ws,
                           const MatrixXd &wy, const MatrixXd &sy, double theta,
                           bool updated);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_ALGO_OPTIM_H_ */
