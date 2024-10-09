//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_OPTIM_H_
#define NURI_ALGO_OPTIM_H_

#include <functional>

#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  constexpr inline double kEpsMach = 2.220446049250313e-16;

  /**
   * nbd == 0x1 if has lower bound,
   *        0x2 if has upper bound,
   *        0x1 | 0x2 if both
   */
  class LbgfsbBounds {
  public:
    LbgfsbBounds(const ArrayXi &nbd, const Array2Xd &bounds)
        : nbd_(&nbd), bds_(&bounds) {
      check_sizes();
    }

    LbgfsbBounds(ArrayXi &&nbd, Array2Xd &&bounds) = delete;

    bool has_bound(int i) const { return nbd()[i] != 0; }

    bool has_lb(int i) const { return (nbd()[i] & 0x1) != 0; }

    bool has_ub(int i) const { return (nbd()[i] & 0x2) != 0; }

    bool has_both(int i) const { return nbd()[i] == (0x1 | 0x2); }

    int raw_nbd(int i) const { return nbd()[i]; }

    double lb(int i) const {
      ABSL_DCHECK(has_lb(i));
      return bds()(0, i);
    }

    double ub(int i) const {
      ABSL_DCHECK(has_ub(i));
      return bds()(1, i);
    }

  private:
    const ArrayXi &nbd() const { return *nbd_; }

    const Array2Xd &bds() const { return *bds_; }

    void check_sizes() const {
      ABSL_DCHECK(nbd().size() == bds().cols());
      ABSL_DCHECK(((nbd() >= 0) && (nbd() <= 3)).all());
      ABSL_DCHECK((bds().row(0) <= bds().row(1)).all());
    }

    const ArrayXi *nbd_;
    const Array2Xd *bds_;
  };

  extern std::pair<bool, bool> lbfgsb_active(MutRef<ArrayXd> &x,
                                             ArrayXi &iwhere,
                                             const LbgfsbBounds &bounds);

  extern double lbfgsb_projgr(ConstRef<ArrayXd> x, const ArrayXd &gx,
                              const LbgfsbBounds &bounds);

  extern bool lbfgsb_bmv(MutRef<VectorXd> &p, MutRef<ArrayXd> &smul,
                         ConstRef<VectorXd> v, ConstRef<MatrixXd> sy,
                         ConstRef<MatrixXd> wtt);

  struct CauchyBrkpt {
    int ibp;
    double tj;
  };

  inline bool operator>(CauchyBrkpt lhs, CauchyBrkpt rhs) {
    return lhs.tj > rhs.tj;
  }

  extern bool lbfgsb_cauchy(
      ArrayXd &xcp, ArrayXi &iwhere, MutRef<VectorXd> &p, MutRef<VectorXd> &v,
      MutRef<VectorXd> &c, ArrayXd &d, MutRef<VectorXd> &wbp,
      MutRef<ArrayXd> &smul, ClearablePQ<CauchyBrkpt, std::greater<>> &brks,
      ConstRef<ArrayXd> x, const ArrayXd &gx, const LbgfsbBounds &bounds,
      ConstRef<MatrixXd> ws, ConstRef<MatrixXd> wy, ConstRef<MatrixXd> sy,
      ConstRef<MatrixXd> wtt, double sbgnrm, double theta);

  extern bool lbfgsb_formk(MutRef<MatrixXd> &wnt, MatrixXd &wn1,
                           Eigen::LLT<MatrixXd> &llt, ConstRef<ArrayXi> free,
                           ConstRef<ArrayXi> bound, ConstRef<ArrayXi> enter,
                           ConstRef<ArrayXi> leave, ConstRef<MatrixXd> ws,
                           ConstRef<MatrixXd> wy, ConstRef<MatrixXd> sy,
                           double theta, bool updated);

  extern bool lbfgsb_cmprlb(MutRef<VectorXd> &r, MutRef<VectorXd> &p,
                            MutRef<ArrayXd> &smul, ConstRef<VectorXd> c,
                            ConstRef<ArrayXi> free, ConstRef<ArrayXd> x,
                            const ArrayXd &z, const ArrayXd &gx,
                            ConstRef<MatrixXd> ws, ConstRef<MatrixXd> wy,
                            ConstRef<MatrixXd> sy, ConstRef<MatrixXd> wtt,
                            double theta, bool constrained);

  extern bool lbfgsb_subsm(ArrayXd &x, ArrayXd &xp, MutRef<VectorXd> &d,
                           MutRef<VectorXd> &wv, ConstRef<MatrixXd> wnt,
                           ConstRef<ArrayXi> free, ConstRef<ArrayXd> xx,
                           const ArrayXd &gg, ConstRef<MatrixXd> ws,
                           ConstRef<MatrixXd> wy, const LbgfsbBounds &bounds,
                           double theta);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_ALGO_OPTIM_H_ */
