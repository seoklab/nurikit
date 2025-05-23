//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TOOLS_GALIGN_H_
#define NURI_TOOLS_GALIGN_H_

//! @cond
#include <vector>

#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"

namespace nuri {
namespace internal {
  class GARotationInfo {
  public:
    static std::vector<GARotationInfo> from(const Molecule &mol,
                                            const Matrix3Xd &ref);

    Matrix3Xd &rotate(Matrix3Xd &pts, double angle) const;

    int origin() const { return origin_; }
    int ref() const { return ref_; }
    double normalizer() const { return normalizer_; }
    const ArrayXi &moving() const { return moving_; }

  private:
    int origin_;
    int ref_;
    double normalizer_;
    ArrayXi moving_;
  };
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TOOLS_GALIGN_H_ */
