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
  struct GARotationInfo {
    int pivot;
    int origin;
    ArrayXi moving;
  };

  extern std::vector<GARotationInfo>
  ga_resolve_rotation_info(const Molecule &mol);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TOOLS_GALIGN_H_ */
