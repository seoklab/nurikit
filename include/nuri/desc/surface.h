//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_DESC_SURFACE_H_
#define NURI_DESC_SURFACE_H_

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"

namespace nuri {
namespace internal {
  enum class SrSasaMethod {
    kAuto = 0,
    kDirect,
    kOctree,
  };
}

/**
 * @brief Calculate the Solvent-Accessible Surface Area (SASA) of a molecule
 *        conformation using the Shrake-Rupley algorithm.
 *
 * @param mol The input molecule.
 * @param conf The conformation matrix.
 * @param nprobe The number of probe spheres. Default is 92.
 * @param rprobe The radius of the probe spheres. Default is 1.4 angstroms.
 * @param method Whether prefer direct or octree method. Default is auto.
 *               This is mainly for testing purpose; in most cases, the auto
 *               method will choose the optimal method.
 * @return The calculated SASA values per atom (in angstroms squared).
 */
extern ArrayXd shrake_rupley_sasa(
    const Molecule &mol, const Matrix3Xd &conf, int nprobe = 92,
    double rprobe = 1.4,
    internal::SrSasaMethod method = internal::SrSasaMethod::kAuto);
}  // namespace nuri

#endif /* NURI_DESC_SURFACE_H_ */
