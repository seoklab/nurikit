//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_CRDGEN_H_
#define NURI_ALGO_CRDGEN_H_

//! @cond
#include <utility>
//! @endcond

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"

namespace nuri {
/**
 * @brief Generate 3D coordinates for the molecule.
 *
 * @param mol The molecule to generate coordinates.
 * @param conf A matrix to store the generated coordinates.
 * @param max_trial The maximum number of trials to generate trial distances.
 * @param seed The seed for the random number generator. Might not be used if
 *        algorithm succeeds in the first trial.
 * @return Whether the coordinates is successfully generated.
 */
extern bool generate_coords(const Molecule &mol, Matrix3Xd &conf,
                            int max_trial = 10, int seed = 0);

/**
 * @brief Generate 3D coordinates for the molecule.
 *
 * @param mol The molecule to generate coordinates.
 * @param max_trial The maximum number of trials to generate trial distances.
 * @param seed The seed for the random number generator. Might not be used if
 *        algorithm succeeds in the first trial.
 * @return Whether the coordinates is successfully generated.
 *
 * @note The generated coordinates is stored as the last conformer of the
 *       molecule, if and only if the coordinates is successfully generated. The
 *       molecule is not modified otherwise.
 */
inline bool generate_coords(Molecule &mol, int max_trial = 10, int seed = 0) {
  Matrix3Xd conf(3, mol.num_atoms());
  bool success = generate_coords(mol, conf, max_trial, seed);
  if (success)
    mol.confs().push_back(std::move(conf));
  return success;
}
}  // namespace nuri

#endif /* NURI_ALGO_CRDGEN_H_ */
