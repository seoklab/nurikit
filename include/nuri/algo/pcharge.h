//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_PCHARGE_H_
#define NURI_ALGO_PCHARGE_H_

#include <absl/base/attributes.h>

#include "nuri/core/molecule.h"

namespace nuri {
/**
 * @brief Assign Marsili-Gasteiger charges to the molecule.
 *
 * @param mol The molecule to assign charges to.
 * @param relaxation_steps The number of relaxation steps to perform after the
 *        initial charge assignment (default: 8). When set to 0, no relaxation
 *        is performed.
 * @return Whether the charge assignment was successful.
 */
ABSL_MUST_USE_RESULT
extern bool assign_charges_gasteiger(Molecule &mol, int relaxation_steps = 8);
}  // namespace nuri

#endif /* NURI_ALGO_PCHARGE_H_ */
