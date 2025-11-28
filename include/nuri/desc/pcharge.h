//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_DESC_PCHARGE_H_
#define NURI_DESC_PCHARGE_H_

//! @cond
#include <absl/base/attributes.h>
//! @endcond

#include "nuri/core/molecule.h"

namespace nuri {
/**
 * @brief Assign Marsili-Gasteiger charges to the molecule.
 *
 * @param mol The molecule to assign charges to.
 * @param relaxation_steps The number of relaxation steps to perform after the
 *        initial charge assignment (default: 12). When set to 0, no relaxation
 *        is performed.
 * @return Whether the charge assignment was successful.
 *
 * This implements the original Gasteiger charge assignment algorithm
 * @cite algo:pcharge:gasteiger, which is based on the charge distribution over
 * the sigma bonds. This is the only version that is implemented in most other
 * cheminformatics libraries, including RDKit and OpenBabel.
 *
 * The Gasteiger parameters are taken from RDKit and support most hybridization
 * states of H, Be, B, C, N, O, F, Mg, Al, Si, P, S, Cl, Br, I, and dummy atoms.
 * Unsupported elements and/or hybridization states will be reported as a
 * warning and the algorithm will abort without any charge assignment.
 *
 * The Gasteiger algorithm requires initial "seed" charges to be assigned to
 * atoms. In this implementation, the initial charges are assigned from the
 * (localized) formal charges of the atoms, then a charge delocalization
 * algorithm is applied to the terminal atoms of a conjugated system with the
 * same Gasteiger type (e.g., oxygens of a carboxylate group will be assigned
 * -0.5 charge each).
 *
 * Sets `mol2_charge_type` to `GASTEIGER` in the molecule properties when
 * successful.
 */
ABSL_MUST_USE_RESULT
extern bool assign_charges_gasteiger(Molecule &mol, int relaxation_steps = 12);
}  // namespace nuri

#endif /* NURI_DESC_PCHARGE_H_ */
