//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_GUESS_H_
#define NURI_ALGO_GUESS_H_

#include <absl/base/attributes.h>

#include "nuri/core/molecule.h"

namespace nuri {
constexpr double kDefaultThreshold = 0.5;

/**
 * @brief Guess bonds, types of atoms, and number of hydrogens of a molecule.
 * @param mut The mutator of the molecule to be guessed.
 * @param conf The index of the conformation used for guessing.
 * @param threshold The threshold for guessing bonds.
 * @return true if the guessing is successful.
 * @note The behavior is undefined if the conformer index is out of range.
 *
 * This function find extra bonds that are not in the input molecule, and add
 * them to the molecule. The information present in the molecule will be
 * preserved, except the newly added bonds and atoms that are connected to them.
 *
 * If connectivity information is already present and is correct, consider using
 * guess_all_types().
 */
ABSL_MUST_USE_RESULT extern bool
guess_everything(MoleculeMutator &mut, int conf = 0,
                 double threshold = kDefaultThreshold);

/**
 * @brief Guess connectivity information of a molecule.
 * @param mut The mutator of the molecule to be guessed.
 * @param conf The index of the conformation used for guessing.
 * @param threshold The threshold for guessing bonds.
 * @note The behavior is undefined if the conformer index is out of range.
 *
 * This function assumes all connectivity information is missing. The
 * information present in the molecule could be overwritten by this function.
 */
extern void guess_connectivity(MoleculeMutator &mut, int conf = 0,
                               double threshold = kDefaultThreshold);

/**
 * @brief Guess types of atoms and bonds, and number of hydrogens of a molecule.
 * @param mol The molecule to be guessed.
 * @param conf The index of the conformation used for guessing.
 * @return true if the guessing is successful.
 * @note The behavior is undefined if the conformer index is out of range.
 *
 * This function assumes all connectivity information is present and correct,
 * and all atom/bond types and implicit hydrogen counts are incorrect. The
 * information present in the molecule could be overwritten by this function.
 */
ABSL_MUST_USE_RESULT extern bool guess_all_types(Molecule &mol, int conf = 0);

/**
 * @brief Guess formal charges of a molecule.
 * @param mol The molecule to be guessed.
 * @pre The molecule must have correct number of hydrogens.
 *
 * This function assumses that all atoms >= group 15 satisfy octet rule, and <=
 * group 13 has no non-bonding electrons. Skip charge calculation on group 3-12
 * & 14 atoms, dummy atom, and atoms with explicit formal charge.
 */
extern void guess_fcharge_2d(Molecule &mol);

/**
 * @brief Guess hydrogens of a molecule.
 * @param mol The molecule to be guessed.
 * @pre The molecule must have correct formal charges.
 *
 * Assumses that all (appropriate) atoms satisfy octet rule. Add hydrogens only
 * to the main group elements.
 */
extern void guess_hydrogens_2d(Molecule &mol);

/**
 * @brief Guess formal charges and implicit hydrogens of a molecule.
 * @param mol The molecule to be guessed.
 *
 * This function is equivalent to calling guess_fcharge() and
 * guess_hydrogens() in sequence.
 */
extern void guess_fcharge_hydrogens_2d(Molecule &mol);

namespace internal {
  ABSL_MUST_USE_RESULT extern bool guess_update_subs(Molecule &mol);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_ALGO_GUESS_H_ */
