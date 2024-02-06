//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_ALGO_GUESS_H_
#define NURI_ALGO_GUESS_H_

#include "nuri/core/molecule.h"

namespace nuri {
constexpr inline double kDefaultThreshold = 0.5;

/**
 * @brief Guess bonds, types of atoms, and number of hydrogens of a molecule.
 * @param mut The mutator of the molecule to be guessed.
 * @param conf The index of the conformation used for guessing.
 * @param threshold The threshold for guessing bonds.
 * @return true if the guessing is successful.
 *
 * This function assumes all connectivity information is missing, and all atom
 * types and implicit hydrogen counts are incorrect. The information present
 * in the molecule could be overwritten by this function.
 *
 * If connectivity information is already present and is correct, consider using
 * guess_all_types().
 */
extern bool guess_bonds(MoleculeMutator &mut, int conf = 0,
                        double threshold = kDefaultThreshold);

/**
 * @brief Guess bonds, types of atoms, and number of hydrogens of a substructure
 *        of a molecule.
 * @param mut The mutator of the molecule to be guessed.
 * @param sub The substructure to be used for guessing.
 * @param conf The index of the conformation used for guessing.
 * @param threshold The threshold for guessing bonds.
 * @return true if the guessing is successful.
 * @note Atoms and bonds not in the substructure are left untouched.
 *
 * This function assumes all connectivity information is missing, and all atom
 * types and implicit hydrogen counts are incorrect. The information present
 * in the molecule could be overwritten by this function.
 *
 * If connectivity information is already present and is correct, consider using
 * guess_all_types_substruct().
 */
extern bool guess_bonds_subtruct(MoleculeMutator &mut, const Substructure &sub,
                                 int conf = 0,
                                 double threshold = kDefaultThreshold);

/**
 * @brief Guess types of atoms and bonds, and number of hydrogens of a molecule.
 * @param mol The molecule to be guessed.
 * @param conf The index of the conformation used for guessing.
 * @return true if the guessing is successful.
 *
 * This function assumes all connectivity information is present and correct,
 * and all atom/bond types and implicit hydrogen counts are incorrect. The
 * information present in the molecule could be overwritten by this function.
 */
extern bool guess_all_types(Molecule &mol, int conf = 0);

/**
 * @brief Guess types of atoms and bonds, and number of hydrogens of a
 *        substructure of a molecule.
 * @param mol The molecule to be guessed.
 * @param sub The substructure to be used for guessing.
 * @param conf The index of the conformation used for guessing.
 * @return true if the guessing is successful.
 * @note Atoms and bonds not in the substructure are left untouched.
 *
 * This function assumes all connectivity information is present and correct,
 * and all atom/bond types and implicit hydrogen counts are incorrect. The
 * information present in the molecule could be overwritten by this function.
 */
extern bool guess_all_types_substruct(Molecule &mol, const Substructure &sub,
                                      int conf = 0);

/**
 * @brief Guess types of atoms and number of hydrogens of a molecule.
 * @param mol The molecule to be guessed.
 * @param conf The index of the conformation used for guessing.
 * @return true if the guessing is successful.
 *
 * This function assumes all connectivity information and bond types are present
 * and correct, and all atom types and implicit hydrogen counts are incorrect.
 * The information present in the molecule could be overwritten by this
 * function.
 */
extern bool guess_atom_types(Molecule &mol, int conf = 0);

/**
 * @brief Guess types of atoms and number of hydrogens of a substructure of a
 *        molecule.
 * @param mol The molecule to be guessed.
 * @param sub The substructure to be used for guessing.
 * @param conf The index of the conformation used for guessing.
 * @return true if the guessing is successful.
 * @note Atoms and bonds not in the substructure are left untouched.
 *
 * This function assumes all connectivity information and bond types are present
 * and correct, and all atom types and implicit hydrogen counts are incorrect.
 * The information present in the molecule could be overwritten by this
 * function.
 */
extern bool guess_atom_types_substruct(Molecule &mol, const Substructure &sub,
                                       int conf = 0);
}  // namespace nuri

#endif /* NURI_ALGO_GUESS_H_ */
