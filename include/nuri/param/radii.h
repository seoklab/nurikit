//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PARAM_RADII_H_
#define NURI_PARAM_RADII_H_

//! @cond
#include <string_view>
//! @endcond

#include "nuri/core/molecule.h"

namespace nuri {
/**
 * @brief The radius returned by a radii scheme for an atom it does not cover.
 *
 * Negative, so callers can test with `< 0` and substitute a fallback such as
 * Element::vdw_radius().
 */
constexpr double kUnknownRadius = -1;

/**
 * @brief United-atom radii of Connolly's MS program for molecular surfaces.
 *
 * Hydrogens are absorbed into the heavy atom they are bonded to, and the heavy
 * atom radius grows with the number of absorbed hydrogens. The set covers the
 * heavy atoms of standard amino acids and selenomethionine:
 *
 * | Atom         | Radius (angstrom) | Atom       | Radius (angstrom) |
 * | ------------ | ---------- | ---------- | ---------- |
 * | C            | 1.80       | N, NH      | 1.65       |
 * | CH           | 1.85       | NH2        | 1.70       |
 * | CH2          | 1.90       | NH3        | 1.75       |
 * | CH3          | 1.95       | O          | 1.60       |
 * | aromatic CH  | 1.90       | OH         | 1.70       |
 * | S, SH, Se    | 1.90       | P          | 1.80       |
 *
 * The values are the default atomic radii of the Molecular Surface Package
 * @cite param:radii:connolly-msp-1993, Table 5 of its user manual
 * @cite param:radii:connolly-msp-manual-2002. The same set, as shipped with
 * Biosym's copy of MS, is listed alongside other radii sets in
 * @cite param:radii:wampler-ms-radii. Carbonyl and carboxyl carbons take the
 * bare-carbon value, and selenium takes the sulfur value. The package's
 * separate nucleic-acid types are not implemented.
 *
 * Both overloads return kUnknownRadius for atoms outside the set.
 */
class ConnollyRadii {
public:
  /**
   * @brief Radius of an atom classified from its chemistry.
   *
   * @param atom The atom; its element, hydrogen count (implicit and explicit)
   *        and aromatic flag select the row.
   * @return The radius in angstroms, or kUnknownRadius for hydrogen, for
   *         elements other than C, N, O, P, S, Se, and for hydrogen counts the
   *         set does not list (CH4, NH4, OH2).
   */
  double operator()(Molecule::Atom atom) const;

  /**
   * @brief Radius of a standard-residue heavy atom by PDB names.
   *
   * @param resname Residue name; one of the 20 standard amino acids or MSE.
   * @param atom_name PDB atom name within the residue, backbone included.
   * @return The radius in angstroms, or kUnknownRadius when the residue or the
   *         atom name is not in the templates.
   *
   * This overload classifies by name alone, so it works on heavy-atom-only
   * structures where hydrogen counts are not known.
   */
  double operator()(std::string_view resname, std::string_view atom_name) const;
};
}  // namespace nuri

#endif /* NURI_PARAM_RADII_H_ */
