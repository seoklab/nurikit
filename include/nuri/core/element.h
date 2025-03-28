//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_ELEMENT_H_
#define NURI_CORE_ELEMENT_H_

//! @cond
#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <absl/container/flat_hash_map.h>
//! @endcond

#include "nuri/utils.h"

namespace nuri {
namespace internal {
  // Value taken from https://physics.nist.gov/cgi-bin/cuu/Value?are
  constexpr double kElectronMass = 5.48579909065e-4;
}  // namespace internal

struct Isotope {
  int atomic_number;
  int mass_number;
  double atomic_weight;
  double abundance;
};

constexpr bool operator==(const Isotope &lhs, const Isotope &rhs) noexcept {
  return &lhs == &rhs;
}

constexpr bool operator!=(const Isotope &lhs, const Isotope &rhs) noexcept {
  return &lhs != &rhs;
}

/**
 * @brief The class for element data.
 *
 * @section ref-notes References and Notes
 *
 * @subsection atwt-and-isotope About the atomic weights & isotopes data
 *
 * @subsubsection ai-notes Notes on atomic weight and isotope data
 *
 *   - If ranges are specified in the standard atomic weights table for an
 *     element, the given value is taken from the abridged table.
 *   - If no value is specified in the standard atomic weights table for an
 *     element, the value given is taken from the radioactive table, where the
 *     value represents the longest-lived isotope of the element.
 *   - If ranges are specified in the isotopic abundances table for an element,
 *     the given value is taken from the NUBASE 2020 table.
 *   - Dummy atom has one *isotope*, which corresponds to the neutron.
 *
 * @subsubsection ai-ref References
 *
 * Values were taken from the following references, unless otherwise noted:
 *   - https://ciaaw.org/atomic-weights.htm (Accessed 2023-05-11)
 *   - https://ciaaw.org/abridged-atomic-weights.htm (Accessed 2023-05-11)
 *   - https://ciaaw.org/radioactive-elements.htm (Accessed 2023-05-11)
 *   - https://ciaaw.org/atomic-masses.htm (Accessed 2023-05-11)
 *   - https://ciaaw.org/isotopic-abundances.htm (Accessed 2023-05-11)
 *
 * For the isotopes that are listed as ranges in the isotopic abundances table,
 * the following reference was used:
 *   - Kondev, F. G. *et al.* *Chinese Phys. C* **2021**, *45* (3), 030001. DOI:
 *     [10.1088/1674-1137/abddae](https://doi.org/10.1088/1674-1137/abddae)
 *
 * For the neutron mass, the following reference was used:
 *   - https://physics.nist.gov/cgi-bin/cuu/Value?arn (Accessed 2023-05-11)
 *
 * @subsubsection ai-copy Atomic weights & isotopes data copyright notice
 *
 * Full copyright text, copied verbatim from the CIAAW website:
 *   > The contents of the website are jointly copyrighted by CIAAW and IUPAC
 *   > and can be freely used for educational purposes. <br>
 *   > © CIAAW, 2007-2022 <br>
 *   > Republication or reproduction of this report or its storage and/or
 *   > dissemination by electronic means is permitted without the need for
 *   > formal IUPAC or CIAAW permission on condition that an acknowledgement,
 *   > with full reference to the source, along with use of the copyright symbol
 *   > ©, the name IUPAC or CIAAW, and the year of publication, are prominently
 *   > visible. Publication of a translation into another language is subject to
 *   > the additional condition of prior approval from the relevant IUPAC
 *   > National Adhering Organization. <br>
 *   > For commercial use of this content please contact CIAAW Secretariat.
 *
 * @subsection r-cov About the covalent radii data
 *
 * Covalent radii were obtained from two sources. The latter provides a more
 * comprehensive set of elements, but the former has performed a more rigorous
 * statistical analysis and carries greater reputation in the field. Notably,
 * the values from the former reference are used in the PubChem and the
 * Wikipedia periodic tables. We henceforth opted to use the values from the
 * former whenever possible. While this may result in some inconsistencies
 * within the dataset, we believe this choice is acceptable considering that the
 * majority of interest lies in the first 96 elements.
 *
 * Cordero *et al.* proposed two covalent radii for manganese (1.39, 1.61), iron
 * (1.32, 1.52), and cobalt (1.26, 1.50), each corresponding to the low-spin and
 * high-spin states. To ensure maximum transferability, we calculated the
 * weighted average of these values based on the number of analyzed bond
 * distances. Carbon also has three values (0.76, 0.73, 0.69) in their dataset,
 * and we selected the value corresponding to \f$\mathrm{sp}^3\f$ hybridized
 * carbon (0.76).
 *
 * @subsubsection r-cov-ref References
 *
 *   - Elements 1-96: Cordero, B. *et al.* *Dalton Trans.* **2008**, *21*,
 *     2832-2838. DOI: [10.1039/B801115J](https://doi.org/10.1039/B801115J)
 *   - Elements 97-118: Pyykko, P.; Atsumi, M. *Chem. Eur. J.* **2009**, *15*
 *     (1), 186-197. DOI:
 *     [10.1002/chem.200800987](https://doi.org/10.1002/chem.200800987)
 *
 * @subsection r-vdw About the Van der Waals radii data
 *
 * @subsubsection r-vdw-ref References
 *
 *   - Alvarez, S. *Dalton Trans.* **2013**, *42*, 8617-8636.
 *     DOI: [10.1039/C3DT50599E](https://doi.org/10.1039/C3DT50599E)
 *
 * @subsection eneg About the electronegativity data
 *
 * This is the Pauling electronegativity.
 *
 * @subsubsection eneg-ref References
 *
 *  - Retrieved from the BODR (Blue Obelisk Data Repository):
 *    https://github.com/BlueObelisk/bodr/blob/29ce17071c71b2d4d5ee81a2a28f0407331f1624/bodr/elements/elements.xml
 *
 * @subsection type-state About the type and standard state data
 *
 * The element type and standard state data were obtained from the PubChem
 * periodic table.
 *
 * The element type represents classification of the element into one of the
 * following categories. The categories are mutually exclusive, and elements
 * that do not fall into any of the categories are classified as metals.
 *
 *  - Unknown (only dummy atoms),
 *  - Nonmetal,
 *  - Metalloid
 *
 * The standard state represents the state of the element at 298.15 K and 1 atm,
 * and is one of gas, liquid, or solid. Unknown is only used for dummy atoms.
 *
 * @subsubsection type-state-ref References
 *
 *  - National Center for Biotechnology Information. Periodic Table of Elements.
 *    https://pubchem.ncbi.nlm.nih.gov/periodic-table. (Accessed 2023-12-05)
 */
class Element {
public:
  enum class Type : std::uint8_t {
    kUnknown,
    kMetal,
    kMetalloid,
    kNonmetal,
  };

  enum class State : std::uint8_t {
    kUnknown,
    kSolid,
    kLiquid,
    kGas,
  };

  Element() = delete;
  ~Element() noexcept = default;

  /**
   * @brief Get the atomic number of the atom.
   * @return The atomic number.
   */
  constexpr int atomic_number() const noexcept { return atomic_number_; }

  /**
   * @brief Get the number of valence electrons of the atom.
   * @return The number of valence electrons.
   */
  constexpr std::int16_t valence_electrons() const noexcept {
    return valence_electrons_;
  }

  /**
   * @brief Get the period of the atom.
   * @return Period of this atom.
   */
  constexpr std::int16_t period() const noexcept { return period_; }

  /**
   * @brief Get the group of the element.
   * @return Group of the element. If the element is a dummy atom, returns 0.
   *         Lanthanides and actinides are treated as group 3.
   */
  constexpr std::int16_t group() const noexcept { return group_; }

  /**
   * @brief Test if the molecule is radioactive. That is, all of its isotopes
   *        has natural abundance of 0.
   * @return `true` if the element is radioactive, `false` otherwise.
   */
  constexpr bool radioactive() const noexcept {
    return internal::check_flag(flags_, ElementFlags::kRadioactive);
  }

  /**
   * @brief Test if the molecule is a main group element.
   * @return `true` if the element is a main group element, `false` otherwise.
   */
  constexpr bool main_group() const noexcept {
    return internal::check_flag(flags_, ElementFlags::kMainGroup);
  }

  /**
   * @brief Test if the molecule is a lanthanide.
   * @return `true` if the element is a lanthanide, `false` otherwise.
   */
  constexpr bool lanthanide() const noexcept {
    return internal::check_flag(flags_, ElementFlags::kLanthanide);
  }

  /**
   * @brief Test if the molecule is an actinide.
   * @return `true` if the element is an actinide, `false` otherwise.
   */
  constexpr bool actinide() const noexcept {
    return internal::check_flag(flags_, ElementFlags::kActinide);
  }

  /**
   * @brief Get the type of the element.
   * @return The type of the element.
   */
  constexpr Type type() const noexcept { return type_; }

  /**
   * @brief Get the standard state of the element.
   * @return The standard state of the element.
   */
  constexpr State state() const noexcept { return state_; }

  /**
   * @brief Get the IUPAC Symbol of the atom.
   * @return The IUPAC Symbol.
   */
  constexpr std::string_view symbol() const noexcept { return symbol_; }

  /**
   * @brief Get the IUPAC Name of the atom.
   * @return The IUPAC Name, in *Titlecase*.
   */
  constexpr std::string_view name() const noexcept { return name_; }

  /**
   * @brief Get the atomic weight of the atom.
   * @return The IUPAC standard atomic weight. See \ref atwt-and-isotope
   *         "references and notes" for more information.
   */
  constexpr double atomic_weight() const noexcept { return atomic_weight_; }

  /**
   * @brief Get the covalent radius of the atom.
   * @return The covalent radius of the atom (in angstroms). See \ref r-cov
   *         "references and notes" for more information.
   */
  constexpr double covalent_radius() const noexcept { return cov_rad_; }

  /**
   * @brief Get the Van der Waals radius of the atom.
   * @return The Van der Waals radius of the atom (in angstroms). See \ref r-vdw
   *         "references and notes" for more information.
   */
  constexpr double vdw_radius() const noexcept { return vdw_rad_; }

  /**
   * @brief Get electronegativity of the element (Pauling scale).
   * @return Electronegativity of the element (Pauling scale). If the element
   *         has no known electronegativity value, returns -1. See \ref eneg
   *         "references and notes" for more information.
   */
  constexpr double eneg() const noexcept { return eneg_; }

  /**
   * @brief Get the representative isotope of the element.
   * @return The representative isotope of the element. I.e., the most abundant
   *         isotope, or the most stable isotope if the element is radioactive.
   *         See \ref atwt-and-isotope "references and notes" for more
   *         information.
   */
  constexpr const Isotope &major_isotope() const noexcept {
    return *major_isotope_;
  }

  /**
   * @brief Find an element with the given mass number.
   * @return A pointer to the isotope, or `nullptr` if no isotopes with the
   *         given mass number is known.
   */
  const Isotope *find_isotope(int mass_number) const noexcept {
    auto it = std::find_if(isotopes_.begin(), isotopes_.end(),
                           [mass_number](const Isotope &iso) {
                             return iso.mass_number == mass_number;
                           });
    if (it == isotopes_.end()) {
      return nullptr;
    }
    return &(*it);
  }

  /**
   * @brief Get all known isotopes of the element.
   * @return Reference to all known isotopes of the element. See \ref
   *         atwt-and-isotope "references and notes" for more information.
   */
  constexpr const std::vector<Isotope> &isotopes() const noexcept {
    return isotopes_;
  }

private:
  enum class ElementFlags : std::uint16_t {
    kRadioactive = 0x1,
    kMainGroup = 0x2,
    kLanthanide = 0x4,
    kActinide = 0x8,
  };

  Element(int atomic_number, std::string_view symbol, std::string_view name,
          double atomic_weight, double cov_rad, double vdw_rad, double eneg,
          std::vector<Isotope> &&isotopes) noexcept;

  Element(const Element &) = default;
  Element(Element &&) noexcept = default;
  Element &operator=(const Element &) = default;
  Element &operator=(Element &&) = default;

  friend class PeriodicTable;

  int atomic_number_;
  std::int16_t valence_electrons_;
  std::int16_t period_;
  std::int16_t group_;
  ElementFlags flags_;
  Type type_;
  State state_;
  std::string_view symbol_;
  std::string_view name_;
  double atomic_weight_;
  double cov_rad_;
  double vdw_rad_;
  double eneg_;
  const Isotope *major_isotope_;
  std::vector<Isotope> isotopes_;
};

constexpr bool operator==(const Element &lhs, const Element &rhs) noexcept {
  return lhs.atomic_number() == rhs.atomic_number();
}

constexpr bool operator!=(const Element &lhs, const Element &rhs) noexcept {
  return lhs.atomic_number() != rhs.atomic_number();
}

/**
 * @brief The periodic table of elements.
 * @note You'd never want to create an instance of this class. Instead, use the
 *       `get()` function to access the singleton instance.
 */
class PeriodicTable final {
public:
  PeriodicTable(const PeriodicTable &) = delete;
  PeriodicTable(PeriodicTable &&) noexcept = delete;
  PeriodicTable &operator=(const PeriodicTable &) = delete;
  PeriodicTable &operator=(PeriodicTable &&) noexcept = delete;

  ~PeriodicTable() noexcept = default;

  /**
   * @brief Get the singleton instance of the periodic table.
   * @return const PeriodicTable & The singleton instance of the periodic table.
   */
  static const PeriodicTable &get() noexcept {
    static const PeriodicTable the_table;

    return the_table;
  }

  /**
   * @brief Get element with the given atomic number.
   *
   * @param atomic_number The atomic number of the element.
   * @return A const reference to the element.
   * @note The behavior is undefined if \p atomic_number is not in range
   *       [0, 118].
   */
  constexpr const Element &operator[](int atomic_number) const noexcept {
    return elements_[atomic_number];
  }

  /**
   * @brief Find element with the given atomic number.
   *
   * @param atomic_number The atomic number of the element.
   * @return A pointer to the element, or `nullptr` if no element with the given
   *         atomic number is known.
   */
  constexpr const Element *find_element(int atomic_number) const noexcept {
    return has_element(atomic_number) ? &elements_[atomic_number] : nullptr;
  }

  /**
   * @brief Find element with the given atomic symbol.
   *
   * @param symbol The atomic symbol of the element.
   * @return A pointer to the element, or `nullptr` if no element with the given
   *         symbol is known.
   * @note The symbol is case-sensitive, but supports three common cases:
   *       Titlecase, UPPERCASE, and lowercase.
   */
  const Element *find_element(std::string_view symbol) const noexcept {
    auto it = symbol_to_element_.find(symbol);
    return it != symbol_to_element_.end() ? it->second : nullptr;
  }

  /**
   * @brief Find element with the given name.
   *
   * @param name The atomic name of the element.
   * @return A pointer to the element, or `nullptr` if no element with the given
   *         name is known.
   * @note The name is case-sensitive, but supports three common cases:
   *       Titlecase, UPPERCASE, and lowercase.
   */
  const Element *find_element_of_name(std::string_view name) const noexcept {
    auto it = name_to_element_.find(name);
    return it != name_to_element_.end() ? it->second : nullptr;
  }

  /**
   * @brief Check if the periodic table has an element with the given atomic
   *        symbol.
   * @param atomic_number The atomic number of the element.
   * @return Whether the periodic table has an element with the given atomic
   *         number.
   */
  constexpr static bool has_element(int atomic_number) noexcept {
    return static_cast<unsigned int>(atomic_number)
           < static_cast<unsigned int>(kElementCount_);
  }

  /**
   * @brief Check if the periodic table has an element with the given atomic
   *        symbol.
   *
   * @param symbol The atomic symbol of the element.
   * @return Whether the periodic table has an element with the given symbol.
   * @note The symbol is case-sensitive, but supports three common cases:
   *       Titlecase, UPPERCASE, and lowercase.
   */
  bool has_element(std::string_view symbol) const noexcept {
    return symbol_to_element_.contains(symbol);
  }

  /**
   * @brief Check if the periodic table has an element with the given atomic
   *        name.
   *
   * @param name The atomic name of the element.
   * @return Whether the periodic table has an element with the given name.
   * @note The name is case-sensitive, but supports three common cases:
   *       Titlecase, UPPERCASE, and lowercase.
   */
  bool has_element_of_name(std::string_view name) const noexcept {
    return name_to_element_.contains(name);
  }

  const Element *begin() const noexcept { return elements_; }
  const Element *end() const noexcept { return elements_ + kElementCount_; }

  // 118 elements + dummy
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr static int kElementCount_ = 118 + 1;

private:
  PeriodicTable() noexcept;

  Element elements_[kElementCount_];
  std::string symb_name_buf_;
  absl::flat_hash_map<std::string_view, const Element *> symbol_to_element_;
  absl::flat_hash_map<std::string_view, const Element *> name_to_element_;
};

// NOLINTNEXTLINE(readability-identifier-naming)
static const PeriodicTable &kPt = PeriodicTable::get();
}  // namespace nuri

#endif /* NURI_CORE_ELEMENT_H_ */
