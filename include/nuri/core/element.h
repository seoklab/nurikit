//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_ELEMENT_H_
#define NURI_CORE_ELEMENT_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include <absl/container/flat_hash_map.h>

namespace nuri {
namespace internal {
  // Value taken from https://physics.nist.gov/cgi-bin/cuu/Value?are
  extern constexpr inline double kElectronMass = 5.48579909065e-4;
}  // namespace internal

struct Isotope {
  int atomic_number;
  int mass_number;
  double atomic_weight;
  double abundance;
};

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
 *     the given value is manually averaged from the given range. (*not*
 *     rounded)
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
 */
class Element {
public:
  Element() = delete;
  ~Element() noexcept = default;

  constexpr int atomic_number() const noexcept { return atomic_number_; }

  constexpr std::int16_t period() const noexcept { return period_; }

  /**
   * @brief Group of the element.
   * @return std::int16_t Group of the element. If the element is a dummy atom,
   *         returns 0. Lanthanides and actinides are treated as group 3.
   */
  constexpr std::int16_t group() const noexcept { return group_; }

  constexpr std::string_view symbol() const noexcept { return symbol_; }

  constexpr std::string_view name() const noexcept { return name_; }

  constexpr double atomic_weight() const noexcept { return atomic_weight_; }

  constexpr double covalent_radius() const noexcept { return cov_rad_; }

  constexpr double vdw_radius() const noexcept { return vdw_rad_; }

  /**
   * @brief Electronegativity of the element (Pauling scale).
   *
   * @return double Electronegativity of the element (Pauling scale). If the
   *         element has no known electronegativity value, returns -1.
   */
  constexpr double eneg() const noexcept { return eneg_; }

  constexpr const Isotope &major_isotope() const noexcept {
    return *major_isotope_;
  }

  constexpr const std::vector<Isotope> &isotopes() const noexcept {
    return isotopes_;
  }

private:
  Element(int atomic_number, std::string_view symbol, std::string_view name,
          double atomic_weight, double cov_rad, double vdw_rad, double eneg,
          std::vector<Isotope> &&isotopes) noexcept;

  Element(const Element &) = default;
  Element(Element &&) noexcept = default;
  Element &operator=(const Element &) = default;
  Element &operator=(Element &&) = default;

  friend class PeriodicTable;

  int atomic_number_;
  std::int16_t period_;
  std::int16_t group_;
  std::string_view symbol_;
  std::string_view name_;
  double atomic_weight_;
  double cov_rad_;
  double vdw_rad_;
  double eneg_;
  const Isotope *major_isotope_;
  std::vector<Isotope> isotopes_;
};

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
   * @note This function is not guaranteed to be safe from the &ldquo;static
   *       initialization order fiasco.&rdquo; Never call this function from the
   *       constructor/destructor of another static object.
   */
  static const PeriodicTable &get() noexcept { return kPeriodicTable_; }

  constexpr const Element &operator[](int atomic_number) const noexcept {
    return *find_element(atomic_number);
  }

  constexpr const Element *find_element(int atomic_number) const noexcept {
    return has_element(atomic_number) ? &elements_[atomic_number] : nullptr;
  }

  const Element *find_element(std::string_view symbol) const noexcept {
    auto it = symbol_to_element_.find(symbol);
    return it != symbol_to_element_.end() ? it->second : nullptr;
  }

  const Element *find_element_of_name(std::string_view name) const noexcept {
    auto it = name_to_element_.find(name);
    return it != name_to_element_.end() ? it->second : nullptr;
  }

  constexpr static bool has_element(int atomic_number) noexcept {
    return atomic_number < kElementCount_;
  }

  bool has_element(std::string_view symbol) const noexcept {
    return symbol_to_element_.contains(symbol);
  }

  bool has_element_of_name(std::string_view name) const noexcept {
    return name_to_element_.contains(name);
  }

  // 118 elements + dummy
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr inline static int kElementCount_ = 118 + 1;

private:
  PeriodicTable() noexcept;

  // NOLINTNEXTLINE(readability-identifier-naming)
  static const PeriodicTable kPeriodicTable_;

  Element elements_[kElementCount_];
  absl::flat_hash_map<std::string_view, const Element *> symbol_to_element_;
  absl::flat_hash_map<std::string_view, const Element *> name_to_element_;
};
}  // namespace nuri

#endif /* NURI_CORE_ELEMENT_H_ */