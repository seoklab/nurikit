//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_ALGO_RINGS_H_
#define NURI_ALGO_RINGS_H_

//! @cond
#include <memory>
#include <utility>
#include <vector>
//! @endcond

#include "nuri/core/molecule.h"

namespace nuri {
using Rings = std::vector<std::vector<int>>;

/**
 * @brief Find all elementary cycles in the molecular graph.
 * @param mol A molecule.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return A pair of (all elementary cycles, success). If success is `false`,
 *         the vector is in an unspecified state. This will fail if and only if
 *         any atom is a member of more than 100 elementary cycles.
 *
 * This is based on the algorithm described in the following paper:
 *    Hanser, Th. *et al.* *J. Chem. Inf. Comput. Sci.* **1996**, *36* (6),
 *    1146-1152. DOI: [10.1021/ci960322f](https://doi.org/10.1021/ci960322f)
 *
 * The time complexity of this function is inherently exponential, but it is
 * expected to run in a reasonable time (\f$\sim\mathcal{O}(V^2)\f$) for most
 * molecules in practice.
 */
extern std::pair<Rings, bool> find_all_rings(const Molecule &mol,
                                             int max_size = -1);

/**
 * @brief Find all elementary cycles in the substructure.
 * @param sub A substructure.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return A pair of (all elementary cycles, success). If success is `false`,
 *         the vector is in an unspecified state. This will fail if and only if
 *         any atom is a member of more than 100 elementary cycles.
 *
 * This is based on the algorithm described in the following paper:
 *    Hanser, Th. *et al.* *J. Chem. Inf. Comput. Sci.* **1996**, *36* (6),
 *    1146-1152. DOI: [10.1021/ci960322f](https://doi.org/10.1021/ci960322f)
 *
 * The time complexity of this function is inherently exponential, but it is
 * expected to run in a reasonable time (\f$\sim\mathcal{O}(V^2)\f$) for most
 * molecules in practice.
 */
extern std::pair<Rings, bool> find_all_rings(const Substructure &sub,
                                             int max_size = -1);

/**
 * @brief Find all elementary cycles in the substructure.
 * @param sub A substructure.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return A pair of (all elementary cycles, success). If success is `false`,
 *         the vector is in an unspecified state. This will fail if and only if
 *         any atom is a member of more than 100 elementary cycles.
 *
 * This is based on the algorithm described in the following paper:
 *    Hanser, Th. *et al.* *J. Chem. Inf. Comput. Sci.* **1996**, *36* (6),
 *    1146-1152. DOI: [10.1021/ci960322f](https://doi.org/10.1021/ci960322f)
 *
 * The time complexity of this function is inherently exponential, but it is
 * expected to run in a reasonable time (\f$\sim\mathcal{O}(V^2)\f$) for most
 * molecules in practice.
 */
extern std::pair<Rings, bool> find_all_rings(const ConstSubstructure &sub,
                                             int max_size = -1);

namespace internal {
  template <class MoleculeLike>
  struct FindRingsCommonData;
}  // namespace internal

/**
 * @brief Wrapper class of the common routines of find_sssr() and
 *        find_relevant_rings().
 * @sa nuri::find_relevant_rings(), nuri::find_sssr()
 *
 * Formally, SSSR (smallest set of smallest rings) is a *minimum cycle basis*
 * of the molecular graph. As discussed in many literatures, there is no unique
 * SSSR for a given molecular graph (even for simple molecules such as
 * 2-oxabicyclo[2.2.2]octane), and the SSSR is often counter-intuitive. For
 * example, the SSSR of cubane (although unique, due to symmetry reasons)
 * contains only five rings, which is not most chemists would expect.
 *
 * On the other hand, union of all SSSRs, sometimes called the *relevant
 * rings* in the literatures, is unique for a given molecule, and is the "all
 * smallest rings" of the molecule, chemically speaking. It is more appropriate
 * for most applications than SSSR.
 *
 * We provide two functions along with this class to find the relevant rings and
 * SSSR, respectively. If both are needed, it is recommended to construct this
 * class first, and call find_relevant_rings() and find_sssr() member functions
 * instead of calling the free functions directly.
 *
 * This is based on the algorithm described in the following paper:
 *    Vismara, P. *Electron. J. Comb.* **1997**, *4* (1), R9.
 *    DOI: [10.37236/1294](https://doi.org/10.37236/1294)
 *
 * Time complexity: theoretically \f$\mathcal{O}(\nu E^3)\f$, where \f$\nu =
 * \mathcal{O}(E)\f$ is size of SSSR. For most molecules, however, this is
 * \f$\mathcal{O}(V^3)\f$.
 */
template <class MoleculeLike>
class RingSetsFinder {
public:
  /**
   * @brief Construct a new Rings Finder object.
   * @param mol A molecule.
   * @param max_size Maximum size of the rings to be found. If negative, all
   *        rings are found.
   */
  explicit RingSetsFinder(const MoleculeLike &mol, int max_size = -1);

  RingSetsFinder(const RingSetsFinder &) = delete;
  RingSetsFinder &operator=(const RingSetsFinder &) = delete;
  RingSetsFinder(RingSetsFinder &&) noexcept;
  RingSetsFinder &operator=(RingSetsFinder &&) noexcept;

  ~RingSetsFinder() noexcept;

  /**
   * @brief Find the relevant rings of the molecule.
   * @return The relevant rings of the molecule.
   * @sa nuri::find_relevant_rings()
   */
  Rings find_relevant_rings() const;

  /**
   * @brief Find the SSSR of the molecule.
   * @return The smallest set of smallest rings (SSSR) of the molecule.
   * @sa nuri::find_sssr()
   * @note This function does not guarantee that the returned set is unique, nor
   * that the result is reproducible even for the same molecule.
   */
  Rings find_sssr() const;

private:
  const MoleculeLike *mol_;
  std::unique_ptr<internal::FindRingsCommonData<MoleculeLike>> data_;
};

template <class MoleculeLike>
RingSetsFinder(const MoleculeLike &, int) -> RingSetsFinder<MoleculeLike>;

extern template class RingSetsFinder<Molecule>;
extern template class RingSetsFinder<Substructure>;
extern template class RingSetsFinder<ConstSubstructure>;

/**
 * @brief Find union of the all SSSRs in the molecular graph.
 * @param mol A molecule.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return Union of the all SSSRs in the molecular graph.
 * @sa find_sssr(), nuri::RingSetsFinder::find_relevant_rings()
 *
 * This is a convenience wrapper of the
 * nuri::RingSetsFinder::find_relevant_rings() member function.
 *
 * @note If both relevant rings and SSSR are needed, it is recommended to use
 * the nuri::RingSetsFinder class instead of the free functions.
 */
inline Rings find_relevant_rings(const Molecule &mol, int max_size = -1) {
  return RingSetsFinder(mol, max_size).find_relevant_rings();
}

/**
 * @brief Find union of the all SSSRs in the substructure.
 * @param sub A substructure.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return Union of the all SSSRs in the substructure.
 * @sa find_sssr(), nuri::RingSetsFinder::find_relevant_rings()
 *
 * This is a convenience wrapper of the
 * nuri::RingSetsFinder::find_relevant_rings() member function.
 *
 * @note If both relevant rings and SSSR are needed, it is recommended to use
 * the nuri::RingSetsFinder class instead of the free functions.
 */
inline Rings find_relevant_rings(const Substructure &sub, int max_size = -1) {
  return RingSetsFinder(sub, max_size).find_relevant_rings();
}

/**
 * @brief Find union of the all SSSRs in the substructure.
 * @param sub A substructure.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return Union of the all SSSRs in the substructure.
 * @sa find_sssr(), nuri::RingSetsFinder::find_relevant_rings()
 *
 * This is a convenience wrapper of the
 * nuri::RingSetsFinder::find_relevant_rings() member function.
 *
 * @note If both relevant rings and SSSR are needed, it is recommended to use
 * the nuri::RingSetsFinder class instead of the free functions.
 */
inline Rings find_relevant_rings(const ConstSubstructure &sub,
                                 int max_size = -1) {
  return RingSetsFinder(sub, max_size).find_relevant_rings();
}

/**
 * @brief Find a smallest set of smallest rings (SSSR) of the molecular graph.
 * @param mol A molecule.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return *A* smallest set of smallest rings (SSSR) of the molecular graph.
 * @sa find_relevant_rings(), nuri::RingSetsFinder::find_sssr()
 * @note This function does not guarantee that the returned set is unique, nor
 *       that the result is reproducible even for the same molecule.
 *
 * This is a convenience wrapper of the nuri::RingSetsFinder::find_sssr() member
 * function.
 *
 * @note If both relevant rings and SSSR are needed, it is recommended to use
 * the nuri::RingSetsFinder class instead of the free functions.
 */
inline Rings find_sssr(const Molecule &mol, int max_size = -1) {
  return RingSetsFinder(mol, max_size).find_sssr();
}

/**
 * @brief Find a smallest set of smallest rings (SSSR) of the substructure.
 * @param sub A substructure.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return *A* smallest set of smallest rings (SSSR) of the substructure.
 * @sa find_relevant_rings(), nuri::RingSetsFinder::find_sssr()
 * @note This function does not guarantee that the returned set is unique, nor
 *       that the result is reproducible even for the same molecule.
 *
 * This is a convenience wrapper of the nuri::RingSetsFinder::find_sssr() member
 * function.
 *
 * @note If both relevant rings and SSSR are needed, it is recommended to use
 * the nuri::RingSetsFinder class instead of the free functions.
 */
inline Rings find_sssr(const Substructure &sub, int max_size = -1) {
  return RingSetsFinder(sub, max_size).find_sssr();
}

/**
 * @brief Find a smallest set of smallest rings (SSSR) of the substructure.
 * @param sub A substructure.
 * @param max_size Maximum size of the rings to be found. If negative, all
 *        rings are found.
 * @return *A* smallest set of smallest rings (SSSR) of the substructure.
 * @sa find_relevant_rings(), nuri::RingSetsFinder::find_sssr()
 * @note This function does not guarantee that the returned set is unique, nor
 *       that the result is reproducible even for the same molecule.
 *
 * This is a convenience wrapper of the nuri::RingSetsFinder::find_sssr() member
 * function.
 *
 * @note If both relevant rings and SSSR are needed, it is recommended to use
 * the nuri::RingSetsFinder class instead of the free functions.
 */
inline Rings find_sssr(const ConstSubstructure &sub, int max_size = -1) {
  return RingSetsFinder(sub, max_size).find_sssr();
}
}  // namespace nuri

#endif /* NURI_ALGO_RINGS_H_ */
