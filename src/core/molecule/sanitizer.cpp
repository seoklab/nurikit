//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/str_cat.h>
#include <Eigen/Dense>

#include "nuri/algo/rings.h"
#include "nuri/core/element.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
// Mainly used in the sanitizer, place here for better optimization
namespace internal {
  int sum_bond_order_raw(Molecule::Atom atom, int implicit_hydrogens,
                         bool aromatic_correct) {
    int sum_order = implicit_hydrogens, num_aromatic = 0, num_multiple_bond = 0;

    for (auto adj: atom) {
      if (adj.edge_data().order() == constants::kAromaticBond) {
        ++num_aromatic;
      } else {
        sum_order += nuri::max(adj.edge_data().order(), constants::kSingleBond);
        num_multiple_bond +=
            static_cast<int>(adj.edge_data().order() > constants::kSingleBond);
      }
    }

    if (num_aromatic == 0) {
      return sum_order;
    }

    ABSL_LOG_IF(INFO, aromatic_correct && !atom.data().is_aromatic())
        << "Non-aromatic atom " << atom.id() << " has " << num_aromatic
        << " aromatic bonds";

    if (num_aromatic == 1) {
      ABSL_LOG(INFO) << "Atom with single aromatic bond; assuming single bond "
                        "for bond order calculation";
      return sum_order + 1;
    }

    if (num_aromatic > 3) {
      // Aromatic atom with >= 4 aromatic bonds is very unlikely;
      // just log it and fall through
      ABSL_LOG(WARNING) << "Cannot correctly determine total bond order for "
                           "aromatic atom with more than 4 aromatic bonds";
    }

    // The logic here:
    //   - for 1 aromatic bond, assume it's a single bond
    //   - for 2 aromatic bonds, each will contribute 1.5 to the total bond
    //     order (e.g. benzene) = 3
    //   - for 3 aromatic bonds, assume 2, 1, 1 for the bond orders (this is
    //     the most common case, e.g. naphthalene) = 4
    //   - others are very unlikely, ignore for now
    // Then, subtract non-aromatic multiple bond; this is for structures like
    // c1(=O)ccccc1 (not aromatic, but the bond order must also be correct for
    // this case)
    sum_order += num_aromatic + 1 - num_multiple_bond;

    if (aromatic_correct) {
      int pie_estimate = count_pi_e(atom, sum_order);
      // pyrrole, again.
      sum_order -= static_cast<int>(pie_estimate != 1);
    }

    return sum_order;
  }

  const Element &effective_element_or_element(const AtomData &data) noexcept {
    const Element *elem = effective_element(data);
    if (ABSL_PREDICT_FALSE(elem == nullptr)) {
      ABSL_LOG(WARNING)
          << "Unexpected atomic number & formal charge combination: "
          << data.atomic_number() << ", " << data.formal_charge()
          << ". The result may be incorrect.";
      elem = &data.element();
    }
    return *elem;
  }

  int nonbonding_electrons(const AtomData &data, const int total_valence) {
    // Don't use effective_element* here, this function must return negative
    // values for any chemically invalid combinations of the three values
    return data.element().valence_electrons() - total_valence
           - data.formal_charge();
  }

  int count_pi_e(Molecule::Atom atom, int total_valence) {
    const int nb_electrons = nonbonding_electrons(atom.data(), total_valence);
    ABSL_DLOG_IF(WARNING, nb_electrons < 0)
        << "Negative nonbonding electrons for atom " << atom.id() << " ("
        << atom.data().element().symbol() << "): " << nb_electrons;

    if (std::any_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
          return nei.edge_data().order() == constants::kAromaticBond;
        })) {
      // Special case, some bonds are aromatic
      // Now we have to check structures like furan, pyrrole, etc.
      const int cv = common_valence(effective_element_or_element(atom));

      // E.g. O in furan, N in pyrrole, ...
      if (cv < total_valence) {
        // Has nonbonding electrons (O, N) -> 2 electrons participate in
        // No nonbonding electrons (B) -> no electrons participate in
        const int pie_estimate = static_cast<int>(nb_electrons > 0) * 2;
        return pie_estimate;
      }

      // Normal case: atoms in pyridine, benzene, ...
      return 1;
    }

    if (std::any_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
          return nei.edge_data().is_ring_bond()
                 && (nei.edge_data().order() == constants::kDoubleBond
                     || nei.edge_data().order() == constants::kTripleBond);
        })) {
      // Normal case, at least one double/triple bond in the ring
      return 1;
    }

    // E.g. N in pyrrole
    const int pie_estimate = nuri::min(nb_electrons, 2);
    return pie_estimate;
  }

  int aromatic_pi_e(Molecule::Atom atom, int total_valence) {
    if (std::any_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
          return !nei.edge_data().is_ring_bond()
                 && nei.edge_data().order() > constants::kSingleBond;
        })) {
      // Exocyclic multiple bond, don't contribute to pi electrons
      return 0;
    }

    return count_pi_e(atom, total_valence);
  }

  int steric_number(const int total_degree, const int nb_electrons) {
    const int lone_pairs = nb_electrons / 2;
    return total_degree + lone_pairs + nb_electrons % 2;
  }

  constants::Hybridization from_degree(const int total_degree,
                                       const int nb_electrons) {
    int sn = steric_number(total_degree, nb_electrons);
    return static_cast<constants::Hybridization>(
        nuri::min(sn, static_cast<int>(constants::kOtherHyb)));
  }
}  // namespace internal

namespace {
  using MutableAtom = Molecule::MutableAtom;
  using MutableBond = Molecule::MutableBond;

  using internal::aromatic_pi_e;
  using internal::effective_element_or_element;
  using internal::from_degree;
  using internal::nonbonding_electrons;
}  // namespace

MoleculeSanitizer::MoleculeSanitizer(Molecule &molecule)
    : mol_(&molecule), valences_(molecule.num_atoms()) {
  // Calculate valence
  for (auto atom: mol()) {
    valences_[atom.id()] = sum_bond_order(atom, false);
  }
}

namespace {
  bool is_conjugated_candidate(Molecule::Atom atom) {
    const AtomData &data = atom.data();
    // Consider main-group atoms only
    return data.element().main_group()
           // Cannot check hybridization here, as it might be incorrect
           // (Most prominent example: amide nitrogen)
           && all_neighbors(atom) <= 3;
  }

  void mark_conjugated(Molecule &mol,
                       const std::vector<std::vector<int>> &sp2_groups) {
    for (const std::vector<int> &group: sp2_groups) {
      ABSL_DCHECK(group.size() > 2) << "Group size: " << group.size();

      for (const int id: group) {
        AtomData &data = mol.atom(id).data();
        data.set_conjugated(true);
      }
    }

    for (auto bond: mol.bonds()) {
      if (bond.src().data().is_conjugated()
          && bond.dst().data().is_conjugated()) {
        bond.data().set_conjugated(true);
      }
    }
  }

  bool path_can_conjugate(const BondData &prev, Molecule::Neighbor curr,
                          const absl::FixedArray<int> &valences) {
    auto src = curr.src(), dst = curr.dst();

    if (prev.order() == constants::kSingleBond) {
      if (curr.edge_data().order() == constants::kSingleBond
          && src.data().atomic_number() != 0
          && dst.data().atomic_number() != 0) {
        // Single - single bond -> conjugated if curr has lone pair and
        // next doesn't, or vice versa, or any of them is dummy
        int src_nbe = nonbonding_electrons(src.data(), valences[src.id()]),
            dst_nbe = nonbonding_electrons(dst.data(), valences[dst.id()]);

        if ((src_nbe > 0 && dst_nbe > 0) || (src_nbe <= 0 && dst_nbe <= 0))
          return false;
      }
    } else if (curr.edge_data().order() != constants::kSingleBond
               && (prev.order() != constants::kAromaticBond
                   || curr.edge_data().order() != constants::kAromaticBond)) {
      // Aromatic - aromatic bond -> conjugated
      // double~triple - double~triple bond -> allene-like
      // aromatic - double~triple bond -> not conjugated (erroneous?)
      return false;
    }

    return true;
  }
}  // namespace

bool MoleculeSanitizer::sanitize_conjugated() {
  absl::flat_hash_set<int> candidates;
  std::vector<std::vector<int>> groups;

  for (auto atom: mol()) {
    atom.data().set_conjugated(false);
    if (is_conjugated_candidate(atom)) {
      candidates.insert(atom.id());
    }
  }

  for (auto bond: mol().bonds()) {
    bond.data().set_conjugated(false);
  }

  auto dfs = [&](auto &self, int curr, const BondData *prev_bond) -> void {
    groups.back().push_back(curr);
    candidates.erase(curr);

    auto src = mol().atom(curr);
    for (auto nei: src) {
      auto dst = nei.dst();
      if (!candidates.contains(dst.id())) {
        continue;
      }

      if (prev_bond == nullptr) {
        prev_bond = &nei.edge_data();
      } else if (!path_can_conjugate(*prev_bond, nei, valences_)) {
        continue;
      }

      self(self, dst.id(), &nei.edge_data());
    }
  };

  // Mark conjugated atoms
  for (auto atom: mol()) {
    if (!candidates.contains(atom.id())) {
      continue;
    }

    groups.emplace_back();
    dfs(dfs, atom.id(), nullptr);

    if (groups.back().size() < 3) {
      candidates.insert(groups.back().begin(), groups.back().end());
      groups.pop_back();
    }
  }

  mark_conjugated(mol(), groups);

  return true;
}

namespace {
  bool is_aromatic_candidate(Molecule::Atom atom) {
    const AtomData &data = atom.data();
    // Consider in-ring, main-group, conjugated (implies hyb <= sp2)
    return data.is_ring_atom() && data.element().main_group()
           && atom.data().is_conjugated()
           // No more than one double bonds, otherwise it's allene-like
           && std::count_if(atom.begin(), atom.end(),
                            [](Molecule::Neighbor nei) {
                              return nei.edge_data().order()
                                     == constants::kDoubleBond;
                            })
                  < 2;
  }

  bool is_ring_aromatic(const Molecule &mol, const std::vector<int> &ring,
                        const std::vector<int> &pi_es,
                        const absl::FixedArray<int> &valences) {
    const int pi_e_sum = std::accumulate(pi_es.begin(), pi_es.end(), 0);
    bool initial = pi_e_sum % 4 == 2
                   // Dummy atoms are always allowed in aromatic rings
                   || std::any_of(ring.begin(), ring.end(), [&](int id) {
                        return mol.atom(id).data().atomic_number() == 0;
                      });
    if (!initial)
      return false;

    const BondData *prev_data = nullptr;

    for (int i = 0; i < ring.size(); ++i) {
      const int src = ring[i],
                dst = ring[(i + 1) % static_cast<int>(ring.size())];

      auto curr = mol.find_neighbor(src, dst);
      ABSL_DCHECK(!curr.end());

      if (prev_data != nullptr
          && !path_can_conjugate(*prev_data, *curr, valences))
        return false;

      prev_data = &curr->edge_data();
    }

    return true;
  }

  void mark_aromatic_ring(Molecule &mol, const std::vector<int> &ring,
                          const absl::flat_hash_map<int, int> &pi_e,
                          const absl::FixedArray<int> &valences) {
    std::vector<int> ring_pi_e;
    ring_pi_e.reserve(ring.size());

    for (int i = 0; i < ring.size(); ++i) {
      auto it = pi_e.find(ring[i]);
      if (it == pi_e.end()) {
        return;
      }

      int ne = it->second;
      ring_pi_e.push_back(ne);
    }

    const bool this_aromatic = is_ring_aromatic(mol, ring, ring_pi_e, valences);
    if (!this_aromatic)
      return;

    for (int i = 0; i < ring.size(); ++i) {
      const int src = ring[i],
                dst = ring[(i + 1) % static_cast<int>(ring.size())];

      mol.atom(src).data().set_aromatic(true);

      auto eit = mol.find_bond(src, dst);
      ABSL_DCHECK(eit != mol.bond_end());
      eit->data().set_aromatic(true);
    }
  }

  void mark_aromatic(Molecule &mol, const absl::FixedArray<int> &valences) {
    absl::flat_hash_map<int, int> pi_e;

    for (auto atom: mol) {
      if (is_aromatic_candidate(atom)) {
        pi_e.insert({ atom.id(), aromatic_pi_e(atom, valences[atom.id()]) });
      }
    }

    auto mark_aromatic_for = [&](const std::vector<std::vector<int>> &rs) {
      for (const std::vector<int> &ring: rs) {
        mark_aromatic_ring(mol, ring, pi_e, valences);
      }
    };

    const bool need_subring = mol.num_sssr()
                              > static_cast<int>(mol.ring_groups().size());
    // Fast path: no need to find subrings
    if (!need_subring) {
      mark_aromatic_for(mol.ring_groups());
      return;
    }

    auto [subrings, success] = find_all_rings(mol, 12);
    // Almost impossible, don't include in coverage report
    // GCOV_EXCL_START
    if (ABSL_PREDICT_TRUE(success)) {
      mark_aromatic_for(subrings);
      return;
    }

    ABSL_LOG(WARNING)
        << "Ring finding exceeds threshold, falling back to SSSR-based "
           "aromaticity detection: the result may be incorrect";

    mark_aromatic_for(find_sssr(mol, 12));
    // GCOV_EXCL_STOP
  }
}  // namespace

bool MoleculeSanitizer::sanitize_aromaticity() {
  for (auto bond: mol().bonds()) {
    if (!bond.data().is_ring_bond()) {
      if (bond.data().order() == constants::kAromaticBond) {
        ABSL_LOG(WARNING)
            << "Bond order between atoms " << bond.src().id() << " and "
            << bond.dst().id()
            << " is set aromatic, but the bond is not a ring bond";
        return false;
      }
    }
  }

  for (auto atom: mol()) {
    atom.data().set_aromatic(false);
  }
  for (auto bond: mol().bonds()) {
    bond.data().set_aromatic(false);
  }

  if (mol().num_sssr() > 0) {
    mark_aromatic(mol(), valences_);
  }

  for (auto atom: mol()) {
    valences_[atom.id()] = sum_bond_order(atom);
  }

  ABSL_LOG_IF(INFO, std::any_of(mol().bond_begin(), mol().bond_end(),
                                [](Molecule::Bond bond) {
                                  return bond.data().order()
                                             == constants::kAromaticBond
                                         && !bond.data().is_aromatic();
                                }))
      << "Bond order of non-aromatic bond is set aromatic; is this intended?";

  return true;
}

namespace {
  bool is_pyrrole_like(Molecule::Atom atom, const Element &effective,
                       const int nbe, const int total_valence) {
    return nbe > 0
           && internal::common_valence(effective) < total_valence
           // inconsistent formatting between 19.1.1 (ubuntu 24.04) vs 19.1.7
           // clang-format off
           && std::any_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
                return nei.edge_data().order() == constants::kAromaticBond;
              });
    // clang-format on
  }

  std::string format_atom_common(Molecule::Atom atom, bool caps) {
    return absl::StrCat(caps ? "A" : "a", "tom ", atom.id(), " (",
                        atom.data().element().symbol(), ")");
  }

  // Set hybridization for others
  bool sanitize_hyb_atom(MutableAtom atom, const int total_valence) {
    const int total_degree = all_neighbors(atom);

    if (atom.data().atomic_number() == 0) {
      // Assume dummy atom always satisfies the octet rule
      const int nbe = nonnegative(8 - total_valence);
      atom.data().set_hybridization(from_degree(total_degree, nbe));
      return true;
    }

    int nbe = nonnegative(nonbonding_electrons(atom.data(), total_valence));
    if (nbe < 0) {
      nbe = 0;
      ABSL_LOG(INFO) << "Valence electrons exceeded for "
                     << format_atom_common(atom, false) << ": total valence "
                     << total_valence << ", "
                     << "formal charge " << atom.data().formal_charge()
                     << "; assuming no lone pair";
    }

    const Element &effective = effective_element_or_element(atom);

    constants::Hybridization hyb = from_degree(total_degree, nbe);
    if (hyb == constants::kSP3 && atom.data().is_conjugated()) {
      hyb = constants::kSP2;
      if (is_pyrrole_like(atom, effective, nbe, total_valence)) {
        // Pyrrole, etc.
        --nbe;
      }
    }

    // Unbound / terminal
    if (total_degree <= 1) {
      atom.data().set_hybridization(
          static_cast<constants::Hybridization>(total_degree));
    } else if (effective.main_group()) {
      atom.data().set_hybridization(hyb);
    } else {
      // Assume non-main-group atoms does not have lone pairs
      atom.data().set_hybridization(
          nuri::min(hyb, from_degree(total_degree, 0)));
    }

    return true;
  }
}  // namespace

bool MoleculeSanitizer::sanitize_hybridization() {
  return std::all_of(mol().begin(), mol().end(), [&](MutableAtom atom) {
    return sanitize_hyb_atom(atom, valences_[atom.id()]);
  });
}

namespace {
  int max_valence(const Element &effective) {
    switch (effective.period()) {
    case 1:
      return 2;
    case 2:
      return 8;
    default:
      return effective.lanthanide() || effective.actinide() ? 32 : 18;
    }
  }

  bool test_valence(Molecule::Atom atom, const Element &effective,
                    const int total_valence_electrons) {
    const int mv = max_valence(effective);
    if (total_valence_electrons > mv) {
      ABSL_LOG(WARNING)
          << format_atom_common(atom, true) << " with charge "
          << atom.data().formal_charge() << " has more than " << mv
          << " valence electrons: " << total_valence_electrons;
      return false;
    }

    return true;
  }

  bool sanitize_val_atom(Molecule::Atom atom, const int total_valence) {
    if (atom.data().atomic_number() == 0) {
      // Assume dummy atom always satisfies the octet rule
      return true;
    }

    int nbe = nonbonding_electrons(atom.data(), total_valence);
    if (nbe < 0) {
      ABSL_LOG(WARNING) << "Valence electrons exceeded for "
                        << format_atom_common(atom, false) << ": total valence "
                        << total_valence << ", "
                        << "formal charge " << atom.data().formal_charge();
      return false;
    }

    const Element &effective = effective_element_or_element(atom);
    if (atom.data().is_conjugated()
        && is_pyrrole_like(atom, effective, nbe, total_valence)) {
      // Pyrrole, etc.
      --nbe;
    }

    return test_valence(atom, effective, nbe + 2 * total_valence);
  }
}  // namespace

bool MoleculeSanitizer::sanitize_valence() {
  return std::all_of(mol().begin(), mol().end(), [&](Molecule::Atom atom) {
    return sanitize_val_atom(atom, valences_[atom.id()]);
  });
}
}  // namespace nuri
