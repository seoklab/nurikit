//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/container/fixed_array.h>
#include <absl/log/absl_log.h>

#include "nuri/algo/guess.h"
#include "nuri/algo/rings.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  template <class Updater, class Scorer>
  void fix_aromatic_ring_common(Molecule &mol,
                                std::vector<int> &adjust_candidates,
                                const std::vector<int> &ring, Updater updater,
                                Scorer scorer) {
    std::vector<int> candids;

    for (int id: ring) {
      auto atom = mol.atom(id);
      if (atom.degree() > 3)
        return;

      if (adjust_candidates[id] != 0)
        candids.push_back(id);
    }

    if (candids.empty())
      return;

    absl::FixedArray<int> priority(candids.size(), 0);
    for (int i = 0; i < candids.size(); ++i) {
      Molecule::MutableAtom atom = mol.atom(candids[i]);
      if (all_neighbors(atom) > 3) {
        atom.data().set_implicit_hydrogens(atom.data().implicit_hydrogens()
                                           - 1);
        priority[i] += 10000;
      }
    }

    int sum_pi_e = 0;
    for (int i = 0; i < ring.size(); ++i) {
      Molecule::Atom atom = mol.atom(ring[i]);
      if (atom.data().atomic_number() == 0)
        return;

      sum_pi_e +=
          internal::aromatic_pi_e(atom, internal::sum_bond_order(atom, false));
    }

    int test = sum_pi_e % 4 - 2;
    if (test == 0)
      return;

    auto heuristic_update = [&](int id) {
      adjust_candidates[id] = 0;
      updater(mol.atom(id), test);
    };

    if (candids.size() == 1) {
      heuristic_update(candids[0]);
      return;
    }

    for (int i = 0; i < candids.size(); ++i)
      priority[i] += scorer(mol.atom(candids[i]));

    int max_idx = static_cast<int>(
        std::max_element(priority.begin(), priority.end()) - priority.begin());
    heuristic_update(candids[max_idx]);
  }

  template <class Updater, class Scorer>
  void fix_aromatic_rings(Molecule &mol, std::vector<int> &adjust_candidates,
                          Updater updater, Scorer scorer) {
    auto rings = find_sssr(mol, 12);
    for (const std::vector<int> &ring: rings)
      fix_aromatic_ring_common(mol, adjust_candidates, ring, updater, scorer);
  }

  void guess_aromatic_fcharge_updater(Molecule::MutableAtom atom, int delta) {
    // eg) pyrrole N will have    +1 formal charge, test = -1 (correct  0)
    //     tropylium ion will have 0 formal charge, test = +1 (correct +1)
    //     -> add test to formal charge
    atom.data().set_formal_charge(atom.data().formal_charge() + delta);
  }

  void guess_aromatic_hydrogens_updater(Molecule::MutableAtom atom,
                                        int /* delta */) {
    // 1. Octet satisfied
    //   - pyrrole N will have 0 hydrogens, and test will be nonzero
    //   - tropylium ion will have 0 hydrogens, and test will be nonzero
    //   -> In both cases, adding 1 hydrogen will yield the correct number of
    //      hydrogens.
    // 2. Octet exceeded
    //   - thiazole S will have larger bond order sum than common valence, and
    //     test will be nonzero
    //   -> Don't add hydrogens in this case
    int sbo = internal::sum_bond_order(atom, false),
        cv = internal::common_valence(
            internal::effective_element_or_element(atom));
    atom.data().set_implicit_hydrogens(atom.data().implicit_hydrogens()
                                       + value_if(sbo <= cv));
  }

  int guess_aromatic_fcharge_scorer(Molecule::Atom atom) {
    // Prioritize:
    // 1) Atom with nonzero formal charge (+100 * (formal charge))
    // 2) Heteroatom (+10)
    // 3) Atom of higher degree (assume degree < 10)
    int priority = 0;
    priority += atom.data().formal_charge() * 100;
    priority += atom.data().atomic_number() == 6 ? 0 : 10;
    priority += atom.degree();
    return priority;
  }

  int guess_aromatic_hydrogens_scorer(Molecule::Atom atom) {
    const Element &effective = internal::effective_element_or_element(atom);

    // Prioritize:
    // 1) (Effective) heteroatom (+10)
    // 2) Atom of lower degree (assume degree < 10)
    int priority = 0;
    priority += effective.atomic_number() == 6 ? 0 : 10;
    priority -= atom.degree();
    return priority;
  }

  int guess_aromatic_fcharge_hydrogens_scorer(Molecule::Atom atom) {
    const Element &effective = internal::effective_element_or_element(atom);

    // Prioritize:
    // 1) Higher formal charge (+100 * abs(formal charge))
    // 2) (Effective) heteroatom (+10)
    // 3) Atom of higher degree (assume degree < 10)
    int priority = 0;
    priority += 100 * std::abs(atom.data().formal_charge());
    priority += effective.atomic_number() == 6 ? 0 : 10;
    priority += atom.degree();
    return priority;
  }

  int guess_fcharge_atom(Molecule::Atom atom, const int group) {
    int cv = internal::common_valence(atom.data().element());
    int sum_bo = internal::sum_bond_order(atom, false);

    // Rationale:
    //  - Group 15-17 has nonbonding electrons, so they must have extra
    //    bonding electrons if sum of bond order > common valence
    //    -> positive formal charge
    //    eg) [N]H3-BF3
    //  - Group 1, 2, 13 does not have nonbonding electrons, so they must have
    //    dative bond(s) if sum of bond order > common valence
    //    -> negative formal charge
    //    eg) NH3-[B]F3
    int fchg = sum_bo - cv;
    fchg = group > 14 ? fchg : -fchg;

    // Check for octet expansion of group 15~17 elements that does not have
    // aromatic bonds. If they have any aromatic bonds, they should not have
    // octet expansion, so we exclude them here.
    // Most of the time, they don't have radicals and using fchg % 2 is enough.
    if (atom.data().element().period() > 2 && group > 14
        && absl::c_any_of(atom, [](Molecule::Neighbor nei) {
             return nei.edge_data().order() == constants::kDoubleBond
                    || nei.edge_data().order() == constants::kTripleBond;
           })) {
      return fchg % 2;
    }

    return fchg;
  }

  void guess_fcharge_aromatic_rings(Molecule &mol,
                                    std::vector<int> &adjust_candidates) {
    fix_aromatic_rings(mol, adjust_candidates, guess_aromatic_fcharge_updater,
                       guess_aromatic_fcharge_scorer);
  }

  bool maybe_aromatic_atom(Molecule::Atom atom) {
    // Cannot use hybridization here because it's might not valid
    return atom.data().is_ring_atom() && all_neighbors(atom) <= 3
           && std::any_of(atom.begin(), atom.end(),
                          [](Molecule::Neighbor nei) {
                            return nei.edge_data().order()
                                   == constants::kAromaticBond;
                          })
           && std::all_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
                return nei.edge_data().is_ring_bond()
                       || nei.edge_data().order() == constants::kSingleBond;
              });
  }

  void guess_hydrogens_aromatic_rings(Molecule &mol,
                                      std::vector<int> &adjust_candidates) {
    fix_aromatic_rings(mol, adjust_candidates, guess_aromatic_hydrogens_updater,
                       guess_aromatic_hydrogens_scorer);
  }

  int predict_unknown_hyb(Molecule::Atom atom) {
    int sum_bo = internal::sum_bond_order(atom, false);
    int neighbors = all_neighbors(atom);
    if (sum_bo >= 4)
      return neighbors;

    int predicted = constants::kSP3 - (sum_bo - neighbors);

    if (predicted == constants::kSP3 && atom.data().is_conjugated())
      predicted = constants::kSP2;

    return predicted;
  }

  int guess_hydrogens_normal_atom(Molecule::Atom atom,
                                  const Element &effective) {
    int hyb_pred = atom.data().hybridization();
    if (hyb_pred == constants::kOtherHyb)
      hyb_pred = predict_unknown_hyb(atom);

    int valence = internal::sum_bond_order(atom, false),
        cv = internal::common_valence(effective),
        max_h = hyb_pred - atom.degree();
    int num_h = nuri::min(max_h, cv - valence);
    return nonnegative(num_h);
  }

  int guess_hydrogens_normal_atom(Molecule::Atom atom) {
    return guess_hydrogens_normal_atom(
        atom, internal::effective_element_or_element(atom));
  }

  int guess_fcharge_from_pcharge(Molecule::Atom atom) {
    return internal::iround(atom.data().partial_charge());
  }

  void fix_nitro_group(Molecule::MutableAtom atom, std::vector<int> &assigned) {
    if (atom.data().atomic_number() != 7 || all_neighbors(atom) > 3)
      return;

    auto is_matching_oxygen = [](Molecule::Neighbor nei) {
      return nei.dst().data().atomic_number() == 8
             && count_heavy(nei.dst()) == 1;
    };

    int terminal_oxygen_count =
        std::count_if(atom.begin(), atom.end(), is_matching_oxygen);
    if (terminal_oxygen_count != 2)
      return;

    assigned[atom.id()] = 1;
    atom.data().set_formal_charge(static_cast<int>(atom.degree() == 3));
    atom.data().set_implicit_hydrogens(0);
    for (auto nei: atom) {
      if (!is_matching_oxygen(nei))
        continue;

      auto dst = nei.dst();
      assigned[dst.id()] = 1;
      if (nei.edge_data().order() == constants::kSingleBond) {
        dst.data().set_formal_charge(-1);
        dst.data().set_implicit_hydrogens(0);
      } else {
        dst.data().set_formal_charge(0);
        dst.data().set_implicit_hydrogens(0);
      }
    }
  }

  void
  guess_fcharge_hydrogens_aromatic_rings(Molecule &mol,
                                         std::vector<int> &adjust_candidates) {
    // Carbon with two single bonds -> update both (eg. Cp-)
    // Carbon -> update formal charge
    // Heteroatom -> update implicit hydrogens
    auto updater = [](Molecule::MutableAtom atom, int delta) {
      if (atom.data().atomic_number() == 6
          || atom.data().formal_charge() != 0) {
        guess_aromatic_fcharge_updater(atom, delta);
      } else {
        guess_aromatic_hydrogens_updater(atom, delta);
      }
    };

    fix_aromatic_rings(mol, adjust_candidates, updater,
                       guess_aromatic_fcharge_hydrogens_scorer);
  }
}  // namespace

void guess_fcharge_2d(Molecule &mol) {
  std::vector<int> adjust_candidates(mol.size(), 0);
  bool has_candidate = false;

  for (auto atom: mol) {
    const int group = atom.data().element().group();
    if (group == 14 || !atom.data().element().main_group()
        || atom.data().atomic_number() == 0
        // Next line is for guadinium nitrogens (already handled; see
        // fix_guadinium)
        || atom.data().formal_charge() != 0) {
      continue;
    }

    // "Aromatic bonds" need special treatment (e.g. pyrrole vs pyridine)
    if (maybe_aromatic_atom(atom)) {
      adjust_candidates[atom.id()] = 1;
      has_candidate = true;
    }

    atom.data().set_formal_charge(guess_fcharge_atom(atom, group));
  }

  if (!has_candidate)
    return;

  guess_fcharge_aromatic_rings(mol, adjust_candidates);
}

void guess_hydrogens_2d(Molecule &mol) {
  std::vector<int> adjust_candidates(mol.size(), 0);
  bool has_candidate = false;

  for (auto atom: mol) {
    // Skip dummy, hydrogen and helium atoms
    if (atom.data().atomic_number() < 3)
      continue;

    const Element *elem = effective_element(atom);
    if (elem == nullptr) {
      ABSL_LOG(WARNING) << "Unexpected combination of element and formal "
                           "charge; cannot add hydrogens";
      continue;
    }
    if (!elem->main_group() || elem->group() == 18)
      continue;

    if (maybe_aromatic_atom(atom)) {
      adjust_candidates[atom.id()] = 1;
      has_candidate = true;
    }

    atom.data().set_implicit_hydrogens(
        guess_hydrogens_normal_atom(atom, *elem));
  }

  if (!has_candidate)
    return;

  guess_hydrogens_aromatic_rings(mol, adjust_candidates);
}

void guess_fcharge_hydrogens_2d(Molecule &mol) {
  std::vector<int> adjust_candidates(mol.size(), 0), assigned(mol.size(), 0);
  bool has_candidate = false;

  for (auto atom: mol) {
    AtomData &data = atom.data();
    if (data.atomic_number() == 0 || !data.element().main_group()
        || assigned[atom.id()] != 0) {
      assigned[atom.id()] = 1;
      continue;
    }

    if (atom.degree() == 0) {
      data.set_formal_charge(guess_fcharge_from_pcharge(atom));
      data.set_implicit_hydrogens(guess_hydrogens_normal_atom(atom));
      assigned[atom.id()] = 1;
      continue;
    }

    // "Aromatic" need special treatment (e.g. pyrrole vs pyridine)
    if (atom.data().is_aromatic() || maybe_aromatic_atom(atom)) {
      adjust_candidates[atom.id()] = 1;
      has_candidate = true;
    }

    if (data.formal_charge() != 0) {
      data.set_implicit_hydrogens(guess_hydrogens_normal_atom(atom));
      assigned[atom.id()] = 1;
      continue;
    }

    if (atom.degree() > 1)
      fix_nitro_group(atom, assigned);

    int cv = internal::common_valence(data.element());
    int sum_bo = internal::sum_bond_order(atom, false);
    int unused_valence = cv - sum_bo;
    if (unused_valence >= 0) {
      data.set_implicit_hydrogens(unused_valence);
    } else if (data.element().period() > 2) {
      ABSL_LOG_FIRST_N(WARNING, 1)
          << "Automatic formal charge & implicit hydrogens assignment might be "
             "incorrect if an atom could have expanded octet; explicitly "
             "assign one of the two properties and use other functions";

      // Just assume expanded octet without charge
      data.set_formal_charge(0);
      data.set_implicit_hydrogens(0);

      int nbe = data.element().valence_electrons() - sum_bo;
      data.set_hybridization(
          internal::from_degree(atom.degree(), nonnegative(nbe)));
    } else {
      int fchg = data.element().group() > 14 ? -unused_valence : unused_valence;
      data.set_formal_charge(fchg);
    }
    assigned[atom.id()] = 1;
  }

  if (!has_candidate)
    return;

  guess_fcharge_hydrogens_aromatic_rings(mol, adjust_candidates);
}
}  // namespace nuri
