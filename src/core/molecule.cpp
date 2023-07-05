//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

#include <algorithm>
#include <stack>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/str_cat.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"
#include "nuri/core/graph.h"
#include "nuri/utils.h"

namespace nuri {
AtomData::AtomData(const Element &element, int implicit_hydrogens,
                   int formal_charge, constants::Hybridization hyb,
                   double partial_charge, int mass_number, bool is_aromatic,
                   bool is_in_ring, bool is_chiral, bool is_right_handed)
  : element_(&element), implicit_hydrogens_(implicit_hydrogens),
    formal_charge_(formal_charge), hyb_(hyb), flags_(static_cast<AtomFlags>(0)),
    partial_charge_(partial_charge), isotope_(nullptr) {
  if (mass_number >= 0) {
    isotope_ = element.find_isotope(mass_number);
    ABSL_LOG_IF(WARNING, ABSL_PREDICT_FALSE(isotope_ == nullptr))
      << "Invalid mass number " << mass_number << " for element "
      << element.symbol();
  }

  internal::set_flag_if(flags_, is_aromatic,
                        AtomFlags::kAromatic | AtomFlags::kConjugated
                          | AtomFlags::kRing);
  internal::set_flag_if(flags_, is_in_ring, AtomFlags::kRing);
  internal::set_flag_if(flags_, is_chiral, AtomFlags::kChiral);
  internal::set_flag_if(flags_, is_right_handed, AtomFlags::kRightHanded);
}

/* Molecule definitions */

void Molecule::clear() noexcept {
  graph_.clear();
  conformers_.clear();
  name_.clear();
}

void Molecule::erase_hydrogens() {
  MoleculeMutator m = mutator();
  for (int i = 0; i < num_atoms(); ++i) {
    auto hydrogen = mutable_atom(i);
    if (hydrogen.data().atomic_number() == 1) {
      m.erase_atom(i);
      for (auto nei: hydrogen) {
        AtomData &data = nei.dst().data();
        data.set_implicit_hydrogens(data.implicit_hydrogens() + 1);
      }
    }
  }
}

namespace {
  void rotate_points(MatrixX3d &coords, const std::vector<int> &moving_idxs,
                     int ref, int pivot, double angle) {
    Vector3d pv = coords.row(pivot);
    AngleAxisd aa(deg2rad(angle), (pv - coords.row(ref)).normalized());

    auto rotate_helper = [&](auto moving) {
      moving = (aa.to_matrix() * (moving.rowwise() - pv).transpose())
                 .transpose()
                 .rowwise()
               + pv;
    };
    rotate_helper(coords(moving_idxs, Eigen::all));
  }
}  // namespace

int Molecule::add_conf(const MatrixX3d &pos) {
  const int ret = static_cast<int>(conformers_.size());
  conformers_.push_back(pos);

  // First conformer, update bond lengths
  if (ret == 0) {
    for (auto bond: graph_.edges()) {
      bond.data().length() = (pos.row(bond.dst()) - pos.row(bond.src())).norm();
    }
  }

  return ret;
}

int Molecule::add_conf(MatrixX3d &&pos) noexcept {
  const int ret = static_cast<int>(conformers_.size());
  conformers_.push_back(std::move(pos));

  // First conformer, update bond lengths
  if (ret == 0) {
    MatrixX3d &the_pos = conformers_[0];
    for (auto bond: graph_.edges()) {
      bond.data().length() =
        (the_pos.row(bond.dst()) - the_pos.row(bond.src())).norm();
    }
  }

  return ret;
}

bool Molecule::rotate_bond(int ref_atom, int pivot_atom, double angle) {
  return rotate_bond(-1, ref_atom, pivot_atom, angle);
}

bool Molecule::rotate_bond(bond_id_type bid, double angle) {
  return rotate_bond(-1, bid, angle);
}

bool Molecule::rotate_bond(int i, int ref_atom, int pivot_atom, double angle) {
  auto bit = find_bond(ref_atom, pivot_atom);
  if (bit == bond_end()) {
    return false;
  }
  return rotate_bond_common(i, *bit, ref_atom, pivot_atom, angle);
}

bool Molecule::rotate_bond(int i, bond_id_type bid, double angle) {
  const Bond b = bond(bid);
  return rotate_bond_common(i, b, b.src(), b.dst(), angle);
}

bool Molecule::rotate_bond_common(int i, Bond b, int ref_atom, int pivot_atom,
                                  double angle) {
  if (!b.data().is_rotable()) {
    return false;
  }

  absl::flat_hash_set<int> connected =
    connected_components(graph_, pivot_atom, ref_atom);
  // GCOV_EXCL_START
  if (ABSL_PREDICT_FALSE(connected.empty())) {
    ABSL_DLOG(WARNING) << ref_atom << " -> " << pivot_atom
                       << " bond is rotable, but the two atoms are connected.";
    return false;
  }
  // GCOV_EXCL_STOP

  std::vector<int> moving_atoms(connected.begin(), connected.end());
  // For faster memory access in the rotate_points function
  std::sort(moving_atoms.begin(), moving_atoms.end());

  if (i < 0) {
    for (int conf = 0; conf < num_conf(); ++conf) {
      rotate_points(conformers_[conf], moving_atoms, ref_atom, pivot_atom,
                    angle);
    }
  } else {
    rotate_points(conformers_[i], moving_atoms, ref_atom, pivot_atom, angle);
  }

  return true;
}

namespace {
  using MutableAtom = Molecule::GraphType::NodeRef;
  using MutableBond = Molecule::GraphType::EdgeRef;

  /*
   * Update the ring information of the molecule
   */
  std::pair<std::vector<std::vector<int>>, int>
  find_rings_count_connected(Molecule::GraphType &graph) {
    absl::FixedArray<int> ids(graph.num_nodes(), -1), lows(ids),
      on_stack(ids.size(), 0);
    std::stack<int, std::vector<int>> stk;
    int id = 0;

    // Find cycles
    auto tarjan = [&](auto &self, const int curr, const int prev) -> void {
      ids[curr] = lows[curr] = id++;
      stk.push(curr);
      on_stack[curr] = 1;

      for (auto adj: graph.node(curr)) {
        const int next = adj.dst().id();
        if (next == prev) {
          continue;
        }

        if (ids[next] == -1) {
          // The next line is *correct*, but clang-tidy doesn't like it
          // NOLINTNEXTLINE(readability-suspicious-call-argument)
          self(self, next, curr);

          lows[curr] = std::min(lows[curr], lows[next]);
          if (ids[curr] < lows[next]) {
            continue;
          }
        } else if (on_stack[next] != 0) {
          lows[curr] = std::min(lows[curr], ids[next]);
        }

        // If we get here, we have a ring bond (i.e, not a bridge)
        adj.edge_data().set_ring_bond(true);
        adj.src().data().set_ring_atom(true);
        adj.dst().data().set_ring_atom(true);
      }

      if (ids[curr] == lows[curr]) {
        int top;
        do {
          top = stk.top();
          stk.pop();
          on_stack[top] = 0;
          lows[top] = ids[curr];
        } while (top != curr);
      }
    };

    int num_connected = 0;
    for (int i = 0; i < graph.num_nodes(); ++i) {
      if (ids[i] == -1) {
        ++num_connected;
        tarjan(tarjan, i, -1);
      }
    }

    absl::flat_hash_map<int, std::vector<int>> components;
    for (int i = 0; i < graph.num_nodes(); ++i) {
      components[lows[i]].push_back(i);
    }

    std::pair<std::vector<std::vector<int>>, int> ret;
    ret.second = num_connected;
    for (auto &[_, comp]: components) {
      if (comp.size() > 2) {
        std::sort(comp.begin(), comp.end(),
                  [&](int a, int b) { return ids[a] < ids[b]; });
        ret.first.push_back(std::move(comp));
      }
    }
    return ret;
  }

  int nonbonding_electrons(Molecule::Atom atom, const int total_valence) {
    return atom.data().element().valence_electrons() - total_valence
           - atom.data().formal_charge();
  }

  bool is_conjugated_candidate(Molecule::Atom atom) {
    const AtomData &data = atom.data();
    // Consider main-group atoms only
    return data.element().main_group()
           // Cannot check hybridization here, as it might be incorrect
           // (Most prominent example: amide nitrogen)
           && all_neighbors(atom) <= 3;
  }

  void mark_conjugated(Molecule::GraphType &graph,
                       const std::vector<std::vector<int>> &sp2_groups) {
    for (const std::vector<int> &group: sp2_groups) {
      ABSL_DCHECK(group.size() > 2) << "Group size: " << group.size();

      for (const int id: group) {
        AtomData &data = graph.node(id).data();
        data.set_conjugated(true);
      }
    }

    for (auto bond: graph.edges()) {
      if (graph.node(bond.src()).data().is_conjugated()
          && graph.node(bond.dst()).data().is_conjugated()) {
        bond.data().set_conjugated(true);
      }
    }
  }

  void sanitize_conjugated(Molecule::GraphType &graph,
                           const absl::FixedArray<int> &valences) {
    absl::flat_hash_set<int> candidates;
    std::vector<std::vector<int>> groups;

    for (auto atom: graph) {
      if (is_conjugated_candidate(atom)) {
        candidates.insert(atom.id());
      }
    }

    auto dfs = [&](auto &self, int curr, const BondData *prev_bond) -> void {
      groups.back().push_back(curr);
      candidates.erase(curr);

      auto src = graph.node(curr);
      for (auto nei: src) {
        auto dst = nei.dst();
        if (!candidates.contains(dst.id())) {
          continue;
        }

        if (prev_bond == nullptr) {
          prev_bond = &nei.edge_data();
        } else {
          if (prev_bond->order() == constants::kSingleBond) {
            if (nei.edge_data().order() == constants::kSingleBond
                && src.data().atomic_number() != 0
                && dst.data().atomic_number() != 0) {
              // Single - single bond -> conjugated if curr has lone pair and
              // next doesn't, or vice versa, or any of them is dummy
              const int src_nbe = nonbonding_electrons(src, valences[curr]),
                        dst_nbe = nonbonding_electrons(dst, valences[dst.id()]);
              if ((src_nbe > 0 && dst_nbe > 0)
                  || (src_nbe <= 0 && dst_nbe <= 0)) {
                continue;
              }
            }
          } else if (nei.edge_data().order() != constants::kSingleBond
                     && (prev_bond->order() != constants::kAromaticBond
                         || nei.edge_data().order()
                              != constants::kAromaticBond)) {
            // Aromatic - aromatic bond -> conjugated
            // double~triple - double~triple bond -> allene-like
            // aromatic - double~triple bond -> not conjugated (erroneous?)
            continue;
          }
        }

        self(self, dst.id(), &nei.edge_data());
      }
    };

    // Mark conjugated atoms
    for (auto atom: graph) {
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

    mark_conjugated(graph, groups);
  }

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
                < 2
           // No exocyclic high-order bonds
           && std::all_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
                return nei.edge_data().is_ring_bond()
                       || nei.edge_data().order() == constants::kSingleBond;
              });
  }

  const Element &effective_element_force_unwrap(Molecule::Atom atom) {
    const Element *elem = effective_element(atom);
    if (ABSL_PREDICT_FALSE(elem == nullptr)) {
      ABSL_LOG(WARNING)
        << "Unexpected atomic number & formal charge combination: "
        << atom.data().atomic_number() << ", " << atom.data().formal_charge()
        << ". The result may be incorrect.";
      elem = &atom.data().element();
    }
    return *elem;
  }

  int octet_valence(Molecule::Atom atom) {
    const Element &elem = effective_element_force_unwrap(atom);

    const int val_electrons = elem.valence_electrons(),
              octet_valence = val_electrons <= 4 ? val_electrons
                                                 : 8 - val_electrons;
    return octet_valence;
  }

  int count_pi_e(Molecule::Atom atom, int total_valence) {
    const int nb_electrons = nonbonding_electrons(atom, total_valence);
    ABSL_DLOG_IF(WARNING, nb_electrons < 0)
      << "Negative nonbonding electrons for atom " << atom.id() << " ("
      << atom.data().element().symbol() << "): " << nb_electrons;

    if (std::any_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
          return nei.edge_data().order() == constants::kAromaticBond;
        })) {
      // Special case, some bonds are aromatic
      // Now we have to check structures like furan, pyrrole, etc.
      const int ov = octet_valence(atom);

      // E.g. O in furan, N in pyrrole, ...
      if (ov < total_valence) {
        // Has nonbonding electrons (O, N) -> 2 electrons participate in
        // No nonbonding electrons (B) -> no electrons participate in
        const int pie_estimate = static_cast<int>(nb_electrons > 0) * 2;
        ABSL_DLOG(INFO) << "Special case: " << atom.id() << " " << pie_estimate
                        << " pi electrons";
        return pie_estimate;
      }

      // Not sure if this condition will ever be true, just in case
      ABSL_LOG_IF(WARNING, ov > total_valence)
        << "Valence smaller than octet valence";

      // Normal case: atoms in pyridine, benzene, ...
      ABSL_DLOG(INFO) << "Normal case: " << atom.id() << " 1 pi electron";
      return 1;
    }

    if (std::any_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
          return nei.edge_data().is_ring_bond()
                 && (nei.edge_data().order() == constants::kDoubleBond
                     || nei.edge_data().order() == constants::kTripleBond);
        })) {
      // Normal case, at least one double/triple bond in the ring
      ABSL_DLOG(INFO) << "Normal case: " << atom.id() << " 1 pi electron";
      return 1;
    }

    // E.g. N in pyrrole
    const int pie_estimate = std::min(nb_electrons, 2);
    ABSL_DLOG(INFO)
      << "Exceptional: " << atom.id() << " " << pie_estimate << " pi electrons";
    return pie_estimate;
  }

  bool test_aromatic(Molecule::GraphType &graph, const std::vector<int> &ring,
                     const int pi_e_sum) {
    return pi_e_sum % 4 == 2
           // Dummy atoms are always allowed in aromatic rings
           || std::any_of(ring.begin(), ring.end(), [&](int id) {
                return graph.node(id).data().atomic_number() == 0;
              });
  }

  void mark_aromatic_ring(Molecule::GraphType &graph,
                          const std::vector<int> &ring,
                          const absl::flat_hash_map<int, int> &pi_e) {
    int pi_e_sum = 0;
    for (int i = 0; i < ring.size(); ++i) {
      auto it = pi_e.find(ring[i]);
      if (it == pi_e.end()) {
        return;
      }
      pi_e_sum += it->second;
    }

    const bool this_aromatic = test_aromatic(graph, ring, pi_e_sum);
    if (!this_aromatic) {
      return;
    }

    for (int i = 0; i < ring.size(); ++i) {
      const int src = ring[i],
                dst = ring[(i + 1) % static_cast<int>(ring.size())];

      graph.node(src).data().set_aromatic(true);

      auto eit = graph.find_edge(src, dst);
      ABSL_DCHECK(eit != graph.edge_end());
      eit->data().set_aromatic(true);
    }
  }

  void mark_aromatic(Molecule::GraphType &graph, const Molecule &mol,
                     const std::vector<std::vector<int>> &rings,
                     const absl::FixedArray<int> &valences,
                     const int circuit_rank) {
    absl::flat_hash_map<int, int> pi_e;

    for (auto atom: graph) {
      if (is_aromatic_candidate(atom)) {
        pi_e.insert({ atom.id(), count_pi_e(atom, valences[atom.id()]) });
      }
    }

    auto mark_aromatic_for = [&](const std::vector<std::vector<int>> &rs) {
      for (const std::vector<int> &ring: rs) {
        mark_aromatic_ring(graph, ring, pi_e);
      }
    };

    const bool need_subring = circuit_rank > static_cast<int>(rings.size());
    // Fast path: no need to find subrings
    if (!need_subring) {
      mark_aromatic_for(rings);
      return;
    }

    auto [subrings, success] = find_all_rings(mol);
    // Almost impossible, don't include in coverage report
    // GCOV_EXCL_START
    if (ABSL_PREDICT_TRUE(success)) {
      mark_aromatic_for(subrings);
      return;
    }

    ABSL_LOG(WARNING)
      << "Ring finding exceeds threshold, falling back to SSSR-based "
         "aromaticity detection: the result may be incorrect";

    mark_aromatic_for(find_sssr(mol));
    // GCOV_EXCL_STOP
  }

  bool sanitize_aromaticity(Molecule::GraphType &graph, const Molecule &mol,
                            const std::vector<std::vector<int>> &rings,
                            const absl::FixedArray<int> &valences,
                            const int circuit_rank) {
    for (auto bond: graph.edges()) {
      if (!bond.data().is_ring_bond()) {
        if (bond.data().order() == constants::kAromaticBond) {
          ABSL_LOG(WARNING)
            << "Bond order between atoms " << bond.src() << " and "
            << bond.dst()
            << " is set aromatic, but the bond is not a ring bond";
          return false;
        }
      }
    }

    if (circuit_rank > 0) {
      mark_aromatic(graph, mol, rings, valences, circuit_rank);
    }

    // NOLINTNEXTLINE(readability-use-anyofallof)
    for (auto bond: graph.edges()) {
      if (bond.data().order() == constants::kAromaticBond
          && !bond.data().is_aromatic()) {
        ABSL_LOG(WARNING) << "Bond order of non-aromatic bond " << bond.src()
                          << " - " << bond.dst() << " is set aromatic";
        return false;
      }
    }

    return true;
  }

  constants::Hybridization from_degree(const int total_degree,
                                       const int nb_electrons) {
    const int lone_pairs = nb_electrons / 2 + nb_electrons % 2,
              steric_number = total_degree + lone_pairs;
    return static_cast<constants::Hybridization>(
      std::min(steric_number, static_cast<int>(constants::kOtherHyb)));
  }

  std::string format_atom_common(Molecule::Atom atom, bool caps) {
    return absl::StrCat(caps ? "A" : "a", "tom ", atom.id(), " (",
                        atom.data().element().symbol(), ")");
  }

  // Set hybridization for others
  bool sanitize_hybridization(MutableAtom atom, const int total_valence) {
    const int total_degree = all_neighbors(atom);

    ABSL_DLOG(INFO)
      << atom.id() << ": " << total_degree << ", " << total_valence;

    if (atom.data().atomic_number() == 0) {
      // Assume dummy atom always satisfies the octet rule
      const int nbe = std::max(8 - total_valence, 0);
      atom.data().set_hybridization(from_degree(total_degree, nbe));
      return true;
    }

    int nbe = nonbonding_electrons(atom, total_valence);
    if (nbe < 0) {
      ABSL_LOG(WARNING)
        << "Valence electrons exceeded for " << format_atom_common(atom, false)
        << ": total valence " << total_valence << ", "
        << "formal charge " << atom.data().formal_charge();
      return false;
    }

    constants::Hybridization hyb = from_degree(total_degree, nbe);
    if (hyb == constants::kSP3 && atom.data().is_conjugated()) {
      hyb = constants::kSP2;
      if (atom.data().is_aromatic() && sum_bond_order(atom) == 4
          && nbe % 2 != 0) {
        // Pyrrole, etc.
        --nbe;
      }
    }

    // Octet verification
    const int total_valence_electrons = nbe + 2 * total_valence;
    const Element &effective = effective_element_force_unwrap(atom);
    int max_valence;
    switch (effective.period()) {
    case 1:
      max_valence = 2;
      break;
    case 2:
      max_valence = 8;
      break;
    default:
      max_valence = effective.lanthanide() || effective.actinide() ? 32 : 18;
      break;
    }

    if (total_valence_electrons > max_valence) {
      ABSL_LOG(WARNING)
        << format_atom_common(atom, true) << " with charge "
        << atom.data().formal_charge() << " has more than " << max_valence
        << " valence electrons: " << total_valence_electrons;
      return false;
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
        std::min(hyb, from_degree(total_degree, 0)));
    }

    return true;
  }

  int sum_bond_order_impl(Molecule::Atom atom, bool aromatic_correct) {
    int sum_order = atom.data().implicit_hydrogens(), num_aromatic = 0;

    for (auto adj: atom) {
      if (adj.edge_data().order() == constants::kAromaticBond
          || (aromatic_correct && adj.edge_data().is_aromatic())) {
        ++num_aromatic;
      } else {
        sum_order += adj.edge_data().order();
      }
    }

    if (num_aromatic == 0) {
      return sum_order;
    }

    ABSL_LOG_IF(INFO, aromatic_correct && !atom.data().is_aromatic())
      << "Non-aromatic atom " << atom.id() << " has " << num_aromatic
      << " aromatic bonds";

    if (num_aromatic < 2) {
      ABSL_LOG(INFO) << "Aromatic atom with less than two aromatic bonds; "
                        "assuming single bond for each aromatic bond";
      return sum_order + num_aromatic;
    }
    if (num_aromatic > 3) {
      // Aromatic atom with >= 4 aromatic bonds is very unlikely;
      // just log it and fall through
      ABSL_LOG(INFO) << "Cannot correctly determine total bond order for "
                        "aromatic atom with more than 4 aromatic bonds";
    }

    // The logic here:
    //   - for 2 aromatic bonds, each will contribute 1.5 to the total bond
    //     order (e.g. benzene) = 3
    //   - for 3 aromatic bonds, assume 2, 1, 1 for the bond orders (this is
    //     the most common case, e.g. naphthalene) = 4
    //   - others are very unlikely, ignore for now
    sum_order += num_aromatic + 1;
    return sum_order;
  }

  std::pair<int, bool> mol_sanitize_impl(Molecule::GraphType &graph,
                                         const Molecule &mol) {
    // Clear all flags
    for (auto atom: graph) {
      atom.data().reset_flags();
    }
    for (auto bond: graph.edges()) {
      bond.data().reset_flags();
    }

    // Find ring atoms & bonds
    auto [rings, num_connected] = find_rings_count_connected(graph);
    const int circuit_rank =
      graph.num_edges() - graph.num_nodes() + num_connected;

    // Calculate valence
    absl::FixedArray<int> valences(graph.num_nodes());
    for (int i = 0; i < graph.num_nodes(); ++i) {
      valences[i] = sum_bond_order_impl(graph.node(i), false);
    }

    // Find conjugation
    sanitize_conjugated(graph, valences);

    // Fix aromaticity
    if (!sanitize_aromaticity(graph, mol, rings, valences, circuit_rank)) {
      return { circuit_rank, false };
    }

    // Validate valence
    for (int i = 0; i < graph.num_nodes(); ++i) {
      if (!sanitize_hybridization(graph.node(i), valences[i])) {
        return { circuit_rank, false };
      }
    }

    // Set rotable bonds
    for (auto bond: graph.edges()) {
      if (bond.data().is_conjugated() || bond.data().is_ring_bond()) {
        continue;
      }

      bond.data().set_rotable(bond.data().order() == constants::kSingleBond);
    }

    // TODO(jnooree): check geometric isomers

    return { circuit_rank, true };
  }
}  // namespace

bool Molecule::sanitize(int /* use_conformer */) {
  std::tie(circuit_rank_, was_valid_) = mol_sanitize_impl(graph_, *this);
  return was_valid_;
}

/* MoleculeMutator definitions */

bool MoleculeMutator::add_bond(int src, int dst, const BondData &bond) {
  if (ABSL_PREDICT_FALSE(src == dst)) {
    return false;
  }

  if (mol().graph_.find_edge(src, dst) != mol().graph_.edge_end()) {
    return false;
  }

  mol().graph_.add_edge(src, dst, bond);
  return true;
}

void MoleculeMutator::erase_bond(int src, int dst) {
  if (ABSL_PREDICT_FALSE(src == dst)) {
    return;
  }

  erased_bonds_.push_back({ src, dst });
}

BondData *MoleculeMutator::bond_data(int src, int dst) {
  auto bit = mol().find_mutable_bond(src, dst);
  return bit != mol().bond_end() ? &bit->data() : nullptr;
}

void MoleculeMutator::discard_erasure() noexcept {
  erased_atoms_.clear();
  erased_bonds_.clear();
}

void MoleculeMutator::finalize() noexcept {
  Molecule::GraphType &g = mol().graph_;
  const int old_size = mol().num_atoms();

  // As per the spec, the order is:
  // 1. Erase bonds
  for (const std::pair<int, int> &ends: erased_bonds_) {
    g.erase_edge_between(ends.first, ends.second);
  }

  // 2. Erase atoms
  auto [last, map] = g.erase_nodes(erased_atoms_.begin(), erased_atoms_.end());

  // Update coordinates
  if (last >= 0) {
    // Only trailing nodes are removed
    for (MatrixX3d &conf: mol().conformers_) {
      conf.conservativeResize(mol().num_atoms(), Eigen::NoChange);
    }
  } else {
    // Select the atom indices
    std::vector<int> idxs;
    idxs.reserve(mol().num_atoms());

    for (int i = 0; i < old_size; ++i) {
      // GCOV_EXCL_START
      ABSL_DCHECK(i < map.size());
      // GCOV_EXCL_STOP
      if (map[i] >= 0) {
        idxs.push_back(i);
      }
    }

    for (MatrixX3d &conf: mol().conformers_) {
      MatrixX3d updated = conf(idxs, Eigen::all);
      conf = std::move(updated);
    }
  }

  if (sanitize_) {
    ABSL_LOG_IF(WARNING, !mol().sanitize(conformer_idx_))
      << "Molecule sanitization failed!";
  }

  discard_erasure();
}

int count_hydrogens(Molecule::Atom atom) {
  int count = atom.data().implicit_hydrogens();
  for (auto adj: atom) {
    if (adj.dst().data().atomic_number() == 1) {
      ++count;
    }
  }
  return count;
}

int sum_bond_order(Molecule::Atom atom) {
  return sum_bond_order_impl(atom, true);
}

const Element *effective_element(Molecule::Atom atom) {
  const int effective_z =
    atom.data().atomic_number() - atom.data().formal_charge();
  return PeriodicTable::get().find_element(effective_z);
}
}  // namespace nuri
