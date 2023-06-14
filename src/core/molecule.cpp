//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"

namespace nuri {
AtomData::AtomData(const Element &element, constants::Hybridization hyb,
                   int implicit_hydrogens, int formal_charge,
                   double partial_charge, int mass_number, bool is_aromatic,
                   bool is_in_ring, bool is_chiral, bool is_right_handed)
  : element_(&element), isotope_(nullptr), hyb_(hyb),
    implicit_hydrogens_(implicit_hydrogens), flags_(0),
    formal_charge_(formal_charge), partial_charge_(partial_charge) {
  if (mass_number >= 0) {
    isotope_ = element.find_isotope(mass_number);
    ABSL_LOG_IF(WARNING, ABSL_PREDICT_FALSE(isotope_ == nullptr))
      << "Invalid mass number " << mass_number << " for element "
      << element.symbol();
  }

  auto set_flag = [this](bool cond, Flags flag) {
    flags_ |= -static_cast<decltype(flags_)>(cond) & flag;
  };
  set_flag(is_aromatic, kAromaticAtom);
  set_flag(is_in_ring, kRingAtom);
  set_flag(is_chiral, kChiralAtom);
  set_flag(is_right_handed, kRightHandedAtom);
}

/* Molecule definitions */

void Molecule::clear() noexcept {
  graph_.clear();
  conformers_.clear();
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
    for (auto bit = graph_.edge_begin(); bit != graph_.edge_end(); ++bit) {
      bit->data().length() = (pos.row(bit->dst()) - pos.row(bit->src())).norm();
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
    for (auto bit = graph_.edge_begin(); bit != graph_.edge_end(); ++bit) {
      bit->data().length() =
        (the_pos.row(bit->dst()) - the_pos.row(bit->src())).norm();
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

  /*
   * Update the ring information of the molecule
   */
  void mark_rings(Molecule::GraphType &graph) {
    absl::FixedArray<int> ids(graph.num_nodes(), -1), lows(ids);
    int id = 0;

    for (auto atom: graph) {
      atom.data().set_ring_atom(false);
    }

    // Find cycle basis
    auto tarjan = [&](auto &self, const int curr, const int prev) -> void {
      ids[curr] = lows[curr] = id++;

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
            adj.edge_data().set_ring_bond(false);
            continue;
          }
        } else {
          lows[curr] = std::min(lows[curr], ids[next]);
        }

        // If we get here, we have a ring bond (i.e, not a bridge)
        adj.edge_data().set_ring_bond(true);
        adj.src().data().set_ring_atom(true);
        adj.dst().data().set_ring_atom(true);
      }
    };

    for (int i = 0; i < graph.num_nodes(); ++i) {
      if (ids[i] == -1) {
        tarjan(tarjan, i, -1);
      }
    }
  }

  constants::Hybridization from_sn(int steric_number) {
    return static_cast<constants::Hybridization>(
      std::min(steric_number, static_cast<int>(constants::kOtherHyb)));
  }

  int octet_valence(const Element &elem) {
    int val_electrons = elem.valence_electrons();
    int octet_valence = val_electrons <= 4 ? val_electrons : 8 - val_electrons;
    return octet_valence;
  }

  int nonbonding_electrons(Molecule::Atom atom, const int total_valence) {
    return atom.data().element().valence_electrons() - total_valence;
  }

  // For 1st - 2nd row elements
  bool sethyb_bound_not_extended(MutableAtom atom, const int total_degree,
                                 const int total_valence) {
    // Special case: helium
    if (atom.data().atomic_number() == 2) {
      return false;
    }

    // Special case: terminal atoms
    if (total_degree == 1) {
      atom.data().set_hybridization(constants::kTerminal);
      return true;
    }

    const int typical_valence =
      octet_valence(PeriodicTable::get()[atom.data().atomic_number()
                                         - atom.data().formal_charge()]);
    if (total_valence != typical_valence) {
      return false;
    }

    const int nb_electrons = nonbonding_electrons(atom, total_valence),
              lone_pairs = nb_electrons / 2 + nb_electrons % 2;
    atom.data().set_hybridization(from_sn(total_degree + lone_pairs));
    return true;
  }

  bool sethyb_bound_maybe_expanded(MutableAtom atom, const int total_degree,
                                   const int total_valence) {
    // Special case: terminal atoms
    if (total_degree == 1) {
      atom.data().set_hybridization(constants::kTerminal);
      return true;
    }

    if (total_valence > atom.data().element().valence_electrons()) {
      atom.data().set_hybridization(constants::kUnbound);
      return false;
    }

    const int nb_electrons = nonbonding_electrons(atom, total_valence),
              lone_pairs = nb_electrons / 2 + nb_electrons % 2;
    atom.data().set_hybridization(from_sn(total_degree + lone_pairs));
    return true;
  }

  bool sethyb(MutableAtom atom, const int total_degree,
              const int total_valence) {
    ABSL_DLOG(INFO)
      << atom.id() << ": " << total_degree << ", " << total_valence;

    if (total_degree == 0) {
      atom.data().set_hybridization(constants::kUnbound);
      return true;
    }

    if (!atom.data().element().main_group()) {
      // Skip non-main group elements from validation, and just assume
      // SN == total degree
      atom.data().set_hybridization(from_sn(total_degree));
      return true;
    }

    if (atom.data().element().period() <= 2) {
      return sethyb_bound_not_extended(atom, total_degree, total_valence);
    }

    return sethyb_bound_maybe_expanded(atom, total_degree, total_valence);
  }

  bool is_aromatic_candidate(const AtomData &data) {
    // Consider in-ring, main-group atoms with sp or sp2 hybridization
    return data.is_ring_atom() && data.hybridization() <= constants::kSP2
           && data.element().main_group();
  }

  int atom_num_pi_e(Molecule::Atom atom, int total_valence) {
    const int nb_electrons = nonbonding_electrons(atom, total_valence);
    // TODO(jnooree): How about radicals?
    return std::min(nb_electrons, 2);
  }

  int bond_num_pi_e(const BondData &bond) {
    if (bond.order() == constants::kAromaticBond) {
      return 1;
    }
    return static_cast<int>(bond.order() >= constants::kDoubleBond) * 2;
  }

  void mark_aromatic(Molecule::GraphType &graph,
                     const absl::FixedArray<int> &valences) {
    // Most rings likely to be small, so use a moderately small initial capacity
    absl::flat_hash_set<int> visited(10);

    // Clear aromaticity flags, might be incorrect
    for (auto atom: graph) {
      atom.data().set_aromatic(false);
    }
    for (auto bit = graph.edge_begin(); bit != graph.edge_end(); ++bit) {
      bit->data().set_aromatic(false);
    }

    for (auto atom: graph) {
      if (atom.data().is_aromatic() || !is_aromatic_candidate(atom.data())) {
        // Already processed or not possible to be aromatic
        continue;
      }

      const int begin = atom.id();
      auto dfs = [&](auto &self, int curr, int pi_cnt) -> bool {
        visited.insert(curr);

        auto curr_atom = graph.node(curr);
        pi_cnt += atom_num_pi_e(curr_atom, valences[curr]);

        for (auto nei: curr_atom) {
          auto dst = nei.dst();
          pi_cnt += bond_num_pi_e(nei.edge_data());

          // Ring formed
          if (dst.id() == begin) {
            const bool is_aromatic = (pi_cnt - 2) % 4 == 0;
            dst.data().set_aromatic(is_aromatic);
            nei.edge_data().set_aromatic(is_aromatic);
            return is_aromatic;
          }

          if (visited.contains(dst.id())
              || !is_aromatic_candidate(dst.data())) {
            continue;
          }

          if (self(self, dst.id(), pi_cnt)) {
            dst.data().set_aromatic(true);
            nei.edge_data().set_aromatic(true);
            return true;
          }
        }

        return false;
      };

      visited.clear();
      dfs(dfs, atom.id(), 0);
    }

    // Must be done here, otherwise will corrupt pi electron count of
    // "Kekulized" molecules
    for (auto bit = graph.edge_begin(); bit != graph.edge_end(); ++bit) {
      if (bit->data().is_aromatic()) {
        bit->data().order() = constants::kAromaticBond;
      }
    }
  }

  bool mol_sanitize_impl(Molecule::GraphType &graph) {
    // Find ring atoms & bonds
    mark_rings(graph);

    // Clear non-ring aromaticity
    for (auto bit = graph.edge_begin(); bit != graph.edge_end(); ++bit) {
      if (!bit->data().is_ring_bond()) {
        bit->data().set_aromatic(false);
        if (bit->data().order() == constants::kAromaticBond) {
          ABSL_LOG(ERROR)
            << "Aromatic bond detected between atoms " << bit->src() << " and "
            << bit->dst() << " but the bond is not a ring bond.";
          return false;
        }
      }
    }
    for (auto atom: graph) {
      if (!atom.data().is_ring_atom()) {
        atom.data().set_aromatic(false);
      }
    }

    // Fix hybridization
    absl::FixedArray<int> valences(graph.num_nodes());
    for (int i = 0; i < graph.num_nodes(); ++i) {
      MutableAtom atom = graph.node(i);
      const int total_degree = all_neighbors(atom),
                total_valence = sum_bond_order(atom);
      valences[i] = total_valence;
      if (ABSL_PREDICT_FALSE(!sethyb(atom, total_degree, total_valence))) {
        return false;
      }
    }

    // Now can find aromatic rings
    mark_aromatic(graph, valences);

    return true;
  }
}  // namespace

bool Molecule::sanitize(int /* use_conformer */) {
  return was_valid_ = mol_sanitize_impl(graph_);
}

/* MoleculeMutator definitions */

bool MoleculeMutator::add_bond(int src, int dst, const BondData &bond) {
  if (ABSL_PREDICT_FALSE(src == dst)) {
    return false;
  }

  const std::pair<int, int> ends = std::minmax(src, dst);
  auto [it, inserted] = new_bonds_set_.insert(ends);
  if (ABSL_PREDICT_FALSE(!inserted)) {
    return false;
  }
  if (src < mol().num_atoms() && dst < mol().num_atoms()
      && ABSL_PREDICT_FALSE(mol().graph_.find_edge(src, dst)
                            != mol().graph_.edge_end())) {
    new_bonds_set_.erase(it);
    return false;
  }

  new_bonds_.push_back({ ends, bond });
  return true;
}

void MoleculeMutator::erase_bond(int src, int dst) {
  if (ABSL_PREDICT_FALSE(src == dst)) {
    return;
  }

  erased_bonds_.push_back(std::minmax(src, dst));
}

int MoleculeMutator::num_atoms() const {
  return next_atom_idx() - static_cast<int>(erased_atoms_.size());
}

void MoleculeMutator::discard() noexcept {
  new_atoms_.clear();
  erased_atoms_.clear();

  new_bonds_.clear();
  new_bonds_set_.clear();
  erased_bonds_.clear();
}

void MoleculeMutator::accept() noexcept {
  Molecule::GraphType &g = mol().graph_;
  const int old_size = mol().num_atoms();

  // As per the spec, the order is:

  // 1. Add atoms
  g.add_node(new_atoms_.begin(), new_atoms_.end());

  // 2. Add bonds
  for (const AddedBond &b: new_bonds_) {
    g.add_edge(b.ends.first, b.ends.second, b.data);
  }

  // 3. Erase bonds
  for (const std::pair<int, int> &ends: erased_bonds_) {
    g.erase_edge_between(ends.first, ends.second);
  }

  // 4. Erase atoms
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
    ABSL_LOG_IF(WARNING, !mol_->sanitize(conformer_idx_))
      << "Molecule sanitization failed!";
  }

  // Update rotable flags
  for (auto bit = g.edge_begin(); bit != g.edge_end(); ++bit) {
    auto &d = bit->data();
    d.set_rotable(!d.is_ring_bond() && !d.is_conjugated()
                  && d.order() <= constants::kSingleBond);
  }

  discard();
}

int MoleculeMutator::next_atom_idx() const {
  return mol().num_atoms() + static_cast<int>(new_atoms_.size());
}

namespace {
  int aromatic_total_bond_order(Molecule::Atom atom) {
    int num_aromatic = 0, sum_order = 0;
    for (auto adj: atom) {
      if (adj.edge_data().is_aromatic()) {
        ++num_aromatic;
      } else {
        sum_order += adj.edge_data().order();
      }
    }

    if (num_aromatic < 2) {
      ABSL_LOG(WARNING) << "Aromatic atom with less than two aromatic bonds; "
                           "assuming single bond for each aromatic bond";
      return sum_order + num_aromatic;
    }
    if (num_aromatic > 3) {
      // Aromatic atom with >= 4 aromatic bonds is very unlikely;
      // just log it and fall through
      ABSL_LOG(WARNING) << "Cannot correctly determine total bond order for "
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
}  // namespace

int sum_bond_order(Molecule::Atom atom) {
  int sum_explicit_order = 0;

  if (atom.data().is_aromatic()) {
    // Now we have to consider aromaticity
    sum_explicit_order = aromatic_total_bond_order(atom);
  } else {
    // Aliphatic atom, just sum up the bond orders
    sum_explicit_order = std::accumulate(
      atom.begin(), atom.end(), 0, [&](int acc, Molecule::Neighbor nei) {
        ABSL_DCHECK(!nei.edge_data().is_aromatic());
        return acc + nei.edge_data().order();
      });
  }

  return sum_explicit_order + atom.data().implicit_hydrogens();
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
}  // namespace nuri
