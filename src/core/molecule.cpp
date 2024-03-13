//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

#include <algorithm>
#include <numeric>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/str_cat.h>

#include "nuri/eigen_config.h"
#include "nuri/algo/rings.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"
#include "nuri/core/graph.h"
#include "nuri/utils.h"

namespace nuri {
using internal::count_pi_e;
using internal::effective_element_or_element;
using internal::from_degree;
using internal::nonbonding_electrons;

AtomData::AtomData(const Element &element, int implicit_hydrogens,
                   int formal_charge, constants::Hybridization hyb,
                   double partial_charge, int mass_number, bool is_aromatic,
                   bool is_in_ring, bool is_chiral, bool is_right_handed)
    : element_(&element), implicit_hydrogens_(implicit_hydrogens),
      formal_charge_(formal_charge), hyb_(hyb),
      flags_(static_cast<AtomFlags>(0)), partial_charge_(partial_charge),
      isotope_(nullptr) {
  if (mass_number >= 0) {
    isotope_ = element.find_isotope(mass_number);
    ABSL_LOG_IF(WARNING, ABSL_PREDICT_FALSE(isotope_ == nullptr))
        << "Invalid mass number " << mass_number << " for element "
        << element.symbol();
  }

  if (implicit_hydrogens_ < 0) {
    ABSL_DLOG(WARNING)
        << "Negative implicit hydrogens for atom " << element.symbol();
    implicit_hydrogens_ = 0;
  }

  internal::set_flag_if(flags_, is_aromatic,
                        AtomFlags::kAromatic | AtomFlags::kConjugated
                            | AtomFlags::kRing);
  internal::set_flag_if(flags_, is_in_ring, AtomFlags::kRing);
  internal::set_flag_if(flags_, is_chiral, AtomFlags::kChiral);
  internal::set_flag_if(flags_, is_right_handed, AtomFlags::kRightHanded);
}

/* Molecule definitions */

Molecule::Molecule(const Molecule &other) noexcept
    : graph_(other.graph_), conformers_(other.conformers_), name_(other.name_),
      props_(other.props_), substructs_(other.substructs_),
      ring_groups_(other.ring_groups_), num_fragments_(other.num_fragments_) {
  rebind_substructs();
}

Molecule::Molecule(Molecule &&other) noexcept
    : graph_(std::move(other.graph_)),
      conformers_(std::move(other.conformers_)), name_(std::move(other.name_)),
      props_(std::move(other.props_)),
      substructs_(std::move(other.substructs_)),
      ring_groups_(std::move(other.ring_groups_)),
      num_fragments_(other.num_fragments_) {
  rebind_substructs();
}

Molecule &Molecule::operator=(const Molecule &other) noexcept {
  graph_ = other.graph_;
  conformers_ = other.conformers_;
  name_ = other.name_;
  props_ = other.props_;
  substructs_ = other.substructs_;
  ring_groups_ = other.ring_groups_;
  num_fragments_ = other.num_fragments_;

  rebind_substructs();

  return *this;
}

Molecule &Molecule::operator=(Molecule &&other) noexcept {
  graph_ = std::move(other.graph_);
  conformers_ = std::move(other.conformers_);
  name_ = std::move(other.name_);
  props_ = std::move(other.props_);
  substructs_ = std::move(other.substructs_);
  ring_groups_ = std::move(other.ring_groups_);
  num_fragments_ = other.num_fragments_;

  rebind_substructs();

  return *this;
}

void Molecule::clear() noexcept {
  graph_.clear();
  conformers_.clear();
  name_.clear();
  props_.clear();
  substructs_.clear();
  ring_groups_.clear();
  num_fragments_ = 0;
}

void Molecule::clear_atoms() noexcept {
  graph_.clear();

  for (Matrix3Xd &conf: conformers_)
    conf.resize(Eigen::NoChange, 0);

  for (Substructure &sub: substructs_)
    sub.clear_atoms();

  ring_groups_.clear();
  num_fragments_ = 0;
}

void Molecule::clear_bonds() noexcept {
  graph_.clear_edges();

  ring_groups_.clear();
  num_fragments_ = num_atoms();
}

void Molecule::erase_hydrogens() {
  MoleculeMutator m = mutator();
  for (auto atom: *this) {
    if (atom.data().atomic_number() == 1) {
      m.mark_atom_erase(atom.id());
      for (auto nei: atom) {
        AtomData &data = nei.dst().data();
        data.set_implicit_hydrogens(data.implicit_hydrogens() + 1);
      }
    }
  }
}

namespace {
  void rotate_points(Matrix3Xd &coords, const std::vector<int> &moving_idxs,
                     int ref, int pivot, double angle) {
    Eigen::Vector3d pv = coords.col(pivot);
    Eigen::Affine3d rotation =
        Eigen::Translation3d(pv)
        * Eigen::AngleAxisd(deg2rad(angle),
                            internal::safe_normalized(pv - coords.col(ref)))
        * Eigen::Translation3d(-pv);

    auto rotate_helper = [&](auto moving) { moving = rotation * moving; };
    rotate_helper(coords(Eigen::all, moving_idxs));
  }
}  // namespace

double Molecule::distsq(int src, int dst, int conf) const {
  const Matrix3Xd &pos = conformers_[conf];
  return (pos.col(dst) - pos.col(src)).squaredNorm();
}

ArrayXd Molecule::bond_lengths(int conf) const {
  ArrayXd lengths(num_bonds());

  auto bit = bond_begin();
  for (int i = 0; i < num_bonds(); ++i) {
    lengths[i] = distance(*bit++, conf);
  }

  return lengths;
}

bool Molecule::rotate_bond(int ref_atom, int pivot_atom, double angle) {
  return rotate_bond_conf(-1, ref_atom, pivot_atom, angle);
}

bool Molecule::rotate_bond(int bid, double angle) {
  return rotate_bond_conf(-1, bid, angle);
}

bool Molecule::rotate_bond_conf(int i, int ref_atom, int pivot_atom,
                                double angle) {
  return rotate_bond_common(i, ref_atom, pivot_atom, angle);
}

bool Molecule::rotate_bond_conf(int i, int bid, double angle) {
  const Bond b = bond(bid);
  return rotate_bond_common(i, b.src().id(), b.dst().id(), angle);
}

void Molecule::rebind_substructs() noexcept {
  for (Substructure &sub: substructs_)
    sub.rebind(graph_);
}

bool Molecule::rotate_bond_common(int i, int ref_atom, int pivot_atom,
                                  double angle) {
  absl::flat_hash_set<int> connected =
      connected_components(graph_, pivot_atom, ref_atom);
  // GCOV_EXCL_START
  if (connected.empty()) {
    ABSL_LOG(INFO) << ref_atom << " -> " << pivot_atom
                   << " two atoms of bond are connected and cannot be rotated";
    return false;
  }
  // GCOV_EXCL_STOP

  std::vector<int> moving_atoms(connected.begin(), connected.end());
  // For faster memory access in the rotate_points function
  std::sort(moving_atoms.begin(), moving_atoms.end());

  if (i < 0) {
    for (int conf = 0; conf < confs().size(); ++conf) {
      rotate_points(conformers_[conf], moving_atoms, ref_atom, pivot_atom,
                    angle);
    }
  } else {
    rotate_points(conformers_[i], moving_atoms, ref_atom, pivot_atom, angle);
  }

  return true;
}

namespace {
  using MutableAtom = Molecule::MutableAtom;
  using MutableBond = Molecule::MutableBond;

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

    std::vector<std::vector<int>> components(id);
    for (int i = 0; i < graph.num_nodes(); ++i) {
      components[lows[i]].push_back(i);
    }

    std::pair<std::vector<std::vector<int>>, int> ret;
    ret.second = num_connected;
    for (std::vector<int> &comp: components) {
      if (comp.size() > 2) {
        std::sort(comp.begin(), comp.end(),
                  [&](int a, int b) { return ids[a] < ids[b]; });
        ret.first.push_back(std::move(comp));
      }
    }
    return ret;
  }
}  // namespace

void Molecule::update_topology() {
  for (auto atom: graph_) {
    atom.data().set_ring_atom(false);
  }

  for (auto bond: graph_.edges()) {
    bond.data().set_ring_bond(false);
  }

  // Find ring atoms & bonds
  std::tie(ring_groups_, num_fragments_) = find_rings_count_connected(graph_);

  // Fix non-ring aromaticity
  for (auto atom: graph_) {
    if (!atom.data().is_ring_atom()) {
      atom.data().set_aromatic(false);
    }
  }

  for (auto bond: graph_.edges()) {
    if (!bond.data().is_ring_bond()) {
      bond.data().set_aromatic(false);
    }
  }
}

/* MoleculeMutator definitions */

void MoleculeMutator::clear_atoms() noexcept {
  mol().clear_atoms();
  prev_num_atoms_ = prev_num_bonds_ = 0;
  discard_erasure();
}

namespace {
  template <class DT>
  std::pair<Molecule::bond_iterator, bool>
  add_bond_impl(Molecule::GraphType &graph, int src, int dst, DT &&bond) {
    auto it = graph.find_edge(src, dst);
    if (it != graph.edge_end()) {
      return std::make_pair(it, false);
    }

    it = graph.add_edge(src, dst, std::forward<DT>(bond));
    return std::make_pair(it, true);
  }
}  // namespace

std::pair<Molecule::bond_iterator, bool>
MoleculeMutator::add_bond(int src, int dst, const BondData &bond) {
  return add_bond_impl(mol().graph_, src, dst, bond);
}

std::pair<Molecule::bond_iterator, bool>
MoleculeMutator::add_bond(int src, int dst, BondData &&bond) noexcept {
  return add_bond_impl(mol().graph_, src, dst, std::move(bond));
}

void MoleculeMutator::mark_bond_erase(int src, int dst) {
  auto it = mol().find_bond(src, dst);
  if (it != mol().bond_end())
    erased_bonds_.push_back(it->id());
}

void MoleculeMutator::clear_bonds() noexcept {
  mol().clear_bonds();
  prev_num_bonds_ = 0;
  erased_bonds_.clear();
}

void MoleculeMutator::clear() noexcept {
  mol().clear();
  prev_num_atoms_ = prev_num_bonds_ = 0;
  discard_erasure();
}

void MoleculeMutator::discard_erasure() noexcept {
  erased_atoms_.clear();
  erased_bonds_.clear();
}

namespace {
  void remap_confs(std::vector<Matrix3Xd> &confs, const int added_size,
                   const int new_size, const bool is_trailing,
                   const std::vector<int> &old_to_new) {
    // Only trailing nodes are removed
    if (is_trailing) {
      for (Matrix3Xd &conf: confs)
        conf.conservativeResize(Eigen::NoChange, new_size);
      return;
    }

    // Select the atom indices
    std::vector<int> idxs;
    idxs.reserve(new_size);

    for (int i = 0; i < added_size; ++i) {
      // GCOV_EXCL_START
      ABSL_DCHECK(i < old_to_new.size());
      // GCOV_EXCL_STOP
      if (old_to_new[i] >= 0)
        idxs.push_back(i);
    }

    for (Matrix3Xd &conf: confs) {
      Matrix3Xd updated = conf(Eigen::all, idxs);
      conf = std::move(updated);
    }
  }

  bool prepare_remap_idxs(int prev_size, int first_erased,
                          std::vector<int> &idxs_map) {
    if (prev_size == first_erased)
      return false;

    if (first_erased < 0)
      return true;

    idxs_map.resize(prev_size);
    std::iota(idxs_map.begin(), idxs_map.begin() + first_erased, 0);
    std::fill(idxs_map.begin() + first_erased, idxs_map.end(), -1);
    return true;
  }
}  // namespace

void MoleculeMutator::finalize() noexcept {
  if (mol().num_atoms() == prev_num_atoms_
      && mol().num_bonds() == prev_num_bonds_  //
      && erased_atoms_.empty()                 //
      && erased_bonds_.empty())
    return;

  Molecule::GraphType &g = mol().graph_;
  const int added_natom = mol().num_atoms();
  const int added_nbond = mol().num_bonds();
  if (added_natom > prev_num_atoms_)
    for (Matrix3Xd &conf: mol().conformers_)
      conf.conservativeResize(Eigen::NoChange, added_natom);

  // As per the spec, the order is:
  // 1. Erase bonds
  std::pair<int, std::vector<int>> bond_info;
  bond_info = g.erase_edges(erased_bonds_.begin(), erased_bonds_.end());
  if (prepare_remap_idxs(added_nbond, bond_info.first, bond_info.second))
    for (Substructure &sub: mol().substructs_)
      sub.graph_.remap_edges(bond_info.second);

  // 2. Erase atoms
  std::pair<int, std::vector<int>> atom_info;
  std::tie(atom_info, bond_info) =
      g.erase_nodes(erased_atoms_.begin(), erased_atoms_.end());

  if (prepare_remap_idxs(added_natom, atom_info.first, atom_info.second)) {
    if (prepare_remap_idxs(added_nbond, bond_info.first, bond_info.second)) {
      for (Substructure &sub: mol().substructs_)
        sub.graph_.remap(atom_info.second, bond_info.second);
    } else {
      for (Substructure &sub: mol().substructs_)
        sub.graph_.remap_nodes(atom_info.second);
    }

    remap_confs(mol().conformers_, added_natom, mol().num_atoms(),
                atom_info.first >= 0, atom_info.second);
  }

  mol().update_topology();

  prev_num_atoms_ = mol().num_atoms();
  prev_num_bonds_ = mol().num_bonds();
  discard_erasure();
}

/* MoleculeSanitizer definitions */

namespace internal {
  int sum_bond_order(Molecule::Atom atom, bool aromatic_correct) {
    int sum_order = atom.data().implicit_hydrogens(), num_aromatic = 0,
        num_multiple_bond = 0;

    for (auto adj: atom) {
      if (adj.edge_data().order() == constants::kAromaticBond) {
        ++num_aromatic;
      } else {
        sum_order += std::max(adj.edge_data().order(), constants::kSingleBond);
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
      ABSL_LOG(INFO) << "Atom with single aromatic bond; assuming double bond "
                        "for bond order calculation";
    } else if (num_aromatic > 3) {
      // Aromatic atom with >= 4 aromatic bonds is very unlikely;
      // just log it and fall through
      ABSL_LOG(WARNING) << "Cannot correctly determine total bond order for "
                           "aromatic atom with more than 4 aromatic bonds";
    }

    // The logic here:
    //   - for 1 aromatic bond, assume it's a double bond (e.g. carboxylate C-O
    //     bond has "aromatic" bond order in some Mol2 files) = 2
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
}  // namespace internal

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
      } else {
        if (prev_bond->order() == constants::kSingleBond) {
          if (nei.edge_data().order() == constants::kSingleBond
              && src.data().atomic_number() != 0
              && dst.data().atomic_number() != 0) {
            // Single - single bond -> conjugated if curr has lone pair and
            // next doesn't, or vice versa, or any of them is dummy
            const int src_nbe =
                          nonbonding_electrons(src.data(), valences_[curr]),
                      dst_nbe =
                          nonbonding_electrons(dst.data(), valences_[dst.id()]);
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

namespace internal {
  const Element &effective_element_or_element(Molecule::Atom atom) noexcept {
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
    const int pie_estimate = std::min(nb_electrons, 2);
    return pie_estimate;
  }

  int steric_number(const int total_degree, const int nb_electrons) {
    const int lone_pairs = nb_electrons / 2;
    return total_degree + lone_pairs;
  }

  constants::Hybridization from_degree(const int total_degree,
                                       const int nb_electrons) {
    int sn = steric_number(total_degree, nb_electrons);
    return static_cast<constants::Hybridization>(
        std::min(sn, static_cast<int>(constants::kOtherHyb)));
  }
}  // namespace internal

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
                  < 2
           // No exocyclic high-order bonds
           && std::all_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
                return nei.edge_data().is_ring_bond()
                       || nei.edge_data().order() == constants::kSingleBond;
              });
  }

  bool is_ring_aromatic(const Molecule &mol, const std::vector<int> &ring,
                        const int pi_e_sum) {
    return pi_e_sum % 4 == 2
           // Dummy atoms are always allowed in aromatic rings
           || std::any_of(ring.begin(), ring.end(), [&](int id) {
                return mol.atom(id).data().atomic_number() == 0;
              });
  }

  void mark_aromatic_ring(Molecule &mol, const std::vector<int> &ring,
                          const absl::flat_hash_map<int, int> &pi_e) {
    int pi_e_sum = 0;
    for (int i = 0; i < ring.size(); ++i) {
      auto it = pi_e.find(ring[i]);
      if (it == pi_e.end()) {
        return;
      }
      pi_e_sum += it->second;
    }

    const bool this_aromatic = is_ring_aromatic(mol, ring, pi_e_sum);
    if (!this_aromatic) {
      return;
    }

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
        pi_e.insert({ atom.id(), count_pi_e(atom, valences[atom.id()]) });
      }
    }

    auto mark_aromatic_for = [&](const std::vector<std::vector<int>> &rs) {
      for (const std::vector<int> &ring: rs) {
        mark_aromatic_ring(mol, ring, pi_e);
      }
    };

    const bool need_subring = mol.num_sssr()
                              > static_cast<int>(mol.ring_groups().size());
    // Fast path: no need to find subrings
    if (!need_subring) {
      mark_aromatic_for(mol.ring_groups());
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
    return nbe > 0 && internal::common_valence(effective) < total_valence
           && std::any_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
                return nei.edge_data().order() == constants::kAromaticBond;
              });
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
          std::min(hyb, from_degree(total_degree, 0)));
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

int count_heavy(Molecule::Atom atom) {
  return std::count_if(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
    return nei.dst().data().atomic_number() != 1;
  });
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

const Element *effective_element(Molecule::Atom atom) {
  const int effective_z =
      atom.data().atomic_number() - atom.data().formal_charge();
  return PeriodicTable::get().find_element(effective_z);
}

std::vector<std::vector<int>> fragments(const Molecule &mol) {
  std::vector<std::vector<int>> result;
  ArrayXb visited = ArrayXb::Constant(mol.num_atoms(), false);

  auto dfs = [&](auto &self, std::vector<int> &sub, int curr) -> void {
    sub.push_back(curr);
    visited[curr] = true;

    for (auto nei: mol.atom(curr)) {
      if (visited[nei.dst().id()])
        continue;

      self(self, sub, nei.dst().id());
    }
  };

  for (auto atom: mol) {
    if (visited[atom.id()])
      continue;

    dfs(dfs, result.emplace_back(), atom.id());
  }

  return result;
}
}  // namespace nuri
