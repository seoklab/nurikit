//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

#include <algorithm>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"
#include "nuri/core/graph.h"
#include "nuri/utils.h"

namespace nuri {
AtomData::AtomData(const Element &element, int implicit_hydrogens,
                   int formal_charge, constants::Hybridization hyb,
                   double partial_charge, int mass_number, bool is_aromatic,
                   bool is_in_ring, bool is_chiral, bool is_clockwise)
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
  internal::set_flag_if(flags_, is_clockwise, AtomFlags::kClockWise);
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
    if (atom.data().atomic_number() != 1 || atom.degree() != 1
        || atom.data().implicit_hydrogens() != 0
        || atom[0].edge_data().order() != constants::kSingleBond
        || atom[0].dst().data().atomic_number() == 1)
      continue;

    m.mark_atom_erase(atom.id());

    auto heavy = atom[0].dst();
    heavy.data().set_implicit_hydrogens(heavy.data().implicit_hydrogens() + 1);

    if (heavy.data().is_chiral()) {
      // hydrogen already marked to be erased and reflected in the
      // implicit_hydrogens count but not in the underlying graph, so must check
      // for degree 4 (3 heavy neighbors + 1 to-be-erased hydrogen) and
      // implicit_hydrogens() of 1
      //
      // When visited more than twice for the same heavy atom (atom has more
      // than 1 explicit hydrogens), the atom is marked chiral but is not
      // chiral in reality.
      //! XXX: Maybe need to set the chiral flag to false here?
      if (heavy.degree() < 4 || heavy.data().implicit_hydrogens() > 1) {
        ABSL_LOG(INFO) << "Atom " << heavy.id()
                       << " has less than 3 heavy neighbors or more than 1 "
                          "implicit hydrogens, but is marked chiral";
        continue;
      }

      // hydrogen neighbor index, from the heavy atom's perspective
      int hni = heavy.find_adjacent(atom) - heavy.begin();
      bool order_consistent = hni == 1 || hni == 3;
      heavy.data().set_clockwise(heavy.data().is_clockwise()
                                 == order_consistent);
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
