//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

#include <algorithm>

#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"

namespace nuri {
namespace internal {
  AtomData::AtomData(const Element &element, constants::Hybridization hyb,
                     int formal_charge, double partial_charge, int mass_number,
                     bool is_aromatic, bool is_in_ring, bool is_chiral,
                     bool is_right_handed)
    : element_(&element), isotope_(nullptr), hyb_(hyb), flags_(0),
      formal_charge_(formal_charge), partial_charge_(partial_charge) {
    if (mass_number >= 0) {
      isotope_ = element.find_isotope(mass_number);
      ABSL_LOG_IF(WARNING, ABSL_PREDICT_FALSE(isotope_ == nullptr))
        << "Invalid mass number " << mass_number << " for element "
        << element.symbol();
    }

    auto set_flag = [this](bool cond, Flags flag) {
      flags_ |= -static_cast<uint32_t>(cond) & flag;
    };
    set_flag(is_aromatic, kAromaticAtom);
    set_flag(is_in_ring, kRingAtom);
    set_flag(is_chiral, kChiralAtom);
    set_flag(is_right_handed, kRightHandedAtom);
  }
}  // namespace internal

/* Molecule definitions */

namespace {
  Matrix3d generate_rotation(const MatrixX3d &coords, int ref, int pivot,
                             double angle) {
    AngleAxisd aa(deg2rad(angle),
                  (coords.row(pivot) - coords.row(ref)).normalized());
    return aa.to_matrix();
  }

  template <class MatrixLike>
  void rotate_helper(const Matrix3d &rot, MatrixLike &&moving,
                     const Vector3d &pivot) {
    for (int i = 0; i < moving.rows(); ++i) {
      moving.row(i) =
        (rot * (moving.row(i) - pivot).transpose()).transpose() + pivot;
    }
  }

  void rotate_points(MatrixX3d &coords, const std::vector<int> &moving_idxs,
                     int ref, int pivot, double angle) {
    Matrix3d rot = generate_rotation(coords, ref, pivot, angle);
    rotate_helper(rot, coords(moving_idxs, Eigen::all), coords.row(pivot));
  }
}  // namespace

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
  Bond b = bond(bid);
  return rotate_bond_common(i, b, b.src(), b.dst(), angle);
}

bool Molecule::rotate_bond_common(int i, Bond b, int ref_atom, int pivot_atom,
                                  double angle) {
  if (!b.data().is_rotable()) {
    return false;
  }

  absl::flat_hash_set<int> connected =
    connected_components(graph_, pivot_atom, ref_atom);
  if (ABSL_PREDICT_FALSE(connected.empty())) {
    ABSL_DLOG(WARNING) << ref_atom << " -> " << pivot_atom
                       << " bond is rotable, but the two atoms are connected.";
    return false;
  }

  std::vector<int> moving_atoms(connected.begin(), connected.end());
  // For faster memory access in the rotate_points function
  std::sort(moving_atoms.begin(), moving_atoms.end());

  if (i < 0) {
    for (int conf = 0; conf < npos(); ++conf) {
      rotate_points(conformers_[conf], moving_atoms, ref_atom, pivot_atom,
                    angle);
    }
  } else {
    rotate_points(conformers_[i], moving_atoms, ref_atom, pivot_atom, angle);
  }

  return true;
}

/* MoleculeMutator definitions */

bool MoleculeMutator::add_bond(int src, int dst,
                               const internal::BondData &bond) {
  if (ABSL_PREDICT_FALSE(src == dst)) {
    return false;
  }

  std::pair<int, int> ends = std::minmax(src, dst);
  auto [it, inserted] = new_bonds_set_.insert(ends);
  if (ABSL_PREDICT_FALSE(!inserted)) {
    return false;
  }
  if (ABSL_PREDICT_FALSE(mol().graph_.find_edge(src, dst)
                         != mol().graph_.edge_end())) {
    new_bonds_set_.erase(it);
    return false;
  }

  new_bonds_.push_back({ ends, bond });
  return true;
}

void MoleculeMutator::remove_bond(int src, int dst) {
  if (ABSL_PREDICT_FALSE(src == dst)) {
    return;
  }

  removed_bonds_.push_back({ src, dst });
}

int MoleculeMutator::num_atoms() const {
  return next_atom_idx() - static_cast<int>(removed_atoms_.size());
}

void MoleculeMutator::discard() noexcept {
  new_atoms_.clear();
  removed_atoms_.clear();

  new_bonds_.clear();
  new_bonds_set_.clear();
  removed_bonds_.clear();
}

void MoleculeMutator::accept() noexcept {
  Molecule::GraphType &g = mol().graph_;

  // As per the spec, the order is:

  // 1. Add atoms
  g.add_node(new_atoms_.begin(), new_atoms_.end());

  // 2. Add bonds
  for (const AddedBond &b: new_bonds_) {
    g.add_edge(b.ends.first, b.ends.second, b.data);
  }

  // 3. Remove bonds
  for (const std::pair<int, int> &ends: removed_bonds_) {
    g.erase_edge_between(ends.first, ends.second);
  }

  // 4. Remove atoms
  auto [last, map] =
    g.erase_nodes(removed_atoms_.begin(), removed_atoms_.end());

  // Update the data
  if (last >= 0) {
    // Only trailing nodes are removed
    for (MatrixX3d &conf: mol().conformers_) {
      conf.conservativeResize(mol().num_atoms(), Eigen::NoChange);
    }
  } else {
    // Select the atom indices
    ArrayXi idxs(mol().num_atoms());
    auto it = idxs.begin();
    for (int i = 0; it != idxs.end(); ++i) {
      ABSL_DCHECK(i < map.size());
      if (map[i] >= 0) {
        *it++ = i;
      }
    }

    for (MatrixX3d &conf: mol().conformers_) {
      conf = conf(idxs, Eigen::all);
    }
  }

  discard();
}

int MoleculeMutator::next_atom_idx() const {
  return mol().num_atoms() + static_cast<int>(new_atoms_.size());
}
}  // namespace nuri
