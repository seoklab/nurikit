//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"

namespace nuri {
void MoleculeMutator::clear_atoms() noexcept {
  mol().clear_atoms();
  prev_num_atoms_ = prev_num_bonds_ = 0;
  discard_erasure();
}

namespace {
  template <class DT>
  std::pair<int, bool> add_bond_impl(Molecule::GraphType &graph, int src,
                                     int dst, DT &&bond) {
    auto it = graph.find_edge(src, dst);
    if (it != graph.edge_end())
      return std::make_pair(it->id(), false);

    int eid = graph.add_edge(src, dst, std::forward<DT>(bond));
    return std::make_pair(eid, true);
  }
}  // namespace

std::pair<int, bool> MoleculeMutator::add_bond(int src, int dst,
                                               const BondData &bond) {
  return add_bond_impl(mol().graph_, src, dst, bond);
}

std::pair<int, bool> MoleculeMutator::add_bond(int src, int dst,
                                               BondData &&bond) noexcept {
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
}  // namespace nuri
