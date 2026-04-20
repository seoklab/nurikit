//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"

namespace nuri {
void MoleculeMutator::clear_atoms() noexcept {
  mol().clear_atoms();
  discard();
}

namespace {
  int find_edge_both(
      const Molecule::GraphType &graph,
      absl::flat_hash_map<int, std::vector<std::pair<int, int>>> &delta_adj,
      int src, int dst) {
    if (auto it = graph.find_edge(src, dst); it != graph.edge_end())
      return it->id();

    auto it = delta_adj.find(src);
    if (it == delta_adj.end())
      return -1;

    for (const auto &[d, e]: it->second) {
      if (d == dst)
        return e;
    }

    return -1;
  }

  template <class DT>
  std::pair<int, bool> register_bond_impl(
      const Molecule::GraphType &graph,
      std::vector<Molecule::GraphType::StoredEdge> &new_bonds,
      absl::flat_hash_map<int, std::vector<std::pair<int, int>>> &delta_adj,
      int src, int dst, DT &&bond) {
    if (int e = find_edge_both(graph, delta_adj, src, dst); e >= 0)
      return std::make_pair(e, false);

    int eid = graph.num_edges() + static_cast<int>(new_bonds.size());
    new_bonds.push_back({ src, dst, std::forward<DT>(bond) });
    delta_adj[src].emplace_back(dst, eid);
    delta_adj[dst].emplace_back(src, eid);
    return std::make_pair(eid, true);
  }
}  // namespace

std::pair<int, bool> MoleculeMutator::register_bond(int src, int dst,
                                                    const BondData &bond) {
  return register_bond_impl(mol().graph_, bond_registry_, delta_adj_, src, dst,
                            bond);
}

std::pair<int, bool> MoleculeMutator::register_bond(int src, int dst,
                                                    BondData &&bond) noexcept {
  return register_bond_impl(mol().graph_, bond_registry_, delta_adj_, src, dst,
                            std::move(bond));
}

void MoleculeMutator::mark_bond_erase(int src, int dst) {
  int e = find_edge_both(mol().graph_, delta_adj_, src, dst);
  if (e >= 0)
    erased_bonds_.push_back(e);
}

void MoleculeMutator::clear_bonds() noexcept {
  mol().clear_bonds();
  discard_bonds();
}

void MoleculeMutator::clear() noexcept {
  mol().clear();
  discard();
}

void MoleculeMutator::discard_bonds() noexcept {
  bond_registry_.clear();
  delta_adj_.clear();
  erased_bonds_.clear();
}

void MoleculeMutator::discard() noexcept {
  prev_num_atoms_ = mol().num_atoms();
  erased_atoms_.clear();
  discard_bonds();
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
  if (mol().num_atoms() == prev_num_atoms_  //
      && bond_registry_.empty()             //
      && erased_atoms_.empty()              //
      && erased_bonds_.empty())
    return;

  Molecule::GraphType &g = mol().graph_;
  const int added_natom = mol().num_atoms();
  if (added_natom > prev_num_atoms_)
    for (Matrix3Xd &conf: mol().conformers_)
      conf.conservativeResize(Eigen::NoChange, added_natom);

  // As per the spec, the order is:
  // 1. Add bonds
  g.add_edges(std::make_move_iterator(bond_registry_.begin()),
              std::make_move_iterator(bond_registry_.end()));

  // 2. Erase bonds
  const int added_nbond = mol().num_bonds();
  std::pair<int, std::vector<int>> bond_info;
  bond_info = g.erase_edges(erased_bonds_.begin(), erased_bonds_.end());
  if (prepare_remap_idxs(added_nbond, bond_info.first, bond_info.second))
    for (Substructure &sub: mol().substructs_)
      sub.graph_.remap_edges(bond_info.second);

  // 3. Erase atoms
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
  discard();
}
}  // namespace nuri
