//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include "nuri/core/molecule.h"

namespace nuri {
namespace {
  int ring_degree(Molecule::Atom atom) {
    return std::accumulate(
      atom.begin(), atom.end(), 0, [](int sum, Molecule::Neighbor nei) {
        return sum + static_cast<int>(nei.edge_data().is_ring_bond());
      });
  }

  std::pair<std::vector<int>, int> ring_degrees(const Molecule &mol) {
    std::pair ret { std::vector<int>(), 0 };
    ret.first.reserve(mol.num_atoms());

    for (auto atom: mol) {
      int degree = ring_degree(atom);
      ret.first[atom.id()] = degree;
      ret.second += static_cast<int>(degree > 0);
    }

    return ret;
  }

  std::vector<int> sort_atoms_by_ring_degree(const Molecule &mol) {
    std::pair dn = ring_degrees(mol);
    std::vector<int> &degrees = dn.first;
    const int num_ring_atoms = dn.second;

    std::vector<int> sorted_atoms;
    sorted_atoms.reserve(num_ring_atoms);
    for (auto atom: mol) {
      if (degrees[atom.id()] > 0) {
        sorted_atoms.push_back(atom.id());
      }
    }

    std::sort(sorted_atoms.begin(), sorted_atoms.end(),
              [&degrees](int i, int j) { return degrees[i] < degrees[j]; });
    return sorted_atoms;
  }

  using PathGraph = absl::flat_hash_map<int, std::vector<std::vector<int>>>;

  PathGraph pathgraph_init(const Molecule &mol,
                           const std::vector<int> &sorted_atoms) {
    PathGraph paths(mol.size());
    for (int i: sorted_atoms) {
      std::vector<std::vector<int>> &path = paths[i];
      for (auto nei: mol.atom(i)) {
        if (nei.edge_data().is_ring_bond() && !paths.contains(nei.dst().id())) {
          path.push_back({ i, nei.dst().id() });
        }
      }
    }
    return paths;
  }

  void find_rings_remove_atom(const int id,
                              std::vector<std::vector<int>> &paths,
                              std::vector<std::vector<int>> &rings,
                              PathGraph &pg,
                              const absl::flat_hash_map<int, int> &atom_order) {
    const int size = static_cast<int>(paths.size());

    for (int i = 0; i < size - 1; ++i) {
      if (paths[i].back() == id) {
        continue;
      }

      for (int j = i + 1; j < size; ++j) {
        if (paths[j].back() == id) {
          continue;
        }

        // path: y-...-x, z-...-x
        const std::vector<int> *x_y = &paths[i], *x_z = &paths[j];
        if (x_y->size() > x_z->size()) {
          std::swap(x_y, x_z);
        }

        // Check if path y-...-x and z-...-x overlap except endpoints
        const absl::flat_hash_set<int> path_set(++x_y->begin(), --x_y->end());
        const bool no_overlap =
          std::none_of(++x_z->begin(), --x_z->end(),
                       [&path_set](int bid) { return path_set.contains(bid); });
        if (!no_overlap) {
          continue;
        }

        int src = x_y->back(), dst = x_z->back();
        auto sit = atom_order.find(src), dit = atom_order.find(dst);
        ABSL_DCHECK(sit != atom_order.end() && dit != atom_order.end());
        if (sit->second > dit->second) {
          std::swap(src, dst);
          std::swap(x_y, x_z);
        }

        // Add src -> x -> dst path to atom src
        auto it = pg.find(src);
        ABSL_DCHECK(it != pg.end());
        std::vector<int> &new_path =
          it->second.emplace_back(x_y->rbegin(), x_y->rend());
        new_path.insert(new_path.end(), ++x_z->begin(), x_z->end());
      }
    }

    for (std::vector<int> &path: paths) {
      if (path.back() == id) {
        path.erase(--path.end());
        rings.push_back(std::move(path));
      }
    }
  }

  constexpr int kMaxRingMembership = 100;
}  // namespace

std::pair<std::vector<std::vector<int>>, bool>
find_all_elementary_rings(const Molecule &mol) {
  std::vector<int> sorted_ring_atoms = sort_atoms_by_ring_degree(mol);
  PathGraph pg = pathgraph_init(mol, sorted_ring_atoms);

  absl::flat_hash_map<int, int> atom_order(sorted_ring_atoms.size());
  for (int i = 0; i < sorted_ring_atoms.size(); ++i) {
    atom_order[sorted_ring_atoms[i]] = i;
  }

  std::pair<std::vector<std::vector<int>>, bool> result;
  result.second = true;

  for (int id: sorted_ring_atoms) {
    auto it = pg.find(id);
    ABSL_DCHECK(it != pg.end());

    if (ABSL_PREDICT_FALSE(it->second.size() > kMaxRingMembership)) {
      ABSL_LOG(INFO)
        << "Stopped finding rings because an atom belongs to more than "
        << kMaxRingMembership << " rings";
      result.second = false;
      break;
    }

    find_rings_remove_atom(id, it->second, result.first, pg, atom_order);
    pg.erase(it);
  }

  return result;
}
}  // namespace nuri
