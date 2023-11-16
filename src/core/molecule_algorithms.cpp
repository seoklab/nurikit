//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include "nuri/core/bool_matrix.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  template <class It, class T>
  bool range_no_overlap(It v_begin, It v_end, It w_begin, It w_end,
                        absl::flat_hash_set<T> &set) {
    auto range_no_overlap_impl = [&set](auto vb, auto ve, auto wb, auto we) {
      set.clear();
      set.insert(vb, ve);
      return std::none_of(wb, we,
                          [&set](const T &x) { return set.contains(x); });
    };

    return v_end - v_begin > w_end - w_begin
               ? range_no_overlap_impl(w_begin, w_end, v_begin, v_end)
               : range_no_overlap_impl(v_begin, v_end, w_begin, w_end);
  }

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
      const int degree = ring_degree(atom);
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
    for (const int i: sorted_atoms) {
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
    absl::flat_hash_set<int> path_set;

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
        if (!range_no_overlap(++x_y->begin(), --x_y->end(), ++x_z->begin(),
                              --x_z->end(), path_set)) {
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
find_all_rings(const Molecule &mol) {
  std::vector<int> sorted_ring_atoms = sort_atoms_by_ring_degree(mol);
  PathGraph pg = pathgraph_init(mol, sorted_ring_atoms);

  absl::flat_hash_map<int, int> atom_order(sorted_ring_atoms.size());
  for (int i = 0; i < sorted_ring_atoms.size(); ++i) {
    atom_order[sorted_ring_atoms[i]] = i;
  }

  std::pair<std::vector<std::vector<int>>, bool> result;
  result.second = true;

  for (const int id: sorted_ring_atoms) {
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

namespace {
  struct Cycle {
    const std::vector<Molecule::Neighbor> *path_rpy;
    const std::vector<Molecule::Neighbor> *path_rqz;
    // nullptr if cycle is odd
    const std::vector<Molecule::Neighbor> *path_ry;
    size_t cycle_length;
  };

  using Dr =
      absl::flat_hash_map<int, std::unique_ptr<std::vector<Molecule::Neighbor>>>;
}  // namespace

namespace internal {
  struct FindRingsCommonData {
    absl::flat_hash_map<int, std::unique_ptr<Dr>> d_rs;
    std::vector<Cycle> c_ip;
  };
}  // namespace internal

RingSetsFinder::RingSetsFinder(RingSetsFinder &&) noexcept = default;
RingSetsFinder &RingSetsFinder::operator=(RingSetsFinder &&) noexcept = default;
RingSetsFinder::~RingSetsFinder() noexcept = default;

namespace {
  template <class T, class C = std::less<>, class S = std::vector<T>>
  struct ClearablePQ: public std::priority_queue<T, S, C> {
  public:
    template <
        class U = T,  //
        std::enable_if_t<
            std::is_same_v<T, U> && std::is_trivially_copyable_v<T>, int> = 0>
    T pop_get() noexcept {
      T v = this->top();
      this->pop();
      return v;
    }

    void clear() noexcept { this->c.clear(); }
  };

  struct IdDist {
    int id, distance;

    // NOLINTNEXTLINE(clang-diagnostic-unused-function)
    friend bool operator>(IdDist lhs, IdDist rhs) {
      return lhs.distance > rhs.distance;
    }
  };

  int extract_sid(Molecule::Neighbor nei) {
    return nei.src().id();
  }

  int extract_did(Molecule::Neighbor nei) {
    return nei.dst().id();
  }

  template <class K, class V>
  V &insert_new(absl::flat_hash_map<K, std::unique_ptr<V>> &map, K key) {
    auto [it, inserted] = map.insert({ key, std::make_unique<V>() });
    ABSL_DCHECK(inserted);
    return *it->second;
  }

  template <class RandomIt>
  using GenericDestIterator =
      internal::TransformIterator<RandomIt, extract_did, int, int>;

  using DestIterator =
      GenericDestIterator<std::vector<Molecule::Neighbor>::const_iterator>;

  void compute_v_r(const Molecule &mol, const int r, std::vector<int> &v_r,
                   Dr &d_r, absl::FixedArray<int> &distances,
                   absl::flat_hash_map<int, Molecule::Neighbor> &backtrace,
                   ClearablePQ<IdDist, std::greater<>> &minheap) {
    v_r.clear();
    std::fill(distances.begin(), distances.end(), mol.num_atoms());
    minheap.clear();
    ABSL_DCHECK(backtrace.empty());

    minheap.push({ r, 0 });
    distances[r] = 0;

    do {
      auto [curr, dist] = minheap.pop_get();
      if (dist != distances[curr]) {
        continue;
      }

      ++dist;

      for (auto nei: mol.atom(curr)) {
        const int dst = nei.dst().id();
        if (nei.edge_data().is_ring_bond() && dst < r
            && distances[dst] > dist) {
          minheap.push({ dst, dist });
          distances[dst] = dist;
          auto [_, inserted] = backtrace.insert({ dst, nei });
          ABSL_DCHECK(inserted);
        }
      }
    } while (!minheap.empty());

    auto find_subpaths = [&](auto &self, const int curr,
                             auto it) -> std::vector<Molecule::Neighbor> & {
      if (it == backtrace.end()) {
        auto dit = d_r.find(curr);
        ABSL_DCHECK(dit != d_r.end());
        return *dit->second;
      }

      const int prev = it->second.src().id();

      v_r.push_back(curr);
      std::vector<Molecule::Neighbor> &path = insert_new(d_r, curr);
      path.reserve(distances[curr]);

      if (prev != r) {
        const std::vector<Molecule::Neighbor> &sub =
            self(self, prev, backtrace.find(prev));
        path.insert(path.end(), sub.begin(), sub.end());
      }
      path.push_back(it->second);

      backtrace.erase(it);
      return path;
    };

    while (!backtrace.empty()) {
      auto it = backtrace.begin();
      find_subpaths(find_subpaths, it->first, it);
    }
  }

  void merge_cycles(std::vector<Cycle> &c_ip,
                    const std::vector<Molecule::Neighbor> &path_rpy,
                    const std::vector<Molecule::Neighbor> &path_rqz,
                    // nullptr if odd
                    const std::vector<Molecule::Neighbor> *path_ry,
                    absl::flat_hash_set<int> &path_set_tmp) {
    if (!range_no_overlap(DestIterator(path_rpy.begin()),
                          DestIterator(--path_rpy.end()),
                          DestIterator(path_rqz.begin()),
                          DestIterator(--path_rqz.end()), path_set_tmp)) {
      return;
    }

    const size_t cycle_size = path_rpy.size() + path_rqz.size()
                              + static_cast<size_t>(path_ry != nullptr) + 1;
    c_ip.push_back({ &path_rpy, &path_rqz, path_ry, cycle_size });
  }

  void compute_c_ip(
      const Molecule &mol, std::vector<Cycle> &c_ip, const Dr &d_r,
      const std::vector<int> &v_r, const absl::FixedArray<int> &distances,
      absl::flat_hash_set<int> &path_set_tmp,
      std::vector<const std::vector<Molecule::Neighbor> *> &r_y_shortest) {
    for (const int y: v_r) {
      r_y_shortest.clear();

      auto yit = d_r.find(y);
      ABSL_DCHECK(yit != d_r.end());
      const std::vector<Molecule::Neighbor> &path_ry = *yit->second;

      for (auto nei: mol.atom(y)) {
        const int z = nei.dst().id();
        if (!nei.edge_data().is_ring_bond()) {
          continue;
        }

        auto zit = d_r.find(z);
        if (zit == d_r.end()) {
          continue;
        }
        const std::vector<Molecule::Neighbor> &path_rz = *zit->second;

        if (distances[z] + 1 == distances[y]) {
          r_y_shortest.push_back(&path_rz);
        } else if (distances[z] != distances[y] + 1 && z < y) {
          // Cycle y -> r -> z -> y
          merge_cycles(c_ip, path_ry, path_rz, nullptr, path_set_tmp);
        }
      }

      for (int i = 0; i < static_cast<int>(r_y_shortest.size()) - 1; ++i) {
        const std::vector<Molecule::Neighbor> &path_rp = *r_y_shortest[i];
        for (int j = i + 1; j < r_y_shortest.size(); ++j) {
          const std::vector<Molecule::Neighbor> &path_rq = *r_y_shortest[j];
          // Path p -> r -> q -> y -> p
          merge_cycles(c_ip, path_rp, path_rq, &path_ry, path_set_tmp);
        }
      }
    }
  }

  std::unique_ptr<internal::FindRingsCommonData>
  prepare_find_ring_sets(const Molecule &mol, const int num_ring_atoms) {
    std::unique_ptr<internal::FindRingsCommonData> data =
        std::make_unique<internal::FindRingsCommonData>();
    auto &d_rs = data->d_rs;
    auto &c_ip = data->c_ip;

    // Cache variables, allocate here to avoid reallocation
    std::vector<int> v_r;
    absl::FixedArray<int> distances(mol.num_atoms());
    absl::flat_hash_map<int, Molecule::Neighbor> backtrace(num_ring_atoms);
    ClearablePQ<IdDist, std::greater<>> minheap;

    absl::flat_hash_set<int> path_set_tmp;
    std::vector<const std::vector<Molecule::Neighbor> *> r_y_shortest;

    for (int r = 0; r < mol.num_atoms(); ++r) {
      Molecule::Atom atom = mol.atom(r);
      if (!atom.data().is_ring_atom()) {
        continue;
      }

      Dr &d_r = insert_new(d_rs, r);
      compute_v_r(mol, r, v_r, d_r, distances, backtrace, minheap);
      compute_c_ip(mol, c_ip, d_r, v_r, distances, path_set_tmp, r_y_shortest);
    }

    std::sort(c_ip.begin(), c_ip.end(), [](const Cycle &a, const Cycle &b) {
      return a.cycle_length < b.cycle_length;
    });

    return data;
  }

  using BondPtr = std::iterator_traits<Molecule::bond_id_type>::pointer;

  void assign_eids(BoolMatrix &m, std::vector<int> &used_edges,
                   const int cycle_idx, const Cycle &cycle,
                   const absl::flat_hash_map<BondPtr, int> &edge_map) {
    const auto mark_edge = [&](Molecule::Neighbor nei) {
      auto it = edge_map.find(&*nei.eid());
      ABSL_DCHECK(it != edge_map.end());
      m.set(cycle_idx, it->second);
      used_edges[it->second] = 1;
    };

    for (Molecule::Neighbor nei: *cycle.path_rpy) {
      mark_edge(nei);
    }
    for (Molecule::Neighbor nei: *cycle.path_rqz) {
      mark_edge(nei);
    }

    if (cycle.path_ry != nullptr) {
      // Even cycle, add edges from r ... p, r ... q, and y - q, y - p edges
      const int p = extract_did(cycle.path_rpy->back()),
                q = extract_did(cycle.path_rqz->back());
      for (Molecule::Neighbor nei: cycle.path_ry->back().dst()) {
        if (nei.dst().id() == p || nei.dst().id() == q) {
          mark_edge(nei);
        }
      }
    } else {
      // Odd cycle, add edges from y ... r, z ... r, and y - z edge
      const int z = extract_did(cycle.path_rqz->back());
      for (Molecule::Neighbor nei: cycle.path_rpy->back().dst()) {
        if (nei.dst().id() == z) {
          mark_edge(nei);
          break;
        }
      }
    }
  }

  template <bool minimal>
  std::vector<int>
  verify_basis(BoolMatrix &m, const int m_idx, std::vector<int> &used_edges,
               const std::vector<Cycle> &c_ip, const int begin, const int end,
               const absl::flat_hash_map<BondPtr, int> &edge_map) {
    for (int i = begin, j = m_idx; i < end; ++i, ++j) {
      assign_eids(m, used_edges, j, c_ip[i], edge_map);
    }

    if constexpr (minimal) {
      // Full reduction to find minimal basis
      return m.gaussian_elimination();
    }

    // Use basis in [0, m_idx) to check [m_idx, m.size())
    return m.partial_reduction(m_idx);
  }

  template <bool minimal>
  std::vector<int> compute_prototype(const Molecule &mol,
                                     const std::vector<Cycle> &c_ip) {
    absl::flat_hash_map<BondPtr, int> edge_map;
    int idx = 0;
    for (auto bond: mol.bonds()) {
      if (bond.data().is_ring_bond()) {
        edge_map[&*bond.id()] = idx++;
      }
    }

    BoolMatrix m(static_cast<BoolMatrix::Index>(edge_map.size()));
    std::vector<int> basis, used_edges(edge_map.size(), 0);
    int size, begin, end = 0, m_idx;

    do {
      begin = end;
      size = static_cast<int>(c_ip[begin].cycle_length);
      end = static_cast<int>(
          std::find_if(c_ip.begin() + begin + 1, c_ip.end(),
                       [&](const Cycle &c) {
                         return static_cast<int>(c.cycle_length) > size;
                       })
          - c_ip.begin());

      // Remove eliminated rows
      for (int original = 0, updated = 0; original < basis.size(); ++original) {
        if (basis[original] != 0 && original != updated++) {
          m.move_row(original, updated);
        }
      }

      m_idx = static_cast<int>(basis.size());
      m.resize(m_idx + end - begin);
      basis = verify_basis<minimal>(m, m_idx, used_edges, c_ip, begin, end,
                                    edge_map);
    } while (std::any_of(used_edges.begin(), used_edges.end(),
                         [](int i) { return i == 0; }));

    return basis;
  }

  using RDestIterator = GenericDestIterator<
      std::vector<Molecule::Neighbor>::const_reverse_iterator>;

  void add_odd_cycle(std::vector<std::vector<int>> &cycles,
                     const std::vector<Molecule::Neighbor> &path_ry,
                     const std::vector<Molecule::Neighbor> &path_rz) {
    std::vector<int> &cycle = cycles.emplace_back();
    cycle.reserve(path_ry.size() + path_rz.size());

    // Add (no r) ... y
    cycle.insert(cycle.end(), DestIterator(path_ry.begin()),
                 DestIterator(path_ry.end()));
    // Add z ... (no r)
    cycle.insert(cycle.end(), RDestIterator(path_rz.rbegin()),
                 RDestIterator(path_rz.rend()));
    // Add r
    cycle.push_back(path_ry.front().src().id());
  }

  void add_even_cycle(std::vector<std::vector<int>> &cycles,
                      const std::vector<Molecule::Neighbor> &path_rp,
                      const std::vector<Molecule::Neighbor> &path_rq,
                      const int y) {
    std::vector<int> &cycle = cycles.emplace_back();
    cycle.reserve(path_rp.size() + path_rq.size() + 1);

    // Add (no r) ... p
    cycle.insert(cycle.end(), DestIterator(path_rp.begin()),
                 DestIterator(path_rp.end()));
    // Add y
    cycle.push_back(y);
    // Add q ... (no r)
    cycle.insert(cycle.end(), RDestIterator(path_rq.rbegin()),
                 RDestIterator(path_rq.rend()));
    // Add r
    cycle.push_back(path_rp.front().src().id());
  }

  void extract_member(std::vector<std::vector<int>> &cycles, const Cycle &c) {
    if (c.path_ry != nullptr) {
      // Even cycle, p ... r, r ... q, q -> y -> p
      add_even_cycle(cycles, *c.path_rpy, *c.path_rqz,
                     c.path_ry->back().dst().id());
    } else {
      // Odd cycle, y ... r, r ... z (z -> y omitted)
      add_odd_cycle(cycles, *c.path_rpy, *c.path_rqz);
    }
  }

  std::vector<std::vector<int>>
  extract_family_traverse(const Dr &d_r, const int src, const int dst) {
    // Path of the form src (r) -> ... -> dst
    const std::vector<Molecule::Neighbor> &path = *d_r.find(dst)->second;

    // Paths of the form (src ->) ... -> dst
    std::vector<std::vector<int>> result;

    // Reverse traversal (dst -> src)
    auto extract_paths = [&](auto &self, Molecule::Atom curr, const int prev,
                             int length) -> void {
      ABSL_DCHECK(length > 0);
      --length;

      for (auto nei: curr) {
        auto next = nei.dst();
        if (next.id() == prev) {
          continue;
        }

        if (next.id() == src) {
          ABSL_DCHECK(length == 0);

          auto &back = result.emplace_back();
          back.reserve(path.size());
          break;
        }

        auto it = d_r.find(next.id());
        if (it == d_r.end() || static_cast<int>(it->second->size()) != length) {
          continue;
        }

        self(self, next, curr.id(), length);
      }

      for (auto &p: result) {
        p.push_back(curr.id());
      }
    };

    extract_paths(extract_paths, path.back().dst(), -1,
                  static_cast<int>(path.size()));

    return result;
  }

  void extract_family(std::vector<std::vector<int>> &cycles,
                      const absl::flat_hash_map<int, std::unique_ptr<Dr>> &d_rs,
                      const int r, const Cycle &c) {
    const Dr &d_r = *d_rs.find(r)->second;

    const int py = extract_did(c.path_rpy->back()),
              qz = extract_did(c.path_rqz->back());

    std::vector paths_rpy = extract_family_traverse(d_r, r, py),
                paths_rqz = extract_family_traverse(d_r, r, qz);

    if (c.path_ry != nullptr) {
      // Even cycle
      const int y = extract_did(c.path_ry->back());

      for (auto &rp: paths_rpy) {
        for (auto &rq: paths_rqz) {
          std::vector<int> &cycle = cycles.emplace_back();
          cycle.reserve(rp.size() + rq.size() + 2);

          cycle.push_back(r);
          cycle.insert(cycle.end(), rp.begin(), rp.end());
          cycle.push_back(y);
          cycle.insert(cycle.end(), rq.rbegin(), rq.rend());
        }
      }
    } else {
      // Odd cycle
      for (auto &ry: paths_rpy) {
        for (auto &rz: paths_rqz) {
          std::vector<int> &cycle = cycles.emplace_back();
          cycle.reserve(ry.size() + rz.size() + 1);

          cycle.push_back(r);
          cycle.insert(cycle.end(), ry.begin(), ry.end());
          cycle.insert(cycle.end(), rz.rbegin(), rz.rend());
        }
      }
    }
  }

  template <bool minimal>
  std::vector<std::vector<int>>
  extract_cycles(const absl::flat_hash_map<int, std::unique_ptr<Dr>> &d_rs,
                 const std::vector<Cycle> &c_ip,
                 const std::vector<int> &prototype) {
    std::vector<std::vector<int>> cycles;

    for (int i = 0; i < prototype.size(); ++i) {
      if (prototype[i] == 0) {
        continue;
      }

      if constexpr (minimal) {
        extract_member(cycles, c_ip[i]);
      } else {
        extract_family(cycles, d_rs, extract_sid(c_ip[i].path_rpy->front()),
                       c_ip[i]);
      }
    }

    return cycles;
  }
}  // namespace

RingSetsFinder::RingSetsFinder(const Molecule &mol): mol_(&mol) {
  if (mol.num_sssr() == 0) {
    // Don't bother if there are no rings
    return;
  }

  const int num_ring_atoms = std::accumulate(
      mol.begin(), mol.end(), 0, [](int sum, Molecule::Atom atom) {
        return sum + static_cast<int>(atom.data().is_ring_atom());
      });

  // d_rs: digraph version of molecule (the edge direction is opposite to
  // the version of the paper)
  // c_ip: A list of unique cycles
  data_ = prepare_find_ring_sets(mol, num_ring_atoms);
}

Rings RingSetsFinder::find_relevant_rings() const {
  if (!data_) {
    return {};
  }

  std::vector<int> prototype = compute_prototype<false>(*mol_, data_->c_ip);
  return extract_cycles<false>(data_->d_rs, data_->c_ip, prototype);
}

Rings RingSetsFinder::find_sssr() const {
  if (!data_) {
    return {};
  }

  std::vector<int> prototype = compute_prototype<true>(*mol_, data_->c_ip);
  return extract_cycles<true>(data_->d_rs, data_->c_ip, prototype);
}
}  // namespace nuri
