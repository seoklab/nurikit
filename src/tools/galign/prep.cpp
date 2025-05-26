//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/container/linear_queue.h"
#include "nuri/core/graph/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/tools/galign.h"

namespace nuri {

namespace {
  auto parent_forest(const Molecule &mol) {
    Graph<std::vector<int>, std::pair<int, int>> forest;

    internal::LinearQueue queue(
        std::vector<std::pair<int, int>>(mol.num_atoms()));
    ArrayXb visited = ArrayXb::Zero(mol.num_atoms());

    ArrayXi roots(mol.num_fragments());
    int ri = 0;
    for (int root = 0; root < mol.num_atoms(); ++root) {
      if (visited[root])
        continue;

      visited[root] = true;

      int root_ti = forest.add_node({ root });
      roots[ri++] = root_ti;

      queue.clear();

      auto ratom = mol[root];
      if (ratom.degree() == 1) {
        const int next = ratom[0].dst().id();
        visited[next] = true;
        forest[root_ti].data().push_back(next);
        if (ratom[0].dst().degree() == 1)
          continue;

        queue.push({ next, root_ti });
      } else {
        queue.push({ root, root_ti });
      }

      do {
        auto [curr, curr_ti] = queue.pop();

        for (auto nei: mol[curr]) {
          const int next = nei.dst().id();
          if (visited[next])
            continue;

          visited[next] = true;

          int next_ti;
          if (nei.edge_data().is_ring_bond() || nei.dst().degree() == 1) {
            next_ti = curr_ti;
            forest[curr_ti].data().push_back(next);
          } else {
            next_ti = forest.add_node({ next });
            forest.add_edge(curr_ti, next_ti,
                            { nei.src().id(), nei.dst().id() });
          }

          if (nei.dst().degree() > 1)
            queue.push({ next, next_ti });
        }
      } while (!queue.empty());
    }

    return std::make_pair(forest, roots);
  }

  struct RotatableBondComp {
    std::vector<int> left_atoms;
    std::vector<int> right_atoms;
  };

  std::vector<RotatableBondComp> split_components_by_bridge(
      const Graph<std::vector<int>, std::pair<int, int>> &pf,
      const ArrayXi &roots) {
    std::vector<RotatableBondComp> rbs(pf.num_edges());

    auto dfs_fill_right = [&](auto &self, int curr,
                              int eid) -> const std::vector<int> * {
      std::vector<int> *const ret = eid < 0 ? nullptr : &rbs[eid].right_atoms;
      if (ret != nullptr)
        *ret = pf[curr].data();

      for (auto nei: pf[curr]) {
        if (nei.eid() == eid)
          continue;

        const std::vector<int> &sub = *self(self, nei.dst().id(), nei.eid());
        if (ret != nullptr)
          ret->insert(ret->end(), sub.begin(), sub.end());
      }

      return ret;
    };

    for (int root: roots)
      dfs_fill_right(dfs_fill_right, root, -1);

    auto dfs_fill_left = [&](auto &self, int curr, int eid) -> void {
      const std::vector<int> *prev_left = eid < 0 ? nullptr
                                                  : &rbs[eid].left_atoms;

      for (auto nei: pf[curr]) {
        if (nei.eid() == eid)
          continue;

        std::vector<int> &left = rbs[nei.eid()].left_atoms;
        left = pf[curr].data();
        if (prev_left != nullptr)
          left.insert(left.end(), prev_left->begin(), prev_left->end());

        for (auto mei: pf[curr]) {
          if (mei.eid() == nei.eid() || mei.eid() == eid)
            continue;

          const std::vector<int> &right = rbs[mei.eid()].right_atoms;
          left.insert(left.end(), right.begin(), right.end());
        }

        self(self, nei.dst().id(), nei.eid());
      }
    };

    for (int root: roots)
      dfs_fill_left(dfs_fill_left, root, -1);

    return rbs;
  }
}  // namespace

namespace internal {
  std::vector<GARotationInfo> GARotationInfo::from(const Molecule &mol,
                                                   const Matrix3Xd &ref) {
    auto [pf, roots] = parent_forest(mol);
    std::vector rbs = split_components_by_bridge(pf, roots);

    std::vector<GARotationInfo> ri(rbs.size());
    for (int i = 0; i < rbs.size(); ++i) {
      RotatableBondComp &rb = rbs[i];
      GARotationInfo &r = ri[i];

      const bool reverse = rb.right_atoms.size() < rb.left_atoms.size();
      (reverse ? std::tie(r.ref_, r.origin_) : std::tie(r.origin_, r.ref_)) =
          pf.edge_data(i);

      r.normalizer_ = 1 / (ref.col(r.origin_) - ref.col(r.ref_)).norm();

      std::vector<int> &moving = reverse ? rb.right_atoms : rb.left_atoms;
      auto pit =
          absl::c_find_if(moving, [&](int id) { return id == r.origin_; });
      ABSL_DCHECK(pit != moving.end());
      std::iter_swap(pit, --moving.end());

      r.moving_ = Eigen::Map<ArrayXi>(
          moving.data(), static_cast<Eigen::Index>(moving.size() - 1));
      absl::c_sort(r.moving_);
    }

    return ri;
  }

  Matrix3Xd &GARotationInfo::rotate(Matrix3Xd &pts, const double angle) const {
    Matrix3d rot = Eigen::AngleAxisd(angle, (pts.col(origin_) - pts.col(ref_))
                                                * normalizer())
                       .toRotationMatrix();
    Vector3d trs = pts.col(origin_);
    Affine3d xform = Translation3d(trs) * rot * Translation3d(-trs);

    for (int i: moving_)
      pts.col(i) = xform * pts.col(i);

    return pts;
  }
}  // namespace internal
}  // namespace nuri
