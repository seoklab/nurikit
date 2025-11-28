//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <benchmark/benchmark.h>
#include <Eigen/Dense>

#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
namespace {
  Matrix3Xd read_1ubq(benchmark::State &state) {
    FileMoleculeReader<> reader("pdb", "test/test_data/1ubqFH.pdb");
    auto stream = reader.stream();
    Molecule mol;
    stream >> mol;

    if (mol.size() == 0) {
      state.SkipWithError("Failed to read molecule from PDB file");
      return {};
    }

    const int natoms = static_cast<int>(state.range(0));
    if (mol.size() < natoms) {
      state.SkipWithMessage("Not enough atoms in the molecule");
      return {};
    }

    return std::move(mol.confs()[0]);
  }

  void octree_distance_query(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    const OCTree tree(pts);
    const double cutoff = 5.0;

    std::vector<int> idxs;
    std::vector<double> dsqs;

    ArrayXi sizes(state.range(0));
    for (int i = 0; i < state.range(0); ++i) {
      tree.find_neighbors_d(pts.col(i), cutoff, idxs, dsqs);
      sizes[i] = static_cast<int>(idxs.size());
    }

    for (auto _: state) {
      for (int i = 0; i < state.range(0); ++i) {
        const Vector3d query_pt = pts.col(i);
        tree.find_neighbors_d(query_pt, cutoff, idxs, dsqs);
      }
    }

    state.SetComplexityN(state.range(0));
    state.counters["nnei"] = sizes.cast<double>().mean();
  }
  BENCHMARK(octree_distance_query)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Complexity(benchmark::oNLogN)
      ->Unit(benchmark::kMicrosecond);

  void octree_count_query(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    const OCTree tree(pts);
    const int count = 10;

    std::vector<int> idxs;
    std::vector<double> dsqs;

    for (int i = 0; i < state.range(0); ++i)
      tree.find_neighbors_k(pts.col(i), count, idxs, dsqs);

    for (auto _: state) {
      for (int i = 0; i < state.range(0); ++i) {
        const Vector3d query_pt = pts.col(i);
        tree.find_neighbors_k(query_pt, count, idxs, dsqs);
      }
    }

    state.SetComplexityN(state.range(0));
  }
  BENCHMARK(octree_count_query)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Complexity(benchmark::oNLogN)
      ->Unit(benchmark::kMicrosecond);
}  // namespace
}  // namespace nuri
