//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <Eigen/Dense>

#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
namespace {
  constexpr double kCutoff = 5.0;

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

  Matrix3Xd create_sample(const Matrix3Xd &full, int64_t ref_size,
                          int query_size = 128) {
    Matrix3Xd sample = full.rightCols(query_size);
    Vector3d ref_cntr = full.leftCols(ref_size).rowwise().mean();
    Vector3d qry_cntr = sample.rowwise().mean();
    sample.colwise() += ref_cntr - qry_cntr;
    return sample;
  }

  void octree_build(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    pts = pts.leftCols(state.range(0));
    for (auto _: state) {
      OCTree tree(pts);
      benchmark::DoNotOptimize(tree);
    }

    state.SetComplexityN(state.range(0));
  }
  BENCHMARK(octree_build)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Complexity(benchmark::oNLogN)
      ->Unit(benchmark::kMicrosecond);

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

  void octree_count_distance_query(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    const OCTree tree(pts);
    const int count = 10;

    std::vector<int> idxs;
    std::vector<double> dsqs;

    for (int i = 0; i < state.range(0); ++i)
      tree.find_neighbors_kd(pts.col(i), count, 5.0, idxs, dsqs);

    for (auto _: state) {
      for (int i = 0; i < state.range(0); ++i) {
        const Vector3d query_pt = pts.col(i);
        tree.find_neighbors_kd(query_pt, count, 5.0, idxs, dsqs);
      }
    }

    state.SetComplexityN(state.range(0));
  }
  BENCHMARK(octree_count_distance_query)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Complexity(benchmark::oNLogN)
      ->Unit(benchmark::kMicrosecond);

  void octree_inter_query(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    Matrix3Xd qpts = create_sample(pts, state.range(0));

    pts.conservativeResize(Eigen::NoChange, state.range(0));
    const OCTree tree(pts);
    const OCTree qtree(qpts);

    std::vector<int> is, js;
    tree.find_neighbors_tree(qtree, kCutoff, is, js);

    for (auto _: state) {
      tree.find_neighbors_tree(qtree, kCutoff, is, js);
    }

    state.counters["npairs"] = static_cast<double>(is.size());
  }
  BENCHMARK(octree_inter_query)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Unit(benchmark::kMicrosecond);

  void octree_intra_query(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    pts.conservativeResize(Eigen::NoChange, state.range(0));
    const OCTree tree(pts);
    const double cutoff = 5.0;

    std::vector<int> is, js;
    tree.find_neighbors_self(cutoff, is, js);

    for (auto _: state) {
      tree.find_neighbors_self(cutoff, is, js);
    }
  }
  BENCHMARK(octree_intra_query)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Unit(benchmark::kMicrosecond);

  void voxel_grid_build(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    pts = pts.leftCols(state.range(0));
    for (auto _: state) {
      VoxelGrid grid(pts, kCutoff);
      benchmark::DoNotOptimize(grid);
    }

    state.SetComplexityN(state.range(0));
  }
  BENCHMARK(voxel_grid_build)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Complexity(benchmark::oN)
      ->Unit(benchmark::kMicrosecond);

  void voxel_grid_distance_query(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    const VoxelGrid grid(pts, kCutoff);

    std::vector<int> idxs;
    std::vector<double> dsqs;

    ArrayXi sizes(state.range(0));
    for (int i = 0; i < state.range(0); ++i) {
      grid.find_neighbors_d(pts.col(i), idxs, dsqs);
      sizes[i] = static_cast<int>(idxs.size());
    }

    for (auto _: state) {
      for (int i = 0; i < state.range(0); ++i) {
        const Vector3d query_pt = pts.col(i);
        grid.find_neighbors_d(query_pt, idxs, dsqs);
      }
    }

    state.SetComplexityN(state.range(0));
    state.counters["nnei"] = sizes.cast<double>().mean();
  }
  BENCHMARK(voxel_grid_distance_query)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Complexity(benchmark::oN)
      ->Unit(benchmark::kMicrosecond);

  void voxel_grid_inter_query(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    Matrix3Xd qpts = create_sample(pts, state.range(0));

    pts.conservativeResize(Eigen::NoChange, state.range(0));
    const VoxelGrid grid(pts, kCutoff);
    const VoxelGrid qgrid(qpts, kCutoff);

    std::vector<int> is, js;
    grid.find_neighbors_grid(qgrid, is, js);

    for (auto _: state) {
      grid.find_neighbors_grid(qgrid, is, js);
    }

    state.SetComplexityN(state.range(0));
    state.counters["npairs"] = static_cast<double>(is.size());
  }
  BENCHMARK(voxel_grid_inter_query)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Complexity(benchmark::oN)
      ->Unit(benchmark::kMicrosecond);

  void voxel_grid_intra_query(benchmark::State &state) {
    Matrix3Xd pts = read_1ubq(state);
    if (state.skipped())
      return;

    pts.conservativeResize(Eigen::NoChange, state.range(0));
    const VoxelGrid grid(pts, kCutoff);

    std::vector<int> is, js;
    grid.find_neighbors_self(is, js);

    for (auto _: state) {
      grid.find_neighbors_self(is, js);
    }

    state.SetComplexityN(state.range(0));
  }
  BENCHMARK(voxel_grid_intra_query)
      ->RangeMultiplier(2)
      ->Range(16, 1024)
      ->Complexity(benchmark::oN)
      ->Unit(benchmark::kMicrosecond);
}  // namespace
}  // namespace nuri
