//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include <benchmark/benchmark.h>
#include <Eigen/Dense>

#include "nuri/core/molecule.h"
#include "nuri/desc/surface.h"
#include "nuri/fmt/pdb.h"

namespace nuri {
namespace {
  std::pair<Matrix3Xd, ArrayXd> read_1ubq(benchmark::State &state) {
    std::ifstream input("test/test_data/1ubqFH.pdb");
    PDBReader reader(input);
    auto result = reader.next()->parse();
    if (!result || result->data().empty()) {
      state.SkipWithError("Failed to parse PDB input");
      return {};
    }
    Molecule mol = std::move(result->data().front());
    if (mol.empty()) {
      state.SkipWithError("Failed to read molecule from PDB file");
      return {};
    }

    const int natoms = static_cast<int>(state.range(0));
    if (mol.size() < natoms) {
      state.SkipWithMessage("Not enough atoms in the molecule");
      return {};
    }

    Matrix3Xd pts = mol.confs()[0].leftCols(natoms);
    ArrayXd radii(natoms);
    for (int i = 0; i < natoms; ++i)
      radii[i] = mol[i].data().element().vdw_radius();

    return std::make_pair(std::move(pts), std::move(radii));
  }

  void sr_sasa_direct(benchmark::State &state) {
    auto [pts, radii] = read_1ubq(state);
    if (state.skipped())
      return;

    ArrayXd sasa(radii.size());
    const int nprobe = static_cast<int>(state.range(1));
    for (auto _: state) {
      sasa = internal::sr_sasa_impl(pts, radii, nprobe,
                                    internal::SrSasaMethod::kDirect);
      benchmark::DoNotOptimize(sasa);
    }
  }
  BENCHMARK(sr_sasa_direct)
      ->ArgsProduct({
          {  10,  50,  100,   500 },
          { 100, 500, 1000, 10000 }
  })
      ->Unit(benchmark::kMillisecond);

  void sr_sasa_octree(benchmark::State &state) {
    auto [pts, radii] = read_1ubq(state);
    if (state.skipped())
      return;

    ArrayXd sasa(radii.size());
    const int nprobe = static_cast<int>(state.range(1));
    for (auto _: state) {
      sasa = internal::sr_sasa_impl(pts, radii, nprobe,
                                    internal::SrSasaMethod::kOctree);
      benchmark::DoNotOptimize(sasa);
    }
  }
  BENCHMARK(sr_sasa_octree)
      ->ArgsProduct({
          {  10,  50,  100,   500 },
          { 100, 500, 1000, 10000 }
  })
      ->Unit(benchmark::kMillisecond);
}  // namespace
}  // namespace nuri
