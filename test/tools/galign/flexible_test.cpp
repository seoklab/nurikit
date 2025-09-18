//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <initializer_list>
#include <vector>

#include <absl/algorithm/container.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "test_utils.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/smiles.h"
#include "nuri/random.h"
#include "nuri/tools/galign.h"

namespace nuri {
namespace internal {
namespace {
TEST(GAlign, Flexible) {
  Molecule templ = read_smiles({
      "O=C(C1CCC(CC(C)OC)CC1)c2cc3c(CC)ccc(C(C)C)c3cc2.Cl.[H][H]",
  });

  Matrix3Xd &tconf = templ.confs().emplace_back(3, templ.num_atoms());
  tconf.transpose() << -2.2395, -1.8709, 7.9552,  //
      -2.7480, -1.6062, 6.8866,                   //
      -4.1759, -2.0027, 6.6126,                   //
      -5.1017, -0.8271, 6.9314,                   //
      -6.5514, -1.2296, 6.6533,                   //
      -6.9257, -2.4261, 7.5305,                   //
      -8.3753, -2.8286, 7.2523,                   //
      -8.7951, -3.9296, 8.2283,                   //
      -10.2124, -4.3969, 7.8911,                  //
      -8.7666, -3.4195, 9.5628,                   //
      -8.4691, -4.4037, 10.5551,                  //
      -5.9998, -3.6016, 7.2117,                   //
      -4.5502, -3.1991, 7.4898,                   //
      -1.9712, -0.8982, 5.8579,                   //
      -0.6559, -0.5350, 6.1151,                   //
      0.0776, 0.1400, 5.1294,                     //
      1.4126, 0.5242, 5.3522,                     //
      2.0811, 0.2130, 6.6665,                     //
      2.6519, -1.2059, 6.6249,                    //
      2.0973, 1.1779, 4.3745,                     //
      1.4970, 1.4690, 3.1509,                     //
      0.2053, 1.1173, 2.9018,                     //
      -0.4257, 1.4467, 1.5735,                    //
      0.4336, 0.8738, 0.4446,                     //
      -0.5247, 2.9656, 1.4185,                    //
      -0.5347, 0.4413, 3.8857,                    //
      -1.8695, 0.0601, 3.6547,                    //
      -2.5665, -0.5926, 4.6184,                   //
      7.6519, 7.9656, 15.555,                     //
      0.0, 0.0, 0.0,                              //
      0.0, 0.0, 1.0;

  Molecule query(templ);
  {
    auto mut = query.mutator();
    mut.mark_atom_erase(29);
    mut.mark_atom_erase(30);
  }
  Matrix3Xd &qconf = query.confs()[0];

  Isometry3d random;
  random.linear() = Quaterniond::UnitRandom().toRotationMatrix();
  random.translation() = Vector3d::Random() * 10;
  tconf = random * tconf;

  GARigidMolInfo qinfo(query, qconf), tinfo(templ, tconf);

  GAGeneticArgs genetic;
  genetic.pool_size = 2;
  genetic.sample_size = 4;
  genetic.max_gen = 5;
  genetic.patience = 2;

  GAMinimizeArgs minimize;
  minimize.ftol = 0.1;
  minimize.max_iters = 50;

  set_thread_seed(42);
  std::vector results =
      flexible_galign_impl(qinfo, tinfo, 1, 0.7, genetic, minimize);
  ASSERT_EQ(results.size(), 1);

  const AlignResult &result = results[0];
  NURI_EXPECT_EIGEN_EQ_TOL(random.matrix(), result.xform.matrix(), 1e-6);
  EXPECT_GE(result.align_score, 0.95);
}
}  // namespace
}  // namespace internal
}  // namespace nuri
