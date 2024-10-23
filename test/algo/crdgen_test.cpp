//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/crdgen.h"

#include <Eigen/Dense>

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/smiles.h"

namespace nuri {
namespace {
TEST(Crdgen, CHEMBL2228334) {
  Molecule mol = read_smiles({ "CC(=O)OC1CCCC2COC(=O)C21" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  Matrix3Xd &conf = mol.confs().emplace_back(3, mol.num_atoms());
  ASSERT_TRUE(generate_coords(mol, conf));

  // XXX: The algorithm is basically deterministic on first pass, so just
  // compare it with the reference output. Should be changed later on?
  MatrixX3d ans {
    { -2.935,  1.012,  0.302 },
    { -1.763,  0.055,  0.158 },
    { -1.909, -1.037,  0.717 },
    { -1.160,  0.101, -0.966 },
    {  0.103,  0.678, -1.212 },
    {  0.163,  2.143, -0.808 },
    {  0.313,  2.278,  0.700 },
    {  1.506,  1.506,  1.230 },
    {  1.613,  0.105,  0.744 },
    {  0.757, -0.897,  1.489 },
    {  0.689, -1.977,  0.565 },
    {  0.957, -1.626, -0.639 },
    {  0.404, -2.209, -1.577 },
    {  1.262, -0.131, -0.703 },
  };

  NURI_EXPECT_EIGEN_EQ_TOL(conf.transpose(), ans, 5e-2);
}
}  // namespace
}  // namespace nuri
