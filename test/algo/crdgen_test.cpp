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
TEST(Strgen, Playground) {
  Molecule mol = read_smiles({ "CC(=O)OC1CCCC2COC(=O)C21" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  Matrix3Xd &conf = mol.confs().emplace_back(3, mol.num_atoms());
  ASSERT_TRUE(generate_coords(mol, conf));

  // XXX: The algorithm is basically deterministic on first pass, so just
  // compare it with the reference output. Should be changed later on?
  MatrixX3d ans {
    { -2.660,  1.200,  0.539 },
    { -1.787,  0.051,  0.036 },
    { -1.671, -0.981,  0.715 },
    { -1.188,  0.148, -1.095 },
    {  0.111,  0.683, -1.280 },
    {  0.163,  2.150, -0.882 },
    {  0.144,  2.286,  0.634 },
    {  1.209,  1.469,  1.313 },
    {  1.509,  0.149,  0.783 },
    {  0.970, -1.042,  1.575 },
    {  0.686, -2.046,  0.589 },
    {  0.840, -1.641, -0.629 },
    {  0.469, -2.281, -1.618 },
    {  1.205, -0.144, -0.679 },
  };

  NURI_EXPECT_EIGEN_EQ_TOL(conf, ans.transpose(), 1e-2);
}
}  // namespace
}  // namespace nuri
