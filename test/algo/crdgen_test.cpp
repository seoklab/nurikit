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
    {  -3.55966,   1.50993, -0.930075 },
    {  -2.75028,  0.486151, -0.150855 },
    {  -3.33442, -0.427297,  0.441262 },
    {  -1.61163,  0.835663,  0.312375 },
    { -0.483807,  0.430077,  -0.44915 },
    {  0.388935,   1.64985, -0.691068 },
    {   1.19272,   2.03227,  0.543011 },
    {   1.99297,  0.859375,   1.08574 },
    {   1.14132, -0.371937,   1.32634 },
    {   2.19254,  -1.47294,   1.25092 },
    {   2.29758,  -1.75128, -0.141185 },
    {   1.26159,  -1.36537, -0.791154 },
    {   1.02142,  -1.68478,  -1.96013 },
    {  0.250725,  -0.72971,  0.153981 },
  };

  NURI_EXPECT_EIGEN_EQ_TOL(conf.transpose(), ans, 5e-2);
}

TEST(Crdgen, CHEMBL2228334Chiral) {
  Molecule mol = read_smiles({ "CC(=O)O[C@H]1CCC[C@@H]2COC(=O)[C@@H]21" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  Matrix3Xd &conf = mol.confs().emplace_back(3, mol.num_atoms());
  ASSERT_TRUE(generate_coords(mol, conf));

  // XXX: The algorithm is basically deterministic on first pass, so just
  // compare it with the reference output. Should be changed later on?
  MatrixX3d ans {
    {  -3.70254,   1.36343,   -0.631601 },
    {  -2.72728,   0.38195, -0.00235632 },
    {  -3.18834, -0.427798,    0.808779 },
    {  -1.57831,  0.856477,     0.29421 },
    { -0.476699,  0.486279,    -0.52175 },
    {  0.397502,   1.72502,   -0.618426 },
    {   1.15448,   1.99352,    0.673924 },
    {   1.94115,  0.777206,     1.13475 },
    {   1.08813, -0.472856,     1.22981 },
    {   2.14736,  -1.55996,     1.09125 },
    {   2.30371,  -1.70828,   -0.315881 },
    {   1.28956,  -1.26622,    -0.96453 },
    {   1.10924,  -1.42605,     -2.1762 },
    {  0.242056, -0.722701,  -0.0019777 },
  };

  NURI_EXPECT_EIGEN_EQ_TOL(conf.transpose(), ans, 5e-2);
}
}  // namespace
}  // namespace nuri
