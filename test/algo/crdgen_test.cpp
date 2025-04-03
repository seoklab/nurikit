//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/crdgen.h"

#include <cmath>

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
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
    {  -3.27734,   1.85008,  0.331878 },
    {  -2.35362,  0.643631,  0.291369 },
    {  -2.16021, 0.0192391,   1.33967 },
    {   -1.8673,   0.31158, -0.842877 },
    { -0.478895,  0.165713,  -1.10243 },
    {  0.250714,   1.49762,  -1.06163 },
    {  0.547288,   1.94428,   0.36239 },
    {   1.27779,  0.871516,   1.15371 },
    {  0.569783, -0.468935,   1.11893 },
    {   1.72724,  -1.41762,   1.40761 },
    {    2.3178,  -1.63682,  0.130978 },
    {   1.50836,  -1.37125, -0.827468 },
    {   1.77025,   -1.5044,   -2.0266 },
    {  0.168125, -0.904638, -0.275532 },
  };

  auto [_, msd] = qcp(conf, ans.transpose(), AlignMode::kMsdOnly);

  double rmsd = std::sqrt(msd);
  EXPECT_LE(rmsd, 0.1) << "RMSD: " << rmsd << ",\n"
                       << "conf:\n"
                       << conf.transpose();
}

TEST(Crdgen, CHEMBL2228334Chiral) {
  Molecule mol = read_smiles({ "CC(=O)O[C@H]1CCC[C@@H]2COC(=O)[C@@H]21" });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  Matrix3Xd &conf = mol.confs().emplace_back(3, mol.num_atoms());
  ASSERT_TRUE(generate_coords(mol, conf));

  // XXX: The algorithm is basically deterministic on first pass, so just
  // compare it with the reference output. Should be changed later on?
  MatrixX3d ans {
    { -3.81639,   1.29168, -0.524913 },
    {  -2.7298,  0.436659,  0.106535 },
    { -2.91186,  -0.75426,  0.380012 },
    { -1.58535,  0.961442,  0.325492 },
    { -0.52508,  0.453655, -0.470913 },
    { 0.356459,   1.61319, -0.902941 },
    {   1.2496,   2.10462,  0.226706 },
    {  2.05839,  0.975216,  0.843873 },
    {  1.19803,  -0.19705,   1.27386 },
    {  2.21753,  -1.32986,   1.25283 },
    {  2.22197,  -1.76199,  -0.10359 },
    {  1.15315,  -1.41896, -0.723512 },
    { 0.890656,  -1.71997,  -1.89185 },
    { 0.222705, -0.654381,  0.208408 },
  };

  auto [_, msd] = qcp(conf, ans.transpose(), AlignMode::kMsdOnly);

  double rmsd = std::sqrt(msd);
  EXPECT_LE(rmsd, 0.1) << "RMSD: " << rmsd << ",\n"
                       << "conf:\n"
                       << conf.transpose();
}
}  // namespace
}  // namespace nuri
