//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/tools/galign.h"

#include <initializer_list>
#include <vector>

#include <absl/algorithm/container.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "test_utils.h"
#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/smiles.h"

namespace nuri {
namespace {
testing::AssertionResult
verify_rotation_info(const internal::GARotationInfo &ri, int ref,
                     const std::initializer_list<int> &expected_moving) {
  if (ri.ref() != ref) {
    return testing::AssertionFailure()
           << "ref mismatch: " << ri.ref() << " != " << ref;
  }

  if (!absl::c_equal(ri.moving(), expected_moving)) {
    return testing::AssertionFailure()
           << "moving atoms: " << ri.moving().transpose();
  }

  return testing::AssertionSuccess();
}

TEST(GAlign, RotationInfo) {
  Molecule mol = read_smiles({
      "O=C(C1CCC(CC(C)OC)CC1)c2cc3c(CC)ccc(C(C)C)c3cc2.Cl.[H][H]",
  });

  Matrix3Xd &conf = mol.confs().emplace_back(3, mol.num_atoms());
  conf.transpose() << -2.2395, -1.8709, 7.9552,  //
      -2.7480, -1.6062, 6.8866,                  //
      -4.1759, -2.0027, 6.6126,                  //
      -5.1017, -0.8271, 6.9314,                  //
      -6.5514, -1.2296, 6.6533,                  //
      -6.9257, -2.4261, 7.5305,                  //
      -8.3753, -2.8286, 7.2523,                  //
      -8.7951, -3.9296, 8.2283,                  //
      -10.2124, -4.3969, 7.8911,                 //
      -8.7666, -3.4195, 9.5628,                  //
      -8.4691, -4.4037, 10.5551,                 //
      -5.9998, -3.6016, 7.2117,                  //
      -4.5502, -3.1991, 7.4898,                  //
      -1.9712, -0.8982, 5.8579,                  //
      -0.6559, -0.5350, 6.1151,                  //
      0.0776, 0.1400, 5.1294,                    //
      1.4126, 0.5242, 5.3522,                    //
      2.0811, 0.2130, 6.6665,                    //
      2.6519, -1.2059, 6.6249,                   //
      2.0973, 1.1779, 4.3745,                    //
      1.4970, 1.4690, 3.1509,                    //
      0.2053, 1.1173, 2.9018,                    //
      -0.4257, 1.4467, 1.5735,                   //
      0.4336, 0.8738, 0.4446,                    //
      -0.5247, 2.9656, 1.4185,                   //
      -0.5347, 0.4413, 3.8857,                   //
      -1.8695, 0.0601, 3.6547,                   //
      -2.5665, -0.5926, 4.6184,                  //
      7.6519, 7.9656, 15.555,                    //
      0.0, 0.0, 0.0,                             //
      0.0, 0.0, 1.0;

  std::vector ri = internal::GARotationInfo::from(mol, conf);
  ASSERT_EQ(ri.size(), 7);

  for (auto &r: ri) {
    auto bit = mol.find_bond(r.origin(), r.ref());
    ASSERT_NE(bit, mol.bond_end());

    EXPECT_TRUE(bit->data().is_rotatable());

    switch (r.origin()) {
    case 2:
      EXPECT_TRUE(
          verify_rotation_info(r, 1, { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
      break;
    case 1:
      EXPECT_TRUE(verify_rotation_info(
          r, 13, { 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
      break;
    case 6:
      EXPECT_TRUE(verify_rotation_info(r, 5, { 7, 8, 9, 10 }));
      break;
    case 17:
      EXPECT_TRUE(verify_rotation_info(r, 16, { 18 }));
      break;
    case 7:
      EXPECT_TRUE(verify_rotation_info(r, 6, { 8, 9, 10 }));
      break;
    case 22:
      EXPECT_TRUE(verify_rotation_info(r, 21, { 23, 24 }));
      break;
    case 9:
      EXPECT_TRUE(verify_rotation_info(r, 7, { 10 }));
      break;
    default:
      FAIL() << "Unexpected origin/ref pair: " << r.origin() << ", " << r.ref();
    }
  }

  ri[0].rotate(conf, deg2rad(30));
  Matrix3Xd rotated(3, mol.num_atoms());
  rotated.transpose() << -2.2395, -1.8709, 7.9552,  //
      -2.7480, -1.6062, 6.8866,                     //
      -4.1759, -2.0027, 6.6126,                     //
      -4.9775, -0.7674, 6.1975,                     //
      -6.4272, -1.1699, 5.9194,                     //
      -7.0402, -1.7744, 7.1843,                     //
      -8.4898, -2.1770, 6.9061,                     //
      -9.1466, -2.6479, 8.2052,                     //
      -10.5660, -3.1373, 7.9110,                    //
      -9.1998, -1.5623, 9.1329,                     //
      -9.1364, -1.9644, 10.5026,                    //
      -6.2385, -3.0097, 7.5993,                     //
      -4.7889, -2.6072, 7.8774,                     //
      -1.9712, -0.8982, 5.8579,                     //
      -0.6559, -0.5350, 6.1151,                     //
      0.0776, 0.1400, 5.1294,                       //
      1.4126, 0.5242, 5.3522,                       //
      2.0811, 0.2130, 6.6665,                       //
      2.6519, -1.2059, 6.6249,                      //
      2.0973, 1.1779, 4.3745,                       //
      1.4970, 1.4690, 3.1509,                       //
      0.2053, 1.1173, 2.9018,                       //
      -0.4257, 1.4467, 1.5735,                      //
      0.4336, 0.8738, 0.4446,                       //
      -0.5247, 2.9656, 1.4185,                      //
      -0.5347, 0.4413, 3.8857,                      //
      -1.8695, 0.0601, 3.6547,                      //
      -2.5665, -0.5926, 4.6184,                     //
      7.6519, 7.9656, 15.5551,                      //
      0.0, 0.0, 0.0,                                //
      0.0, 0.0, 1.0;

  NURI_EXPECT_EIGEN_EQ_TOL(conf, rotated, 1e-4);
}
}  // namespace
}  // namespace nuri
