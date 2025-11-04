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
#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/smiles.h"
#include "nuri/tools/galign.h"

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

  E::Matrix3Xf conf(3, mol.num_atoms());
  conf.transpose() << -2.2395F, -1.8709F, 7.9552F,  //
      -2.7480F, -1.6062F, 6.8866F,                  //
      -4.1759F, -2.0027F, 6.6126F,                  //
      -5.1017F, -0.8271F, 6.9314F,                  //
      -6.5514F, -1.2296F, 6.6533F,                  //
      -6.9257F, -2.4261F, 7.5305F,                  //
      -8.3753F, -2.8286F, 7.2523F,                  //
      -8.7951F, -3.9296F, 8.2283F,                  //
      -10.2124F, -4.3969F, 7.8911F,                 //
      -8.7666F, -3.4195F, 9.5628F,                  //
      -8.4691F, -4.4037F, 10.5551F,                 //
      -5.9998F, -3.6016F, 7.2117F,                  //
      -4.5502F, -3.1991F, 7.4898F,                  //
      -1.9712F, -0.8982F, 5.8579F,                  //
      -0.6559F, -0.5350F, 6.1151F,                  //
      0.0776F, 0.1400F, 5.1294F,                    //
      1.4126F, 0.5242F, 5.3522F,                    //
      2.0811F, 0.2130F, 6.6665F,                    //
      2.6519F, -1.2059F, 6.6249F,                   //
      2.0973F, 1.1779F, 4.3745F,                    //
      1.4970F, 1.4690F, 3.1509F,                    //
      0.2053F, 1.1173F, 2.9018F,                    //
      -0.4257F, 1.4467F, 1.5735F,                   //
      0.4336F, 0.8738F, 0.4446F,                    //
      -0.5247F, 2.9656F, 1.4185F,                   //
      -0.5347F, 0.4413F, 3.8857F,                   //
      -1.8695F, 0.0601F, 3.6547F,                   //
      -2.5665F, -0.5926F, 4.6184F,                  //
      7.6519F, 7.9656F, 15.555F,                    //
      0.0F, 0.0F, 0.0F,                             //
      0.0F, 0.0F, 1.0F;

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

  ri[0].rotate(conf, deg2rad(30.0F));
  E::Matrix3Xf rotated(3, mol.num_atoms());
  rotated.transpose() << -2.2395F, -1.8709F, 7.9552F,  //
      -2.7480F, -1.6062F, 6.8866F,                     //
      -4.1759F, -2.0027F, 6.6126F,                     //
      -4.9775F, -0.7674F, 6.1975F,                     //
      -6.4272F, -1.1699F, 5.9194F,                     //
      -7.0402F, -1.7744F, 7.1843F,                     //
      -8.4898F, -2.1770F, 6.9061F,                     //
      -9.1466F, -2.6479F, 8.2052F,                     //
      -10.5660F, -3.1373F, 7.9110F,                    //
      -9.1998F, -1.5623F, 9.1329F,                     //
      -9.1364F, -1.9644F, 10.5026F,                    //
      -6.2385F, -3.0097F, 7.5993F,                     //
      -4.7889F, -2.6072F, 7.8774F,                     //
      -1.9712F, -0.8982F, 5.8579F,                     //
      -0.6559F, -0.5350F, 6.1151F,                     //
      0.0776F, 0.1400F, 5.1294F,                       //
      1.4126F, 0.5242F, 5.3522F,                       //
      2.0811F, 0.2130F, 6.6665F,                       //
      2.6519F, -1.2059F, 6.6249F,                      //
      2.0973F, 1.1779F, 4.3745F,                       //
      1.4970F, 1.4690F, 3.1509F,                       //
      0.2053F, 1.1173F, 2.9018F,                       //
      -0.4257F, 1.4467F, 1.5735F,                      //
      0.4336F, 0.8738F, 0.4446F,                       //
      -0.5247F, 2.9656F, 1.4185F,                      //
      -0.5347F, 0.4413F, 3.8857F,                      //
      -1.8695F, 0.0601F, 3.6547F,                      //
      -2.5665F, -0.5926F, 4.6184F,                     //
      7.6519F, 7.9656F, 15.5551F,                      //
      0.0F, 0.0F, 0.0F,                                //
      0.0F, 0.0F, 1.0F;

  NURI_EXPECT_EIGEN_EQ_TOL(conf, rotated, 1e-3F);
}
}  // namespace
}  // namespace nuri
