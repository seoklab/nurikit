//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/desc/surface.h"

#include <gtest/gtest.h>

#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/sdf.h"

namespace nuri {
namespace {
double sphere_surface_exposed(const Matrix3Xd &pts, const E::Array2d &r) {
  const double d = (pts.col(0) - pts.col(1)).norm();
  E::Array2d r2 = r.square();

  double hidden;
  if (d > r[0] + r[1]) {
    hidden = 0.0;
  } else if (d + r[0] < r[1]) {
    hidden = 4 * constants::kPi * r2[0];
  } else if (d + r[1] < r[0]) {
    hidden = 4 * constants::kPi * r2[1];
  } else {
    hidden = constants::kPi / d
             * (r[0] * (r2[1] - (d - r[0]) * (d - r[0]))
                + r[1] * (r2[0] - (d - r[1]) * (d - r[1])));
  }

  return 4 * constants::kPi * (r2.sum()) - hidden;
}

testing::AssertionResult test_sr_sasa_both(const Matrix3Xd &pts,
                                           const E::Array2d &radii,
                                           const int nprobe) {
  const double ref = sphere_surface_exposed(pts, radii);
  const double tol = ref * 1e-3;

  ArrayXd sasa = internal::sr_sasa_impl(pts, radii, nprobe,
                                        internal::SrSasaMethod::kDirect);
  if (std::abs(sasa.sum() - ref) > tol) {
    return testing::AssertionFailure()
           << "Direct method: expected " << ref << ", got " << sasa.sum();
  }

  sasa = internal::sr_sasa_impl(pts, radii, nprobe,
                                internal::SrSasaMethod::kOctree);
  if (std::abs(sasa.sum() - ref) > tol) {
    return testing::AssertionFailure()
           << "Octree method: expected " << ref << ", got " << sasa.sum();
  }

  return testing::AssertionSuccess();
}

TEST(SRSasaImpl, TwoSpheres) {
  constexpr int nprobe = 5000;

  E::Array2d radii { 1, 2 };
  radii += 1.4;
  Matrix3Xd pts = Matrix3Xd::Zero(3, 2);

  pts(0, 1) = 2.0;
  EXPECT_TRUE(test_sr_sasa_both(pts, radii, nprobe));

  pts(0, 1) = 0.0;
  pts(1, 1) = 2.0;
  EXPECT_TRUE(test_sr_sasa_both(pts, radii, nprobe));

  pts(1, 1) = 0.0;
  pts(2, 1) = 2.0;
  EXPECT_TRUE(test_sr_sasa_both(pts, radii, nprobe));
}

TEST(SRSasa, Phenol) {
  Molecule mol = read_sdf({
      "",
      "     RDKit          3D",
      "",
      " 13 13  0  0  0  0  0  0  0  0999 V2000",
      "    2.4823    0.0194   -0.3935 O   0  0  0  0  0  0  0  0  0  0  0  0",
      "    1.1083    0.0083   -0.2127 C   0  0  0  0  0  0  0  0  0  0  0  0",
      "    0.4038   -1.1740   -0.1422 C   0  0  0  0  0  0  0  0  0  0  0  0",
      "   -0.9783   -1.1878    0.0395 C   0  0  0  0  0  0  0  0  0  0  0  0",
      "   -1.6848   -0.0082    0.1548 C   0  0  0  0  0  0  0  0  0  0  0  0",
      "   -0.9667    1.1675    0.0823 C   0  0  0  0  0  0  0  0  0  0  0  0",
      "    0.4040    1.1892   -0.0978 C   0  0  0  0  0  0  0  0  0  0  0  0",
      "    3.1213    0.0076    0.3925 H   0  0  0  0  0  0  0  0  0  0  0  0",
      "    0.9144   -2.1260   -0.2274 H   0  0  0  0  0  0  0  0  0  0  0  0",
      "   -1.5181   -2.1041    0.0934 H   0  0  0  0  0  0  0  0  0  0  0  0",
      "   -2.7594   -0.0193    0.2961 H   0  0  0  0  0  0  0  0  0  0  0  0",
      "   -1.5030    2.1147    0.1708 H   0  0  0  0  0  0  0  0  0  0  0  0",
      "    0.9759    2.1126   -0.1557 H   0  0  0  0  0  0  0  0  0  0  0  0",
      "  1  2  1  0",
      "  2  3  2  0",
      "  3  4  1  0",
      "  4  5  2  0",
      "  5  6  1  0",
      "  6  7  2  0",
      "  7  2  1  0",
      "  1  8  1  0",
      "  3  9  1  0",
      "  4 10  1  0",
      "  5 11  1  0",
      "  6 12  1  0",
      "  7 13  1  0",
      "M  END",
  });
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  // Values from RDKit
  ArrayXd ref(mol.num_atoms());
  ref << 25.363962448022551, 5.0511280666653553, 13.89060218332973,
      17.678948233328743, 16.416166216662404, 15.153384199996067,
      13.89060218332973, 31.431006180635155, 22.936139645328353,
      23.785626298859032, 23.785626298859032, 26.334086259451073,
      21.237166338266992;

  ArrayXd sasa = shrake_rupley_sasa(mol, mol.confs()[0], 100, 1.4,
                                    internal::SrSasaMethod::kDirect);
  for (int i = 0; i < sasa.size(); ++i)
    EXPECT_NEAR(sasa[i], ref[i], 1e-3) << "Atom " << i;

  sasa = shrake_rupley_sasa(mol, mol.confs()[0], 100, 1.4,
                            internal::SrSasaMethod::kOctree);
  for (int i = 0; i < sasa.size(); ++i)
    EXPECT_NEAR(sasa[i], ref[i], 1e-3) << "Atom " << i;
}

}  // namespace
}  // namespace nuri
