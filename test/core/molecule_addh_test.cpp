//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <absl/algorithm/container.h>
#include <absl/log/globals.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/algo/crdgen.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"

namespace nuri {
namespace {
using Hyb = constants::Hybridization;
using Ord = constants::BondOrder;

testing::AssertionResult test_xh_len(const Molecule &mol, int src, int dst,
                                     double tol = 5e-3) {
  double dist = mol.distance(src, dst);
  double rcov_sum = mol[src].data().element().covalent_radius()
                    + mol[dst].data().element().covalent_radius();
  double diff = std::abs(dist - rcov_sum);

  if (diff > tol) {
    return testing::AssertionFailure()
           << src << "-" << dst << " distance " << dist << " != " << rcov_sum
           << ", tol = " << tol;
  }

  return testing::AssertionSuccess();
}

testing::AssertionResult test_axb_angle(const Molecule &mol, int x, int a,
                                        int b, double expected,
                                        double tol = 0.5) {
  const Matrix3Xd &conf = mol.confs()[0];
  Vector3d xa = (conf.col(a) - conf.col(x)).normalized(),
           xb = (conf.col(b) - conf.col(x)).normalized();

  double vcos = xa.dot(xb), vsin = xa.cross(xb).norm();
  double angle = rad2deg(std::abs(std::atan2(vsin, vcos)));
  double diff = std::abs(angle - expected);

  if (diff > tol) {
    return testing::AssertionFailure()
           << a << "-" << x << "-" << b << " angle " << angle
           << " != " << expected << ", tol = " << tol;
  }

  return testing::AssertionSuccess();
}

testing::AssertionResult validate_sp3d_angle_atom(const Molecule &mol, int x,
                                                  int a, int b, double tol) {
  testing::AssertionResult res_90 = test_axb_angle(mol, x, a, b, 90, tol);
  testing::AssertionResult res_120 = test_axb_angle(mol, x, a, b, 120, tol);
  testing::AssertionResult res_180 = test_axb_angle(mol, x, a, b, 180, tol);
  if (!res_90 && !res_120 && !res_180) {
    return testing::AssertionFailure()
           << "Angle " << a << "-" << x << "-" << b
           << " is none of 90, 120, 180: " << res_90.message() << ", "
           << res_120.message() << ", " << res_180.message();
  }

  return testing::AssertionSuccess();
}

testing::AssertionResult validate_sp3d2_angle_atom(const Molecule &mol, int x,
                                                   int a, int b, double tol) {
  testing::AssertionResult res_90 = test_axb_angle(mol, x, a, b, 90, tol);
  testing::AssertionResult res_180 = test_axb_angle(mol, x, a, b, 180, tol);
  if (!res_90 && !res_180) {
    return testing::AssertionFailure()
           << "Angle " << a << "-" << x << "-" << b
           << " is none of 90, 180: " << res_90.message() << ", "
           << res_180.message();
  }

  return testing::AssertionSuccess();
}

testing::AssertionResult addh_validate_geometry(Molecule &mol,
                                                bool optimize = true) {
  const int h_begin = mol.size();
  const int expected_size =
      absl::c_accumulate(mol, mol.size(), [](int sum, Molecule::Atom atom) {
        return sum + atom.data().implicit_hydrogens();
      });

  if (!mol.add_hydrogens(true, optimize))
    return testing::AssertionFailure() << "Failed to add hydrogens";

  if (expected_size != mol.size())
    return testing::AssertionFailure()
           << "Expected " << expected_size << " atoms, got " << mol.size();

  for (int i = 0; i < h_begin; ++i) {
    for (auto nei: mol[i]) {
      if (nei.dst().data().atomic_number() == 1) {
        testing::AssertionResult res = test_xh_len(mol, i, nei.dst().id());
        if (!res)
          return res;
      }
    }

    testing::AssertionResult (*validator)(const Molecule &, int, int, int,
                                          double);

    switch (mol[i].data().hybridization()) {
    case Hyb::kUnbound:
    case Hyb::kTerminal:
      if (mol[i].degree() > 1) {
        return testing::AssertionFailure()
               << "Hybridization mismatch: atom " << i << " has "
               << mol[i].degree() << " neighbors but is "
               << mol[i].data().hybridization();
      }
      continue;

    case Hyb::kSP:
      validator = [](const Molecule &m, int x, int a, int b, double tol) {
        return test_axb_angle(m, x, a, b, 180, tol);
      };
      break;
    case Hyb::kSP2:
      validator = [](const Molecule &m, int x, int a, int b, double tol) {
        return test_axb_angle(m, x, a, b, 120, tol);
      };
      break;
    case Hyb::kSP3:
      validator = [](const Molecule &m, int x, int a, int b, double tol) {
        return test_axb_angle(m, x, a, b, 109.5, tol);
      };
      break;
    case Hyb::kSP3D:
      if (mol[i].degree() == 3) {
        // only 90 and 180
        validator = validate_sp3d2_angle_atom;
      } else if (mol[i].degree() == 2) {
        validator = [](const Molecule &m, int x, int a, int b, double tol) {
          return test_axb_angle(m, x, a, b, 180, tol);
        };
      } else {
        validator = validate_sp3d_angle_atom;
      }
      break;
    case Hyb::kSP3D2:
      if (mol[i].degree() == 2) {
        validator = [](const Molecule &m, int x, int a, int b, double tol) {
          return test_axb_angle(m, x, a, b, 180, tol);
        };
      } else {
        validator = validate_sp3d2_angle_atom;
      }
      break;
    case Hyb::kOtherHyb:
      return testing::AssertionFailure()
             << "Unexpected hybridization: " << mol[i].data().hybridization();
    }

    for (int j = 0; j < mol.num_neighbors(i) - 1; ++j) {
      for (int k = j + 1; k < mol.num_neighbors(i); ++k) {
        if (mol[i][j].dst().data().atomic_number() != 1
            && mol[i][k].dst().data().atomic_number() != 1)
          continue;

        testing::AssertionResult res =
            validator(mol, i, mol[i][j].dst().id(), mol[i][k].dst().id(), 5);
        if (!res)
          return res;
      }
    }
  }

  return testing::AssertionSuccess();
}

TEST(AddHTerminal, NoNbe) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[1], 1, 0, Hyb::kTerminal });
  }
  mol.confs().push_back(Matrix3Xd::Zero(3, 1));

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHTerminal, HasNbe) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 1, 0, Hyb::kTerminal });
  }
  mol.confs().push_back(Matrix3Xd::Zero(3, 1));

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp, Fixed0) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[4], 2, 0, Hyb::kSP });
    mol.confs().emplace_back(Matrix3Xd::Zero(3, 1));
  }

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp, Fixed1) {
  const double cn_bl = 1.16;

  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[6], 1, 0, Hyb::kSP });
    mut.add_atom({ kPt[7], 0, 0, Hyb::kTerminal });
    mut.add_bond(0, 1, BondData(Ord::kTripleBond));

    mol.confs().emplace_back(Matrix3Xd::Zero(3, 2)).col(1) =
        Vector3d::Random().normalized() * cn_bl;
  }

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp2, Fixed0E01) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[5], 3, 0, Hyb::kSP2 });
    mut.add_atom({ kPt[6], 2, 0, Hyb::kSP2 });

    mol.confs().emplace_back(Matrix3Xd::Zero(3, 2)).col(1) =
        Vector3d::UnitZ() * 5;
  }

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp2, Fixed1) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[5], 2, 0, Hyb::kSP2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(0, 1, BondData(Ord::kSingleBond));

    mol.confs().emplace_back(Matrix3Xd::Zero(3, 2)).col(1) =
        Vector3d::Random().normalized()
        * (kPt[5].covalent_radius() + kPt[9].covalent_radius());
  }

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp2, Fixed1E1) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[6], 1, 0, Hyb::kSP2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(0, 1, BondData(Ord::kSingleBond));

    mol.confs().emplace_back(Matrix3Xd::Zero(3, 2)).col(1) =
        Vector3d::Random().normalized()
        * (kPt[6].covalent_radius() + kPt[9].covalent_radius());
  }

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp2, Fixed2) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[5], 1, 0, Hyb::kSP2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
  }

  ASSERT_TRUE(generate_coords(mol));
  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3, Fixed0) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[6], 4, 0, Hyb::kSP3 });

    mol.confs().emplace_back(Matrix3Xd::Zero(3, 1));
  }

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3, Fixed0E1) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[7], 3, 0, Hyb::kSP3 });
  }

  mol.confs().emplace_back(Matrix3Xd::Zero(3, 1));

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3, Fixed0E2) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[8], 2, 0, Hyb::kSP3 });
  }

  mol.confs().emplace_back(Matrix3Xd::Zero(3, 1));

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3, Fixed123) {
  Molecule mol = internal::read_first("smi", "CC(CC)C");
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  mol.confs().emplace_back(3, mol.size()).transpose() << -0.0127, 1.0858,
      0.0080, -0.7288, 1.5792, 1.2668, 0.0111, 1.0724, 2.5063, -0.0755, -0.4541,
      2.5634, -0.7496, 3.1091, 1.2752;

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3, Fixed1E12) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[7], 2, 0, Hyb::kSP3 });
    mut.add_atom({ kPt[8], 1, 0, Hyb::kSP3 });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
  }

  mol.confs().emplace_back(Matrix3Xd::Zero(3, 2)).col(1) =
      Vector3d::Random().normalized()
      * (kPt[7].covalent_radius() + kPt[8].covalent_radius());

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3, Fixed2E1) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[7], 1, 0, Hyb::kSP3 });
    mut.add_atom({ kPt[6], 3, 0, Hyb::kSP3 });
    mut.add_atom({ kPt[6], 3, 0, Hyb::kSP3 });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
  }

  ASSERT_TRUE(generate_coords(mol));
  EXPECT_TRUE(addh_validate_geometry(mol));
}

// artificial molecules used for testing

TEST(AddHSp3d, Fixed0E0123) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[15], 5, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[15], 4, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[15], 3, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[15], 2, 0, Hyb::kSP3D });
  }

  mol.confs().emplace_back(Matrix3Xd::Zero(3, 4)).row(2) << -5, 0, 5, 10;
  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed1E0123) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[15], 4, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(0, 1, BondData(Ord::kSingleBond));

    mut.add_atom({ kPt[15], 3, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));

    mut.add_atom({ kPt[15], 2, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(4, 5, BondData(Ord::kSingleBond));

    mut.add_atom({ kPt[15], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(6, 7, BondData(Ord::kSingleBond));
  }

  double bl = kPt[15].covalent_radius() + kPt[9].covalent_radius();
  mol.confs().emplace_back(Matrix3Xd::Zero(3, 8)).row(2) << -10, bl - 10, 0, bl,
      10, 10 + bl, 20, 20 + bl;
  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed2AxE012) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[15], 3, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[15], 2, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[15], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(1, 2, BondData(Ord::kSingleBond));
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
  }

  double ble = kPt[15].covalent_radius() + kPt[9].covalent_radius(),
         bli = kPt[15].covalent_radius() * 2;
  mol.confs().emplace_back(Matrix3Xd::Zero(3, 5)).row(2) << 0, ble, ble + bli,
      ble + bli * 2, ble * 2 + bli * 2;

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed2AxEqE012) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[15], 3, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[17], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(1, 2, BondData(Ord::kSingleBond));
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
  }

  mol.confs().emplace_back(3, 5).transpose() << 3.6415, 3.7358, -0.0126, 1.9718,
      3.7130, 0.0032, 2.0005, 1.5933, -0.0085, -0.0192, 1.5658, 0.0107, 0.0021,
      -0.0041, 0.0020;

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed2EqE012) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[15], 3, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[17], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(1, 2, BondData(Ord::kSingleBond));
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
  }

  Array4d bls { kPt[9].covalent_radius() + kPt[15].covalent_radius(),
                kPt[15].covalent_radius() + kPt[16].covalent_radius(),
                kPt[16].covalent_radius() + kPt[17].covalent_radius(),
                kPt[17].covalent_radius() + kPt[9].covalent_radius() };

  Vector3d x = Vector3d::UnitX(),
           xy1 = { constants::kCos60, constants::kCos30, 0 };
  Matrix3Xd &conf = mol.confs().emplace_back(Array3Xd::Zero(3, 5));
  conf.col(0) = -x * bls[0];
  conf.col(2) = xy1 * bls[1];
  conf.col(3) = conf.col(2) + x * bls[2];
  conf.col(4) = conf.col(3) + xy1 * bls[3];

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed3Eq3E01) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[15], 2, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(1, 2, BondData(Ord::kSingleBond));
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
    mut.add_bond(3, 5, BondData(Ord::kSingleBond));
  }

  Array3d bls { kPt[9].covalent_radius() + kPt[15].covalent_radius(),
                kPt[15].covalent_radius() + kPt[16].covalent_radius(),
                kPt[16].covalent_radius() + kPt[9].covalent_radius() };

  Vector3d y = Vector3d::UnitY(),
           xy1 = { -constants::kCos30, -constants::kCos60, 0 },
           xy2 = { +constants::kCos30, -constants::kCos60, 0 };

  Matrix3Xd &conf = mol.confs().emplace_back(Array3Xd::Zero(3, 6));
  conf.col(0) = y * bls[0];
  conf.col(1) = xy1 * bls[0];
  conf.col(3) = xy2 * bls[1];
  conf.col(4) = conf.col(3) - y * bls[2];
  conf.col(5) = conf.col(3) - xy1 * bls[2];

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed3Eq2E01) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[15], 2, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(1, 2, BondData(Ord::kSingleBond));
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
    mut.add_bond(3, 5, BondData(Ord::kSingleBond));
  }

  mol.confs().emplace_back(3, 6).transpose() << -2.5352, 1.5770, -0.9663,
      -0.3977, 3.0594, -2.2313, -1.0971, 1.6012, -1.8150, -0.0198, 1.6058,
      0.0109, 0.0021, -0.0041, 0.0020, 1.5900, 1.6277, -0.0044;

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed3Eq1E01) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[15], 2, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(1, 2, BondData(Ord::kSingleBond));
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
    mut.add_bond(3, 5, BondData(Ord::kSingleBond));
  }

  Array3d bls { kPt[9].covalent_radius() + kPt[15].covalent_radius(),
                kPt[15].covalent_radius() + kPt[16].covalent_radius(),
                kPt[16].covalent_radius() + kPt[9].covalent_radius() };

  Vector3d x = Vector3d::UnitX(), y = Vector3d::UnitY();

  Matrix3Xd &conf = mol.confs().emplace_back(Array3Xd::Zero(3, 6));
  conf.col(0) = y * bls[0];
  conf.col(1) = -x * bls[0];
  conf.col(3) = -y * bls[1];
  conf.col(4) = conf.col(3) + x * bls[2];
  conf.col(5) = conf.col(3) - y * bls[2];

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed4Ax) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[15], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[17], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[17], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[17], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[17], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(0, 3, BondData(Ord::kSingleBond));
    mut.add_bond(0, 4, BondData(Ord::kSingleBond));
  }

  mol.confs().emplace_back(Matrix3Xd::Zero(3, 5)).transpose() << -0.0261,
      2.0757, 0.0135, 0.0021, -0.0041, 0.0020, 2.0536, 2.1040, -0.0062, -1.0832,
      2.0712, -1.7779, -1.0488, 2.0519, 1.8245;

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d, Fixed4Eq) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[15], 1, 0, Hyb::kSP3D });
    mut.add_atom({ kPt[17], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[17], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[17], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[17], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(0, 3, BondData(Ord::kSingleBond));
    mut.add_bond(0, 4, BondData(Ord::kSingleBond));
  }

  mol.confs().emplace_back(Matrix3Xd::Zero(3, 5)).transpose() << -0.0261,
      2.0757, 0.0135, 0.0021, -0.0041, 0.0020, -0.0543, 4.1555, 0.0249, 2.0536,
      2.1040, -0.0062, -1.0832, 2.0712, -1.7779;

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d2, Fixed0E01234) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[16], 6, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 5, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 4, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 3, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D2 });
  }

  mol.confs().emplace_back(Matrix3Xd::Zero(3, 5)).row(2) << -10, -5, 0, 5, 10;
  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d2, Fixed1E01234) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[16], 5, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(0, 1, BondData(Ord::kSingleBond));

    mut.add_atom({ kPt[16], 4, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));

    mut.add_atom({ kPt[16], 3, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(4, 5, BondData(Ord::kSingleBond));

    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(6, 7, BondData(Ord::kSingleBond));

    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_bond(8, 9, BondData(Ord::kSingleBond));
  }

  double bl = kPt[16].covalent_radius() + kPt[9].covalent_radius();
  mol.confs().emplace_back(Matrix3Xd::Zero(3, 10)).row(2) << -10, bl - 10, -5,
      bl - 5, 0, bl, 5, 5 + bl, 10, 10 + bl;
  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d2, Fixed2AxE0123) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[16], 4, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 3, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(1, 2, BondData(Ord::kSingleBond));
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
    mut.add_bond(4, 5, BondData(Ord::kSingleBond));
  }

  double ble = kPt[16].covalent_radius() + kPt[9].covalent_radius(),
         bli = kPt[16].covalent_radius() * 2;
  mol.confs().emplace_back(Matrix3Xd::Zero(3, 6)).row(2) << 0, ble, ble + bli,
      ble + bli * 2, ble + bli * 3, ble * 2 + bli * 3;
  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSp3d2, Fixed2AxEqE0123) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[16], 4, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 3, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(1, 2, BondData(Ord::kSingleBond));
    mut.add_bond(2, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
    mut.add_bond(4, 5, BondData(Ord::kSingleBond));
  }

  Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd::Zero(3, 6));

  Vector3d x = Vector3d::UnitX(), y = Vector3d::UnitY();
  double ble = kPt[16].covalent_radius() + kPt[9].covalent_radius(),
         bli = kPt[16].covalent_radius() * 2;
  conf.col(0) = y * ble;
  conf.col(2) = x * bli;
  conf.col(3) = conf.col(2) - y * bli;
  conf.col(4) = conf.col(3) + x * bli;
  conf.col(5) = conf.col(4) - y * ble;

  EXPECT_TRUE(addh_validate_geometry(mol, false));
}

TEST(AddHSp3d2, Fixed3MerE012) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[16], 3, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(0, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
    mut.add_bond(3, 5, BondData(Ord::kSingleBond));
    mut.add_bond(5, 6, BondData(Ord::kSingleBond));
    mut.add_bond(5, 7, BondData(Ord::kSingleBond));
  }

  Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd::Zero(3, 8));

  Vector3d x = Vector3d::UnitX(), y = Vector3d::UnitY(), z = Vector3d::UnitZ();
  double ble = kPt[16].covalent_radius() + kPt[9].covalent_radius(),
         bli = kPt[16].covalent_radius() * 2;

  conf.col(1) = -z * ble;
  conf.col(2) = -x * ble;

  conf.col(3) = z * bli;
  conf.col(4) = conf.col(3) + y * ble;

  conf.col(5) = conf.col(3) + z * bli;
  conf.col(6) = conf.col(5) + z * ble;
  conf.col(7) = conf.col(5) + x * ble;

  EXPECT_TRUE(addh_validate_geometry(mol, false));
}

TEST(AddHSp3d2, Fixed3FacE012) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[16], 3, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(0, 3, BondData(Ord::kSingleBond));
    mut.add_bond(3, 4, BondData(Ord::kSingleBond));
    mut.add_bond(3, 5, BondData(Ord::kSingleBond));
    mut.add_bond(5, 6, BondData(Ord::kSingleBond));
    mut.add_bond(5, 7, BondData(Ord::kSingleBond));
  }

  Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd::Zero(3, 8));

  Vector3d x = Vector3d::UnitX(), y = Vector3d::UnitY(), z = Vector3d::UnitZ();
  double ble = kPt[16].covalent_radius() + kPt[9].covalent_radius(),
         bli = kPt[16].covalent_radius() * 2;

  conf.col(1) = -y * ble;
  conf.col(2) = -x * ble;

  conf.col(3) = z * bli;
  conf.col(4) = conf.col(3) + y * ble;
  conf.col(5) = conf.col(3) + x * bli;

  conf.col(6) = conf.col(5) + z * ble;
  conf.col(7) = conf.col(5) + y * ble;

  EXPECT_TRUE(addh_validate_geometry(mol, false));
}

TEST(AddHSp3d2, Fixed4PlanarE01) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(0, 3, BondData(Ord::kSingleBond));
    mut.add_bond(0, 4, BondData(Ord::kSingleBond));
    mut.add_bond(4, 5, BondData(Ord::kSingleBond));
    mut.add_bond(4, 6, BondData(Ord::kSingleBond));
    mut.add_bond(4, 7, BondData(Ord::kSingleBond));
  }

  Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd::Zero(3, 8));

  Vector3d x = Vector3d::UnitX(), y = Vector3d::UnitY();
  double ble = kPt[16].covalent_radius() + kPt[9].covalent_radius(),
         bli = kPt[16].covalent_radius() * 2;

  conf.col(1) = -y * ble;
  conf.col(2) = -x * ble;
  conf.col(3) = y * ble;

  conf.col(4) = x * bli;
  conf.col(5) = conf.col(4) + x * bli;
  conf.col(6) = conf.col(4) - y * ble;
  conf.col(7) = conf.col(4) + y * ble;

  EXPECT_TRUE(addh_validate_geometry(mol, false));
}

TEST(AddHSp3d2, Fixed4NonplanarE01) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[16], 2, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(0, 3, BondData(Ord::kSingleBond));
    mut.add_bond(0, 4, BondData(Ord::kSingleBond));
    mut.add_bond(4, 5, BondData(Ord::kSingleBond));
    mut.add_bond(4, 6, BondData(Ord::kSingleBond));
    mut.add_bond(4, 7, BondData(Ord::kSingleBond));
  }

  Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd::Zero(3, 8));

  Vector3d x = Vector3d::UnitX(), y = Vector3d::UnitY(), z = Vector3d::UnitZ();
  double ble = kPt[16].covalent_radius() + kPt[9].covalent_radius(),
         bli = kPt[16].covalent_radius() * 2;

  conf.col(1) = -y * ble;
  conf.col(2) = -x * ble;
  conf.col(3) = z * ble;

  conf.col(4) = x * bli;
  conf.col(5) = conf.col(4) + x * bli;
  conf.col(6) = conf.col(4) - y * ble;
  conf.col(7) = conf.col(4) + z * ble;

  EXPECT_TRUE(addh_validate_geometry(mol, false));
}

TEST(AddHSp3d2, Fixed5E0) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom({ kPt[16], 1, 0, Hyb::kSP3D2 });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });
    mut.add_atom({ kPt[9], 0, 0, Hyb::kTerminal });

    mut.add_bond(0, 1, BondData(Ord::kSingleBond));
    mut.add_bond(0, 2, BondData(Ord::kSingleBond));
    mut.add_bond(0, 3, BondData(Ord::kSingleBond));
    mut.add_bond(0, 4, BondData(Ord::kSingleBond));
    mut.add_bond(0, 5, BondData(Ord::kSingleBond));
  }

  Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd::Zero(3, 6));

  Vector3d x = Vector3d::UnitX(), y = Vector3d::UnitY(), z = Vector3d::UnitZ();
  double bl = kPt[16].covalent_radius() + kPt[9].covalent_radius();

  conf.col(1) = -x * bl;
  conf.col(2) = -y * bl;
  conf.col(3) = -z * bl;
  conf.col(4) = x * bl;
  conf.col(5) = y * bl;

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSample, MixedSp2Sp3) {
  Molecule mol = internal::read_first("smi", "CCCC(C)=C(C(C=C)O)N");
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  mol.confs().emplace_back(3, mol.size()).transpose() << -2.2563, 0.7428,
      -2.6736, -0.7895, 1.1676, -2.7686, 0.0930, 0.1038, -2.1124, 1.5378,
      0.5222, -2.2060, 2.3467, 0.1698, -3.4277, 2.0947, 1.1973, -1.2179, 1.2515,
      1.6796, -0.0657, 1.2721, 3.1859, -0.0242, 0.1545, 3.8597, -0.1390, 1.7768,
      1.1621, 1.1583, 3.4579, 1.4631, -1.2401;

  EXPECT_TRUE(addh_validate_geometry(mol));
}

TEST(AddHSubstr, WillAddH) {
  Molecule mol = internal::read_first("smi", "CCCC");
  ASSERT_TRUE(MoleculeSanitizer(mol).sanitize_all());

  mol.substructures().push_back(mol.atom_substructure({ 0 }));
  mol.substructures().push_back(mol.atom_substructure({ 0, 1 }));
  mol.substructures().push_back(mol.atom_substructure({ 2 }));

  ASSERT_TRUE(mol.add_hydrogens());

  for (const auto &sub: mol.substructures()) {
    for (auto sa: sub) {
      int shcnt = absl::c_count_if(sa, [&](Substructure::Neighbor n) {
        return n.dst().data().atomic_number() == 1;
      });
      int hcnt = absl::c_count_if(sa.as_parent(), [&](Molecule::Neighbor n) {
        return n.dst().data().atomic_number() == 1;
      });

      EXPECT_EQ(shcnt, hcnt) << sa.as_parent().id();
    }
  }
}
}  // namespace
}  // namespace nuri
