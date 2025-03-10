
#include "nuri/fmt/sdf.h"

#include <string>
#include <string_view>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "fmt_test_common.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/core/property_map.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
using SDFTest = internal::StringFormatTest<SDFReader>;

TEST_F(SDFTest, BasicMolecule) {
  set_test_string(R"sdf(L-Alanine
  ABCDEFGH09071717443D
Exported
  6  5  0  0  1  0              3 V2000
   -0.6622    0.5342    0.0000 C   0  0  2  0  0  0
    0.6622   -0.3000    0.0000 C   0  0  0  0  0  0
   -0.7207    2.0817    0.0000 C   1  0  0  0  0  0
   -1.8622   -0.3695    0.0000 N   0  3  0  0  0  0
    0.6220   -1.8037    0.0000 O   0  0  0  0  0  0
    1.9464    0.4244    0.0000 O   0  5  0
  1  2  1  0  0  0
  1  3  1  1  0  0
  1  4  1  0  0  0
  2  5  2  0  0  0
  2  6  1  0  0  0
M  CHG  2   4   1   6  -1
M  ISO  1   3  13
M  END
> 25  <MELTING.POINT>
179.0 - 183.0

> 25  <DESCRIPTION>
PW(W)

$$$$
L-Alanine
 OpenBabel03182412213D
Exported
  0  0  0     0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 6 5 0 0 1
M  V30 BEGIN ATOM
M  V30 1 C -0.6622 0.5342 0 0
M  V30 2 C 0.6622 -0.3 0 0
M  V30 3 C -0.7207 2.0817 0 0 MASS=13
M  V30 4 N -1.8622 -0.3695 0 0 CHG=1
M  V30 5 O 0.622 -1.8037 0 0
M  V30 6 O 1.9464 0.4244 0 0 CHG=-1
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 1 3 CFG=1
M  V30 3 1 1 4
M  V30 4 2 2 5
M  V30 5 1 2 6
M  V30 END BOND
M  V30 END CTAB
M  END
>  <MELTING.POINT>
179.0 - 183.0

>  <DESCRIPTION>
PW(W)

$$$$
)sdf");

  auto verify_mol = [this]() {
    EXPECT_TRUE(mol()[1].data().is_conjugated());

    EXPECT_EQ(mol()[2].data().isotope().mass_number, 13);

    EXPECT_EQ(mol()[3].data().formal_charge(), +1);
    EXPECT_EQ(mol()[3].data().implicit_hydrogens(), 3);

    EXPECT_TRUE(mol()[4].data().is_conjugated());

    EXPECT_EQ(mol()[5].data().formal_charge(), -1);
    EXPECT_TRUE(mol()[5].data().is_conjugated());

    auto mp = internal::find_key(mol().props(), "MELTING.POINT");
    ASSERT_NE(mp, mol().props().end());
    EXPECT_EQ(mp->second, "179.0 - 183.0");

    auto desc = internal::find_key(mol().props(), "DESCRIPTION");
    ASSERT_NE(desc, mol().props().end());
    EXPECT_EQ(desc->second, "PW(W)");
  };

  std::string sdf;

  for (int vers = 2000; vers <= 3000; vers += 1000) {
    NURI_FMT_TEST_NEXT_MOL("L-Alanine", 6, 5);

    SCOPED_TRACE(absl::StrCat("Initial read, version: ", vers));
    verify_mol();

    bool success = write_sdf(
        sdf, mol(), -1, vers == 2000 ? SDFVersion::kV2000 : SDFVersion::kV3000);
    ASSERT_TRUE(success);
  }

  set_test_string(sdf);

  for (int vers = 2000; vers <= 3000; vers += 1000) {
    NURI_FMT_TEST_NEXT_MOL("L-Alanine", 6, 5);

    SCOPED_TRACE(absl::StrCat("Re-read, version: ", vers));
    verify_mol();
  }
}

TEST_F(SDFTest, HistidineCharged) {
  set_test_string(R"sdf(HIS
  -OEChem-03082421403D

 11 11  0     1  0  0  0  0  0999 V2000
   33.4720   42.6850   -4.6100 N   0  0  0  0  0  0  0  0  0  0  0  0
   33.4140   41.6860   -5.6730 C   0  0  2  0  0  0  0  0  0  0  0  0
   33.7730   42.2790   -7.0400 C   0  0  0  0  0  0  0  0  0  0  0  0
   33.4970   43.4440   -7.3370 O   0  0  0  0  0  0  0  0  0  0  0  0
   32.0050   41.0800   -5.7340 C   0  0  0  0  0  0  0  0  0  0  0  0
   31.8880   39.9020   -6.6510 C   0  0  0  0  0  0  0  0  0  0  0  0
   32.5390   38.7100   -6.4140 N   0  3  0  0  0  0  0  0  0  0  0  0
   31.1990   39.7340   -7.8040 C   0  0  0  0  0  0  0  0  0  0  0  0
   32.2510   37.8570   -7.3820 C   0  0  0  0  0  0  0  0  0  0  0  0
   31.4390   38.4530   -8.2370 N   0  0  0  0  0  0  0  0  0  0  0  0
   34.3820   41.4550   -7.8790 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  5  1  0  0  0  0
  3  4  2  0  0  0  0
  3 11  1  0  0  0  0
  5  6  1  0  0  0  0
  6  7  1  0  0  0  0
  6  8  2  0  0  0  0
  7  9  2  0  0  0  0
  8 10  1  0  0  0  0
  9 10  1  0  0  0  0
M  CHG  1   7   1
M  END
$$$$
HIS
 OpenBabel03182412583D

  0  0  0     0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 11 11 0 0 1
M  V30 BEGIN ATOM
M  V30 1 N 33.472 42.685 -4.61 0
M  V30 2 C 33.414 41.686 -5.673 0
M  V30 3 C 33.773 42.279 -7.04 0
M  V30 4 O 33.497 43.444 -7.337 0
M  V30 5 C 32.005 41.08 -5.734 0
M  V30 6 C 31.888 39.902 -6.651 0
M  V30 7 N 32.539 38.71 -6.414 0 CHG=1
M  V30 8 C 31.199 39.734 -7.804 0
M  V30 9 C 32.251 37.857 -7.382 0
M  V30 10 N 31.439 38.453 -8.237 0
M  V30 11 O 34.382 41.455 -7.879 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 2 3
M  V30 3 1 2 5
M  V30 4 2 3 4
M  V30 5 1 3 11
M  V30 6 1 5 6
M  V30 7 1 6 7
M  V30 8 2 6 8
M  V30 9 2 7 9
M  V30 10 1 8 10
M  V30 11 1 9 10
M  V30 END BOND
M  V30 END CTAB
M  END
$$$$
)sdf");

  auto verify_mol = [this]() {
    EXPECT_EQ(mol()[6].data().atomic_number(), 7);
    EXPECT_TRUE(mol()[6].data().is_aromatic());
    EXPECT_EQ(mol()[6].data().hybridization(), constants::kSP2);
    EXPECT_EQ(mol()[6].data().implicit_hydrogens(), 1);
    EXPECT_EQ(mol()[6].data().formal_charge(), 1);

    EXPECT_EQ(mol()[9].data().atomic_number(), 7);
    EXPECT_TRUE(mol()[9].data().is_aromatic());
    EXPECT_EQ(mol()[9].data().hybridization(), constants::kSP2);
    EXPECT_EQ(mol()[9].data().implicit_hydrogens(), 1);
  };

  std::string sdf;

  for (int vers = 2000; vers <= 3000; vers += 1000) {
    NURI_FMT_TEST_NEXT_MOL("HIS", 11, 11);

    SCOPED_TRACE(absl::StrCat("Initial read, version: ", vers));
    verify_mol();

    bool success = write_sdf(
        sdf, mol(), -1, vers == 2000 ? SDFVersion::kV2000 : SDFVersion::kV3000);
    ASSERT_TRUE(success);
  }

  set_test_string(sdf);

  for (int vers = 2000; vers <= 3000; vers += 1000) {
    NURI_FMT_TEST_NEXT_MOL("HIS", 11, 11);

    SCOPED_TRACE(absl::StrCat("Re-read, version: ", vers));
    verify_mol();
  }
}

TEST_F(SDFTest, HistidineNeutral) {
  set_test_string(R"sdf(HIS
  -OEChem-03082421403D

 11 11  0     1  0  0  0  0  0999 V2000
   33.4720   42.6850   -4.6100 N   0  0  0  0  0  0  0  0  0  0  0  0
   33.4140   41.6860   -5.6730 C   0  0  2  0  0  0  0  0  0  0  0  0
   33.7730   42.2790   -7.0400 C   0  0  0  0  0  0  0  0  0  0  0  0
   33.4970   43.4440   -7.3370 O   0  0  0  0  0  0  0  0  0  0  0  0
   32.0050   41.0800   -5.7340 C   0  0  0  0  0  0  0  0  0  0  0  0
   31.8880   39.9020   -6.6510 C   0  0  0  0  0  0  0  0  0  0  0  0
   32.5390   38.7100   -6.4140 N   0  0  0  0  0  0  0  0  0  0  0  0
   31.1990   39.7340   -7.8040 C   0  0  0  0  0  0  0  0  0  0  0  0
   32.2510   37.8570   -7.3820 C   0  0  0  0  0  0  0  0  0  0  0  0
   31.4390   38.4530   -8.2370 N   0  0  0  0  0  0  0  0  0  0  0  0
   34.3820   41.4550   -7.8790 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  5  1  0  0  0  0
  3  4  2  0  0  0  0
  3 11  1  0  0  0  0
  5  6  1  0  0  0  0
  6  7  1  0  0  0  0
  6  8  2  0  0  0  0
  7  9  2  0  0  0  0
  8 10  1  0  0  0  0
  9 10  1  0  0  0  0
M  END
$$$$
HIS
 OpenBabel03182412583D

  0  0  0     0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 11 11 0 0 1
M  V30 BEGIN ATOM
M  V30 1 N 33.472 42.685 -4.61 0
M  V30 2 C 33.414 41.686 -5.673 0
M  V30 3 C 33.773 42.279 -7.04 0
M  V30 4 O 33.497 43.444 -7.337 0
M  V30 5 C 32.005 41.08 -5.734 0
M  V30 6 C 31.888 39.902 -6.651 0
M  V30 7 N 32.539 38.71 -6.414 0
M  V30 8 C 31.199 39.734 -7.804 0
M  V30 9 C 32.251 37.857 -7.382 0
M  V30 10 N 31.439 38.453 -8.237 0
M  V30 11 O 34.382 41.455 -7.879 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 2 3
M  V30 3 1 2 5
M  V30 4 2 3 4
M  V30 5 1 3 11
M  V30 6 1 5 6
M  V30 7 1 6 7
M  V30 8 2 6 8
M  V30 9 2 7 9
M  V30 10 1 8 10
M  V30 11 1 9 10
M  V30 END BOND
M  V30 END CTAB
M  END
$$$$
)sdf");

  auto verify_mol = [this]() {
    EXPECT_EQ(mol()[6].data().atomic_number(), 7);
    EXPECT_TRUE(mol()[6].data().is_aromatic());
    EXPECT_EQ(mol()[6].data().hybridization(), constants::kSP2);
    EXPECT_EQ(mol()[6].data().implicit_hydrogens(), 0);
    EXPECT_EQ(mol()[6].data().formal_charge(), 0);

    EXPECT_EQ(mol()[9].data().atomic_number(), 7);
    EXPECT_TRUE(mol()[9].data().is_aromatic());
    EXPECT_EQ(mol()[9].data().hybridization(), constants::kSP2);
    EXPECT_EQ(mol()[9].data().implicit_hydrogens(), 1);
  };

  std::string sdf;

  for (int vers = 2000; vers <= 3000; vers += 1000) {
    NURI_FMT_TEST_NEXT_MOL("HIS", 11, 11);

    SCOPED_TRACE(absl::StrCat("Initial read, version: ", vers));
    verify_mol();

    bool success = write_sdf(
        sdf, mol(), -1, vers == 2000 ? SDFVersion::kV2000 : SDFVersion::kV3000);
    ASSERT_TRUE(success);
  }

  set_test_string(sdf);

  for (int vers = 2000; vers <= 3000; vers += 1000) {
    NURI_FMT_TEST_NEXT_MOL("HIS", 11, 11);

    SCOPED_TRACE(absl::StrCat("Re-read, version: ", vers));
    verify_mol();
  }
}

TEST_F(SDFTest, Tryptophan) {
  set_test_string(R"sdf(TRP
  -OEChem-03012415303D

 15 16  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 N   0  0  0  0  0  0  0  0  0  0  0  0
   74.4000   61.7350   32.1140 C   0  0  2  0  0  0  0  0  0  0  0  0
   73.5880   61.4110   30.8400 C   0  0  0  0  0  0  0  0  0  0  0  0
   72.9390   62.2920   30.2770 O   0  0  0  0  0  0  0  0  0  0  0  0
   75.6840   62.4730   31.7060 C   0  0  0  0  0  0  0  0  0  0  0  0
   76.6750   62.7270   32.8320 C   0  0  0  0  0  0  0  0  0  0  0  0
   77.7530   61.9640   33.1570 C   0  0  0  0  0  0  0  0  0  0  0  0
   76.6460   63.8050   33.7770 C   0  0  0  0  0  0  0  0  0  0  0  0
   78.4030   62.4940   34.2470 N   0  0  0  0  0  0  0  0  0  0  0  0
   77.7410   63.6250   34.6500 C   0  0  0  0  0  0  0  0  0  0  0  0
   75.7960   64.9020   33.9740 C   0  0  0  0  0  0  0  0  0  0  0  0
   78.0140   64.4990   35.7090 C   0  0  0  0  0  0  0  0  0  0  0  0
   76.0650   65.7760   35.0310 C   0  0  0  0  0  0  0  0  0  0  0  0
   77.1680   65.5650   35.8840 C   0  0  0  0  0  0  0  0  0  0  0  0
   73.4950   60.4700   30.4380 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  5  1  0  0  0  0
  3  4  2  0  0  0  0
  3 15  1  0  0  0  0
  5  6  1  0  0  0  0
  6  7  2  0  0  0  0
  6  8  1  0  0  0  0
  7  9  1  0  0  0  0
  8 10  2  0  0  0  0
  8 11  1  0  0  0  0
  9 10  1  0  0  0  0
 10 12  1  0  0  0  0
 11 13  2  0  0  0  0
 12 14  2  0  0  0  0
 13 14  1  0  0  0  0
M  END
$$$$
TRP
 OpenBabel03182413003D

  0  0  0     0  0            999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 15 16 0 0 1
M  V30 BEGIN ATOM
M  V30 1 N 74.708 60.512 32.843 0
M  V30 2 C 74.4 61.735 32.114 0
M  V30 3 C 73.588 61.411 30.84 0
M  V30 4 O 72.939 62.292 30.277 0
M  V30 5 C 75.684 62.473 31.706 0
M  V30 6 C 76.675 62.727 32.832 0
M  V30 7 C 77.753 61.964 33.157 0
M  V30 8 C 76.646 63.805 33.777 0
M  V30 9 N 78.403 62.494 34.247 0
M  V30 10 C 77.741 63.625 34.65 0
M  V30 11 C 75.796 64.902 33.974 0
M  V30 12 C 78.014 64.499 35.709 0
M  V30 13 C 76.065 65.776 35.031 0
M  V30 14 C 77.168 65.565 35.884 0
M  V30 15 O 73.495 60.47 30.438 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 2 3
M  V30 3 1 2 5
M  V30 4 2 3 4
M  V30 5 1 3 15
M  V30 6 1 5 6
M  V30 7 2 6 7
M  V30 8 1 6 8
M  V30 9 1 7 9
M  V30 10 2 8 10
M  V30 11 1 8 11
M  V30 12 1 9 10
M  V30 13 1 10 12
M  V30 14 2 11 13
M  V30 15 2 12 14
M  V30 16 1 13 14
M  V30 END BOND
M  V30 END CTAB
M  END
$$$$
)sdf");

  auto verify_mol = [this]() {
    EXPECT_EQ(mol()[8].data().atomic_number(), 7);
    EXPECT_TRUE(mol()[8].data().is_aromatic());
    EXPECT_EQ(mol()[8].data().hybridization(), constants::kSP2);
    EXPECT_EQ(mol()[8].data().implicit_hydrogens(), 1);
  };

  std::string sdf;

  for (int vers = 2000; vers <= 3000; vers += 1000) {
    NURI_FMT_TEST_NEXT_MOL("TRP", 15, 16);

    SCOPED_TRACE(absl::StrCat("Initial read, version: ", vers));
    verify_mol();

    bool success = write_sdf(
        sdf, mol(), -1, vers == 2000 ? SDFVersion::kV2000 : SDFVersion::kV3000);
    ASSERT_TRUE(success);
  }

  set_test_string(sdf);

  for (int vers = 2000; vers <= 3000; vers += 1000) {
    NURI_FMT_TEST_NEXT_MOL("TRP", 15, 16);

    SCOPED_TRACE(absl::StrCat("Re-read, version: ", vers));
    verify_mol();
  }
}

TEST_F(SDFTest, SpecialHydrogens) {
  set_test_string(R"sdf(D


  1  0  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 D   0  0  0  0  0  0  0  0  0  0  0  0
M END
$$$$
T


  1  0  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 T   0  0  0  0  0  0  0  0  0  0  0  0
M END
$$$$
)sdf");

  NURI_FMT_TEST_NEXT_MOL("D", 1, 0);
  EXPECT_EQ(mol()[0].data().isotope().mass_number, 2);

  NURI_FMT_TEST_NEXT_MOL("T", 1, 0);
  EXPECT_EQ(mol()[0].data().isotope().mass_number, 3);
}

TEST_F(SDFTest, V3000Continuation) {
  set_test_string(R"sdf(test


  1  0  0     0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 1 0 0 0 0
M  V30 BEGIN ATOM
M  V30 1 N 74.708 60.512 32.843 0 CHG=1 -
M  V30  CHG=2
M  V30 END ATOM
M  V30 END CTAB
$$$$
test


  1  0  0     0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 1 0 0 0 0
M  V30 BEGIN ATOM
M  V30 1 N 74.708 60.512 32.843 0 CHG=1 -
M  V30  CHG=2 -
M  V30  MASS=12
M  V30 END ATOM
M  V30 END CTAB
$$$$
)sdf");

  NURI_FMT_TEST_NEXT_MOL("test", 1, 0);

  EXPECT_EQ(mol()[0].data().atomic_number(), 7);
  EXPECT_EQ(mol()[0].data().formal_charge(), 2);

  NURI_FMT_TEST_NEXT_MOL("test", 1, 0);

  EXPECT_EQ(mol()[0].data().atomic_number(), 7);
  EXPECT_EQ(mol()[0].data().formal_charge(), 2);
  EXPECT_EQ(mol()[0].data().isotope().mass_number, 12);
}

TEST_F(SDFTest, WarningsNoFailure) {
  set_test_string(R"sdf(test


  1  0  0     0  0  0  0  0  0999 V2000
   -0.6622    0.5342    0.0000 C   0  0  0  0  0  0
M END
test invalid
> 25  <MELTING.POINT>
179.0 - 183.0

$$$$
test


  1  0  0     0  0  0  0  0  0999 V2000
   -0.6622    0.5342    0.0000 C   0  0  0  0  0  0
M END
> MELTING.POINT
179.0 - 183.0

$$$$
test


  1  0  0     0  0  0  0  0  0999 V2000
   -0.6622    0.5342    0.0000 C   0  0  0  0  0  0
> 25  <MELTING.POINT>
179.0 - 183.0

$$$$
test


  0  0  0     0  0  0  0  0  0999 V2000
   -0.6622    0.5342    0.0000 C   0  0  0  0  0  0
   -0.6622    0.5342    0.0000 C   0  0  0  0  0  0

  1  2  1  0  0  0  0

$$$$
test


  1  0  0  0  1  0              3 V2000
   -0.6622    0.5342    0.0000 C   0  0  2  0  0  0
M  CHG
M  END
$$$$
test


  1  0  0  0  1  0              3 V2000
   -0.6622    0.5342    0.0000 C   0  0  2  0  0  0
M  CHG  1   1
M  END
$$$$
test


  1  0  0  0  1  0              3 V2000
   -0.6622    0.5342    0.0000 C   0  0  2  0  0  0
M  CHG  1   10  1
M  END
$$$$
)sdf");

  for (int i = 1; i <= 3; ++i) {
    NURI_FMT_TEST_NEXT_MOL("test", 1, 0);

    auto mp = internal::find_key(mol().props(), "MELTING.POINT");
    ASSERT_NE(mp, mol().props().end()) << "Test: " << i;
    EXPECT_EQ(mp->second, "179.0 - 183.0") << "Test: " << i;
  }

  NURI_FMT_TEST_NEXT_MOL("test", 2, 1);
  NURI_FMT_TEST_NEXT_MOL("test", 1, 0);
  NURI_FMT_TEST_NEXT_MOL("test", 1, 0);
  NURI_FMT_TEST_NEXT_MOL("test", 1, 0);
}

TEST_F(SDFTest, MalformedParsing) {
  set_test_string(R"sdf(test
$$$$
test


 15 16  0     1  0  0  0  0  0999 V1000
$$$$
test


  a 16  0     1  0  0  0  0  0999 V2000
$$$$
test


 15 16  0     1  0  0  0  0  0999
$$$$
test


 15 16  0     1  0  0  0  0  0999 Vaaaa
$$$$
test


 15 16  0     1  0  0  0  0  0999 V2000
M  V30 BEGIN CTAB
M  V30 COUNTS 6 5 0 0 1
M  V30 BEGIN ATOM
M  V30 END ATOM
M  V30 END CTAB
M  END
$$$$
test


  1  0  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 ZZ  0  0  0  0  0  0  0  0  0  0  0  0
M END
$$$$
test


  2  1  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  1  1  0  0  0  0
M END
$$$$
test


  2  1  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
  a  1  1  0  0  0  0
M END
$$$$
test


  2  1  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  a  0  0  0  0
M END
$$$$
test


  2  1  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2 10  0  0  0  0
M END
$$$$
test


  2  1  0     1  0  0  0  0  0999 V2000
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
   74.7080   60.5120   32.8430 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  1  1  0  0  0  0
M END
$$$$
)sdf");

  for (int i = 0; i < 12; ++i)
    NURI_FMT_TEST_PARSE_FAIL();
}

TEST_F(SDFTest, Write2D) {
  Molecule m;
  {
    auto mut = m.mutator();
    mut.add_atom(kPt[6]);
  }

  std::string sdf;
  ASSERT_TRUE(write_sdf(sdf, m));
  ASSERT_TRUE(write_sdf(sdf, m, -1, SDFVersion::kV2000));
  ASSERT_TRUE(write_sdf(sdf, m, -1, SDFVersion::kV3000));

  set_test_string(sdf);

  NURI_FMT_TEST_NEXT_MOL("", 1, 0);
  EXPECT_EQ(mol()[0].data().atomic_number(), 6);
  NURI_FMT_TEST_NEXT_MOL("", 1, 0);
  EXPECT_EQ(mol()[0].data().atomic_number(), 6);
  NURI_FMT_TEST_NEXT_MOL("", 1, 0);
  EXPECT_EQ(mol()[0].data().atomic_number(), 6);
}

TEST_F(SDFTest, EscapeUnsafeChars) {
  Molecule m;
  m.mutator().add_atom(kPt[6]);
  m.add_prop("> unsafe key\n\n", "$$$$\n> a\n\nb\n");

  std::string sdf;
  ASSERT_TRUE(write_sdf(sdf, m));

  set_test_string(sdf);

  NURI_FMT_TEST_NEXT_MOL("", 1, 0);

  ASSERT_EQ(mol().props().size(), 2);

  auto it = mol().props().find("? unsafe key  ");
  ASSERT_NE(it, mol().props().end());
  EXPECT_EQ(it->second, "?$$$\n> a\nb");
}

TEST(SDFFormatTest, AutoDetect) {
  Molecule mol;
  mol.reserve(1000);
  {
    auto mut = mol.mutator();
    for (int i = 0; i < 999; ++i)
      mut.add_atom(kPt[6]);
  }

  std::string sdf;
  ASSERT_TRUE(write_sdf(sdf, mol));

  std::vector<std::string_view> lines =
      absl::StrSplit(sdf, absl::MaxSplits('\n', 5));
  EXPECT_PRED2(absl::EndsWith, lines[3], " V2000");

  mol.mutator().add_atom(kPt[6]);
  sdf.clear();
  ASSERT_TRUE(write_sdf(sdf, mol));

  lines = absl::StrSplit(sdf, absl::MaxSplits('\n', 5));
  EXPECT_PRED2(absl::EndsWith, lines[3], " V3000");
}

TEST(SDFFormatTest, V2000Correct) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[6]);
    mut.add_bond(0, 1, BondData(constants::kSingleBond));
  }

  mol[1].data().set_isotope(13).set_formal_charge(1);

  Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd(3, 2));
  conf.transpose() << 1, 2, 3, 4, 5, 6;

  std::string sdf;
  ASSERT_TRUE(write_sdf(sdf, mol, -1, SDFVersion::kV2000));

  std::vector<std::string_view> lines = absl::StrSplit(sdf, '\n');
  ASSERT_EQ(lines.size(), 12);

  EXPECT_EQ(safe_substr(lines[1], 2, 8), " NuriKit");
  EXPECT_EQ(safe_substr(lines[1], 20, 2), "3D");

  EXPECT_EQ(safe_substr(lines[3], 0, 3), "  2");
  EXPECT_EQ(safe_substr(lines[3], 3, 3), "  1");
  EXPECT_EQ(safe_substr(lines[3], 33, 6), " V2000");

  EXPECT_EQ(  //
      lines[4],
      "    1.0000    2.0000    3.0000 C   0  0  0  0  0  0  0  0  0  0  0  0");
  EXPECT_EQ(  //
      lines[5],
      "    4.0000    5.0000    6.0000 C   1  3  0  0  0  0  0  0  0  0  0  0");
  EXPECT_EQ(  //
      lines[6], "  1  2  1  0  0  0  0");

  for (int i = 7; i < 9; ++i) {
    auto line = lines[i];
    if (absl::StartsWith(line, "M  CHG")) {
      EXPECT_EQ(line, "M  CHG  1   2   1");
    } else if (absl::StartsWith(line, "M  ISO")) {
      EXPECT_EQ(line, "M  ISO  1   2  13");
    } else {
      FAIL() << "Unexpected line: " << line;
    }
  }

  EXPECT_EQ(lines[9], "M  END");
  EXPECT_EQ(lines[10], "$$$$");
}

TEST(SDFFormatTest, V3000Correct) {
  Molecule mol;
  {
    auto mut = mol.mutator();
    mut.add_atom(kPt[6]);
    mut.add_atom(kPt[6]);
    mut.add_bond(0, 1, BondData(constants::kSingleBond));
  }

  mol[1].data().set_isotope(13).set_formal_charge(1);

  Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd(3, 2));
  conf.transpose() << 1, 2, 3, 4, 5, 6;

  std::string sdf;
  ASSERT_TRUE(write_sdf(sdf, mol, -1, SDFVersion::kV3000));

  std::vector<std::string_view> lines = absl::StrSplit(sdf, '\n');
  ASSERT_EQ(lines.size(), 17);

  EXPECT_EQ(safe_substr(lines[1], 2, 8), " NuriKit");
  EXPECT_EQ(safe_substr(lines[1], 20, 2), "3D");

  EXPECT_EQ(safe_substr(lines[3], 0, 3), "  0");
  EXPECT_EQ(safe_substr(lines[3], 3, 3), "  0");
  EXPECT_EQ(safe_substr(lines[3], 33, 6), " V3000");

  EXPECT_EQ(lines[4], "M  V30 BEGIN CTAB");
  EXPECT_EQ(lines[5], "M  V30 COUNTS 2 1 0 0 0");

  EXPECT_EQ(lines[6], "M  V30 BEGIN ATOM");
  EXPECT_EQ(lines[7], "M  V30 1 C 1.0000 2.0000 3.0000 0");
  EXPECT_EQ(lines[8], "M  V30 2 C 4.0000 5.0000 6.0000 0 CHG=1 MASS=13");
  EXPECT_EQ(lines[9], "M  V30 END ATOM");

  EXPECT_EQ(lines[10], "M  V30 BEGIN BOND");
  EXPECT_EQ(lines[11], "M  V30 1 1 1 2");
  EXPECT_EQ(lines[12], "M  V30 END BOND");

  EXPECT_EQ(lines[13], "M  V30 END CTAB");
  EXPECT_EQ(lines[14], "M  END");
  EXPECT_EQ(lines[15], "$$$$");
}
}  // namespace
}  // namespace nuri
