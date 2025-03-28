//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/mol2.h"

#include <string>

#include <absl/strings/str_cat.h>

#include <gtest/gtest.h>

#include "fmt_test_common.h"
#include "test_utils.h"
#include "nuri/algo/guess.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/core/property_map.h"

namespace nuri {
namespace {
using Mol2Test = internal::StringFormatTest<Mol2Reader>;

TEST_F(Mol2Test, BasicParsing) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
methane
1 0 0 0 0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1            -0.0127     1.0858     0.0080 C.3
)mol2");

  NURI_FMT_TEST_NEXT_MOL("methane", 1, 0);
}

TEST_F(Mol2Test, MalformedParsing) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
@<TRIPOS>MOLECULE
empty error
@<TRIPOS>MOLECULE
******
a b c d
@<TRIPOS>MOLECULE
@<TRIPOS>BOND
     1     1     2    1
@<TRIPOS>MOLECULE
*****
 1 0 0 0 0
SMALL
GASTEIGER


@<TRIPOS>ATOM
      1 N           a    0.0000    0.0000 N.4     1  UNL1        1.0000
@<TRIPOS>BOND
@<TRIPOS>MOLECULE
*****
 1 0 0 0 0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
      1 N           0    0.0000    0.0000 N.3
      2 N           0    0.0000    0.0000 N.3
@<TRIPOS>BOND
     1     1     2    error
@<TRIPOS>MOLECULE
*****
 1 1 0 0 0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
      1 N           0    0.0000    0.0000 N.3
@<TRIPOS>BOND
     1     1     1    1
@<TRIPOS>MOLECULE
*****
 1 0 0 0 0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
      1 N           0    0.0000    0.0000 N.3
      2 N           0    0.0000    0.0000 N.3
@<TRIPOS>BOND
     1     1     2    1
     2     2     1    1
@<TRIPOS>MOLECULE
corina - carboxylic acid
   4    3    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>MOLECULE
*****
 1 0 0 0 0
SMALL
GASTEIGER

@<TRIPOS>UNITY_ATOM_ATTR
1 1
charge 1
@<TRIPOS>BOND
@<TRIPOS>ATOM
      1 N           0.0000    0.0000    0.0000 N.4     1  UNL1        1.0000
@<TRIPOS>MOLECULE
*****
 1 0 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 N           0.0000    0.0000    0.0000 N.4     1  UNL1        1.0000
@<TRIPOS>BOND
@<TRIPOS>UNITY_ATOM_ATTR
1 error
charge 1
@<TRIPOS>MOLECULE
*****
 1 0 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 N           0.0000    0.0000    0.0000 N.4     1  UNL1        1.0000
@<TRIPOS>BOND
@<TRIPOS>UNITY_ATOM_ATTR
2 1
charge 1
)mol2");

  for (int i = 0; i < 12; ++i) {
    NURI_FMT_TEST_PARSE_FAIL();
  }
}

/* From here, some molecules are written twice: one for openbabel, the other
 * for corina */

TEST_F(Mol2Test, QuaternaryAmmonium) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
quaternary ammonium salt
 5 4 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C           0.0000    0.0000    0.0000 C.3     1  UNL1        0.1712
      2 N           0.0000    0.0000    0.0000 N.4     1  UNL1        0.3151
      3 C           0.0000    0.0000    0.0000 C.3     1  UNL1        0.1712
      4 C           0.0000    0.0000    0.0000 C.3     1  UNL1        0.1712
      5 C           0.0000    0.0000    0.0000 C.3     1  UNL1        0.1712
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
     3     2     4    1
     4     2     5    1
@<TRIPOS>MOLECULE
quaternary ammonium salt
   5    4    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1             0.0021    -0.0041     0.0020 C.3
   2 N2            -0.0178     1.4648     0.0101 N.4
   3 C3             1.3603     1.9732    -0.0003 C.3
   4 C4            -0.7283     1.9514    -1.1800 C.3
   5 C5            -0.7054     1.9385     1.2187 C.3
@<TRIPOS>BOND
   1    1    2 1
   2    2    3 1
   3    2    4 1
   4    2    5 1
)mol2");

  auto verify_mol = [this]() {
    for (auto atom: mol()) {
      EXPECT_EQ(atom.data().hybridization(), constants::kSP3) << atom.id();

      if (atom.id() == 1) {
        EXPECT_EQ(atom.data().formal_charge(), 1) << atom.id();
        EXPECT_EQ(atom.data().implicit_hydrogens(), 0) << atom.id();
      } else {
        EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();
        EXPECT_EQ(atom.data().implicit_hydrogens(), 3) << atom.id();
      }
    }
  };

  std::string mol2;
  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("quaternary ammonium salt", 5, 4);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();

    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("quaternary ammonium salt", 5, 4);

    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, GuadiniumLike) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
guadinium-like
 7 6 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 N           0.0000    0.0000    0.0000 N.pl3   1  UNL1        0.1360
      2 C           0.0000    0.0000    0.0000 C.cat   1  UNL1        0.5446
      3 N           0.0000    0.0000    0.0000 N.pl3   1  UNL1        0.0168
      4 C           0.0000    0.0000    0.0000 C.cat   1  UNL1        0.5446
      5 N           0.0000    0.0000    0.0000 N.pl3   1  UNL1        0.1360
      6 N           0.0000    0.0000    0.0000 N.pl3   1  UNL1        0.1360
      7 N           0.0000    0.0000    0.0000 N.pl3   1  UNL1        0.1360
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
     3     3     4    1
     4     4     5    1
     5     4     6    2
     6     2     7    2
#	Name: NoName

#	Program:	corina 4.4.0 0026  12.08.2021

@<TRIPOS>MOLECULE
guadinium-like
  7   6    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 N1             1.1391     3.0624    -0.0827 N.pl3
   2 C2            -0.0145     3.6922     0.1261 C.cat
   3 N3            -0.8308     3.3054     1.1618 N.pl3
   4 C4            -0.3907     2.3758     2.0732 C.cat
   5 N5            -1.2588     1.7768     2.8847 N.pl3
   6 N6             0.9036     2.0745     2.1443 N.pl3
   7 N9            -0.3776     4.6964    -0.6680 N.pl3
@<TRIPOS>BOND
   1    1    2 ar
   2    2    3 ar
   3    2    7 ar
   4    3    4 ar
   5    4    5 ar
   6    4    6 ar

#	End of record
#	Name: NoName

#	Program:	corina 4.4.0 0026  12.08.2021

@<TRIPOS>MOLECULE
guadinium
   4    3    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 N1            -0.0111     0.9658     0.0074 N.pl3
   2 C2            -1.1814     1.6205     0.0165 C.cat
   3 N3            -2.3336     0.9343     0.0182 N.pl3
   4 N4            -1.1996     2.9614     0.0238 N.pl3
@<TRIPOS>BOND
   1    1    2 1
   2    2    3 1
   3    2    4 1

#	End of record
)mol2");

  auto verify_head = [this]() {
    int total_fchg = 0, total_hcount = 0, bo_sum = 0;

    for (auto atom: mol()) {
      EXPECT_EQ(atom.data().hybridization(), constants::kSP2) << atom.id();

      total_fchg += atom.data().formal_charge();
      EXPECT_GE(atom.data().formal_charge(), 0) << atom.id();

      total_hcount += atom.data().implicit_hydrogens();
      EXPECT_GE(atom.data().implicit_hydrogens(), 0) << atom.id();
    }

    for (auto bond: mol().bonds())
      bo_sum += bond.data().order();

    EXPECT_EQ(total_fchg, 2);
    EXPECT_EQ(total_hcount, 9);
    EXPECT_EQ(bo_sum, 8);
  };

  auto verify_tail = [this]() {
    int total_fchg = 0, total_hcount = 0, bo_sum = 0;

    for (auto atom: mol()) {
      EXPECT_EQ(atom.data().hybridization(), constants::kSP2) << atom.id();

      total_fchg += atom.data().formal_charge();
      EXPECT_GE(atom.data().formal_charge(), 0) << atom.id();

      total_hcount += atom.data().implicit_hydrogens();
      EXPECT_GE(atom.data().implicit_hydrogens(), 0) << atom.id();
    }

    for (auto bond: mol().bonds())
      bo_sum += bond.data().order();

    EXPECT_EQ(total_fchg, 1);
    EXPECT_EQ(total_hcount, 6);
    EXPECT_EQ(bo_sum, 4);
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("guadinium-like", 7, 6);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_head();

    write_mol2(mol2, mol());
  }

  NURI_FMT_TEST_NEXT_MOL("guadinium", 4, 3);
  {
    SCOPED_TRACE("Initial read - 2");
    verify_tail();
  }
  write_mol2(mol2, mol());

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("guadinium-like", 7, 6);
    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_head();
  }

  NURI_FMT_TEST_NEXT_MOL("guadinium", 4, 3);
  SCOPED_TRACE(absl::StrCat("Re-read - 2"));
  verify_tail();
}

TEST_F(Mol2Test, Nitrobenzene) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
nitrobenzene
 9 9 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 O           0.0000    0.0000    0.0000 O.2     1  UNL1        0.0415
      2 N           0.0000    0.0000    0.0000 N.pl3   1  UNL1        0.0809
      3 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.2901
      4 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0753
      5 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0063
      6 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0004
      7 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0063
      8 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0753
      9 O           0.0000    0.0000    0.0000 O.co2   1  UNL1       -0.5760
@<TRIPOS>BOND
     1     1     2    2
     2     2     3    1
     3     3     4   ar
     4     4     5   ar
     5     5     6   ar
     6     6     7   ar
     7     7     8   ar
     8     3     8   ar
     9     2     9    1
#	Name: NoName

#	Program:	corina 4.4.0 0026  12.08.2021

@<TRIPOS>MOLECULE
nitrobenzene
   9    9    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 O1             2.4833    -0.1572    -0.0217 O.2
   2 N2             2.4668     1.0606    -0.0154 N.pl3
   3 C3             1.1750     1.7832     0.0004 C.ar
   4 C4            -0.0126     1.0758     0.0080 C.ar
   5 C5            -1.2190     1.7506     0.0173 C.ar
   6 C6            -1.2377     3.1328     0.0290 C.ar
   7 C7            -0.0501     3.8402     0.0223 C.ar
   8 C8             1.1563     3.1654     0.0081 C.ar
   9 O9             3.5132     1.6839    -0.0220 O.2
@<TRIPOS>BOND
   1    1    2 ar
   2    2    3 1
   3    2    9 ar
   4    3    8 ar
   5    3    4 ar
   6    4    5 ar
   7    5    6 ar
   8    6    7 ar
   9    7    8 ar

#	End of record
)mol2");

  auto verify_mol = [this]() {
    int oxygen_fchg = 0, total_hcount = 0, bo_sum = 0;

    for (auto atom: mol()) {
      switch (atom.data().atomic_number()) {
      case 8:
        oxygen_fchg += atom.data().formal_charge();
        EXPECT_EQ(atom.data().hybridization(), constants::kTerminal)
            << atom.id();
        break;
      default:
        EXPECT_EQ(atom.data().hybridization(), constants::kSP2) << atom.id();
        EXPECT_EQ(atom.data().formal_charge(),
                  atom.data().atomic_number() == 7 ? 1 : 0)
            << atom.id();
      }

      total_hcount += atom.data().implicit_hydrogens();
      EXPECT_GE(atom.data().implicit_hydrogens(), 0) << atom.id();

      bo_sum += sum_bond_order(atom);
    }

    EXPECT_EQ(oxygen_fchg, -1);
    EXPECT_EQ(total_hcount, 5);
    EXPECT_EQ(bo_sum, 13 * 2 + 5);
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("nitrobenzene", 9, 9);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();

    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("nitrobenzene", 9, 9);
    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, SF6) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
sf6
 7 6 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 F           0.0000    0.0000    0.0000 F       1  UNL1       -0.0722
      2 S           0.0000    0.0000    0.0000 S.3     1  UNL1        0.4335
      3 F           0.0000    0.0000    0.0000 F       1  UNL1       -0.0722
      4 F           0.0000    0.0000    0.0000 F       1  UNL1       -0.0722
      5 F           0.0000    0.0000    0.0000 F       1  UNL1       -0.0722
      6 F           0.0000    0.0000    0.0000 F       1  UNL1       -0.0722
      7 F           0.0000    0.0000    0.0000 F       1  UNL1       -0.0722
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
     3     2     4    1
     4     2     5    1
     5     2     6    1
     6     2     7    1
#	Name: NoName

#	Program:	corina 4.4.0 0026  12.08.2021

@<TRIPOS>MOLECULE
sf6
   7    6    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 F1             0.0021    -0.0041     0.0020 F
   2 S2            -0.0198     1.6058     0.0109 S.oh
   3 F3            -0.0416     3.2156     0.0197 F
   4 F4             1.5900     1.6277    -0.0044 F
   5 F5            -1.6295     1.5838     0.0261 F
   6 F6            -0.0351     1.6144    -1.5990 F
   7 F7            -0.0044     1.5971     1.6208 F
@<TRIPOS>UNITY_ATOM_ATTR
2 1
charge 0
@<TRIPOS>BOND
   1    1    2 1
   2    2    3 1
   3    2    4 1
   4    2    5 1
   5    2    6 1
   6    2    7 1

#	End of record
)mol2");

  auto verify_mol = [this]() {
    for (auto atom: mol()) {
      if (atom.id() == 1) {
        EXPECT_EQ(atom.data().hybridization(), constants::kSP3D2) << atom.id();
      } else {
        EXPECT_EQ(atom.data().hybridization(), constants::kTerminal)
            << atom.id();
      }
      EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();
      EXPECT_EQ(atom.data().implicit_hydrogens(), 0) << atom.id();
    }

    for (auto bond: mol().bonds()) {
      EXPECT_EQ(bond.data().order(), constants::kSingleBond);
    }
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("sf6", 7, 6);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();

    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("sf6", 7, 6);
    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, CarboxylicAcid) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
carboxylic acid
   4    3    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1            -0.0127     1.0858     0.0080 C.3
   2 C2            -0.7181     1.5718     1.2479 C.2
   3 O3            -0.8566     2.8886     1.4687 O.3
   4 O4            -1.1590     0.7762     2.0432 O.2
@<TRIPOS>BOND
   1    1    2 1
   2    2    3 1
   3    2    4 2
@<TRIPOS>MOLECULE
carboxylic acid
 4 3 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C           0.0000    0.0000    0.0000 C.3     1  UNL1        0.0331
      2 C           0.0000    0.0000    0.0000 C.2     1  UNL1        0.3016
      3 O           0.0000    0.0000    0.0000 O.3     1  UNL1       -0.4808
      4 O           0.0000    0.0000    0.0000 O.2     1  UNL1       -0.2513
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
     3     2     4    2
@<TRIPOS>MOLECULE
carboxylate
   4    3    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1            -0.0127     1.0858     0.0080 C.3
   2 C2            -0.7181     1.5718     1.2479 C.2
   3 O3            -0.0841     1.7780     2.2687 O.co2
   4 O4            -1.9227     1.7588     1.2302 O.co2
@<TRIPOS>BOND
   1    1    2 1
   2    2    3 ar
   3    2    4 ar
@<TRIPOS>MOLECULE
carboxylate
 4 3 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C           0.0000    0.0000    0.0000 C.3     1  UNL1        0.1258
      2 C           0.0000    0.0000    0.0000 C.2     1  UNL1        0.3654
      3 O           0.0000    0.0000    0.0000 O.co2   1  UNL1       -0.2456
      4 O           0.0000    0.0000    0.0000 O.co2   1  UNL1       -0.2456
@<TRIPOS>BOND
     1     1     2    1
     2     2     3   ar
     3     2     4   ar
)mol2");

  auto verify_head = [this]() {
    auto test_atom = [](Molecule::Atom atom, int hcount) {
      EXPECT_EQ(atom.data().implicit_hydrogens(), hcount) << atom.id();
      EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();
    };

    test_atom(mol().atom(0), 3);
    test_atom(mol().atom(1), 0);
    test_atom(mol().atom(2), 1);
    test_atom(mol().atom(3), 0);
  };

  auto verify_tail = [this]() {
    for (auto atom: mol())
      EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();

    EXPECT_EQ(mol().atom(0).data().implicit_hydrogens(), 3);
    EXPECT_EQ(mol().atom(1).data().implicit_hydrogens(), 0);

    EXPECT_TRUE((mol().atom(2).data().implicit_hydrogens() == 0
                 && mol().atom(3).data().implicit_hydrogens() == 1)
                || (mol().atom(2).data().implicit_hydrogens() == 1
                    && mol().atom(3).data().implicit_hydrogens() == 0))
        << "2: " << mol().atom(2).data().implicit_hydrogens()
        << ", 3: " << mol().atom(3).data().implicit_hydrogens();
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("carboxylic acid", 4, 3);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_head();
    write_mol2(mol2, mol());
  }

  for (int i = 2; i < 4; ++i) {
    NURI_FMT_TEST_NEXT_MOL("carboxylate", 4, 3);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_tail();
    write_mol2(mol2, mol());
  }
}

TEST_F(Mol2Test, Thiazole) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
Z337709406
11 12 1 0 0
SMALL
AMBER ff14SB


@<TRIPOS>ATOM
      1 N          -0.9225    6.1660    2.9696 N.pl3     1 UNK0       -0.3375
      2 C          -0.8738    5.1881    2.0021 C.ar      1 UNK0        0.1550
      3 N          -0.8295    3.9080    2.2425 N.ar      1 UNK0       -0.2302
      4 C          -0.7861    3.0943    1.2197 C.ar      1 UNK0        0.0608
      5 C          -0.7219    1.5760    1.2745 C.3       1 UNK0       -0.0086
      6 C          -0.0127    1.0858    0.0080 C.3       1 UNK0       -0.0404
      7 C          -0.7443    1.5884   -1.2367 C.3       1 UNK0        0.0118
      8 C          -0.7689    3.1007   -1.2410 C.2       1 UNK0        0.1710
      9 O          -0.7601    3.7258   -2.2829 O.2       1 UNK0       -0.2935
     10 C          -0.8018    3.7604    0.0037 C.ar      1 UNK0        0.0796
     11 S          -0.8636    5.4845    0.3181 S.2       1 UNK0       -0.0491
@<TRIPOS>BOND
     1    1    2 1
     2    2    3 ar
     3    2   11 ar
     4    3    4 ar
     5    4    5 1
     6    4   10 ar
     7    5    6 1
     8    6    7 1
     9    7    8 1
    10    8    9 2
    11    8   10 1
    12   10   11 ar
@<TRIPOS>SUBSTRUCTURE
     1 UNK0        1 RESIDUE           4 A     UNK     0 ROOT
@<TRIPOS>MOLECULE
Z337709406
19 20 1 0 0
SMALL
AMBER ff14SB


@<TRIPOS>ATOM
      1 N          -0.9225    6.1660    2.9696 N.pl3     1 UNK0       -0.3375
      2 C          -0.8738    5.1881    2.0021 C.ar      1 UNK0        0.1550
      3 N          -0.8295    3.9080    2.2425 N.ar      1 UNK0       -0.2302
      4 C          -0.7861    3.0943    1.2197 C.ar      1 UNK0        0.0608
      5 C          -0.7219    1.5760    1.2745 C.3       1 UNK0       -0.0086
      6 C          -0.0127    1.0858    0.0080 C.3       1 UNK0       -0.0404
      7 C          -0.7443    1.5884   -1.2367 C.3       1 UNK0        0.0118
      8 C          -0.7689    3.1007   -1.2410 C.2       1 UNK0        0.1710
      9 O          -0.7601    3.7258   -2.2829 O.2       1 UNK0       -0.2935
     10 C          -0.8018    3.7604    0.0037 C.ar      1 UNK0        0.0796
     11 S          -0.8636    5.4845    0.3181 S.2       1 UNK0       -0.0491
     12 HN1        -0.9255    5.9101    3.9466 H         1 UNK0        0.1447
     13 HN2        -0.9553    7.1395    2.7025 H         1 UNK0        0.1447
     14 HC2        -0.1637    1.2611    2.1562 H         1 UNK0        0.0333
     15 HC3        -1.7313    1.1664    1.3127 H         1 UNK0        0.0333
     16 HC4         0.0002   -0.0041    0.0020 H         1 UNK0        0.0275
     17 HC5         1.0114    1.4591    0.0002 H         1 UNK0        0.0275
     18 HC6        -0.2279    1.2325   -2.1282 H         1 UNK0        0.0351
     19 HC7        -1.7663    1.2093   -1.2342 H         1 UNK0        0.0351
@<TRIPOS>BOND
     1    1    2 1
     2    2    3 ar
     3    2   11 ar
     4    3    4 ar
     5    4    5 1
     6    4   10 ar
     7    5    6 1
     8    6    7 1
     9    7    8 1
    10    8    9 2
    11    8   10 1
    12   10   11 ar
    13   12    1 1
    14   13    1 1
    15   14    5 1
    16   15    5 1
    17   16    6 1
    18   17    6 1
    19   18    7 1
    20   19    7 1
@<TRIPOS>SUBSTRUCTURE
     1 UNK0        1 RESIDUE           4 A     UNK     0 ROOT
  )mol2");

  auto check_thiazole = [&]() {
    for (auto atom: mol()) {
      EXPECT_EQ(atom.data().formal_charge(), 0) << "Atom id: " << atom.id();
    }

    EXPECT_TRUE(MoleculeSanitizer(mol()).sanitize_all());
    for (int i: { 1, 2, 3, 9, 10 }) {
      EXPECT_TRUE(mol().atom(i).data().is_aromatic());
    }
  };

  std::string mol2;

  {
    NURI_FMT_TEST_NEXT_MOL("Z337709406", 11, 12);
    SCOPED_TRACE("Initial read - 1");
    check_thiazole();
    write_mol2(mol2, mol());
  }
  {
    NURI_FMT_TEST_NEXT_MOL("Z337709406", 19, 20);
    SCOPED_TRACE("Initial read - 2");
    check_thiazole();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  {
    NURI_FMT_TEST_NEXT_MOL("Z337709406", 11, 12);
    SCOPED_TRACE("Re-read - 1");
    check_thiazole();
  }
  {
    NURI_FMT_TEST_NEXT_MOL("Z337709406", 19, 20);
    SCOPED_TRACE("Re-read - 2");
    check_thiazole();
  }
}

TEST_F(Mol2Test, ExplicitFchgCpm) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
Cyclopentadienyl anion
 5 5 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C           0.0000    0.0000    0.0000 C.ar    1  UNL1       -0.2250
      2 C           0.0000    0.0000    0.0000 C.2     1  UNL1       -0.4635
      3 C           0.0000    0.0000    0.0000 C.ar    1  UNL1       -0.2250
      4 C           0.0000    0.0000    0.0000 C.ar    1  UNL1       -0.0433
      5 C           0.0000    0.0000    0.0000 C.ar    1  UNL1       -0.0433
@<TRIPOS>UNITY_ATOM_ATTR
2 1
charge -1
@<TRIPOS>BOND
     1     1     2   ar
     2     2     3   ar
     3     3     4   ar
     4     4     5   ar
     5     1     5   ar
#	Name: NoName

#	Program:	corina 4.4.0 0026  12.08.2021

@<TRIPOS>MOLECULE
Cyclopentadienyl anion
   5    5    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1             1.0980     1.9060     0.0018 C.2
   2 C2            -0.0126     1.0758     0.0080 C.2
   3 C4            -1.1453     1.8755     0.0170 C.2
   4 C5            -0.7347     3.1999     0.0254 C.2
   5 C6             0.6517     3.2188     0.0126 C.2
@<TRIPOS>BOND
   1    1    5 2
   2    1    2 1
   3    2    3 1
   4    3    4 2
   5    4    5 1
@<TRIPOS>UNITY_ATOM_ATTR
2 1
charge -1

#	End of record


)mol2");

  auto verify_mol = [this]() {
    int total_fchg = 0, bo_sum = 0;

    for (auto atom: mol()) {
      total_fchg += atom.data().formal_charge();
      EXPECT_EQ(atom.data().implicit_hydrogens(), 1) << atom.id();
      bo_sum += sum_bond_order(atom);
    }

    EXPECT_EQ(total_fchg, -1);
    EXPECT_EQ(bo_sum, 19);
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("Cyclopentadienyl anion", 5, 5);
    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("Cyclopentadienyl anion", 5, 5);
    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, AllImplicitCpm) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
Cyclopentadienyl anion
 5 5 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C           0.0000    0.0000    0.0000 C.ar    1  UNL1       -0.2250
      2 C           0.0000    0.0000    0.0000 C.2     1  UNL1       -0.4635
      3 C           0.0000    0.0000    0.0000 C.ar    1  UNL1       -0.2250
      4 C           0.0000    0.0000    0.0000 C.ar    1  UNL1       -0.0433
      5 C           0.0000    0.0000    0.0000 C.ar    1  UNL1       -0.0433
@<TRIPOS>BOND
     1     1     2   ar
     2     2     3   ar
     3     3     4   ar
     4     4     5   ar
     5     1     5   ar
#	Name: NoName

#	Program:	corina 4.4.0 0026  12.08.2021

@<TRIPOS>MOLECULE
Cyclopentadienyl anion
   5    5    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1             1.0980     1.9060     0.0018 C.ar
   2 C2            -0.0126     1.0758     0.0080 C.ar
   3 C4            -1.1453     1.8755     0.0170 C.ar
   4 C5            -0.7347     3.1999     0.0254 C.ar
   5 C6             0.6517     3.2188     0.0126 C.ar
@<TRIPOS>BOND
   1    1    5 2
   2    1    2 1
   3    2    3 1
   4    3    4 2
   5    4    5 1
#	End of record


)mol2");

  auto verify_mol = [this]() {
    int total_fchg = 0, bo_sum = 0;

    for (auto atom: mol()) {
      total_fchg += atom.data().formal_charge();
      EXPECT_EQ(atom.data().implicit_hydrogens(), 1) << atom.id();
      bo_sum += sum_bond_order(atom);
    }

    EXPECT_EQ(total_fchg, -1);
    EXPECT_EQ(bo_sum, 19);
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("Cyclopentadienyl anion", 5, 5);
    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("Cyclopentadienyl anion", 5, 5);
    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, ExplicitFchgTropylium) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
Tropylium cation
 7 7 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.2549
      2 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0555
      3 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0045
      4 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0045
      5 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0555
      6 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.2549
      7 C           0.0000    0.0000    0.0000 C.cat   1  UNL1        0.3702
@<TRIPOS>UNITY_ATOM_ATTR
7 1
charge 1
@<TRIPOS>BOND
     1     1     2   ar
     2     2     3   ar
     3     3     4   ar
     4     4     5   ar
     5     5     6   ar
     6     6     7   ar
     7     1     7   ar
@<TRIPOS>MOLECULE
Tropylium cation
   7    7    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1            -0.0167     1.3835     0.0096 C.2
   2 C2             1.0564     2.2634     0.0041 C.2
   3 C3             2.4133     1.9731    -0.0095 C.2
   4 C4             3.0323     0.7311    -0.0219 C.2
   5 C5             2.4472    -0.5273    -0.0233 C.2
   6 C6             1.0986    -0.8545    -0.0126 C.2
   7 C7             0.0021    -0.0041     0.0020 C.cat
@<TRIPOS>UNITY_ATOM_ATTR
7 1
charge 1
@<TRIPOS>BOND
   1    1    7 1
   2    1    2 2
   3    2    3 1
   4    3    4 2
   5    4    5 1
   6    5    6 2
   7    6    7 1
)mol2");

  auto verify_mol = [this]() {
    int total_fchg = 0, bo_sum = 0;

    for (auto atom: mol()) {
      total_fchg += atom.data().formal_charge();
      EXPECT_EQ(atom.data().implicit_hydrogens(), 1) << atom.id();
      bo_sum += sum_bond_order(atom);
    }

    EXPECT_EQ(total_fchg, 1);
    EXPECT_EQ(bo_sum, 6 * 4 + 3);
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("Tropylium cation", 7, 7);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("Tropylium cation", 7, 7);

    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, AllImplicitTropylium) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
Tropylium cation
 7 7 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.2549
      2 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0555
      3 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0045
      4 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0045
      5 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0555
      6 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.2549
      7 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.3702
@<TRIPOS>BOND
     1     1     2   ar
     2     2     3   ar
     3     3     4   ar
     4     4     5   ar
     5     5     6   ar
     6     6     7   ar
     7     1     7   ar
@<TRIPOS>MOLECULE
Tropylium cation
   7    7    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1            -0.0167     1.3835     0.0096 C.ar
   2 C2             1.0564     2.2634     0.0041 C.ar
   3 C3             2.4133     1.9731    -0.0095 C.ar
   4 C4             3.0323     0.7311    -0.0219 C.ar
   5 C5             2.4472    -0.5273    -0.0233 C.ar
   6 C6             1.0986    -0.8545    -0.0126 C.ar
   7 C7             0.0021    -0.0041     0.0020 C.ar
@<TRIPOS>BOND
   1    1    7 1
   2    1    2 2
   3    2    3 1
   4    3    4 2
   5    4    5 1
   6    5    6 2
   7    6    7 1
)mol2");

  auto verify_mol = [this]() {
    int total_fchg = 0, bo_sum = 0;

    for (auto atom: mol()) {
      total_fchg += atom.data().formal_charge();
      EXPECT_EQ(atom.data().implicit_hydrogens(), 1) << atom.id();
      bo_sum += sum_bond_order(atom);
    }

    EXPECT_EQ(total_fchg, 1);
    EXPECT_EQ(bo_sum, 6 * 4 + 3);
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("Tropylium cation", 7, 7);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("Tropylium cation", 7, 7);

    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, ExplicitFchgImidazole) {
  set_test_string(R"mol2(
#	Program:	corina 4.4.0 0026  12.08.2021

@<TRIPOS>MOLECULE
imidazole
   5    5    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 N1             0.0021    -0.0041     0.0020 N.pl3
   2 C3            -0.0165     1.3646     0.0095 C.2
   3 C4             1.2671     1.7717    -0.0005 C.2
   4 N5             2.0482     0.6814    -0.0138 N.2
   5 C6             1.2973    -0.3859    -0.0124 C.2
@<TRIPOS>BOND
   1    1    5 1
   2    1    2 1
   3    2    3 2
   4    3    4 1
   5    4    5 2
@<TRIPOS>UNITY_ATOM_ATTR
1 1
charge 0
#	End of record

@<TRIPOS>MOLECULE
imidazole
 5 5 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 N           0.0000    0.0000    0.0000 N.ar    1  UNL1       -0.2208
      2 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.1213
      3 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.1213
      4 N           0.0000    0.0000    0.0000 N.ar    1  UNL1       -0.2208
      5 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.1990
@<TRIPOS>BOND
     1     1     2   ar
     2     2     3   ar
     3     3     4   ar
     4     4     5   ar
     5     1     5   ar
@<TRIPOS>UNITY_ATOM_ATTR
1 1
charge 0
)mol2");

  auto verify_mol = [this]() {
    int nitrogen_hcount = 0, bo_sum = 0;

    for (auto atom: mol()) {
      EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();
      if (atom.data().atomic_number() == 7) {
        nitrogen_hcount += atom.data().implicit_hydrogens();
      } else {
        EXPECT_EQ(atom.data().implicit_hydrogens(), 1) << atom.id();
      }
      bo_sum += sum_bond_order(atom);
    }

    EXPECT_EQ(nitrogen_hcount, 1);
    EXPECT_EQ(bo_sum, 18);
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("imidazole", 5, 5);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("imidazole", 5, 5);

    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, AllImplicitImidazole) {
  set_test_string(R"mol2(
#	Program:	corina 4.4.0 0026  12.08.2021

@<TRIPOS>MOLECULE
imidazole
   5    5    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 N1             0.0021    -0.0041     0.0020 N.pl3
   2 C3            -0.0165     1.3646     0.0095 C.2
   3 C4             1.2671     1.7717    -0.0005 C.2
   4 N5             2.0482     0.6814    -0.0138 N.2
   5 C6             1.2973    -0.3859    -0.0124 C.2
@<TRIPOS>BOND
   1    1    5 1
   2    1    2 1
   3    2    3 2
   4    3    4 1
   5    4    5 2
#	End of record

@<TRIPOS>MOLECULE
imidazole
 5 5 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 N           0.0000    0.0000    0.0000 N.ar    1  UNL1       -0.2208
      2 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.1213
      3 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.1213
      4 N           0.0000    0.0000    0.0000 N.ar    1  UNL1       -0.2208
      5 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.1990
@<TRIPOS>BOND
     1     1     2   ar
     2     2     3   ar
     3     3     4   ar
     4     4     5   ar
     5     1     5   ar
)mol2");

  auto verify_mol = [this]() {
    int nitrogen_hcount = 0, bo_sum = 0;

    for (auto atom: mol()) {
      EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();
      if (atom.data().atomic_number() == 7) {
        nitrogen_hcount += atom.data().implicit_hydrogens();
      } else {
        EXPECT_EQ(atom.data().implicit_hydrogens(), 1) << atom.id();
      }
      bo_sum += sum_bond_order(atom);
    }

    EXPECT_EQ(nitrogen_hcount, 1);
    EXPECT_EQ(bo_sum, 18);
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("imidazole", 5, 5);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("imidazole", 5, 5);

    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, AromaticIndole) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
indole
 9 10 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0192
      2 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0015
      3 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0001
      4 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0015
      5 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0191
      6 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0814
      7 N           0.0000    0.0000    0.0000 N.ar    1  UNL1       -0.2438
      8 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0989
      9 C           0.0000    0.0000    0.0000 C.ar    1  UNL1        0.0221
@<TRIPOS>BOND
     1     1     2   ar
     2     2     3   ar
     3     3     4   ar
     4     4     5   ar
     5     5     6   ar
     6     1     6   ar
     7     6     7   ar
     8     7     8   ar
     9     8     9   ar
    10     1     9   ar
@<TRIPOS>MOLECULE
indole
   9   10    0    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 C1            -0.6876     3.2303     0.0251 C.ar
   2 C2            -1.3581     4.4545     0.0376 C.ar
   3 C3            -0.6436     5.6195     0.0369 C.ar
   4 C4             0.7440     5.5954     0.0233 C.ar
   5 C5             1.4261     4.3982     0.0109 C.ar
   6 C6             0.7191     3.2004     0.0119 C.ar
   7 N7             1.0971     1.8764     0.0017 N.pl3
   8 C8            -0.0126     1.0758     0.0080 C.2
   9 C9            -1.1221     1.8313     0.0168 C.2
@<TRIPOS>BOND
   1    1    6 ar
   2    1    9 1
   3    1    2 ar
   4    2    3 ar
   5    3    4 ar
   6    4    5 ar
   7    5    6 ar
   8    6    7 1
   9    7    8 1
  10    8    9 2

#	End of record
)mol2");

  auto verify_mol = [this]() {
    int bo_sum = 0;

    for (auto atom: mol()) {
      EXPECT_TRUE(atom.data().is_aromatic());
      EXPECT_EQ(atom.data().formal_charge(), 0) << atom.id();
      EXPECT_EQ(atom.data().implicit_hydrogens(),
                static_cast<int>(atom.degree() == 2))
          << atom.id();
      bo_sum += sum_bond_order(atom);
    }

    EXPECT_EQ(bo_sum, 8 * 4 + 3);

    for (auto bond: mol().bonds())
      EXPECT_TRUE(bond.data().is_aromatic()) << bond.id();
  };

  std::string mol2;

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("indole", 9, 10);

    SCOPED_TRACE(absl::StrCat("Initial read - ", i));
    verify_mol();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  for (int i = 0; i < 2; ++i) {
    NURI_FMT_TEST_NEXT_MOL("indole", 9, 10);

    SCOPED_TRACE(absl::StrCat("Re-read - ", i));
    verify_mol();
  }
}

TEST_F(Mol2Test, Substructure) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
7yoo-cut
44 44 4 0 0
SMALL
USER_CHARGES


@<TRIPOS>ATOM
      1 N         187.2360  135.0290  183.1390 N.am      1 ARG3   -0.3024
      2 CA        186.8300  135.1630  181.7490 C.3       1 ARG3    0.0995
      3 C         186.0740  136.4690  181.5650 C.2       1 ARG3    0.2339
      4 O         185.2830  136.8650  182.4260 O.2       1 ARG3   -0.2738
      5 CB        185.9560  133.9820  181.3090 C.3       1 ARG3   -0.0243
      6 CG        185.7360  133.8940  179.8110 C.3       1 ARG3   -0.0305
      7 CD        184.5070  133.0640  179.4850 C.3       1 ARG3    0.0512
      8 NE        184.0100  133.3320  178.1420 N.2       1 ARG3   -0.2178
      9 CZ        182.8960  132.8180  177.6400 C.cat     1 ARG3    0.1058
     10 NH1       182.1330  131.9990  178.3450 N.2       1 ARG3   -0.2337
     11 NH2       182.5370  133.1350  176.3990 N.2       1 ARG3   -0.2337
     12 N         186.3220  137.1350  180.4400 N.am      2 GLN4   -0.3025
     13 CA        185.6800  138.3980  180.1070 C.3       2 GLN4    0.0999
     14 C         185.0270  138.2790  178.7360 C.2       2 GLN4    0.2340
     15 O         185.1500  137.2600  178.0500 O.2       2 GLN4   -0.2738
     16 CB        186.6840  139.5570  180.1350 C.3       2 GLN4   -0.0177
     17 CG        187.7500  139.4280  181.2050 C.3       2 GLN4    0.0259
     18 CD        188.8590  140.4420  181.0420 C.2       2 GLN4    0.2106
     19 OE1       188.6130  141.5970  180.6960 O.2       2 GLN4   -0.2760
     20 NE2       190.0910  140.0150  181.2890 N.am      2 GLN4   -0.3293
     21 N         184.3260  139.3340  178.3350 N.am      3 ARG5   -0.3025
     22 CA        183.6250  139.3410  177.0590 C.3       3 ARG5    0.0995
     23 C         184.0570  140.5160  176.1890 C.2       3 ARG5    0.2340
     24 O         184.1980  140.3800  174.9740 O.2       3 ARG5   -0.2738
     25 CB        182.1130  139.3800  177.2810 C.3       3 ARG5   -0.0243
     26 CG        181.3190  138.5900  176.2570 C.3       3 ARG5   -0.0305
     27 CD        179.8770  138.4090  176.6950 C.3       3 ARG5    0.0512
     28 NE        179.6300  137.0830  177.2500 N.2       3 ARG5   -0.2178
     29 CZ        179.5300  135.9740  176.5300 C.cat     3 ARG5    0.1058
     30 NH1       179.6460  135.9920  175.2120 N.2       3 ARG5   -0.2337
     31 NH2       179.3040  134.8190  177.1470 N.2       3 ARG5   -0.2337
     32 N         184.2650  141.6700  176.8150 N.am      4 TYC6   -0.3022
     33 CA        184.6950  142.8550  176.0950 C.3       4 TYC6    0.1031
     34 C         185.9800  143.4000  176.7080 C.2       4 TYC6    0.2311
     35 O         186.7410  142.6660  177.2410 O.2       4 TYC6   -0.2740
     36 CB        183.5930  143.9090  176.0940 C.3       4 TYC6   -0.0007
     37 CG        182.3450  143.4670  175.3360 C.ar      4 TYC6   -0.0449
     38 CD1       181.2290  143.0390  176.0350 C.ar      4 TYC6   -0.0551
     39 CD2       182.3300  143.4910  173.9530 C.ar      4 TYC6   -0.0551
     40 CE1       180.0950  142.6360  175.3490 C.ar      4 TYC6   -0.0199
     41 CE2       181.1970  143.0870  173.2680 C.ar      4 TYC6   -0.0199
     42 OH        178.9350  142.2540  173.2750 O.3       4 TYC6   -0.5068
     43 CZ        180.0800  142.6610  173.9660 C.ar      4 TYC6    0.1169
     44 NXT       186.2700  144.8190  176.6290 N.am      4 TYC6   -0.3278
@<TRIPOS>BOND
     1   23   22 1
     2   23   24 2
     3   23   32 am
     4   34   33 1
     5   34   35 2
     6   34   44 am
     7   33   32 1
     8   33   36 1
     9   36   37 1
    10   38   37 ar
    11   38   40 ar
    12   39   37 ar
    13   39   41 ar
    14   40   43 ar
    15   41   43 ar
    16   43   42 1
    17    1    2 1
    18    2    3 1
    19    2    5 1
    20    3    4 2
    21    5    6 1
    22    6    7 1
    23    7    8 1
    24    8    9 1
    25    9   10 1
    26    9   11 1
    27   12   13 1
    28   13   16 1
    29   13   14 1
    30   14   15 2
    31   16   17 1
    32   17   18 1
    33   18   19 2
    34   18   20 am
    35    3   12 am
    36   21   22 1
    37   22   25 1
    38   25   26 1
    39   26   27 1
    40   27   28 1
    41   28   29 1
    42   29   30 1
    43   29   31 1
    44   14   21 am
@<TRIPOS>SUBSTRUCTURE
     1 ARG3     2 RESIDUE           4 A     ARG3     1 ROOT
     2 GLN4    13 RESIDUE           4 A     GLN4     2
     3 ARG5    22 RESIDUE           4 A     ARG5     2
     4 TYC6    33 RESIDUE           4 A     TYC6     1
)mol2");

  auto verify_sub = [this]() {
    ASSERT_EQ(mol().substructures().size(), 4);

    for (auto &sub: mol().substructures()) {
      switch (sub.id()) {
      case 1:
        EXPECT_EQ(sub.name(), "ARG3");
        EXPECT_EQ(sub.num_atoms(), 11);
        break;
      case 2:
        EXPECT_EQ(sub.name(), "GLN4");
        EXPECT_EQ(sub.num_atoms(), 9);
        break;
      case 3:
        EXPECT_EQ(sub.name(), "ARG5");
        EXPECT_EQ(sub.num_atoms(), 11);
        break;
      case 4:
        EXPECT_EQ(sub.name(), "TYC6");
        EXPECT_EQ(sub.num_atoms(), 13);
        break;
      default:
        FAIL() << "Unexpected substructure id: " << sub.id();
      }
    }
  };

  std::string mol2;

  NURI_FMT_TEST_NEXT_MOL("7yoo-cut", 44, 44);
  {
    SCOPED_TRACE("Initial read");
    verify_sub();
    write_mol2(mol2, mol());
  }

  set_test_string(mol2);

  NURI_FMT_TEST_NEXT_MOL("7yoo-cut", 44, 44);
  {
    SCOPED_TRACE("Re-read");
    verify_sub();
  }
}

TEST_F(Mol2Test, ExtraProp) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
****
 1 0 0 0 0
SMALL
GASTEIGER
****

@<TRIPOS>ATOM
      1 N           0.0000    0.0000    0.0000 N.4     1  UNL1        1.0000
@<TRIPOS>UNITY_ATOM_ATTR
1 1
test prop
)mol2");

  NURI_FMT_TEST_NEXT_MOL("", 1, 0);
  auto atom = mol()[0];
  auto it = internal::find_key(atom.data().props(), "test");
  EXPECT_NE(it, atom.data().props().end());
  EXPECT_EQ(it->second, "prop");

  std::string mol2;
  write_mol2(mol2, mol());
  set_test_string(mol2);

  NURI_FMT_TEST_NEXT_MOL("", 1, 0);
  atom = mol()[0];
  it = internal::find_key(atom.data().props(), "test");
  EXPECT_NE(it, atom.data().props().end());
  EXPECT_EQ(it->second, "prop");
}

TEST_F(Mol2Test, WriteFormatCorrect) {
  set_test_string(R"mol2(
@<TRIPOS>MOLECULE
imidazole
   5    5    1    0    0
SMALL
NO_CHARGES


@<TRIPOS>ATOM
   1 N1             0.0021    -0.0041     0.0020 N.pl3 1 UNK 0.000
   2 C3            -0.0165     1.3646     0.0095 C.2   1 UNK 0.000
   3 C4             1.2671     1.7717    -0.0005 C.2   1 UNK 0.000
   4 N5             2.0482     0.6814    -0.0138 N.2   1 UNK 0.000
   5 C6             1.2973    -0.3859    -0.0124 C.2   1 UNK 0.000
@<TRIPOS>BOND
   1    1    5 1
   2    1    2 1
   3    2    3 2
   4    3    4 1
   5    4    5 2
@<TRIPOS>SUBSTRUCTURE
1 UNK 1
)mol2");

  NURI_FMT_TEST_NEXT_MOL("imidazole", 5, 5);

  std::string mol2;
  write_mol2(mol2, mol());
  NURI_EXPECT_STRTRIM_EQ(mol2, R"mol2(@<TRIPOS>MOLECULE
imidazole
5 5 1 0 0
SMALL
NO_CHARGES
****

@<TRIPOS>ATOM
1 N1  0.002 -0.004  0.002 N.ar  1 UNK 0.000
2 C1 -0.017  1.365  0.009 C.ar  1 UNK 0.000
3 C2  1.267  1.772 -0.001 C.ar  1 UNK 0.000
4 N2  2.048  0.681 -0.014 N.ar  1 UNK 0.000
5 C3  1.297 -0.386 -0.012 C.ar  1 UNK 0.000
@<TRIPOS>BOND
1 1 5 1
2 1 2 1
3 2 3 2
4 3 4 1
5 4 5 2
@<TRIPOS>SUBSTRUCTURE
1 UNK 1
)mol2");

  mol2.clear();
  write_mol2(mol2, mol(), -1, false);
  NURI_EXPECT_STRTRIM_EQ(mol2, R"mol2(@<TRIPOS>MOLECULE
imidazole
5 5 1 0 0
SMALL
NO_CHARGES
****

@<TRIPOS>ATOM
1 N1  0.002 -0.004  0.002 N.ar  1 UNK 0.000
2 C1 -0.017  1.365  0.009 C.ar  1 UNK 0.000
3 C2  1.267  1.772 -0.001 C.ar  1 UNK 0.000
4 N2  2.048  0.681 -0.014 N.ar  1 UNK 0.000
5 C3  1.297 -0.386 -0.012 C.ar  1 UNK 0.000
@<TRIPOS>BOND
1 1 5 1
2 1 2 1
3 2 3 2
4 3 4 1
5 4 5 2
)mol2");
}

TEST_F(Mol2Test, Write2D) {
  Molecule m;
  {
    auto mut = m.mutator();
    for (int i = 0; i < 6; ++i)
      mut.add_atom(kPt[6]);
    for (int i = 0; i < 5; ++i)
      mut.add_bond(i, i + 1, BondData { constants::kSingleBond });
  }

  guess_hydrogens_2d(m);
  ASSERT_TRUE(MoleculeSanitizer(m).sanitize_all());

  std::string mol2;
  write_mol2(mol2, m);
  set_test_string(mol2);

  NURI_FMT_TEST_NEXT_MOL("", 6, 5);

  ASSERT_EQ(mol().confs().size(), 1);
  EXPECT_TRUE((mol().confs()[0].array().abs2() < 1e-12).all());
}
}  // namespace
}  // namespace nuri
