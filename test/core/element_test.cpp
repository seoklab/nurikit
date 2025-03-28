//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/element.h"

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

namespace {
using nuri::Element;
using nuri::Isotope;
using nuri::PeriodicTable;

class PeriodicTableTest: public ::testing::Test {
public:
  // NOLINTNEXTLINE(*-ref-data-members,readability-identifier-naming)
  const PeriodicTable &table_ = nuri::kPt;
};

TEST_F(PeriodicTableTest, AtomicNumberTest) {
  for (int i = 0; i < PeriodicTable::kElementCount_; ++i) {
    ASSERT_TRUE(table_.has_element(i));
    const Element &elem = *table_.find_element(i);
    EXPECT_EQ(elem.atomic_number(), i);
    for (const Isotope &isotope: elem.isotopes()) {
      EXPECT_EQ(isotope.atomic_number, i);
    }
  }

  EXPECT_FALSE(table_.has_element(200));
  EXPECT_EQ(table_.find_element(200), nullptr);
}

TEST_F(PeriodicTableTest, SymbolTest) {
  ASSERT_TRUE(table_.has_element("He"));

  const Element &helium = *table_.find_element("He");
  EXPECT_EQ(helium.atomic_number(), 2);
  EXPECT_EQ(helium.symbol(), "He");
  EXPECT_EQ(helium.name(), "Helium");

  EXPECT_EQ(table_.find_element("HE"), &helium);
  EXPECT_EQ(table_.find_element("he"), &helium);

  EXPECT_FALSE(table_.has_element("Aa"));
  EXPECT_EQ(table_.find_element("Aa"), nullptr);
}

TEST_F(PeriodicTableTest, NameTest) {
  ASSERT_TRUE(table_.has_element_of_name("Helium"));

  const Element &helium = *table_.find_element_of_name("Helium");
  EXPECT_EQ(helium.atomic_number(), 2);
  EXPECT_EQ(helium.symbol(), "He");
  EXPECT_EQ(helium.name(), "Helium");

  EXPECT_EQ(table_.find_element_of_name("HELIUM"), &helium);
  EXPECT_EQ(table_.find_element_of_name("helium"), &helium);

  EXPECT_FALSE(table_.has_element_of_name("Random"));
  EXPECT_EQ(table_.find_element_of_name("Random"), nullptr);
}

TEST_F(PeriodicTableTest, ExtraSymbolNameTest) {
  ASSERT_TRUE(table_.has_element("X"));
  ASSERT_TRUE(table_.has_element("*"));

  const Element *dummy = table_.find_element("X");
  EXPECT_EQ(dummy, table_.find_element("*"));
  EXPECT_EQ(dummy->atomic_number(), 0);

  const Element *og = table_.find_element("Og");
  EXPECT_EQ(og, table_.find_element("Uuo"));
  EXPECT_EQ(og->atomic_number(), 118);
}

TEST_F(PeriodicTableTest, MajorIsotopeTest) {
  for (const Element &elem: table_) {
    const Isotope *major = &elem.major_isotope();
    ASSERT_NE(major, nullptr);
    ASSERT_TRUE(std::any_of(elem.isotopes().begin(), elem.isotopes().end(),
                            [=](const Isotope &iso) { return &iso == major; }));
  }
}

TEST_F(PeriodicTableTest, IsotopesTest) {
  for (const Element &elem: table_) {
    const auto &isotopes = elem.isotopes();
    if (std::any_of(isotopes.begin(), isotopes.end(),
                    [](const Isotope &iso) { return iso.abundance > 0; })) {
      double avg_wt = 0, total_abundance = 0;
      for (const Isotope &iso: isotopes) {
        avg_wt += iso.atomic_weight * iso.abundance;
        total_abundance += iso.abundance;
      }
      avg_wt /= total_abundance;

      double tol = 1e-3;
      // Lead has a large uncertainty
      if (elem.atomic_number() == 82) {
        tol = 1e-2;
      }
      EXPECT_NEAR(total_abundance, 1.0, tol)
          << elem.atomic_number() << elem.symbol();

      tol = 5e-3;
      // Some elements have larger uncertainty
      switch (elem.atomic_number()) {
      case 3:  // Li
        tol = 0.06;
        break;
      case 16:  // S
      case 44:  // Ru
        tol = 0.02;
        break;
      case 76:  // Os
        tol = 0.03;
        break;
      case 82:  // Pb
        tol = 1.1;
        break;
      default:
        break;
      }

      EXPECT_NEAR(elem.atomic_weight(), avg_wt, tol)
          << elem.atomic_number() << elem.symbol();
      EXPECT_FALSE(elem.radioactive());
    } else {
      EXPECT_EQ(elem.atomic_weight(), elem.major_isotope().mass_number)
          << elem.atomic_number() << elem.symbol();
      EXPECT_TRUE(elem.radioactive());
    }
  }
}

TEST_F(PeriodicTableTest, PeriodTest) {
  for (int i = 1; i <= 2; ++i) {
    EXPECT_EQ(table_.find_element(i)->period(), 1);
  }
  for (int i = 3; i <= 10; ++i) {
    EXPECT_EQ(table_.find_element(i)->period(), 2);
  }
  for (int i = 11; i <= 18; ++i) {
    EXPECT_EQ(table_.find_element(i)->period(), 3);
  }
  for (int i = 19; i <= 36; ++i) {
    EXPECT_EQ(table_.find_element(i)->period(), 4);
  }
  for (int i = 37; i <= 54; ++i) {
    EXPECT_EQ(table_.find_element(i)->period(), 5);
  }
  for (int i = 55; i <= 86; ++i) {
    EXPECT_EQ(table_.find_element(i)->period(), 6);
  }
  for (int i = 87; i <= 118; ++i) {
    EXPECT_EQ(table_.find_element(i)->period(), 7);
  }
}

TEST_F(PeriodicTableTest, GroupTest) {
  std::vector<int> used(PeriodicTable::kElementCount_, 0);

  auto check_group = [&](int z, int group) {
    used[z] = 1;
    EXPECT_EQ(table_.find_element(z)->group(), group);
  };

  auto check_group_for = [&](int group, const std::vector<int> &zs) {
    for (int z: zs) {
      check_group(z, group);
    }
  };

  // dummy
  check_group(0, 0);

  // s-block
  check_group_for(1, { 1, 3, 11, 19, 37, 55, 87 });
  check_group_for(2, { 4, 12, 20, 38, 56, 88 });

  // Group 3 includes lanthanides and actinides
  check_group_for(3, { 21, 39 });
  for (int i = 57; i <= 71; ++i) {
    check_group(i, 3);
  }
  for (int i = 89; i <= 103; ++i) {
    check_group(i, 3);
  }

  // Other d-block
  check_group_for(4, { 22, 40, 72, 104 });
  check_group_for(5, { 23, 41, 73, 105 });
  check_group_for(6, { 24, 42, 74, 106 });
  check_group_for(7, { 25, 43, 75, 107 });
  check_group_for(8, { 26, 44, 76, 108 });
  check_group_for(9, { 27, 45, 77, 109 });
  check_group_for(10, { 28, 46, 78, 110 });
  check_group_for(11, { 29, 47, 79, 111 });
  check_group_for(12, { 30, 48, 80, 112 });

  // p-block
  check_group_for(13, { 5, 13, 31, 49, 81, 113 });
  check_group_for(14, { 6, 14, 32, 50, 82, 114 });
  check_group_for(15, { 7, 15, 33, 51, 83, 115 });
  check_group_for(16, { 8, 16, 34, 52, 84, 116 });
  check_group_for(17, { 9, 17, 35, 53, 85, 117 });
  check_group_for(18, { 2, 10, 18, 36, 54, 86, 118 });

  // Check all elements are used
  EXPECT_TRUE(
      std::all_of(used.begin(), used.end(), [](int x) { return x == 1; }));
}

TEST_F(PeriodicTableTest, LanActTest) {
  for (const auto &elem: table_) {
    if (elem.atomic_number() >= 57 && elem.atomic_number() <= 71) {
      EXPECT_TRUE(elem.lanthanide());
      EXPECT_FALSE(elem.actinide());
    } else if (elem.atomic_number() >= 89 && elem.atomic_number() <= 103) {
      EXPECT_FALSE(elem.lanthanide());
      EXPECT_TRUE(elem.actinide());
    } else {
      EXPECT_FALSE(elem.lanthanide());
      EXPECT_FALSE(elem.actinide());
    }
  }
}
}  // namespace
