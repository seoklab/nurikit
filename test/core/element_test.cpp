//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/element.h"

#include <algorithm>

#include <gtest/gtest.h>

namespace {
using nuri::Element;
using nuri::Isotope;
using nuri::PeriodicTable;

class PeriodicTableTest: public ::testing::Test {
public:
  // NOLINTNEXTLINE(*-ref-data-members,readability-identifier-naming)
  const PeriodicTable &table_ = PeriodicTable::get();
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
  ASSERT_TRUE(table_.has_element("H"));

  const Element &hydrogen = *table_.find_element("H");
  EXPECT_EQ(hydrogen.atomic_number(), 1);
  EXPECT_EQ(hydrogen.symbol(), "H");
  EXPECT_EQ(hydrogen.name(), "Hydrogen");

  EXPECT_FALSE(table_.has_element("Aa"));
  EXPECT_EQ(table_.find_element("Aa"), nullptr);
}

TEST_F(PeriodicTableTest, NameTest) {
  ASSERT_TRUE(table_.has_element_of_name("Helium"));

  const Element &helium = *table_.find_element_of_name("Helium");
  EXPECT_EQ(helium.atomic_number(), 2);
  EXPECT_EQ(helium.symbol(), "He");
  EXPECT_EQ(helium.name(), "Helium");

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
      }

      EXPECT_NEAR(elem.atomic_weight(), avg_wt, tol)
        << elem.atomic_number() << elem.symbol();
    } else {
      EXPECT_EQ(elem.atomic_weight(), elem.major_isotope().mass_number)
        << elem.atomic_number() << elem.symbol();
    }
  }
}
}  // namespace
