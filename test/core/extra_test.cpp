//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>

#include <gtest/gtest.h>

#include "nuri/core/container/bool_matrix.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
struct CastBase {
  virtual ~CastBase() = default;
};

struct CastDerived: CastBase { };

TEST(DownCastTest, DowncastsPointers) {
  CastDerived derived;
  CastBase *base = &derived;
  EXPECT_EQ(down_cast<CastDerived *>(base), &derived);

  const CastBase *const_base = &derived;
  EXPECT_EQ(down_cast<const CastDerived *>(const_base), &derived);
}

TEST(DownCastTest, DowncastsNullPointers) {
  CastBase *base = nullptr;
  EXPECT_EQ(down_cast<CastDerived *>(base), nullptr);

  const CastBase *const_base = nullptr;
  EXPECT_EQ(down_cast<const CastDerived *>(const_base), nullptr);
}

TEST(DownCastTest, DowncastsReferences) {
  CastDerived derived;
  CastBase &base = derived;
  EXPECT_EQ(&down_cast<CastDerived &>(base), &derived);

  const CastBase &const_base = derived;
  EXPECT_EQ(&down_cast<const CastDerived &>(const_base), &derived);

  CastDerived &&rvalue = down_cast<CastDerived &&>(std::move(base));
  EXPECT_EQ(&rvalue, &derived);

  const CastDerived &&const_rvalue =
      down_cast<const CastDerived &&>(std::move(const_base));
  EXPECT_EQ(&const_rvalue, &derived);
}

TEST(DownCastTest, TransfersUniqueOwnership) {
  struct Counted: CastBase {
    explicit Counted(int &destroyed): destroyed_(&destroyed) { }
    ~Counted() override { ++*destroyed_; }

    int *destroyed_;
  };

  int destroyed = 0;
  std::unique_ptr<CastBase> base = std::make_unique<Counted>(destroyed);
  auto *original = base.get();
  {
    auto derived = down_cast<Counted>(std::move(base));
    EXPECT_EQ(base, nullptr);
    EXPECT_EQ(derived.get(), original);
    EXPECT_EQ(destroyed, 0);
  }
  EXPECT_EQ(destroyed, 1);
}

TEST(DownCastTest, DowncastsConstUniqueOwnership) {
  std::unique_ptr<const CastBase> base = std::make_unique<const CastDerived>();
  const auto *original = base.get();
  auto derived = down_cast<const CastDerived>(std::move(base));
  EXPECT_EQ(base, nullptr);
  EXPECT_EQ(derived.get(), original);
}

TEST(DownCastTest, DowncastsNullUniqueOwnership) {
  std::unique_ptr<CastBase> base;
  EXPECT_EQ(down_cast<CastDerived>(std::move(base)), nullptr);
}

TEST(BoolMatrixTest, EliminationTest) {
  /*
   * [[0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
   *  [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
   *  [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
   *  [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
   *  [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
   *  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]] (transposed)
   */
  BoolMatrix mat(12, 6);

  int c = 0;
  auto assign_col = [&, c](std::vector<bool> col) mutable {
    for (int i = 0; i < col.size(); ++i)
      mat.assign(i, c, col[i]);
    ++c;
  };

  assign_col({ false, true, false, false, true, false, false, true, true, true,
               false, true });
  assign_col({ true, false, false, false, false, false, false, true, false,
               true, false, false });
  assign_col({ false, true, false, true, true, false, false, false, false,
               false, false, false });
  assign_col({ true, true, false, true, true, false, false, true, true, false,
               false, true });
  // Dependent col (0 ^ 2 = 4)
  assign_col({ false, false, false, true, false, false, false, true, true, true,
               false, true });
  assign_col({ false, false, false, true, true, true, false, false, false, true,
               true, false });

  std::vector<int> result = mat.gaussian_elimination();
  for (int i = 0; i < result.size(); ++i)
    EXPECT_EQ(static_cast<bool>(result[i]), i != 4);
}

TEST(PowerSetTest, Correctness) {
  internal::PowersetStream ps(7);

  std::vector<unsigned int> gen;
  gen.reserve(1U << 7);

  unsigned int i;
  while (ps >> i)
    gen.push_back(i);

  EXPECT_EQ(gen.size(), (1U << 7) - 1);

  absl::c_sort(gen);
  for (i = 0; i < gen.size(); ++i)
    EXPECT_EQ(gen[i], i + 1) << i;
}
}  // namespace
}  // namespace nuri
