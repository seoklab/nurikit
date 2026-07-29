//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/parse_result.h"

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace nuri {
namespace {
using Result = ParseResult<std::vector<int>>;

TEST(ParseResultTest, HoldValue) {
  Result res = std::vector { 1, 2, 3 };

  EXPECT_EQ(res.status(), ParseStatus::kValid);
  EXPECT_TRUE(res);
  EXPECT_EQ(res->size(), 3);
  EXPECT_EQ((*res)[1], 2);

  (*res)[1] = 20;
  EXPECT_EQ(res->at(1), 20);
}

TEST(ParseResultTest, HoldError) {
  Result res = Result::error("bad input");

  EXPECT_EQ(res.status(), ParseStatus::kError);
  EXPECT_FALSE(res);
  EXPECT_EQ(res.error_msg(), "bad input");
}

TEST(ParseResultTest, HoldEof) {
  Result res = Result::eof();

  EXPECT_EQ(res.status(), ParseStatus::kEOF);
  EXPECT_FALSE(res);
}

TEST(ParseResultTest, FormatError) {
  Result res = Result::error("cannot parse line ", 42, ": ", "x y z");
  ASSERT_EQ(res.status(), ParseStatus::kError);
  EXPECT_EQ(res.error_msg(), "cannot parse line 42: x y z");
}

TEST(ParseResultTest, Move) {
  Result res = std::vector { 1, 2, 3 };
  std::vector<int> moved = *std::move(res);
  EXPECT_EQ(moved.size(), 3);

  Result other = Result::error("boom");
  Result moved_res = std::move(other);
  ASSERT_EQ(moved_res.status(), ParseStatus::kError);
  EXPECT_EQ(moved_res.error_msg(), "boom");

  moved_res = Result::eof();
  EXPECT_EQ(moved_res.status(), ParseStatus::kEOF);
}

TEST(ParseResultTest, ConstAccess) {
  const Result res = std::vector { 1, 2, 3 };
  EXPECT_EQ(res->size(), 3);
  EXPECT_EQ((*res)[0], 1);
}
}  // namespace
}  // namespace nuri
