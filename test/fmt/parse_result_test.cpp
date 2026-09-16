//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/parse_result.h"

#include <string>
#include <string_view>
#include <type_traits>
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

TEST(ParseResultTest, MoveErrorString) {
  const std::string expected(128, 'x');
  std::string message = expected;
  const char *original = message.data();
  auto result = Result::error(std::move(message));
  ASSERT_EQ(result.status(), ParseStatus::kError);
  EXPECT_EQ(result.error_msg(), expected);
  EXPECT_EQ(result.error_msg().data(), original);

  auto extracted = std::move(result).error_msg();
  static_assert(std::is_same_v<decltype(extracted), std::string>);
  EXPECT_EQ(extracted, expected);
  EXPECT_EQ(extracted.data(), original);
  result.reset();
  EXPECT_EQ(extracted, expected);
}

TEST(ParseResultTest, BorrowErrorString) {
  const std::string expected(128, 'x');
  std::string message = expected;
  auto result = Result::error(message);
  message.clear();

  auto borrowed = result.error_msg();
  static_assert(std::is_same_v<decltype(borrowed), std::string_view>);
  EXPECT_EQ(borrowed, expected);

  const auto &const_result = result;
  auto const_borrowed = const_result.error_msg();
  static_assert(std::is_same_v<decltype(const_borrowed), std::string_view>);
  EXPECT_EQ(const_borrowed, expected);
  EXPECT_EQ(const_borrowed.data(), borrowed.data());
}

TEST(ParseResultTest, TemporaryErrorString) {
  auto message = Result::error("bad input").error_msg();
  static_assert(std::is_same_v<decltype(message), std::string>);
  EXPECT_EQ(message, "bad input");
}
}  // namespace
}  // namespace nuri
