//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/base.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace nuri {
namespace {
TEST(ReversedStreamTest, HandleEmptyFile) {
  std::istringstream input("");
  ReversedStream reversed(input, '\n', 7);

  std::string line;
  ASSERT_FALSE(reversed.getline(line));
}

TEST(ReversedStreamTest, HandleSingleLineFile) {
  std::istringstream input("single line");
  ReversedStream reversed(input, '\n', 7);

  std::string line;
  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "single line");

  ASSERT_FALSE(reversed.getline(line));
}

TEST(ReversedStreamTest, HandleMultipleNewlines) {
  std::istringstream input("single line\n\n");
  ReversedStream reversed(input, '\n', 7);

  std::string line;
  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "");

  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "single line");

  ASSERT_FALSE(reversed.getline(line));
}

TEST(ReversedStreamTest, ReadBackwardsLines) {
  std::istringstream input("line1\nline2\nline3\nline4");
  ReversedStream reversed(input, '\n', 7);

  std::string line;
  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "line4");

  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "line3");

  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "line2");

  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "line1");

  ASSERT_FALSE(reversed.getline(line));
}

TEST(ReversedStreamTest, ReadBackwardsMixed) {
  std::istringstream iss(" a   bcd ");
  std::string tok;

  std::vector<std::string> forward;
  while (std::getline(iss, tok, ' ')) {
    forward.push_back(tok);
  }

  iss.clear();
  ReversedStream rs(iss, ' ', 2);
  std::vector<std::string> backward;
  while (rs.getline(tok)) {
    backward.push_back(tok);
  }

  for (size_t i = 0; i < forward.size(); ++i) {
    EXPECT_EQ(forward[i], backward[backward.size() - i - 1]);
  }
}
}  // namespace
}  // namespace nuri
