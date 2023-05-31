//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include <gtest/gtest.h>

namespace {
TEST(TempTest, A) {
  uint32_t i = -static_cast<uint32_t>(true);
  EXPECT_EQ(UINT_MAX, i);
}
}  // namespace
