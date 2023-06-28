//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/bool_matrix.h"

#include <vector>

#include <gtest/gtest.h>

namespace {
using nuri::BoolMatrix;

TEST(BoolMatrixTest, EliminationTest) {
  /*
   * [[0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
   *  [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
   *  [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
   *  [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
   *  [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
   *  [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]]
   */
  BoolMatrix mat(6, 12);

  int r = 0;
  auto assign_row = [&, r](std::vector<bool> row) mutable {
    for (int j = 0; j < row.size(); ++j) {
      mat.assign(r, j, row[j]);
    }
    ++r;
  };

  assign_row({ false, true, false, false, true, false, false, true, true, true,
               false, true });
  assign_row({ true, false, false, false, false, false, false, true, false,
               true, false, false });
  assign_row({ false, true, false, true, true, false, false, false, false,
               false, false, false });
  assign_row({ true, true, false, true, true, false, false, true, true, false,
               false, true });
  // Dependent row (0 ^ 2 = 4)
  assign_row({ false, false, false, true, false, false, false, true, true, true,
               false, true });
  assign_row({ false, false, false, true, true, true, false, false, false, true,
               true, false });

  std::vector<int> result = mat.gaussian_elimination();
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_EQ(static_cast<bool>(result[i]), i != 4);
  }
}
}  // namespace
