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
}  // namespace
