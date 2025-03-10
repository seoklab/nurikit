//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/graph.h"

#include <vector>

#include <absl/algorithm/container.h>

#include "nuri/utils.h"

namespace nuri {
namespace internal {
  void IndexSet::remap(const std::vector<int> &old_to_new) {
    auto first =
        absl::c_find_if(*this, [&](int id) { return old_to_new[id] < 0; });

    for (auto it = begin(); it < first; ++it)
      *it = old_to_new[*it];

    for (auto it = first; it < end(); ++it) {
      int new_id = old_to_new[*it];
      *first = new_id;
      first += value_if(new_id >= 0);
    }

    erase(first, end());
  }
}  // namespace internal
}  // namespace nuri
