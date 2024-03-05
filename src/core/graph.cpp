//
// Project nurikit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/graph.h"

#include <vector>

#include <absl/algorithm/container.h>

#include "nuri/utils.h"

namespace nuri {
namespace internal {
  void SortedIdxs::remap(const std::vector<int> &old_to_new) {
    auto first =
        absl::c_find_if(idxs_, [&](int id) { return old_to_new[id] < 0; });

    for (auto it = idxs_.begin(); it < first; ++it)
      *it = old_to_new[*it];

    for (auto it = first; it < idxs_.end(); ++it) {
      int new_id = old_to_new[*it];
      *first = new_id;
      first += value_if(new_id >= 0);
    }

    idxs_.erase(first, idxs_.end());
  }
}  // namespace internal
}  // namespace nuri
