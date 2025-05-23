//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_CONTAINER_INDEX_SET_H_
#define NURI_CORE_CONTAINER_INDEX_SET_H_

//! @cond
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <boost/container/flat_set.hpp>
//! @endcond

#include "nuri/core/container/container_ext.h"

namespace nuri {
namespace internal {
  class IndexSet
      : public boost::container::flat_set<int, std::less<>, std::vector<int>> {
  private:
    using Base = boost::container::flat_set<int, std::less<>, std::vector<int>>;

  public:
    using Base::Base;

    explicit IndexSet(std::vector<int> &&vec) noexcept {
      adopt_sequence(std::move(vec));
    }

    IndexSet(boost::container::ordered_unique_range_t tag,
             std::vector<int> &&vec) noexcept {
      adopt_sequence(tag, std::move(vec));
    }

    template <class UnaryPred>
    void erase_if(UnaryPred &&pred) {
      std::vector<int> work = extract_sequence();
      nuri::erase_if(work, std::forward<UnaryPred>(pred));
      adopt_sequence(boost::container::ordered_unique_range, std::move(work));
    }

    void union_with(const IndexSet &other) {
      std::vector<int> result;
      result.reserve(size() + other.size());
      absl::c_set_union(*this, other, std::back_inserter(result));
      adopt_sequence(boost::container::ordered_unique_range, std::move(result));
    }

    void difference(const IndexSet &other) {
      std::vector<int> result;
      result.reserve(size());
      absl::c_set_difference(*this, other, std::back_inserter(result));
      adopt_sequence(boost::container::ordered_unique_range, std::move(result));
    }

    int operator[](int idx) const { return sequence()[idx]; }

    int find_index(int id) const {
      auto it = find(id);
      return static_cast<int>(it - begin());
    }

    void remap(const std::vector<int> &old_to_new);
  };
}  // namespace internal
}  // namespace nuri

#endif /* NURI_CORE_CONTAINER_INDEX_SET_H_ */
