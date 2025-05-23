//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_CONTAINER_COMPACT_MAP_H_
#define NURI_CORE_CONTAINER_COMPACT_MAP_H_

//! @cond
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
//! @endcond

namespace nuri {
namespace internal {
  template <class K, class V, V sentinel = -1>
  class CompactMap {
  public:
    static_assert(std::is_convertible_v<K, size_t>,
                  "Key must be an integer-like type");

    using key_type = K;
    using mapped_type = V;
    using value_type = V;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = V &;
    using const_reference = const V &;
    using pointer = V *;
    using const_pointer = const V *;

    CompactMap(size_t cap): data_(cap, sentinel) { }

    template <class... Args>
    std::pair<pointer, bool> try_emplace(key_type key, Args &&...args) {
      handle_resize(key);

      reference v = data_[key];
      bool isnew = v == sentinel;
      if (isnew) {
        v = V(std::forward<Args>(args)...);
      }
      return { &v, isnew };
    }

    pointer find(key_type key) {
      if (key < data_.size()) {
        return data_[key] == sentinel ? nullptr : &data_[key];
      }
      return nullptr;
    }

    const_pointer find(key_type key) const {
      if (key < data_.size()) {
        return data_[key] == sentinel ? nullptr : &data_[key];
      }
      return nullptr;
    }

  private:
    void handle_resize(key_type new_key) {
      if (new_key >= data_.size()) {
        data_.resize(new_key + 1, sentinel);
      }
    }

    std::vector<V> data_;
  };
}  // namespace internal
}  // namespace nuri

#endif /* NURI_CORE_CONTAINER_COMPACT_MAP_H_ */
