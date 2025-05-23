//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_CONTAINER_CONTAINER_EXT_H_
#define NURI_CORE_CONTAINER_CONTAINER_EXT_H_

//! @cond
#include <cstddef>
#include <functional>
#include <queue>
#include <vector>

#include <absl/algorithm/container.h>
//! @endcond

namespace nuri {
namespace internal {
  template <class T, class C = std::less<>, class S = std::vector<T>>
  struct ClearablePQ: public std::priority_queue<T, S, C> {
    using Base = std::priority_queue<T, S, C>;

  public:
    using Base::Base;

    T pop_get() noexcept {
      T v = std::move(data().front());
      this->pop();
      return v;
    }

    void clear() noexcept { data().clear(); }

    auto &data() { return this->c; }

    void rebuild() noexcept { absl::c_make_heap(data(), C()); }
  };
}  // namespace internal

#if __cplusplus >= 202002L
using std::erase;
using std::erase_if;
#else
template <class T, class Alloc, class U>
typename std::vector<T, Alloc>::size_type erase(std::vector<T, Alloc> &c,
                                                const U &value) {
  auto it = std::remove(c.begin(), c.end(), value);
  auto r = std::distance(it, c.end());
  c.erase(it, c.end());
  return r;
}

template <class T, class Alloc, class Pred>
typename std::vector<T, Alloc>::size_type erase_if(std::vector<T, Alloc> &c,
                                                   Pred &&pred) {
  auto it = std::remove_if(c.begin(), c.end(), std::forward<Pred>(pred));
  auto r = std::distance(it, c.end());
  c.erase(it, c.end());
  return r;
}
#endif

template <class T, class Alloc, class Pred>
typename std::vector<T, Alloc>::iterator erase_first(std::vector<T, Alloc> &c,
                                                     Pred &&pred) {
  auto it = std::find_if(c.begin(), c.end(), std::forward<Pred>(pred));
  if (it != c.end()) {
    return c.erase(it);
  }
  return it;
}

template <class T, size_t N>
constexpr size_t array_size(T (& /* arr */)[N]) {
  return N;
}
}  // namespace nuri

#endif /* NURI_CORE_CONTAINER_CONTAINER_EXT_H_ */
