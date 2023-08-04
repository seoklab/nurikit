//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_UTILS_H_
#define NURI_UTILS_H_

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>

#include "nuri/eigen_config.h"

namespace nuri {
namespace internal {
  // Use of std::underlying_type_t on non-enum types is UB until C++20.

#if __cplusplus >= 202002L
  using std::underlying_type;
  using std::underlying_type_t;
#else
  template <class E, bool = std::is_enum_v<E>>
  struct underlying_type { };

  template <class E>
  struct underlying_type<E, false> { };

  template <class E>
  struct underlying_type<E, true> {
    using type = std::underlying_type_t<E>;
  };

  template <class T>
  using underlying_type_t = typename underlying_type<T>::type;
#endif
}  // namespace internal

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E operator|(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E &operator|=(E &self, E rhs) {
  return self = self | rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E operator&(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E &operator&=(E &self, E rhs) {
  return self = self & rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E operator^(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) ^ static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E &operator^=(E &self, E rhs) {
  return self = self ^ rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E operator~(E val) {
  return static_cast<E>(~static_cast<U>(val));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0,
          std::enable_if_t<std::is_unsigned_v<U>, int> = 0>
constexpr inline E operator-(E val) {
  return static_cast<E>(-static_cast<U>(val));
}

namespace internal {
  template <class T, bool = std::is_enum_v<T>>
  struct extract_if_enum { };

  template <class T>
  struct extract_if_enum<T, true> {
    using type = std::underlying_type_t<T>;
  };

  template <class T>
  struct extract_if_enum<T, false> {
    using type = T;
  };

  template <class T>
  using extract_if_enum_t = typename extract_if_enum<T>::type;

  template <class E>
  constexpr bool check_flag(E flags, E flag) {
    return static_cast<bool>(flags & flag);
  }

  template <class E, class U = extract_if_enum_t<E>,
            std::enable_if_t<std::is_unsigned_v<U>, U> = 0>
  constexpr E &update_flag(E &flags, bool cond, E flag) {
    E mask = -static_cast<E>(cond);
    flags = (flags & ~flag) | (mask & flag);
    return flags;
  }

  template <class E, class U = extract_if_enum_t<E>,
            std::enable_if_t<std::is_unsigned_v<U>, U> = 0>
  constexpr E &set_flag_if(E &flags, bool cond, E flag) {
    E mask = -static_cast<E>(cond);
    flags |= mask & flag;
    return flags;
  }

  template <class E, class U = extract_if_enum_t<E>,
            std::enable_if_t<std::is_unsigned_v<U>, U> = 0>
  constexpr E &unset_flag_if(E &flags, bool cond, E flag) {
    E mask = -static_cast<E>(cond);
    flags &= ~(mask & flag);
    return flags;
  }

  template <bool is_const, class T>
  struct const_if {
    using type = std::conditional_t<is_const, const T, T>;
  };

  template <bool is_const, class T>
  using const_if_t = typename const_if<is_const, T>::type;

  template <class Iterator, class T, class U = T>
  using enable_if_compatible_iter_t = std::enable_if_t<std::is_constructible_v<
    U, typename std::iterator_traits<Iterator>::reference>>;

  template <class F>
  int iround(F x) {
    return static_cast<int>(std::lround(x));
  }
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
                                                   Pred pred) {
  auto it = std::remove_if(c.begin(), c.end(), pred);
  auto r = std::distance(it, c.end());
  c.erase(it, c.end());
  return r;
}
#endif

template <class T, class Alloc, class Pred>
typename std::vector<T, Alloc>::iterator erase_first(std::vector<T, Alloc> &c,
                                                     Pred pred) {
  auto it = std::find_if(c.begin(), c.end(), pred);
  if (it != c.end()) {
    return c.erase(it);
  }
  return it;
}

template <class T, class Alloc, class Comp>
typename std::vector<T, Alloc>::iterator
insert_sorted(std::vector<T, Alloc> &c, const T &value, Comp comp) {
  return c.insert(std::upper_bound(c.begin(), c.end(), value, comp), value);
}

template <class T, class Alloc>
typename std::vector<T, Alloc>::iterator insert_sorted(std::vector<T, Alloc> &c,
                                                       const T &value) {
  return insert_sorted(c, value, std::less<>());
}

inline absl::FixedArray<int> generate_index(int size) {
  absl::FixedArray<int> result(size);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

template <class Container, class Comp>
absl::FixedArray<int> argsort(const Container &container, Comp op) {
  absl::FixedArray<int> idxs = generate_index(container.size());
  std::sort(idxs.begin(), idxs.end(),
            [&](int i, int j) { return op(container[i], container[j]); });
  return idxs;
}

template <class Container, class Comp>
absl::FixedArray<int> argpartition(const Container &container, int count,
                                   Comp op) {
  absl::FixedArray<int> idxs = generate_index(container.size());
  std::nth_element(idxs.begin(), idxs.begin() + count - 1, idxs.end(),
                   [&](int i, int j) {
                     return op(container[i], container[j]);
                   });
  return idxs;
}

template <class Container>
void mask_to_map(Container &mask) {
  typename Container::value_type idx = 0;
  for (int i = 0; i < mask.size(); ++i) {
    mask[i] = mask[i] ? idx++ : -1;
  }
}

template <typename Derived, typename Base>
std::unique_ptr<Derived>
static_unique_ptr_cast(std::unique_ptr<Base> &&p) noexcept {
  auto d = static_cast<Derived *>(p.release());
  return std::unique_ptr<Derived>(d);
}

inline std::string_view extension_no_dot(const std::filesystem::path &ext) {
  const std::string_view ext_view = ext.native();
  if (ABSL_PREDICT_TRUE(!ext_view.empty())) {
    return ext_view.substr(1);
  }
  return ext_view;
}

inline MatrixX3d stack(const std::vector<Vector3d> &vs) {
  MatrixX3d m(vs.size(), 3);
  for (int i = 0; i < vs.size(); ++i) {
    m.row(i) = vs[i];
  }
  return m;
}
}  // namespace nuri

#endif /* NURI_UTILS_H_ */
