//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_META_H_
#define NURI_META_H_

/// @cond
#include <iterator>
#include <type_traits>
/// @endcond

namespace nuri {
namespace internal {
  // Use of std::underlying_type_t on non-enum types is UB until C++20.
#if __cplusplus >= 202002L
  using std::underlying_type;
  using std::underlying_type_t;

  using std::remove_cvref;
  using std::remove_cvref_t;
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

  template <class T>
  struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };

  template <class T>
  using remove_cvref_t = typename remove_cvref<T>::type;
#endif

  template <class To, class From>
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr inline bool is_implicitly_constructible_v =
      std::is_constructible_v<To, From> && std::is_convertible_v<From, To>;

  template <class To, class From>
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr inline bool is_explicitly_constructible_v =
      std::is_constructible_v<To, From> && !std::is_convertible_v<From, To>;

  template <bool is_const, class T>
  struct const_if {
    using type = std::conditional_t<is_const, const T, T>;
  };

  template <bool is_const, class T>
  using const_if_t = typename const_if<is_const, T>::type;

  template <class Iterator, class T, class IfTrue = int>
  using enable_if_compatible_iter_t =
      std::enable_if_t<is_implicitly_constructible_v<
                           T, typename std::iterator_traits<Iterator>::reference>,
                       IfTrue>;

  template <class Iter, class IteratorTag, class IfTrue = int>
  using enable_if_iter_category_t = std::enable_if_t<
      std::is_same_v<typename std::iterator_traits<Iter>::iterator_category,
                     IteratorTag>,
      IfTrue>;

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
}  // namespace internal
}  // namespace nuri

#endif /* NURI_META_H_ */
