//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_META_H_
#define NURI_META_H_

//! @cond
#include <iterator>
#include <type_traits>
//! @endcond

//! @privatesection

#if defined(__clang_analyzer__) && __clang_major__ >= 18
#define NURI_CLANG_ANALYZER_NOLINT       [[clang::suppress]]
#define NURI_CLANG_ANALYZER_NOLINT_BEGIN [[clang::suppress]] {
#define NURI_CLANG_ANALYZER_NOLINT_END   }
#else
#define NURI_CLANG_ANALYZER_NOLINT
#define NURI_CLANG_ANALYZER_NOLINT_BEGIN
#define NURI_CLANG_ANALYZER_NOLINT_END
#endif

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

  template <class T, std::enable_if_t<
                         std::is_enum_v<T> || std::is_arithmetic_v<T>, int> = 0>
  constexpr auto extract_if_enum_v(T val) {
    return static_cast<extract_if_enum_t<T>>(val);
  }

  /**
   * This class and the overload_cast variable template are imported from
   * pybind11. The original code is licensed under the following license:
   *
   * \code{.unparsed}
   * Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights
   * reserved.
   *
   * Redistribution and use in source and binary forms, with or without
   * modification, are permitted provided that the following conditions are met:
   *
   * 1. Redistributions of source code must retain the above copyright notice,
   * this list of conditions and the following disclaimer.
   *
   * 2. Redistributions in binary form must reproduce the above copyright
   * notice, this list of conditions and the following disclaimer in the
   * documentation and/or other materials provided with the distribution.
   *
   * 3. Neither the name of the copyright holder nor the names of its
   * contributors may be used to endorse or promote products derived from this
   * software without specific prior written permission.
   *
   * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   * POSSIBILITY OF SUCH DAMAGE.
   *
   * Please also refer to the file .github/CONTRIBUTING.md, which clarifies
   * licensing of external contributions to this project including patches, pull
   * requests, etc.
   * \endcode
   */
  template <class... Args>
  struct overload_cast_impl {
    template <class Return>
    constexpr decltype(auto)
    operator()(Return (*func)(Args...)) const noexcept {
      return func;
    }

    template <typename Return, typename Class>
    constexpr decltype(auto)
    operator()(Return (Class::*pmf)(Args...),
               std::false_type /* dispatch */ = {}) const noexcept {
      return pmf;
    }

    template <typename Return, typename Class>
    constexpr decltype(auto)
    operator()(Return (Class::*pmf)(Args...) const,
               std::true_type /* dispatch */) const noexcept {
      return pmf;
    }
  };
}  // namespace internal

template <class... Args>
// NOLINTNEXTLINE(readability-identifier-naming)
constexpr inline static internal::overload_cast_impl<Args...> overload_cast;
}  // namespace nuri

#endif /* NURI_META_H_ */
