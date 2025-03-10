//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TEST_TEST_UTILS_H_
#define NURI_TEST_TEST_UTILS_H_

#include <string_view>
#include <utility>

#include <absl/log/absl_check.h>
#include <absl/strings/ascii.h>
#include <absl/strings/str_split.h>

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

#define NURI_EXPECT_EIGEN_EQ(a, b)                                             \
  EXPECT_PRED2(                                                                \
      (nuri::internal::eigen_eq<std::remove_reference_t<decltype(a)>,          \
                                std::remove_reference_t<decltype(b)>>),        \
      (a), (b))
#define NURI_EXPECT_EIGEN_NE(a, b)                                             \
  EXPECT_PRED2(                                                                \
      (nuri::internal::eigen_ne<std::remove_reference_t<decltype(a)>,          \
                                std::remove_reference_t<decltype(b)>>),        \
      (a), (b))

#define NURI_EXPECT_EIGEN_EQ_TOL(a, b, tol)                                    \
  EXPECT_PRED3(                                                                \
      (nuri::internal::eigen_eq_tol<std::remove_reference_t<decltype(a)>,      \
                                    std::remove_reference_t<decltype(b)>>),    \
      (a), (b), (tol))
#define NURI_EXPECT_EIGEN_NE_TOL(a, b, tol)                                    \
  EXPECT_PRED3(                                                                \
      (nuri::internal::eigen_ne_tol<std::remove_reference_t<decltype(a)>,      \
                                    std::remove_reference_t<decltype(b)>>),    \
      (a), (b), (tol))

#define NURI_EXPECT_STRTRIM_EQ(a, b)                                           \
  EXPECT_PRED2(nuri::internal::expect_line_eq_trim, (a), (b))

#define NURI_WRITE_ONCE(func, ...)                                             \
  nuri::internal::write_once_impl(                                             \
      [](auto &&...args) -> decltype(auto) {                                   \
        return (func)(std::forward<decltype(args)>(args)...);                  \
      },                                                                       \
      ##__VA_ARGS__)

namespace nuri {
namespace internal {
template <class M, class N>
bool eigen_eq_tol(const M &a, const N &b, double tol) {
  auto diff = a - b;
  return (diff.array().abs() < tol).all();
}

template <class M, class N>
bool eigen_ne_tol(const M &a, const N &b, double tol) {
  return !eigen_eq_tol(a, b, tol);
}

template <class M, class N>
bool eigen_eq(const M &a, const N &b) {
  return eigen_eq_tol(a, b, 1e-6);
}

template <class M, class N>
bool eigen_ne(const M &a, const N &b) {
  return !eigen_eq(a, b);
}

inline bool expect_line_eq_trim(std::string_view lhs, std::string_view rhs) {
  auto lhs_split = absl::StrSplit(lhs, '\n'),
       rhs_split = absl::StrSplit(rhs, '\n');

  auto lit = lhs_split.begin(), rit = rhs_split.begin();
  for (; lit != lhs_split.end() && rit != rhs_split.end(); ++lit, ++rit) {
    if (absl::StripAsciiWhitespace(*lit) != absl::StripAsciiWhitespace(*rit))
      return false;
  }

  return lit == lhs_split.end() && rit == rhs_split.end();
}

inline Molecule read_first(std::string_view fmt, std::string_view data) {
  StringMoleculeReader<> reader(fmt, std::string { data });
  MoleculeStream<> stream = reader.stream();
  ABSL_CHECK(stream.advance());
  return stream.current();
}

template <class Func, class... Args>
std::string write_once_impl(Func &&func, Args &&...args) {
  std::string out;
  std::invoke(std::forward<Func>(func), out, std::forward<Args>(args)...);
  return out;
}
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TEST_TEST_UTILS_H_ */
