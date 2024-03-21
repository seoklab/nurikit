//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TEST_TEST_UTILS_H_
#define NURI_TEST_TEST_UTILS_H_

#include <string_view>

#include <absl/strings/ascii.h>
#include <absl/strings/str_split.h>

#define NURI_EXPECT_EIGEN_EQ(a, b)                                             \
  EXPECT_PRED2((nuri::internal::eigen_eq<decltype(a), decltype(b)>), (a), (b))
#define NURI_EXPECT_EIGEN_NE(a, b)                                             \
  EXPECT_PRED2((nuri::internal::eigen_ne<decltype(a), decltype(b)>), (a), (b))

#define NURI_EXPECT_EIGEN_EQ_TOL(a, b, tol)                                    \
  EXPECT_PRED3((nuri::internal::eigen_eq_tol<decltype(a), decltype(b)>), (a),  \
               (b), (tol))
#define NURI_EXPECT_EIGEN_NE_TOL(a, b, tol)                                    \
  EXPECT_PRED3((nuri::internal::eigen_ne_tol<decltype(a), decltype(b)>), (a),  \
               (b), (tol))

#define NURI_EXPECT_STRTRIM_EQ(a, b)                                           \
  EXPECT_PRED2(nuri::internal::expect_line_eq_trim, (a), (b))

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
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TEST_TEST_UTILS_H_ */
