//
// Project nurikit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TEST_TEST_UTILS_H_
#define NURI_TEST_TEST_UTILS_H_

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
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TEST_TEST_UTILS_H_ */
