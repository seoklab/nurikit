//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_EIGEN_CONFIG_H_
#define NURI_EIGEN_CONFIG_H_

#include <type_traits>

#include <Eigen/Dense>

#include <absl/log/absl_check.h>

namespace nuri {
template <class DT, int Rows, int Cols, int... Extra>
using Matrix = Eigen::Matrix<DT, Rows, Cols, Eigen::RowMajor, Extra...>;

template <class DT>
using Matrix3 = Matrix<DT, 3, 3>;
using Matrix3b = Matrix3<bool>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;

template <class DT, int... Extra>
using MatrixX3 = Matrix<DT, Eigen::Dynamic, 3, Extra...>;
using MatrixX3f = MatrixX3<float>;
using MatrixX3d = MatrixX3<double>;

template <class DT, int... Extra>
using MatrixX = Matrix<DT, Eigen::Dynamic, Eigen::Dynamic, Extra...>;
using MatrixXf = MatrixX<float>;
using MatrixXd = MatrixX<double>;

template <class DT, int Cols, int... Extra>
using Vector = Eigen::Matrix<DT, 1, Cols, Eigen::RowMajor, Extra...>;

template <class DT>
using Vector4 = Vector<DT, 4>;
using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;

template <class DT>
using Vector3 = Vector<DT, 3>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

template <class DT, int... Extra>
using VectorX = Vector<DT, Eigen::Dynamic, Extra...>;
using VectorXf = VectorX<float>;
using VectorXd = VectorX<double>;

template <class DT, int Rows, int Cols, int... Extra>
using Array = Eigen::Array<DT, Rows, Cols, Eigen::RowMajor, Extra...>;

template <class DT>
using Array2 = Array<DT, 1, 2>;
using Array2b = Array2<bool>;
using Array2i = Array2<int>;
using Array2f = Array2<float>;
using Array2d = Array2<double>;

template <class DT>
using Array3 = Array<DT, 1, 3>;
using Array3b = Array3<bool>;
using Array3i = Array3<int>;
using Array3f = Array3<float>;
using Array3d = Array3<double>;

template <class DT, int... Extra>
using ArrayX = Array<DT, 1, Eigen::Dynamic, Extra...>;
using ArrayXb = ArrayX<bool>;
using ArrayXi = ArrayX<int>;
using ArrayXf = ArrayX<float>;
using ArrayXd = ArrayX<double>;

template <class DT, int... Extra>
using ArrayXX = Array<DT, Eigen::Dynamic, Eigen::Dynamic, Extra...>;
using ArrayXXi = ArrayXX<int>;
using ArrayXXf = ArrayXX<float>;
using ArrayXXd = ArrayXX<double>;

template <class MT>
using Ref = Eigen::Ref<MT, Eigen::RowMajor>;

template <class MT>
using Map = Eigen::Map<MT, Eigen::RowMajor>;

template <
    class ReturnType, class MatrixLike,
    std::enable_if_t<
        std::is_same_v<typename MatrixLike::Scalar, typename ReturnType::Scalar>
            && MatrixLike::RowsAtCompileTime == ReturnType::ColsAtCompileTime
            && MatrixLike::ColsAtCompileTime == ReturnType::RowsAtCompileTime
            && MatrixLike::IsRowMajor != ReturnType::IsRowMajor,
        int> = 0>
auto swap_axis(Eigen::PlainObjectBase<MatrixLike> &m) {
  return Eigen::Map<ReturnType, Eigen::AlignedMax>(m.data(), m.cols(),
                                                   m.rows());
}

/**
 * @brief Cyclic indexer for Eigen types with a given offset.
 * @tparam O Offset of the cyclic index. If 0, it is determined at runtime.
 * @tparam N Size of the cyclic index. If 0, it is determined at runtime.
 *
 * This will produce an index that is cyclic with a given offset, i.e.,
 * (O, O+1, ..., N-1, 0, 1, ..., O-1).
 */
template <int O = 0, int N = 0>
class CyclicIndex {
public:
  static_assert(N > 0, "N must be positive");

  constexpr static int size() { return N; }

  constexpr int operator[](int i) const {
    int before = O + i, after = i - kZeroAt;
    return i < kZeroAt ? before : after;
  }

private:
  constexpr static int kZeroAt = (N - O) % N;
  static_assert(kZeroAt >= 0, "Offset must be smaller than size");
};

template <int O>
class CyclicIndex<O, 0> {
public:
  constexpr CyclicIndex(int n): n_(n), zero_at_((n - O) % n) {
    ABSL_DCHECK(zero_at_ >= 0);
  }

  constexpr int size() const { return n_; }

  constexpr int operator[](int i) const {
    int before = O + i, after = i - zero_at_;
    return i < zero_at_ ? before : after;
  }

private:
  int n_;
  int zero_at_;
};

template <>
class CyclicIndex<0, 0> {
public:
  constexpr CyclicIndex(int n, int offset)
      : n_(n), offset_(offset), zero_at_((n - offset) % n) {
    ABSL_DCHECK(zero_at_ >= 0);
  }

  constexpr int size() const { return n_; }

  constexpr int operator[](int i) const {
    int before = offset_ + i, after = i - zero_at_;
    return i < zero_at_ ? before : after;
  }

private:
  int n_;
  int offset_;
  int zero_at_;
};
}  // namespace nuri

#endif /* NURI_EIGEN_CONFIG_H_ */
