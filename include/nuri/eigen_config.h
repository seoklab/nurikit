//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_EIGEN_CONFIG_H_
#define NURI_EIGEN_CONFIG_H_

#include <type_traits>

#include <Eigen/Dense>

namespace nuri {
template <class DT, Eigen::Index Rows, Eigen::Index Cols>
using Matrix = Eigen::Matrix<DT, Rows, Cols, Eigen::RowMajor>;

template <class DT>
using Matrix3 = Matrix<DT, 3, 3>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;

template <class DT>
using MatrixX3 = Matrix<DT, Eigen::Dynamic, 3>;
using MatrixX3f = MatrixX3<float>;
using MatrixX3d = MatrixX3<double>;

template <class DT>
using MatrixX = Matrix<DT, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXf = MatrixX<float>;
using MatrixXd = MatrixX<double>;

template <class DT, Eigen::Index Cols>
using Vector = Eigen::Matrix<DT, 1, Cols, Eigen::RowMajor>;

template <class DT>
using Vector4 = Vector<DT, 4>;

template <class DT>
using Vector3 = Vector<DT, 3>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

template <class DT, Eigen::Index Rows, Eigen::Index Cols>
using Array = Eigen::Array<DT, Rows, Cols, Eigen::RowMajor>;

template <class DT>
using ArrayX = Array<DT, 1, Eigen::Dynamic>;
using ArrayXi = ArrayX<int>;
using ArrayXf = ArrayX<float>;
using ArrayXd = ArrayX<double>;

template <class DT>
using ArrayXX = Array<DT, Eigen::Dynamic, Eigen::Dynamic>;
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
}  // namespace nuri

#endif /* NURI_EIGEN_CONFIG_H_ */
