//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_CORE_BOOL_MATRIX_H_
#define NURI_CORE_BOOL_MATRIX_H_

//! @cond
#include <cstdint>
#include <limits>
#include <vector>

#include <absl/base/optimization.h>
#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"

namespace nuri {
namespace internal {
  using Block = uint64_t;
  constexpr Eigen::Index kBitsPerBlock = std::numeric_limits<Block>::digits;
}  // namespace internal

class BoolMatrixKey {
private:
  using Index = Eigen::Index;

public:
  BoolMatrixKey() = delete;

  BoolMatrixKey(const Index idx)
      : BoolMatrixKey(idx / internal::kBitsPerBlock,
                      idx % internal::kBitsPerBlock) {
    ABSL_ASSUME(idx >= 0);
  }

  BoolMatrixKey(const Index blk, const Index bit)
      : blk_(blk), bit_(static_cast<int>(bit)), mask_(1ULL << bit_) { }

  Index to_index() const { return blk_ * internal::kBitsPerBlock + bit_; }

  Index blk() const { return blk_; }
  int bit() const { return bit_; }
  internal::Block mask() const { return mask_; }

private:
  Index blk_;
  int bit_;
  internal::Block mask_;
};

class BoolMatrix {
public:
  using Index = Eigen::Index;

  BoolMatrix() = delete;

  explicit BoolMatrix(Index rows, Index cols = 0)
      : data_((rows - 1) / internal::kBitsPerBlock + 1, cols), rows_(rows) { }

  bool operator()(BoolMatrixKey row, Index col) const {
    return (data_(row.blk(), col) & row.mask()) != 0;
  }

  Index rows() const { return rows_; }

  Index cols() const { return data_.cols(); }

  std::vector<int> gaussian_elimination() { return col_reduction_impl<true>(); }

  std::vector<int> partial_reduction(const int reduce_begin) {
    return col_reduction_impl<false>(reduce_begin);
  }

  void colwise_xor(Index c1, Index c2) {
    for (Index blk = 0; blk < data_.rows(); ++blk)
      data_(blk, c1) ^= data_(blk, c2);
  }

  void zero() { data_.setZero(); }

  void resize(int cols) {
    const Index oldcols = data_.cols();
    data_.conservativeResize(Eigen::NoChange, cols);
    if (cols > oldcols)
      data_.rightCols(cols - oldcols).setZero();
  }

  // Move col c1 to c2; c2 is overwritten, c1 is zeroed
  void move_col(Index c1, Index c2) {
    data_.col(c2) = data_.col(c1);
    data_.col(c1).setZero();
  }

  void set(BoolMatrixKey row, Index col) {
    data_(row.blk(), col) |= row.mask();
  }

  void unset(BoolMatrixKey row, Index col) {
    data_(row.blk(), col) &= ~row.mask();
  }

  void assign(BoolMatrixKey row, Index col, bool value) {
    internal::Block &data = data_(row.blk(), col);
    data = (data & ~row.mask())
           | (static_cast<internal::Block>(value) << row.bit());
  }

private:
  template <bool full>
  std::vector<int> col_reduction_impl(const int reduce_begin = 0) {
    std::vector<int> used(data_.cols(), 0);

    for (int row = 0; row < rows_; ++row) {
      const int p = static_cast<int>(find_pivot(row, used));
      // Try basis in [0, begin)
      if (p < 0)
        continue;

      used[p] = 1;

      if constexpr (!full)
        if (p >= reduce_begin)
          continue;

      for (Index i = reduce_begin; i < data_.cols(); ++i)
        if (used[i] == 0 && (*this)(row, i))
          colwise_xor(i, p);
    }

    return used;
  }

  Index find_pivot(BoolMatrixKey row, const std::vector<int> &used) const {
    for (Index i = 0; i < data_.cols(); ++i)
      if (used[i] == 0 && (*this)(row, i))
        return i;

    return -1;
  }

  // Col-wise compressed data
  Matrix<internal::Block, Eigen::Dynamic, Eigen::Dynamic> data_;
  Eigen::Index rows_;
};
}  // namespace nuri

#endif /* NURI_CORE_BOOL_MATRIX_H_ */
