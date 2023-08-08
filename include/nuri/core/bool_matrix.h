//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_CORE_BOOL_MATRIX_H_
#define NURI_CORE_BOOL_MATRIX_H_

#include <climits>
#include <vector>

#include <Eigen/Dense>

#include <absl/base/attributes.h>
#include <absl/base/config.h>
#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

namespace nuri {
// NOLINTBEGIN(google-runtime-int)

namespace internal {
  template <class N>
  ABSL_ATTRIBUTE_CONST_FUNCTION int ffsll(N i) {
#if ABSL_HAVE_BUILTIN(__builtin_ffsll) || defined(__GNUC__)
    return __builtin_ffsll(static_cast<long long>(i)) - 1;
#else
    for (int j = 0; j < sizeof(i) * CHAR_BIT; ++j) {
      if ((i & static_cast<N>(1ULL << j)) != 0) {
        return j;
      }
    }
    return -1;
#endif
  }

  using Block = unsigned long long;
  constexpr inline Eigen::Index kBitsPerBlock = sizeof(Block) * CHAR_BIT;
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

  explicit BoolMatrix(Index cols)
      : data_(0, (cols - 1) / internal::kBitsPerBlock + 1), cols_(cols) { }

  BoolMatrix(Index rows, Index cols)
      : data_(rows, (cols - 1) / internal::kBitsPerBlock + 1), cols_(cols) { }

  bool operator()(Index row, BoolMatrixKey col) const {
    return (data_(row, col.blk()) & col.mask()) != 0;
  }

  Index rows() const { return data_.rows(); }

  Index cols() const { return cols_; }

  std::vector<int> gaussian_elimination() { return row_reduction_impl<true>(); }

  std::vector<int> partial_reduction(const int reduce_begin) {
    return row_reduction_impl<false>(reduce_begin);
  }

  void rowwise_xor(Index r1, Index r2) {
    for (Index blk = 0; blk < data_.cols(); ++blk) {
      data_(r1, blk) ^= data_(r2, blk);
    }
  }

  void zero() { data_.setZero(); }

  void resize(int rows) {
    const Index oldrows = data_.rows();
    data_.conservativeResize(rows, Eigen::NoChange);
    if (rows > oldrows) {
      data_.bottomRows(rows - oldrows).setZero();
    }
  }

  // Move row r1 to r2; r2 is overwritten, r1 is zeroed
  void move_row(Index r1, Index r2) {
    data_.row(r2) = data_.row(r1);
    data_.row(r1).setZero();
  }

  void set(Index row, BoolMatrixKey col) {
    data_(row, col.blk()) |= col.mask();
  }

  void unset(Index row, BoolMatrixKey col) {
    data_(row, col.blk()) &= ~col.mask();
  }

  void assign(Index row, BoolMatrixKey col, bool value) {
    internal::Block &data = data_(row, col.blk());
    data = (data & ~col.mask())
           | (static_cast<internal::Block>(value) << col.bit());
  }

private:
  template <bool full>
  std::vector<int> row_reduction_impl(const int reduce_begin = 0) {
    std::vector<int> used(data_.rows(), 0);

    for (int col = 0; col < cols_; ++col) {
      const int p = static_cast<int>(find_pivot(col, used));
      // Try basis in [0, begin)
      if (p < 0) {
        continue;
      }

      used[p] = 1;

      if constexpr (!full) {
        if (p >= reduce_begin) {
          continue;
        }
      }

      for (Index i = reduce_begin; i < data_.rows(); ++i) {
        if (used[i] == 0 && (*this)(i, col)) {
          rowwise_xor(i, p);
        }
      }
    }

    return used;
  }

  Index find_pivot(BoolMatrixKey col, const std::vector<int> &used) const {
    for (Index i = 0; i < data_.rows(); ++i) {
      if (used[i] == 0 && (data_(i, col.blk()) & col.mask()) != 0) {
        return i;
      }
    }
    return -1;
  }

  // Row-wise compressed data
  Eigen::Matrix<internal::Block, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      data_;
  Eigen::Index cols_;
};
// NOLINTEND(google-runtime-int)
}  // namespace nuri

#endif /* NURI_CORE_BOOL_MATRIX_H_ */
