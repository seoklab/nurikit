//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TOOLS_TM_H_
#define NURI_TOOLS_TM_H_

/// @cond
#include <cstdint>
#include <utility>

#include <Eigen/Dense>

#include <absl/base/attributes.h>
/// @endcond

#include "nuri/eigen_config.h"

namespace nuri {
namespace internal {
  class AlignedXY {
  public:
    AlignedXY(ConstRef<Matrix3Xd> x, ConstRef<Matrix3Xd> y, const int l_min)
        : x_(x), y_(y), xtm_(3, l_min), ytm_(3, l_min), y2x_(y.cols()),
          l_ali_(0) { }

    void remap(ArrayXi &y2x) noexcept;

    void remap_final(ConstRef<Matrix3Xd> x_aln, double score_d8sq) noexcept;

    void swap_align_with(ArrayXi &y2x) noexcept;

    ConstRef<Matrix3Xd> x() const { return x_; }

    auto xtm() { return xtm_.leftCols(l_ali_); }
    auto xtm() const { return xtm_.leftCols(l_ali_); }

    ConstRef<Matrix3Xd> y() const { return y_; }

    auto ytm() { return ytm_.leftCols(l_ali_); }
    auto ytm() const { return ytm_.leftCols(l_ali_); }

    int l_ali() const { return l_ali_; }

    int l_min() const { return static_cast<int>(xtm_.cols()); }

    ArrayXi &y2x() { return y2x_; }
    const ArrayXi &y2x() const { return y2x_; }

  private:
    Eigen::Ref<const Matrix3Xd> x_;
    Eigen::Ref<const Matrix3Xd> y_;

    Matrix3Xd xtm_;
    Matrix3Xd ytm_;
    ArrayXi y2x_;

    int l_ali_;
  };

  enum class SecStr : char {
    kCoil = 'C',
    kHelix = 'H',
    kStrand = 'E',
    kTurn = 'T',
  };

  /**
   * @brief Assign secondary structure to a given distance matrix.
   *
   * @param dists The distance matrix to assign secondary structure, each column
   *        containing 1-3, 1-4, 1-5 distances in that order. Only first (3 -
   * col) rows of each column are used.
   * @return The assigned secondary structure.
   */
  extern SecStr assign_secstr_approx(ConstRef<Array33d> dists);

  /**
   * @brief Assign secondary structure to the given points.
   *
   * @param pts The points to assign secondary structure.
   * @param buf A buffer matrix to store 1-3, 1-4, 1-5 distances. Must have at
   *        least (n - 2) columns, where n is the number of points.
   * @return The assigned secondary structure. First and last two residues are
   *         always assigned as coil. The returned array contains the ASCII code
   *         point of the assigned secondary structure.
   */
  extern ArrayXc assign_secstr_approx_full(ConstRef<Matrix3Xd> pts,
                                           Matrix3Xd &buf);

  ABSL_MUST_USE_RESULT
  extern double tm_initial_gt(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                              ConstRef<Matrix3Xd> x, ConstRef<Matrix3Xd> y,
                              ArrayXi &y2x, double d0sq_inv,
                              double d0sq_search);

  extern void tm_initial_ss(ArrayXi &y2x, ArrayXXc &path, ArrayXXd &val,
                            ConstRef<ArrayXc> secx, ConstRef<ArrayXc> secy);

  ABSL_MUST_USE_RESULT
  extern double tm_initial_local(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                                 ArrayXXc &path, ArrayXXd &val, AlignedXY &xy,
                                 ArrayXi &y2x, ArrayXi &buf, double d0sq_inv,
                                 double d01sq_inv, double d0sq_search);

  ABSL_MUST_USE_RESULT
  extern bool tm_initial_ssplus(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXXc &path,
                                ArrayXXd &val, const AlignedXY &xy,
                                ArrayXi &y2x, ConstRef<ArrayXc> secx,
                                ConstRef<ArrayXc> secy, double d01sq_inv);

  ABSL_MUST_USE_RESULT
  extern double tm_initial_fgt(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                               ConstRef<Matrix3Xd> x, ConstRef<Matrix3Xd> y,
                               ArrayXi &y2x, double dcu0_sq, double d0sq_inv,
                               double d0sq_search);

  extern double tm_realign_calculate_msd(AlignedXY &xy, Matrix3Xd &rx,
                                         Matrix3Xd &ry, const Affine3d &xform,
                                         double score_d8sq);
}  // namespace internal

class TMAlign {
public:
  TMAlign(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ);

  enum class InitFlags : std::uint32_t {
    kNone = 0x0,

    kGaplessThreading = 0x1,
    kSecStr = 0x2,
    kLocal = 0x4,
    kLocalPlusSecStr = 0x8,
    kFragmentGaplessThreading = 0x10,

    kDefault = kGaplessThreading | kSecStr | kLocal | kLocalPlusSecStr
               | kFragmentGaplessThreading,
  };

  ABSL_MUST_USE_RESULT
  bool initialize(InitFlags flags = InitFlags::kDefault);

  ABSL_MUST_USE_RESULT
  bool initialize(InitFlags flags, ConstRef<ArrayXc> secx,
                  ConstRef<ArrayXc> secy);

  bool initialized() const { return xy_.l_ali() > 0; }

  std::pair<Affine3d, double> tm_score(int l_norm = -1, double d0 = -1);

  const ArrayXi &templ_to_query() const & { return xy_.y2x(); }

  ArrayXi &&templ_to_query() && { return std::move(xy_.y2x()); }

  int l_ali() const { return xy_.l_ali(); }

  double aligned_msd() const { return aligned_msd_; }

private:
  ConstRef<Matrix3Xd> query() const { return xy_.x(); }
  ConstRef<Matrix3Xd> templ() const { return xy_.y(); }

  int l_min() const { return l_minmax_.first; }
  int l_max() const { return l_minmax_.second; }

  ArrayXi &y2x_local() { return y2x_buf1_; }
  ArrayXi &y2x_buf() { return y2x_buf2_; }

  ArrayXi &i_ali() { return y2x_buf1_; }
  ArrayXi &j_ali() { return y2x_buf2_; }

  std::pair<int, int> l_minmax_;
  internal::AlignedXY xy_;

  Affine3d best_xform_;
  double aligned_msd_;

  Matrix3Xd rx_, ry_;
  ArrayXd dsqs_;
  ArrayXi y2x_buf1_, y2x_buf2_;
};

struct TMAlignResult {
  Affine3d xform;
  ArrayXi templ_to_query;
  double msd;
  double tm_score = -1;
};

extern TMAlignResult
tm_align(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
         TMAlign::InitFlags flags = TMAlign::InitFlags::kDefault,
         int l_norm = -1, double d0 = -1);

extern TMAlignResult
tm_align(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
         ConstRef<ArrayXc> secx, ConstRef<ArrayXc> secy,
         TMAlign::InitFlags flags = TMAlign::InitFlags::kDefault,
         int l_norm = -1, double d0 = -1);

// test utils
namespace internal {
  extern std::pair<int, double>
  tmalign_score_fun8(const Matrix3Xd &x, const Matrix3Xd &y, ArrayXi &aligned,
                     double d_cutoff, double d0sq_inv,
                     double score_d8sq_cutoff);

  extern std::pair<int, double>
  tmalign_score_fun8(const Matrix3Xd &x, const Matrix3Xd &y, ArrayXi &aligned,
                     double d_cutoff, double d0sq_inv);

  extern std::pair<Affine3d, double>
  tmalign_tmscore8_search(const AlignedXY &xy, int simplify_step,
                          double local_d0_search, double score_d8sq_cutoff,
                          double d0sq_inv);

  extern void tmalign_dp_iter(Affine3d &xform_best, double &tmscore_max,
                              AlignedXY &xy, ArrayXi &y2x_best, int g1, int g2,
                              int max_iter, int simplify_step,
                              double local_d0_search, double score_d8sq_cutoff,
                              double d0sq_inv);

  extern double tmalign_get_score_fast(const Matrix3Xd &x, const Matrix3Xd &y,
                                       double d0sq_inv, double d0_search);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TOOLS_TM_H_ */
