//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TOOLS_TM_H_
#define NURI_TOOLS_TM_H_

//! @cond
#include <cstdint>
#include <utility>

#include <absl/base/attributes.h>
#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"

namespace nuri {
namespace internal {
  class AlignedXY;

  template <class Pred>
  void remap_helper(AlignedXY &xy, const Pred &pred) noexcept;

  class AlignedXY {
  public:
    AlignedXY(ConstRef<Matrix3Xd> x, ConstRef<Matrix3Xd> y, const int l_min)
        : x_(x), y_(y), xtm_(3, l_min), ytm_(3, l_min), y2x_(y.cols()),
          l_ali_(0) { }

    void remap(ConstRef<ArrayXi> y2x) noexcept;

    void remap(ArrayXi &&y2x) noexcept;

    void remap_final(ConstRef<Matrix3Xd> x_aln, double score_d8sq) noexcept;

    void swap_remap(ArrayXi &y2x) noexcept;

    void swap_align_with(ArrayXi &y2x) noexcept;

    void reset() noexcept {
      y2x_.setConstant(-1);
      l_ali_ = 0;
    }

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
    template <class Pred>
    friend void remap_helper(AlignedXY &xy, const Pred &pred) noexcept;

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

/**
 * @brief TM-align algorithm.
 *
 * This is a ground-up reimplementation of TM-align algorithm based on the
 * original TM-align code (version 20220412) by Yang Zhang. This implementation
 * aims to reproduce the results of the original code while providing improved
 * user interface and maintainability. Refer to the following paper for details
 * of the algorithm.
 *
 * Reference:
 * - Y Zhang and J Skolnick. *Nucleic Acids Res.* **2005**, *33*, 2302-2309.
 *   DOI:[10.1093/nar/gki524](https://doi.org/10.1093/nar/gki524)
 *
 * Here follows the full license text for the TM-align code:
 *
 * \code{.unparsed}
 * TM-align: sequence-independent structure alignment of monomer proteins by
 * TM-score superposition. Please report issues to yangzhanglab@umich.edu
 *
 * References to cite:
 * Y Zhang, J Skolnick. Nucl Acids Res 33, 2302-9 (2005)
 *
 * DISCLAIMER:
 * Permission to use, copy, modify, and distribute the Software for any
 * purpose, with or without fee, is hereby granted, provided that the
 * notices on the head, the reference information, and this copyright
 * notice appear in all copies or substantial portions of the Software.
 * It is provided "as is" without express or implied warranty.
 * \endcode
 *
 * @sa tm_align()
 */
class TMAlign {
public:
  /**
   * @brief Initialization flags for TM-align algorithm.
   */
  enum class InitFlags : std::uint32_t {
    kNone = 0x0,

    //! Enable gapless threading.
    kGaplessThreading = 0x1,

    //! Enable secondary structure-based alignment. Requires secondary structure
    //! assignment.
    kSecStr = 0x2,

    //! Enable alignment based on local superposition. This initialization is
    //! the most time-consuming method due to the exhaustive pairwise distance
    //! calculation.
    kLocal = 0x4,

    //! Enable local superposition with secondary structure-based alignment.
    //! Requires secondary structure assignment.
    kLocalPlusSecStr = 0x8,

    //! Enable fragment gapless threading.
    kFragmentGaplessThreading = 0x10,

    //! Default initialization flags, combination of all initialization methods.
    kDefault = kGaplessThreading | kSecStr | kLocal | kLocalPlusSecStr
               | kFragmentGaplessThreading,
  };

  /**
   * @brief Prepare TM-align algorithm with the given structures.
   *
   * @param query The query structure.
   * @param templ The template structure.
   * @note If any of the structures contain less than 5 residues, all
   *       initialization and alignment attempts will fail.
   */
  TMAlign(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ);

  /**
   * @brief Initialize the TM-align algorithm.
   *
   * @param flags Initialization flags.
   * @return Whether the initialization was successful.
   * @note If any of kSecStr or kLocalPlusSecStr flags are set, the secondary
   *       structures of the input structures will be assigned using the
   *       approximate secondary structure assignment algorithm in the
   *       TM-align code.
   */
  ABSL_MUST_USE_RESULT
  bool initialize(InitFlags flags = InitFlags::kDefault);

  /**
   * @brief Initialize the TM-align algorithm with user-provided secondary
   *        structures.
   *
   * @param flags Initialization flags.
   * @param secx Secondary structure of the query structure.
   * @param secy Secondary structure of the template structure.
   * @return Whether the initialization was successful.
   * @note If none of kSecStr or kLocalPlusSecStr flags are set, secondary
   *       structures are ignored.
   */
  ABSL_MUST_USE_RESULT
  bool initialize(InitFlags flags, ConstRef<ArrayXc> secx,
                  ConstRef<ArrayXc> secy);

  /**
   * @brief Initialize the TM-align algorithm with user-provided alignment.
   *
   * @param y2x A map of the template structure to the query structure. Negative
   *        values indicate that the corresponding residue in the template
   *        structure is not aligned to any residue in the query structure.
   * @return Whether the initialization was successful.
   * @note If size of y2x is not equal to the length of the template structure
   *       or any value of y2x is larger than or equal to the length of the
   *       query structure, the behavior is undefined.
   */
  ABSL_MUST_USE_RESULT
  bool initialize(ConstRef<ArrayXi> y2x);

  bool initialized() const { return xy_.l_ali() > 0; }

  /**
   * @brief Calculate TM-score using the current alignment.
   *
   * @param l_norm Length normalization factor. If negative, the length of the
   *        template structure is used.
   * @param d0 Distance scale factor. If negative, the default value is
   *        calculated based on the length normalization factor.
   * @return A pair of best transformation matrix and TM-score. If the alignment
   *         failed for any reason, the TM-score is set to a negative value and
   *         the transformation matrix is left unspecified.
   */
  std::pair<Affine3d, double> tm_score(int l_norm = -1, double d0 = -1);

  /**
   * @brief Final alignment of the structures.
   * @return A map of the template structure to the query structure. Negative
   *         values indicate that the corresponding residue in the template
   *         structure is not aligned to any residue in the query structure.
   */
  const ArrayXi &templ_to_query() const & { return xy_.y2x(); }

  /**
   * @brief Final alignment of the structures (move version).
   * @return A map of the template structure to the query structure. Negative
   *         values indicate that the corresponding residue in the template
   *         structure is not aligned to any residue in the query structure.
   * @note TMAlign class is invalidated after this call.
   */
  ArrayXi &&templ_to_query() && { return std::move(xy_.y2x()); }

  /**
   * @brief Get the length of the aligned region.
   * @return The length of the aligned region.
   */
  int l_ali() const { return xy_.l_ali(); }

  /**
   * @brief Get the mean square deviation of the aligned region.
   * @return The mean square deviation of the aligned region.
   */
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

/**
 * @brief Result of TM-align algorithm.
 * @sa tm_align()
 * @note The TM-score is set to a negative value if the alignment failed for any
 *       reason. In such cases, other fields will be left unspecified.
 */
struct TMAlignResult {
  //! The transformation matrix.
  Affine3d xform;

  //! A map of the template structure to the query structure. Negative values
  //! indicate no alignment.
  ArrayXi templ_to_query;

  //! The mean square deviation of the aligned region.
  double msd;

  //! The TM-score of the alignment.
  double tm_score = -1;
};

/**
 * @brief Align two structures using TM-align algorithm.
 *
 * @param query The query structure.
 * @param templ The template structure.
 * @param flags Initialization flags.
 * @param l_norm Length normalization factor. If negative, the length of the
 *        template structure is used.
 * @param d0 Distance scale factor. If negative, the default value is calculated
 *        based on the length normalization factor.
 * @return The result of the alignment. If the alignment failed for any reason,
 *         the TM-score is set to a negative value.
 * @note If any of kSecStr or kLocalPlusSecStr flags are set, the secondary
 *       structures of the input structures will be assigned using the
 *       approximate secondary structure assignment algorithm in the TM-align
 *       code.
 */
extern TMAlignResult
tm_align(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
         TMAlign::InitFlags flags = TMAlign::InitFlags::kDefault,
         int l_norm = -1, double d0 = -1);

/**
 * @brief Align two structures using TM-align algorithm.
 *
 * @param query The query structure.
 * @param templ The template structure.
 * @param secx Secondary structure of the query structure.
 * @param secy Secondary structure of the template structure.
 * @param flags Initialization flags.
 * @param l_norm Length normalization factor. If negative, the length of the
 *        template structure is used.
 * @param d0 Distance scale factor. If negative, the default value is calculated
 *        based on the length normalization factor.
 * @return The result of the alignment. If the alignment failed for any reason,
 *         the TM-score is set to a negative value.
 * @note If none of kSecStr or kLocalPlusSecStr flags are set, the secondary
 *       structures will be ignored.
 */
extern TMAlignResult
tm_align(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
         ConstRef<ArrayXc> secx, ConstRef<ArrayXc> secy,
         TMAlign::InitFlags flags = TMAlign::InitFlags::kDefault,
         int l_norm = -1, double d0 = -1);

/**
 * @brief Align two structures using TM-align algorithm, with the given
 *        alignment. This is also known as the "TM-score" program in the
 *        TM-tools suite.
 *
 * @param query The query structure.
 * @param templ The template structure.
 * @param y2x A map of the template structure to the query structure. Negative
 *        values indicate that the corresponding residue in the template
 *        structure is not aligned to any residue in the query structure.
 * @param l_norm Length normalization factor. If negative, the length of the
 *        template structure is used.
 * @param d0 Distance scale factor. If negative, the default value is calculated
 *        based on the length normalization factor.
 * @return The result of the alignment. If the alignment failed for any reason,
 *         the TM-score is set to a negative value.
 *
 * @note If size of y2x is not equal to the length of the template structure or
 *       any value of y2x is larger than or equal to the length of the query
 *       structure, the behavior is undefined.
 */
extern TMAlignResult tm_align(ConstRef<Matrix3Xd> query,
                              ConstRef<Matrix3Xd> templ, ConstRef<ArrayXi> y2x,
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
