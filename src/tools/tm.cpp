//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/tools/tm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  using Array6d = Array<double, 6, 1>;
  using Array6i = Array<int, 6, 1>;
}  // namespace

namespace internal {
  namespace {
    // 1-3, 1-4, 1-5, 2-4, 2-5, 3-5
    const Array6d kHelixRefs { 5.45, 5.18, 6.37, 5.45, 5.18, 5.45 };
    // 1-5, 1-4, 1-3, 2-5, 2-4, 3-5
    const Array6d kStrandRefs { 6.1, 10.4, 13, 6.1, 10.4, 6.1 };
  }  // namespace

  SecStr assign_secstr_approx(ConstRef<Array33d> dists) {
    Array6d ds;
    ds.head<5>() = dists.reshaped().head<5>();
    ds[5] = dists(0, 2);

    if (((ds - kHelixRefs).abs() < 2.1).all())
      return SecStr::kHelix;

    if (((ds - kStrandRefs).abs() < 1.42).all())
      return SecStr::kStrand;

    if (ds[2] < 8)
      return SecStr::kTurn;

    return SecStr::kCoil;
  }

  ArrayXc assign_secstr_approx_full(ConstRef<Matrix3Xd> pts, Matrix3Xd &buf) {
    const int n = static_cast<int>(pts.cols());

    auto dists = buf.leftCols(n - 2).array();
    for (int i = 0; i < n - 4; ++i) {
      // (2, 3, 4) - 0, ..., (n-3, n-2, n-1) - n-5
      dists.col(i) = (pts.middleCols<3>(i + 2).colwise() - pts.col(i))
                         .colwise()
                         .squaredNorm()
                         .transpose();
    }
    // (n-2, n-1, x) - n-4
    dists.col(n - 4).head<2>() = (pts.rightCols<2>().colwise() - pts.col(n - 4))
                                     .colwise()
                                     .squaredNorm()
                                     .transpose();
    // (n-1, x, x) - n-3
    dists(0, n - 3) = (pts.col(n - 1) - pts.col(n - 3)).squaredNorm();
    dists = dists.sqrt();

    ArrayXc sec(n);
    sec.head<2>().setConstant(
        static_cast<std::int8_t>(internal::SecStr::kCoil));
    for (int i = 2; i < pts.cols() - 2; ++i) {
      // 2 <- 0, 1, 2, ..., n-3 <- n-5, n-4, n-3
      sec[i] = static_cast<std::int8_t>(
          assign_secstr_approx(dists.middleCols<3>(i - 2)));
    }
    sec.tail<2>().setConstant(
        static_cast<std::int8_t>(internal::SecStr::kCoil));

    return sec;
  }

  template <class Pred>
  void remap_helper(AlignedXY &xy, const Pred &pred) noexcept {
    ABSL_DCHECK_EQ(xy.y2x_.size(), xy.y_.cols());
    ABSL_DCHECK((xy.y2x_ < static_cast<int>(xy.x_.cols())).all());

    xy.l_ali_ = 0;
    for (int j = 0; j < xy.y2x_.size(); ++j) {
      const int i = xy.y2x_[j];
      if (i >= 0) {
        if (pred(i, j)) {
          xy.xtm_.col(xy.l_ali_) = xy.x_.col(i);
          xy.ytm_.col(xy.l_ali_) = xy.y_.col(j);
          ++xy.l_ali_;
        } else {
          xy.y2x_[j] = -1;
        }
      }
    }
  }

  void AlignedXY::remap(ConstRef<ArrayXi> y2x) noexcept {
    y2x_ = y2x;
    remap_helper(*this, [](int, int) { return true; });
  }

  void AlignedXY::remap(ArrayXi &&y2x) noexcept {
    y2x_ = std::move(y2x);
    remap_helper(*this, [](int, int) { return true; });
  }

  void AlignedXY::swap_remap(ArrayXi &y2x) noexcept {
    swap_align_with(y2x);
    remap_helper(*this, [](int, int) { return true; });
  }

  void AlignedXY::remap_final(ConstRef<Matrix3Xd> x_aln,
                              const double score_d8sq) noexcept {
    remap_helper(*this, [&](int i, int j) {
      return (x_aln.col(i) - y_.col(j)).squaredNorm() <= score_d8sq;
    });
  }

  void AlignedXY::swap_align_with(ArrayXi &y2x) noexcept {
    const auto my_sz = y2x_.size(), other_sz = y2x.size();
    ABSL_ASSUME(my_sz == other_sz);
    y2x_.swap(y2x);
  }

  namespace {
    template <class AL>
    double raw_tmscore(AL &&dsqs, const double d0sq_inv) {
      double tmscore = (1 + std::forward<AL>(dsqs) * d0sq_inv).inverse().sum();
      return tmscore;
    }

    template <bool squared, class AL>
    std::pair<int, double> find_aligned_cutoff(const AL &dsqs,
                                               double d_or_dsq) {
      double dsq_cutoff = dsqs.minCoeff();
      if constexpr (squared) {
        dsq_cutoff = nuri::max(d_or_dsq, dsq_cutoff);
      } else {
        dsq_cutoff = nuri::max(d_or_dsq * d_or_dsq, dsq_cutoff);
      }

      while (true) {
        const int selected = (dsqs <= dsq_cutoff).template cast<int>().sum();
        if (selected >= 3 || ABSL_PREDICT_FALSE(dsqs.size() <= 3))
          return { selected, dsq_cutoff };

        if constexpr (squared) {
          dsq_cutoff += 0.5;
        } else {
          d_or_dsq += 0.5;
          dsq_cutoff = d_or_dsq * d_or_dsq;
        }
      }
    }

    template <bool use_d8sq, class AL, class ML1, class ML2>
    std::pair<int, double>
    collect_res_tmscore(const ML1 &x, const ML2 &y, AL dsqs, ArrayXi &aligned,
                        const double d_cutoff, const double score_d8sq_cutoff,
                        const double d0sq_inv) {
      dsqs = (x - y).colwise().squaredNorm().transpose();

      auto [n_sel, dsq_cutoff] = find_aligned_cutoff<false>(dsqs, d_cutoff);
      for (int i = 0, j = 0; i < n_sel; ++j) {
        if (dsqs[j] <= dsq_cutoff)
          aligned[i++] = j;
      }

      double tmscore;
      if constexpr (use_d8sq) {
        tmscore = 0;
        for (int i = 0; i < dsqs.size(); ++i) {
          if (dsqs[i] <= score_d8sq_cutoff)
            tmscore += 1 / (1 + dsqs[i] * d0sq_inv);
        }
      } else {
        tmscore = raw_tmscore(dsqs, d0sq_inv);
      }

      return { n_sel, tmscore };
    }

    template <bool use_d8sq, class ML1, class ML2, class AL1, class AL2>
    int tmscore_greedy_iter(std::pair<Affine3d, double> &result, ML1 rx, ML1 ry,
                            AL1 &&dsqs, const AL2 &i_ali, ArrayXi &j_ali,
                            const ML2 &x, const ML2 &y, const double d_cutoff,
                            const double score_d8sq_cutoff,
                            const double d0sq_inv) {
      rx.leftCols(i_ali.size()) = x(Eigen::all, i_ali);
      ry.leftCols(i_ali.size()) = y(Eigen::all, i_ali);

      auto [xform, flag] = qcp_inplace(rx.leftCols(i_ali.size()),
                                       ry.leftCols(i_ali.size()),
                                       AlignMode::kXformOnly);
      if (ABSL_PREDICT_FALSE(flag < 0)) {
        ABSL_LOG(WARNING) << "Alignment failed while optimizing TM-score";
        return 0;
      }

      inplace_transform(rx, xform, x);

      auto [n_ali, tmscore] =
          collect_res_tmscore<use_d8sq>(rx, y, std::forward<AL1>(dsqs), j_ali,
                                        d_cutoff, score_d8sq_cutoff, d0sq_inv);

      if (tmscore > result.second) {
        result.first = xform;
        result.second = tmscore;
      }

      return n_ali;
    }

    template <bool use_d8sq>
    std::pair<Affine3d, double>
    tmscore_greedy_search(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                          ArrayXi &i_ali, ArrayXi &j_ali, const AlignedXY &xy,
                          const int simplify_step, const double local_d0_search,
                          const double score_d8sq_cutoff,
                          const double d0sq_inv) {
      // original TMalign iterates 20 times *after* initialization
      // this implementation merges the initialization and iteration
      constexpr int max_iter = 20 + 1;
      constexpr int n_init_max = 6;

      const double d_cutoff_init = local_d0_search - 1,
                   d_cutoff_sub = local_d0_search + 1;

      std::pair<Affine3d, double> result;
      result.second = -1;

      const int l_ali = xy.l_ali();
      const int l_ini_min = nuri::min(l_ali, 4);

      auto rx_ali = rx.leftCols(l_ali), ry_ali = ry.leftCols(l_ali);
      auto dsqs_ali = dsqs.head(l_ali);

      unsigned int n_ali_start = l_ali;
      for (int i_init = 0; i_init < n_init_max; ++i_init) {
        int l_frag = static_cast<int>(n_ali_start >> i_init);
        if (l_frag <= l_ini_min || i_init == n_init_max - 1)
          l_frag = l_ini_min;

        int il_max = l_ali - l_frag + 1;
        for (int i = 0;;) {
          double d_cutoff = d_cutoff_init;
          int n_ali = l_frag;

          auto i_frag = i_ali.head(l_frag);
          absl::c_iota(i_frag, i);

          for (int iter = 0; iter < max_iter; ++iter) {
            int m_ali = tmscore_greedy_iter<use_d8sq>(
                result, rx_ali, ry_ali, dsqs_ali, i_ali.head(n_ali), j_ali,
                xy.xtm(), xy.ytm(), d_cutoff, score_d8sq_cutoff, d0sq_inv);

            if (iter > 0 && absl::c_equal(i_ali.head(n_ali), j_ali.head(m_ali)))
              break;

            n_ali = m_ali;
            d_cutoff = d_cutoff_sub;

            const auto my_sz = i_ali.size(), other_sz = j_ali.size();
            ABSL_ASSUME(my_sz == other_sz);
            i_ali.swap(j_ali);
          }

          if (i == il_max - 1)
            break;

          i = nuri::min(i + simplify_step, il_max - 1);
        }

        if (l_frag == l_ini_min)
          break;
      }

      return result;
    }

    constexpr std::int8_t kPathDiag = 0, kPathVert = 1, kPathHorz = 2;

    template <class Scorer>
    // path, val arrays are transposed (x -> cols, y -> rows)
    void tm_nwdp(ArrayXi &y2x, ArrayXXc &path, ArrayXXd &val,
                 const double gap_open, Scorer scorer) {
      const int lx = static_cast<int>(path.cols() - 1),
                ly = static_cast<int>(path.rows() - 1);

      ABSL_DCHECK_EQ(lx + 1, val.cols());

      ABSL_DCHECK_EQ(ly, y2x.size());
      ABSL_DCHECK_EQ(ly + 1, val.rows());

      for (int i = 0; i < lx; ++i) {
        for (int j = 0; j < ly; ++j) {
          Array3d vals;

          vals[kPathDiag] = val(j, i) + scorer(i, j);

          vals[kPathHorz] = val(j + 1, i);
          if (path(j + 1, i) == kPathDiag)
            vals[kPathHorz] += gap_open;

          vals[kPathVert] = val(j, i + 1);
          if (path(j, i + 1) == kPathDiag)
            vals[kPathVert] += gap_open;

          std::int8_t p;
          val(j + 1, i + 1) = vals.maxCoeff(&p);
          path(j + 1, i + 1) = p;
        }
      }

      y2x.fill(-1);
      for (int i = lx - 1, j = ly - 1; i >= 0 && j >= 0;) {
        switch (path(j + 1, i + 1)) {
        case kPathDiag:
          y2x[j] = i;
          --i;
          --j;
          break;
        case kPathVert:
          --j;
          break;
        case kPathHorz:
          --i;
          break;
        default:
          ABSL_UNREACHABLE();
        }
      }
    }

    void tm_find_best_alignment(Affine3d &xform_best, double &tmscore_max,
                                Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                                ArrayXXc &path, ArrayXXd &val, AlignedXY &xy,
                                ArrayXi &y2x_best, ArrayXi &buf1, ArrayXi &buf2,
                                const int g1, const int g2, const int max_iter,
                                const int simplify_step,
                                const double local_d0_search,
                                const double score_d8sq_cutoff,
                                const double d0sq_inv) {
      constexpr double gap_open[] = { -0.6, 0 };

      ABSL_DCHECK_GE(g1, 0);
      ABSL_DCHECK_LE(g2, 2);
      ABSL_DCHECK_LT(g1, g2);

      Affine3d xform = xform_best;
      double tmscore_old = 0;
      for (int g = g1; g < g2; ++g) {
        for (int iter = 0; iter < max_iter; ++iter) {
          inplace_transform(rx, xform, xy.x());

          tm_nwdp(buf1, path, val, gap_open[g], [&](int i, int j) {
            double dsq = (rx.col(i) - xy.y().col(j)).squaredNorm();
            return 1 / (1 + dsq * d0sq_inv);
          });
          xy.swap_remap(buf1);

          double tmscore;
          std::tie(xform, tmscore) = tmscore_greedy_search<true>(
              rx, ry, dsqs, buf1, buf2, xy, simplify_step, local_d0_search,
              score_d8sq_cutoff, d0sq_inv);
          if (tmscore > tmscore_max) {
            tmscore_max = tmscore;
            xform_best = xform;
            xy.swap_align_with(y2x_best);
          }

          if (iter > 0 && std::abs(tmscore - tmscore_old) < 1e-6)
            break;

          tmscore_old = tmscore;
        }
      }
    }

    template <class AL, class ML1, class ML2, class ML3>
    double tmscore_fast(ML1 x0, ML1 y0, AL dsqs, const ML2 &x, const ML3 &y,
                        const double d0sq_inv, const double d0sq_search) {
      auto align_tmscore = [&](auto &&x0_sel, auto &&y0_sel) {
        auto [xform_sel, flag_sel] =
            qcp_inplace(x0_sel, y0_sel, AlignMode::kXformOnly);
        if (ABSL_PREDICT_FALSE(flag_sel < 0)) {
          ABSL_LOG(WARNING) << "Alignment failed while calculating TM-score";
          return -2.0;
        }

        inplace_transform(x0, xform_sel, x);

        dsqs = (x0 - y).colwise().squaredNorm().transpose();
        return raw_tmscore(dsqs, d0sq_inv);
      };

      auto align_sub_tmscore = [&](const int n_sel, const double dsq_cutoff) {
        auto x0_sel = x0.leftCols(n_sel), y0_sel = y0.leftCols(n_sel);

        int j = 0;
        for (int k = 0; k < dsqs.size(); ++k) {
          if (dsqs[k] <= dsq_cutoff) {
            x0_sel.col(j) = x.col(k);
            y0_sel.col(j) = y.col(k);
            ++j;
          }
        }

        return align_tmscore(x0_sel, y0_sel);
      };

      int n_ali = static_cast<int>(x0.cols());
      ABSL_DCHECK_EQ(n_ali, y0.cols());
      ABSL_DCHECK_EQ(n_ali, x.cols());
      ABSL_DCHECK_EQ(n_ali, y.cols());

      x0 = x;
      y0 = y;
      const double tmscore = align_tmscore(x0, y0);

      auto [n_sel, dsq_cutoff] = find_aligned_cutoff<true>(dsqs, d0sq_search);
      if (n_sel == n_ali)
        return tmscore;

      const double tmscore1 = align_sub_tmscore(n_sel, dsq_cutoff);

      std::tie(n_sel, dsq_cutoff) =
          find_aligned_cutoff<true>(dsqs, d0sq_search + 1);
      const double tmscore2 =
          n_sel == n_ali ? tmscore : align_sub_tmscore(n_sel, dsq_cutoff);

      return std::max({ tmscore, tmscore1, tmscore2 });
    }

    template <class ML1, class ML2>
    std::pair<int, double> tm_gt_find_best_alignment(
        Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
        // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
        ML1 &&x, ML2 &&y, const double d0sq_inv, const double d0sq_search,
        const int min_frag) {
      const int lx = static_cast<int>(x.cols());
      const int ly = static_cast<int>(y.cols());

      int n1 = min_frag - ly, n2 = lx - min_frag + 1;

      int k_max = n1;
      double tmscore_max = -1;
      for (int k = n1; k < n2; ++k) {
        const int x_start = nuri::clamp(k, 0, lx),
                  y_start = nuri::clamp(-k, 0, ly);
        const int n_overlap = nuri::min(lx - x_start, ly - y_start);

        double tmscore = tmscore_fast(
            rx.leftCols(n_overlap), ry.leftCols(n_overlap),
            dsqs.head(n_overlap), x.middleCols(x_start, n_overlap),
            y.middleCols(y_start, n_overlap), d0sq_inv, d0sq_search);
        if (tmscore > tmscore_max) {
          tmscore_max = tmscore;
          k_max = k;
        }
      }

      return { k_max, tmscore_max };
    }

    void fill_map_fragment(ArrayXi &y2x, const int x_begin, const int y_begin,
                           const int n_frag) {
      const int ly = static_cast<int>(y2x.size());

      y2x.head(y_begin).setConstant(-1);

      auto aligned = y2x.segment(y_begin, n_frag);
      absl::c_iota(aligned, x_begin);

      y2x.tail(ly - y_begin - n_frag).setConstant(-1);
    }
  }  // namespace

  double tm_initial_gt(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                       ConstRef<Matrix3Xd> x, ConstRef<Matrix3Xd> y,
                       ArrayXi &y2x, const double d0sq_inv,
                       const double d0sq_search) {
    const int lx = static_cast<int>(x.cols());
    const int ly = static_cast<int>(y.cols());
    ABSL_DCHECK_EQ(y2x.size(), ly);

    const int l_min = static_cast<int>(dsqs.size());
    ABSL_DCHECK_EQ(l_min, nuri::min(lx, ly));

    const int min_frag = nuri::max(l_min / 2, 5);
    auto [k, tmscore] = tm_gt_find_best_alignment(rx, ry, dsqs, x, y, d0sq_inv,
                                                  d0sq_search, min_frag);
    if (tmscore > 0) {
      const int x_begin = nuri::clamp(k, 0, lx),
                y_begin = nuri::clamp(-k, 0, ly);
      const int n_overlap = nuri::min(lx - x_begin, ly - y_begin);
      fill_map_fragment(y2x, x_begin, y_begin, n_overlap);
    }

    return tmscore;
  }

  void tm_initial_ss(ArrayXi &y2x, ArrayXXc &path, ArrayXXd &val,
                     ConstRef<ArrayXc> secx, ConstRef<ArrayXc> secy) {
    tm_nwdp(y2x, path, val, -1,
            [&](int i, int j) { return value_if(secx[i] == secy[j], 1.0); });
  }

  double tm_initial_local(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                          ArrayXXc &path, ArrayXXd &val, AlignedXY &xy,
                          ArrayXi &y2x, ArrayXi &buf, const double d0sq_inv,
                          const double d01sq_inv, const double d0sq_search) {
    const int lx = static_cast<int>(rx.cols()),  //
        ly = static_cast<int>(ry.cols()),        //
        l_min = xy.l_min();
    ABSL_DCHECK_EQ(l_min, dsqs.size());

    // l > 250 -> 45, 250 >= l > 200 -> 35, 200 >= l > 150 -> 25, 150 >= l -> 15
    // then bounded by l / 3 (as in the original TMalign code)
    const int jmp_x = nuri::min(lx / 3,
                                nuri::clamp((lx - 101) / 50 * 10 + 15, 15, 45)),
              jmp_y = nuri::min(ly / 3,
                                nuri::clamp((ly - 101) / 50 * 10 + 15, 15, 45));

    double gl_max = 0;
    for (const int n_frag:
         { nuri::min(20, l_min / 3), nuri::min(100, l_min / 2) }) {
      auto rx_frag = rx.leftCols(n_frag), ry_frag = ry.leftCols(n_frag);

      for (int i = 0; i < lx - n_frag + 1; i += jmp_x) {
        for (int j = 0; j < ly - n_frag + 1; j += jmp_y) {
          rx_frag = xy.x().middleCols(i, n_frag);
          ry_frag = xy.y().middleCols(j, n_frag);

          auto [xform, flag] =
              qcp_inplace(rx_frag, ry_frag, AlignMode::kXformOnly);
          if (ABSL_PREDICT_FALSE(flag < 0)) {
            ABSL_LOG(WARNING) << "Alignment failed while calculating TM-score";
            continue;
          }

          inplace_transform(rx, xform, xy.x());

          auto scores = val.bottomRightCorner(ly, lx);
          cdistsq(scores, xy.y(), rx);
          scores = (1 + scores * d01sq_inv).inverse();

          tm_nwdp(buf, path, val, 0,
                  [&](int xi, int yi) { return scores(yi, xi); });
          xy.swap_remap(buf);

          const double gl = tmscore_fast(rx.leftCols(xy.l_ali()),
                                         ry.leftCols(xy.l_ali()),
                                         dsqs.head(xy.l_ali()), xy.xtm(),
                                         xy.ytm(), d0sq_inv, d0sq_search);
          if (gl > gl_max) {
            gl_max = gl;
            xy.swap_align_with(y2x);
          }
        }
      }
    }

    return gl_max;
  }

  bool tm_initial_ssplus(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXXc &path,
                         ArrayXXd &val, const AlignedXY &xy, ArrayXi &y2x,
                         ConstRef<ArrayXc> secx, ConstRef<ArrayXc> secy,
                         const double d01sq_inv) {
    rx.leftCols(xy.l_ali()) = xy.xtm();
    ry.leftCols(xy.l_ali()) = xy.ytm();

    auto [xform, flag] = qcp_inplace(rx.leftCols(xy.l_ali()),
                                     ry.leftCols(xy.l_ali()),
                                     AlignMode::kXformOnly);
    if (ABSL_PREDICT_FALSE(flag < 0)) {
      ABSL_LOG(WARNING) << "Alignment failed while calculating TM-score";
      return false;
    }

    inplace_transform(rx, xform, xy.x());

    auto scores = val.bottomRightCorner(xy.y().cols(), xy.x().cols());
    cdistsq(scores, xy.y(), rx);
    scores = (1 + scores * d01sq_inv).inverse();

    tm_nwdp(y2x, path, val, -1, [&](int i, int j) {
      return scores(j, i) + value_if(secx[i] == secy[j], 0.5);
    });

    return true;
  }

  namespace {
    std::pair<int, int> find_max_frag(ConstRef<Matrix3Xd> pts,
                                      const double dcu0_sq,
                                      const int min_frag) {
      constexpr double multipler = 1.1 * 1.1;

      const int n = static_cast<int>(pts.cols());
      const int r_min = nuri::min(min_frag, n / 3);

      int max_begin = 0, max_len = 1;
      for (double dcu_cut = dcu0_sq; max_len < r_min; dcu_cut *= multipler) {
        int begin = 0, len = 1;
        for (int i = 1; i < n; ++i) {
          if ((pts.col(i) - pts.col(i - 1)).squaredNorm() < dcu_cut) {
            ++len;
          } else {
            if (len > max_len) {
              max_begin = begin;
              max_len = len;
            }

            begin = i;
            len = 1;
          }
        }
        if (len > max_len) {
          max_begin = begin;
          max_len = len;
        }
      }

      // this is to avoid redundant initial alignment with the full structure
      if (max_begin == 0 && max_len == n) {
        // see the original TMalign code for the following numbers
        max_begin = static_cast<int>(n * 0.1);
        max_len = static_cast<int>(n * 0.89) - max_begin + 1;
      }

      return { max_begin, max_len };
    }
  }  // namespace

  double tm_initial_fgt(Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                        ConstRef<Matrix3Xd> x, ConstRef<Matrix3Xd> y,
                        ArrayXi &y2x, const double dcu0_sq,
                        const double d0sq_inv, const double d0sq_search) {
    constexpr int min_frag = 4;

    const int lx = static_cast<int>(x.cols()), ly = static_cast<int>(y.cols());

    auto [xf_begin, xf_len] = find_max_frag(x, dcu0_sq, min_frag);
    auto [yf_begin, yf_len] = find_max_frag(y, dcu0_sq, min_frag);

    const int min_ali =
        nuri::max(nuri::min(xf_len, yf_len) * 2 / 5, min_frag - 1);

    double tmscore_max = 0;
    int x_begin = 0, y_begin = 0;

    // Both if statements should run when xf_len == yf_len and lx == ly to avoid
    // asymmetric results

    if (std::make_pair(xf_len, lx) <= std::make_pair(yf_len, ly)) {
      auto [k, tmscore] = tm_gt_find_best_alignment(
          rx, ry, dsqs, x.middleCols(xf_begin, xf_len), y, d0sq_inv,
          d0sq_search, min_ali);
      if (tmscore > tmscore_max) {
        tmscore_max = tmscore;

        x_begin = nuri::clamp(k, 0, xf_len) + xf_begin;
        y_begin = nuri::clamp(-k, 0, ly);
      }
    }

    if (std::make_pair(xf_len, lx) >= std::make_pair(yf_len, ly)) {
      auto [k, tmscore] = tm_gt_find_best_alignment(
          rx, ry, dsqs, x, y.middleCols(yf_begin, yf_len), d0sq_inv,
          d0sq_search, min_ali);
      if (tmscore > tmscore_max) {
        tmscore_max = tmscore;

        x_begin = nuri::clamp(k, 0, lx);
        y_begin = nuri::clamp(-k, 0, yf_len) + yf_begin;
      }
    }

    if (tmscore_max > 0) {
      const int n_overlap = nuri::min(lx - x_begin, ly - y_begin);
      fill_map_fragment(y2x, x_begin, y_begin, n_overlap);
    }

    return tmscore_max;
  }

  double tm_realign_calculate_msd(AlignedXY &xy, Matrix3Xd &rx, Matrix3Xd &ry,
                                  const Affine3d &xform,
                                  const double score_d8sq) {
    inplace_transform(rx, xform, xy.x());
    xy.remap_final(rx, score_d8sq);

    rx.leftCols(xy.l_ali()) = xy.xtm();
    ry.leftCols(xy.l_ali()) = xy.ytm();
    // kabsch never fails for MSD calculation
    auto [_, msd] = kabsch(rx.leftCols(xy.l_ali()), ry.leftCols(xy.l_ali()),
                           AlignMode::kMsdOnly);
    return msd;
  }
}  // namespace internal

TMAlign::TMAlign(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ)
    : l_minmax_(nuri::minmax(static_cast<int>(query.cols()),
                             static_cast<int>(templ.cols()))),
      xy_(query, templ, l_min()), rx_(3, query.cols()), ry_(3, templ.cols()),
      dsqs_(l_min()), y2x_buf1_(templ.cols()), y2x_buf2_(templ.cols()) { }

namespace {
  template <class AL>
  bool tmalign_conclude_init(Affine3d &best_xform, double &aligned_msd,
                             internal::AlignedXY &xy, AL &&y2x_best,
                             Matrix3Xd &rx, Matrix3Xd &ry, ArrayXd &dsqs,
                             ArrayXi &i_ali, ArrayXi &j_ali, double tm_max,
                             const double d0_search, const double score_d8sq,
                             const double d0sq_inv) {
    xy.remap(std::forward<AL>(y2x_best));

    if (ABSL_PREDICT_FALSE(xy.l_ali() <= 0)) {
      ABSL_LOG(ERROR) << "No alignment found between the input structures";
      ABSL_DLOG_IF(WARNING, tm_max > 0)
          << "TMscore is positive despite no alignment found";
      return false;
    }

    std::tie(best_xform, tm_max) = internal::tmscore_greedy_search<true>(
        rx, ry, dsqs, i_ali, j_ali, xy, 1, d0_search, score_d8sq, d0sq_inv);
    if (ABSL_PREDICT_FALSE(tm_max <= 0)) {
      ABSL_LOG(ERROR) << "TMscore is zero or negative after initialization";
      xy.reset();
      return false;
    }

    aligned_msd =
        internal::tm_realign_calculate_msd(xy, rx, ry, best_xform, score_d8sq);
    return true;
  }
}  // namespace

bool TMAlign::initialize(const InitFlags flags) {
  ArrayXc secx, secy;

  if ((flags & (InitFlags::kSecStr | InitFlags::kLocalPlusSecStr))
      != InitFlags::kNone) {
    secx = internal::assign_secstr_approx_full(query(), rx_);
    secy = internal::assign_secstr_approx_full(templ(), ry_);
  }

  return initialize(flags, secx, secy);
}

bool TMAlign::initialize(const InitFlags flags, ConstRef<ArrayXc> secx,
                         ConstRef<ArrayXc> secy) {
  const int lx = static_cast<int>(query().cols()),
            ly = static_cast<int>(templ().cols());

  if (ABSL_PREDICT_FALSE(flags == InitFlags::kNone)) {
    ABSL_LOG(WARNING) << "No initialization flags are set";
    return false;
  }

  if (ABSL_PREDICT_FALSE(l_min() < 5)) {
    ABSL_LOG(ERROR) << "One of the input structures has too few residues ("
                    << l_min() << " < 5)";
    return false;
  }

  if ((flags & (InitFlags::kSecStr | InitFlags::kLocalPlusSecStr))
          != InitFlags::kNone
      && ABSL_PREDICT_FALSE(secx.size() != query().cols()
                            || secy.size() != templ().cols())) {
    ABSL_LOG(ERROR) << "Secondary structures must be consistent with the input "
                       "structures when secondary structure-based "
                       "initialization is requested (query: "
                    << lx << " vs " << secx.size() << ", templ: "  //
                    << ly << " vs " << secy.size() << ")";
    return false;
  }

  constexpr double dcu0 = 4.25, dcu0_sq = dcu0 * dcu0;
  constexpr int simplify_step = 40;
  constexpr int max_iter_full = 30, max_iter_short = 2;

  const double d0 = l_min() < 20 ? 0.968 : 1.24 * std::cbrt(l_min() - 15) - 1,
               d01 = d0 + 1.5;
  const double d0sq_inv = 1 / (d0 * d0), d01sq_inv = 1 / (d01 * d01);

  const double d0_search = nuri::clamp(d0, 4.5, 8.0),
               d0sq_search = d0_search * d0_search;
  const double score_d8 = 1.5 * std::pow(l_min(), 0.3) + 3.5,
               score_d8sq = score_d8 * score_d8;

  const double ddcc = l_min() <= 40 ? 0.1 : 0.4;

  ArrayXi y2x_best = ArrayXi::Constant(ly, -1);

  ArrayXXc path(ly + 1, lx + 1);
  path.col(0).fill(internal::kPathHorz);
  path.row(0).fill(internal::kPathVert);

  ArrayXXd val(ly + 1, lx + 1);
  val.col(0).fill(0);
  val.row(0).fill(0);

  internal::AllowEigenMallocScoped<false> ems;

  double raw_tm_max = -1;
  auto tm_try_init = [&](InitFlags test, auto init, const int g1, const int g2,
                         const int max_iter, const double cutoff_coeff) {
    if ((flags & test) == InitFlags::kNone)
      return;

    if (!init())
      return;

    xy_.swap_remap(y2x_local());

    auto [xform, tmscore] = internal::tmscore_greedy_search<true>(
        rx_, ry_, dsqs_, i_ali(), j_ali(), xy_, simplify_step, d0_search,
        score_d8sq, d0sq_inv);
    if (tmscore > raw_tm_max) {
      best_xform_ = xform;
      raw_tm_max = tmscore;
      xy_.swap_align_with(y2x_best);
    }

    const double try_align_cutoff = raw_tm_max * cutoff_coeff;
    if (tmscore > try_align_cutoff) {
      internal::tm_find_best_alignment(xform, raw_tm_max, rx_, ry_, dsqs_, path,
                                       val, xy_, y2x_best, i_ali(), j_ali(), g1,
                                       g2, max_iter, simplify_step, d0_search,
                                       score_d8sq, d0sq_inv);
    }
  };

  tm_try_init(
      InitFlags::kGaplessThreading,
      [&]() {
        return internal::tm_initial_gt(rx_, ry_, dsqs_, query(), templ(),
                                       y2x_local(), d0sq_inv, d0sq_search)
               > 0;
      },
      0, 2, max_iter_full, 1.0);

  tm_try_init(
      InitFlags::kSecStr,
      [&]() {
        internal::tm_initial_ss(y2x_local(), path, val, secx, secy);
        return true;
      },
      0, 2, max_iter_full, 0.2);

  tm_try_init(
      InitFlags::kLocal,
      [&]() {
        return internal::tm_initial_local(rx_, ry_, dsqs_, path, val, xy_,
                                          y2x_local(), y2x_buf(), d0sq_inv,
                                          d01sq_inv, d0sq_search)
               > 0;
      },
      0, 2, max_iter_short, ddcc);

  tm_try_init(
      InitFlags::kLocalPlusSecStr,
      [&]() {
        xy_.remap(y2x_best);
        return internal::tm_initial_ssplus(rx_, ry_, path, val, xy_,
                                           y2x_local(), secx, secy, d01sq_inv);
      },
      0, 2, max_iter_full, ddcc);

  tm_try_init(
      InitFlags::kFragmentGaplessThreading,
      [&]() {
        return internal::tm_initial_fgt(rx_, ry_, dsqs_, query(), templ(),
                                        y2x_local(), dcu0_sq, d0sq_inv,
                                        d0sq_search)
               > 0;
      },
      1, 2, max_iter_short, ddcc);

  return tmalign_conclude_init(best_xform_, aligned_msd_, xy_,
                               std::move(y2x_best), rx_, ry_, dsqs_,
                               y2x_local(), y2x_buf(), raw_tm_max, d0_search,
                               score_d8sq, d0sq_inv);
}

bool TMAlign::initialize(ConstRef<ArrayXi> y2x) {
  const double d0 = l_min() < 20 ? 0.968 : 1.24 * std::cbrt(l_min() - 15) - 1;
  const double d0sq_inv = 1 / (d0 * d0);

  const double d0_search = nuri::clamp(d0, 4.5, 8.0);
  const double score_d8 = 1.5 * std::pow(l_min(), 0.3) + 3.5,
               score_d8sq = score_d8 * score_d8;

  return tmalign_conclude_init(best_xform_, aligned_msd_, xy_, y2x, rx_, ry_,
                               dsqs_, y2x_local(), y2x_buf(), -1, d0_search,
                               score_d8sq, d0sq_inv);
}

std::pair<Affine3d, double> TMAlign::tm_score(int l_norm, double d0) {
  std::pair<Affine3d, double> result;
  result.second = -1;

  if (ABSL_PREDICT_FALSE(!initialized())) {
    ABSL_LOG(ERROR)
        << "TMAlign is not initialized or previous initialization failed";
    return result;
  }

  if (ABSL_PREDICT_FALSE(l_norm == 0)) {
    ABSL_LOG(ERROR) << "l_norm must be positive or set to a negative value to "
                       "request automatic l_norm selection (got "
                    << l_norm << ")";
    return result;
  }

  // Default is to use the length of the reference structure
  if (l_norm < 0)
    l_norm = static_cast<int>(templ().cols());

  if (d0 <= 0) {
    constexpr double d0_min = 0.5;
    d0 = nuri::max(l_norm <= 21 ? 0.5 : 1.24 * std::cbrt(l_norm - 15) - 1.8,
                   d0_min);
  }

  const double d0sq_inv = 1 / (d0 * d0);
  const double d0_search = nuri::clamp(d0, 4.5, 8.0);

  result = internal::tmscore_greedy_search<false>(
      rx_, ry_, dsqs_, i_ali(), j_ali(), xy_, 1, d0_search, 0, d0sq_inv);
  result.second /= l_norm;
  return result;
}

namespace {
  TMAlignResult tm_align_wrapper(TMAlign &&tm, int l_norm, double d0) {
    TMAlignResult result;
    result.msd = tm.aligned_msd();
    std::tie(result.xform, result.tm_score) = tm.tm_score(l_norm, d0);
    result.templ_to_query = std::move(tm).templ_to_query();
    return result;
  }
}  // namespace

TMAlignResult tm_align(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
                       TMAlign::InitFlags flags, int l_norm, double d0) {
  TMAlign tm(query, templ);
  if (!tm.initialize(flags))
    return {};

  return tm_align_wrapper(std::move(tm), l_norm, d0);
}

TMAlignResult tm_align(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
                       ConstRef<ArrayXc> secx, ConstRef<ArrayXc> secy,
                       TMAlign::InitFlags flags, int l_norm, double d0) {
  TMAlign tm(query, templ);
  if (!tm.initialize(flags, secx, secy))
    return {};

  return tm_align_wrapper(std::move(tm), l_norm, d0);
}

TMAlignResult tm_align(ConstRef<Matrix3Xd> query, ConstRef<Matrix3Xd> templ,
                       ConstRef<ArrayXi> y2x, int l_norm, double d0) {
  TMAlign tm(query, templ);
  if (!tm.initialize(y2x))
    return {};

  return tm_align_wrapper(std::move(tm), l_norm, d0);
}

namespace internal {
  std::pair<int, double> tmalign_score_fun8(const Matrix3Xd &x,
                                            const Matrix3Xd &y,
                                            ArrayXi &aligned, double d_cutoff,
                                            double d0sq_inv,
                                            double score_d8sq_cutoff) {
    ArrayXd dsqs(x.cols());
    return collect_res_tmscore<true, ArrayXd &>(x, y, dsqs, aligned, d_cutoff,
                                                score_d8sq_cutoff, d0sq_inv);
  }

  std::pair<int, double> tmalign_score_fun8(const Matrix3Xd &x,
                                            const Matrix3Xd &y,
                                            ArrayXi &aligned, double d_cutoff,
                                            double d0sq_inv) {
    ArrayXd dsqs(x.cols());
    return collect_res_tmscore<false, ArrayXd &>(x, y, dsqs, aligned, d_cutoff,
                                                 0, d0sq_inv);
  }

  std::pair<Affine3d, double> tmalign_tmscore8_search(const AlignedXY &xy,
                                                      int simplify_step,
                                                      double local_d0_search,
                                                      double score_d8sq_cutoff,
                                                      double d0sq_inv) {
    Matrix3Xd rx(3, xy.l_ali()), ry(3, xy.l_ali());
    ArrayXd dsqs(xy.l_ali());
    ArrayXi i_ali(xy.l_ali()), j_ali(xy.l_ali());
    return tmscore_greedy_search<true>(rx, ry, dsqs, i_ali, j_ali, xy,
                                       simplify_step, local_d0_search,
                                       score_d8sq_cutoff, d0sq_inv);
  }

  void tmalign_dp_iter(Affine3d &xform_best, double &tmscore_max, AlignedXY &xy,
                       ArrayXi &y2x_best, int g1, int g2, int max_iter,
                       int simplify_step, double local_d0_search,
                       double score_d8sq_cutoff, double d0sq_inv) {
    Matrix3Xd rx(3, xy.x().cols()), ry(3, xy.y().cols());

    ArrayXXc path(xy.y().cols() + 1, xy.x().cols() + 1);
    path.col(0).fill(kPathHorz);
    path.row(0).fill(kPathVert);

    ArrayXXd val(xy.y().cols() + 1, xy.x().cols() + 1);
    val.col(0).fill(0);
    val.row(0).fill(0);

    ArrayXd dsqs(xy.l_min());
    ArrayXi buf1(xy.y().cols()), buf2(xy.y().cols());

    tm_find_best_alignment(xform_best, tmscore_max, rx, ry, dsqs, path, val, xy,
                           y2x_best, buf1, buf2, g1, g2, max_iter,
                           simplify_step, local_d0_search, score_d8sq_cutoff,
                           d0sq_inv);
  }

  double tmalign_get_score_fast(const Matrix3Xd &x, const Matrix3Xd &y,
                                double d0sq_inv, double d0_search) {
    const auto n_overlap = x.cols();

    Matrix3Xd rx(3, n_overlap), ry(3, n_overlap);
    ArrayXd dsqs(n_overlap);
    return tmscore_fast<ArrayXd &, Matrix3Xd &>(rx, ry, dsqs, x, y, d0sq_inv,
                                                d0_search * d0_search);
  }
}  // namespace internal
}  // namespace nuri
