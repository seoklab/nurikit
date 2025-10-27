//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/tools/galign.h"

namespace nuri {
namespace internal {
  namespace {
    class AtomMatching {
    public:
      AtomMatching(ArrayXi &&mapping, bool left_mapped)
          : mapping_(std::move(mapping)), left_mapped_(left_mapped) { }

      std::pair<Array3i, Array3i> select_triad(const Array3i &idxs) const {
        Array3i ret[2] { idxs, mapping_(idxs) };
        return std::make_pair(ret[left_mapped_], ret[!left_mapped_]);
      }

      auto size() const { return mapping_.size(); }

    private:
      ArrayXi mapping_;
      bool left_mapped_;
    };

    AtomMatching find_best_match(const GARigidMolInfo &query,
                                 const GARigidMolInfo &templ,
                                 const double scale, bool left_mapped = true) {
      if (query.n() < templ.n())
        return find_best_match(templ, query, scale, !left_mapped);

      const MatrixXd &qnv = query.nv(), &tnv = templ.nv();

      ArrayXi match(templ.n());
      for (int i = 0; i < templ.n(); ++i) {
        double best = -1.0;

        for (int j = 0; j < query.n(); ++j) {
          double sim = qnv.col(j).dot(tnv.col(i));
          if (query.atom_types()[j] != templ.atom_types()[i])
            sim *= scale;

          if (sim > best) {
            match[i] = j;
            best = sim;
          }
        }
      }

      return { std::move(match), left_mapped };
    }

    void maybe_replace_candidate(std::vector<GAlignResult> &results,
                                 GAlignResult &candidate,
                                 const double min_msd) {
      if (candidate.align_score * 2 < results.front().align_score)
        return;

      for (int i = 0; i < results.size(); ++i) {
        if (candidate.align_score > results[i].align_score) {
          std::swap(candidate, results[i]);

          if (candidate.align_score < 0)
            return;
        }

        if (msd(candidate.conf, results[i].conf) < min_msd)
          return;
      }
    }

    template <bool kOnlySimilar>
    void align_add_candidate(std::vector<GAlignResult> &results,
                             GAlignResult &candidate,
                             const std::pair<Array3i, Array3i> &coms,
                             const GARigidMolInfo &query,
                             const GARigidMolInfo &templ, const double scale,
                             const double min_msd) {
      if (kOnlySimilar) {
        for (int i = 0; i < 2; ++i) {
          for (int j = i + 1; j < 3; ++j) {
            if (std::abs(query.dists()(coms.first[j], coms.first[i])
                         - templ.dists()(coms.second[j], coms.second[i]))
                > 2.0) {
              return;
            }
          }
        }
      }

      Matrix3d qpts = query.ref()(Eigen::all, coms.first);
      Matrix3d tpts = templ.ref()(Eigen::all, coms.second);

      candidate.xform = qcp_inplace(qpts, tpts, AlignMode::kXformOnly).first;

      inplace_transform(candidate.conf, candidate.xform, query.ref());
      candidate.align_score =
          align_score(query, candidate.conf, templ, templ.ref(), scale);

      maybe_replace_candidate(results, candidate, min_msd);
    }

    std::vector<GAlignResult>
    align_triad(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                const AtomMatching &mapping, const int max_conf,
                const double scale, const double min_msd) {
      std::vector<GAlignResult> results(max_conf, { query.ref() });
      GAlignResult candidate { query.ref() };

      auto do_align = [&](auto align_eval) -> void {
        int row = 0;
        Array3i choose;
        choose[row] = 0;
        do {
          for (int rp1 = row + 1; row < 3 - 1; row = rp1++)
            choose[rp1] = choose[row] + 1;

          for (; choose[2] < mapping.size(); ++choose[2]) {
            align_eval(results, candidate, mapping.select_triad(choose), query,
                       templ, scale, min_msd);
          }

          for (row = 1; row >= 0 && ++choose[row] >= mapping.size() - (2 - row);
               --row)
            ;

        } while (row >= 0);
      };

      do_align(align_add_candidate<true>);

      if (absl::c_all_of(results, [](const GAlignResult &r) {
            return r.align_score < 0;
          })) {
        do_align(align_add_candidate<false>);
      }

      results.erase(absl::c_find_if(results,
                                    [](const GAlignResult &r) {
                                      return r.align_score < 0;
                                    }),
                    results.end());

      return results;
    }
  }  // namespace

  std::vector<GAlignResult> rigid_galign_impl(const GARigidMolInfo &query,
                                              const GARigidMolInfo &templ,
                                              const int max_conf,
                                              const double scale,
                                              const double min_msd) {
    AtomMatching match = find_best_match(query, templ, scale);
    return align_triad(query, templ, match, max_conf, scale, min_msd);
  }
}  // namespace internal
}  // namespace nuri
