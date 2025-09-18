//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/log/absl_check.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/algo/optim.h"
#include "nuri/core/geometry.h"
#include "nuri/random.h"
#include "nuri/tools/galign.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    Vector3d random_unit() {
      double u = draw_urd(-1.0, 1.0), v = std::sqrt(1.0 - u * u),
             t = draw_urd(0.0, constants::kTwoPi);
      return { v * std::cos(t), v * std::sin(t), u };
    }

    Vector3d random_translation(double max_trs) {
      return Vector3d::NullaryExpr([&] { return draw_urd(-max_trs, max_trs); });
    }

    AngleAxisd random_rotation(double max_angle) {
      auto icdf = [max_angle](double p) {
        double q = 2 * p - 1;
        return std::copysign(std::pow(std::abs(q), 1.0 / 3.0), q) * max_angle;
      };

      Vector3d axis = random_unit();
      double angle = icdf(draw_urd(0.0, 1.0));
      return AngleAxisd(angle, axis);
    }

    double clash_penalty(const Matrix3Xd &pts, ArrayXd &pdsq) {
      pdistsq(pdsq, pts);
      double penalty = pdsq.unaryExpr([](double d) {
                             return (d < 5.0) * 1.0 + (d < 4.0) * 9.0;
                           })
                           .sum();
      return penalty / static_cast<double>(pts.cols());
    }

    std::pair<double, double> flex_score(const GARigidMolInfo &query,
                                         const GARigidMolInfo &templ,
                                         const Matrix3Xd &conf,
                                         const Matrix3Xd &ref, ArrayXXd &cd,
                                         ArrayXd &pdsq, double base_clash,
                                         double templ_overlap, double scale) {
      cdist(cd, conf, ref);
      double align_score =
          shape_overlap_impl(query, templ, cd, scale) / templ_overlap;
      double clash = nonnegative(clash_penalty(conf, pdsq) - base_clash);
      return std::make_pair(align_score, clash);
    }

    Vector3d rotation_vector(const Quaterniond &q) {
      AngleAxisd aa(q);
      return aa.angle() * aa.axis();
    }

    AngleAxisd angle_axis_from_rotvec(const Vector3d &rotvec) {
      double angle = rotvec.norm();

      Vector3d axis;
      if (angle > 1e-6) {
        axis = rotvec / angle;
      } else {
        angle = 0.0;
        axis = Vector3d::UnitX();
      }

      return AngleAxisd(angle, axis);
    }

    class GeneticConf {
    public:
      GeneticConf(const GARigidMolInfo &query, const Matrix3Xd &query_ref)
          : conf_(query_ref), rigid_(Isometry3d::Identity()),
            torsion_(ArrayXd::Zero(
                static_cast<Eigen::Index>(query.rot_info().size()))),
            align_score_(-1), clash_penalty_(0) { }

      GeneticConf(const GARigidMolInfo &query, const Vector3d &templ_cntr,
                  AlignResult &&rigid)
          : conf_(std::move(rigid.conf)), rigid_(std::move(rigid.xform)),
            torsion_(ArrayXd::Zero(
                static_cast<Eigen::Index>(query.rot_info().size()))),
            align_score_(rigid.align_score), clash_penalty_(0) {
        conf_.colwise() -= templ_cntr;
        rigid_.translation() += rigid_.linear() * query.cntr() - templ_cntr;
      }

      Matrix3Xd &&conf() && { return std::move(conf_); }

      const Matrix3Xd &conf() const & { return conf_; }

      const Isometry3d &rigid() const & { return rigid_; }

      const ArrayXd &torsion() const & { return torsion_; }

      void update_scores(const GARigidMolInfo &query,
                         const GARigidMolInfo &templ, const Matrix3Xd &ref,
                         ArrayXXd &cd, ArrayXd &pdsq, double base_clash,
                         double templ_overlap, double scale) {
        std::tie(align_score_, clash_penalty_) =
            flex_score(query, templ, conf(), ref, cd, pdsq, base_clash,
                       templ_overlap, scale);
      }

      void update_from_simplex(ConstRef<ArrayXd> x, const GARigidMolInfo &query,
                               const Matrix3Xd &query_ref,
                               const GARigidMolInfo &templ,
                               const Matrix3Xd &templ_ref, ArrayXXd &cd,
                               ArrayXd &pdsq, double base_clash,
                               double templ_overlap, double scale,
                               bool save_transforms) {
        Translation3d trs(x.head<3>());
        AngleAxisd aa = angle_axis_from_rotvec(x.segment<3>(3).matrix());

        Isometry3d xform = trs * aa;
        inplace_transform(conf_, xform, query_ref);

        for (int i = 0; i < torsion_.size(); ++i)
          query.rot_info()[i].rotate(conf_, x[i + 6]);

        update_scores(query, templ, templ_ref, cd, pdsq, base_clash,
                      templ_overlap, scale);

        if (save_transforms) {
          rigid_ = xform;
          torsion_ = x.tail(torsion_.size());
        }
      }

      double align_score() const { return align_score_; }

      double score() const { return align_score_ - clash_penalty_; }

      friend bool operator<(const GeneticConf &lhs, const GeneticConf &rhs) {
        return lhs.score() < rhs.score();
      }

      friend bool operator>(const GeneticConf &lhs, const GeneticConf &rhs) {
        return lhs.score() > rhs.score();
      }

    private:
      Matrix3Xd conf_;
      Isometry3d rigid_;
      ArrayXd torsion_;

      double align_score_;
      double clash_penalty_;

      friend class Mutator;
    };

    class Mutator {
    public:
      Mutator(const GARigidMolInfo &mol, GeneticConf &gconf)
          : mol_(&mol), gconf_(&gconf), delta_(Isometry3d::Identity()) { }

      ~Mutator() noexcept { finalize(); }

      Mutator(const Mutator &) = delete;
      Mutator(Mutator &&) = delete;
      Mutator &operator=(const Mutator &) = delete;
      Mutator &operator=(Mutator &&) = delete;

      int ndim() const { return static_cast<int>(2 + gconf_->torsion_.size()); }

      Mutator &crossover(const GeneticConf &other) {
        if (gconf_ == &other)
          return *this;

        int sel = draw_uid(ndim());
        switch (sel) {
        case 0:
          update_trs(other.rigid_.translation() - gconf_->rigid_.translation());
          break;
        case 1:
          update_rot(other.rigid_.linear()
                     * gconf_->rigid_.linear().transpose());
          break;
        default:
          sel -= 2;
          update_tors(sel, other.torsion_[sel] - gconf_->torsion_[sel]);
          break;
        }

        return *this;
      }

      Mutator &random(double max_trs, double max_rot, double max_tors) {
        int sel = draw_uid(ndim());
        switch (sel) {
        case 0:
          update_trs(random_translation(max_trs));
          break;
        case 1:
          update_rot(random_rotation(max_rot).toRotationMatrix());
          break;
        default:
          update_tors(sel - 2, draw_urd(-max_tors, max_tors));
          break;
        }
        return *this;
      }

      void finalize() noexcept {
        if (rot_updated_) {
          inplace_transform(conf(), delta_, conf());
          gconf_->rigid_ = delta_ * gconf_->rigid_;
        } else if (trs_updated_) {
          conf().colwise() += delta_.translation();
          gconf_->rigid_.translation() += delta_.translation();
        }
      }

    private:
      Matrix3Xd &conf() { return gconf_->conf_; }

      void update_trs(const Vector3d &delta) {
        delta_.translation() += delta;
        trs_updated_ = true;
      }

      void update_rot(const Matrix3d &delta) {
        delta_.linear() = delta * delta_.linear();
        rot_updated_ = true;
      }

      void update_tors(int i, double delta) {
        mol_->rot_info()[i].rotate(conf(), delta);
        gconf_->torsion_[i] += delta;
      }

      Nonnull<const GARigidMolInfo *> mol_;
      Nonnull<GeneticConf *> gconf_;

      Isometry3d delta_;

      bool trs_updated_ = false;
      bool rot_updated_ = false;
    };

    double exp_cumsum_scores(MutRef<ArrayXd> cutoffs,
                             const std::vector<GeneticConf> &confs,
                             const double start = 0) {
      const int n = static_cast<int>(cutoffs.size());
      ABSL_DCHECK_GE(confs.size(), n);

      double total = start;
      for (int i = 0; i < n; ++i)
        cutoffs[i] = (total += std::exp(confs[i].score()));
      return total;
    }

    void fill_initial(std::vector<GeneticConf> &pool,
                      const GARigidMolInfo &query, const GARigidMolInfo &templ,
                      const Matrix3Xd &templ_ref, const GAGeneticArgs &genetic,
                      ArrayXXd &cd, ArrayXd &pdsq, const double base_clash,
                      const double templ_overlap, const double scale) {
      if (pool.size() >= genetic.pool_size)
        return;

      ArrayXd cutoffs(genetic.pool_size);
      exp_cumsum_scores(cutoffs.head(pool.size()), pool);

      for (int i = static_cast<int>(pool.size()); i < genetic.pool_size; ++i) {
        const GeneticConf &sel = pool[weighted_select(cutoffs.head(i))];

        GeneticConf &newconf = pool.emplace_back(sel);
        {
          Mutator mut(query, newconf);
          mut.random(genetic.max_trs, genetic.max_rot, genetic.max_tors);
        }
        newconf.update_scores(query, templ, templ_ref, cd, pdsq, base_clash,
                              templ_overlap, scale);

        cutoffs[i] = cutoffs[i - 1] + newconf.score();
      }
    }

    void minimize_one_conf(GeneticConf &gconf, const GARigidMolInfo &query,
                           const Matrix3Xd &query_ref,
                           const GARigidMolInfo &templ,
                           const Matrix3Xd &templ_ref, NelderMead &nm,
                           MutRef<ArrayXXd> simplex, ArrayXXd &cd,
                           ArrayXd &pdsq, double base_clash,
                           double templ_overlap, double scale,
                           const GAGeneticArgs &genetic,
                           const GAMinimizeArgs &minimize) {
      nm.reset();

      Quaterniond q0(gconf.rigid().linear());

      simplex.col(0).head<3>() = gconf.rigid().translation();
      simplex.col(0).segment<3>(3) = rotation_vector(q0);
      simplex.col(0).tail(gconf.torsion().size()) = gconf.torsion();
      simplex.rightCols(nm.n()).colwise() = simplex.col(0);

      for (int i = 0; i < 3; ++i)
        simplex.col(i + 1)[i] += genetic.max_trs;

      for (int i = 0; i < 3; ++i) {
        simplex.col(i + 4).segment<3>(3) = rotation_vector(
            AngleAxisd(genetic.max_rot, Vector3d::Unit(i)) * q0);
      }

      for (int i = 0; i < gconf.torsion().size(); ++i)
        simplex.col(i + 7)[i + 6] += genetic.max_tors;

      auto eval_func = [&](ConstRef<ArrayXd> x) {
        gconf.update_from_simplex(x, query, query_ref, templ, templ_ref, cd,
                                  pdsq, base_clash, templ_overlap, scale,
                                  false);
        return -gconf.score();
      };

      auto [_, idx] = nm.minimize(eval_func, minimize.max_iters, minimize.ftol,
                                  minimize.alpha, minimize.gamma, minimize.rho,
                                  minimize.sigma);

      gconf.update_from_simplex(simplex.col(idx), query, query_ref, templ,
                                templ_ref, cd, pdsq, base_clash, templ_overlap,
                                scale, true);
    }

    void genetic_sampling(std::vector<GeneticConf> &pool_sample,
                          const std::vector<GeneticConf> &initial_pool,
                          ArrayXd &cutoffs, const GARigidMolInfo &query,
                          const GARigidMolInfo &templ,
                          const Matrix3Xd &templ_ref, ArrayXXd &cd,
                          ArrayXd &pdsq, double base_clash,
                          double templ_overlap, double scale,
                          const GAGeneticArgs &genetic) {
      const double initial_sum = cutoffs[genetic.pool_size - 1];

      auto prev_cutoffs = cutoffs.tail(genetic.pool_size);
      exp_cumsum_scores(prev_cutoffs, pool_sample, initial_sum);

      for (int i = genetic.pool_size; i < pool_sample.size(); ++i) {
        const GeneticConf &seed =
            pool_sample[weighted_select(prev_cutoffs, initial_sum)];

        auto sel = weighted_select(cutoffs);
        const GeneticConf &other = sel < genetic.pool_size
                                       ? initial_pool[sel]
                                       : pool_sample[sel - genetic.pool_size];

        GeneticConf &newconf = pool_sample[i] = seed;

        {
          Mutator mut(query, newconf);
          mut.crossover(other);
          for (int j = 0; j < genetic.mut_cnt; ++j)
            if (draw_urd(0.0, 1.0) <= genetic.mut_prob)
              mut.random(genetic.max_trs, genetic.max_rot, genetic.max_tors);
        }

        newconf.update_scores(query, templ, templ_ref, cd, pdsq, base_clash,
                              templ_overlap, scale);
      }
    }
  }  // namespace

  std::vector<AlignResult>
  flexible_galign_impl(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                       int max_conf, double scale, const GAGeneticArgs &genetic,
                       const GAMinimizeArgs &minimize, const int rigid_max_conf,
                       const double rigid_min_msd) {
    if (query.rot_info().empty())
      return rigid_galign_impl(query, templ, max_conf, scale, rigid_min_msd);

    std::vector rigid_result =
        rigid_galign_impl(query, templ, nuri::max(max_conf, rigid_max_conf),
                          scale, rigid_min_msd);

    const Matrix3Xd query_centered = query.ref().colwise() - query.cntr();
    const Matrix3Xd templ_centered = templ.ref().colwise() - templ.cntr();

    ArrayXXd cd(query.n(), templ.n());
    ArrayXd pdsq(query.n() * (query.n() - 1) / 2);
    const double base_clash = clash_penalty(query.ref(), pdsq);
    const double templ_overlap = templ.overlap();

    int ndimp1 = 6 + static_cast<int>(query.rot_info().size()) + 1;
    ArrayXXd simplexf(ndimp1, ndimp1);
    NelderMead nm(simplexf);
    auto simplex = simplexf.topRows(nm.n());

    std::vector<GeneticConf> pool;
    pool.reserve(genetic.pool_size + genetic.sample_size);

    for (AlignResult &r: rigid_result)
      pool.push_back(GeneticConf(query, templ.cntr(), std::move(r)));

    fill_initial(pool, query, templ, templ_centered, genetic, cd, pdsq,
                 base_clash, templ_overlap, scale);

    for (GeneticConf &conf: pool) {
      minimize_one_conf(conf, query, query_centered, templ, templ_centered, nm,
                        simplex, cd, pdsq, base_clash, templ_overlap, scale,
                        genetic, minimize);
    }

    const std::vector<GeneticConf> initial_pool(pool);
    ArrayXd cutoffs(genetic.pool_size * 2);
    exp_cumsum_scores(cutoffs.head(genetic.pool_size), initial_pool);

    pool.resize(genetic.pool_size + genetic.sample_size,
                GeneticConf(query, query_centered));

    int patience = genetic.patience;
    double prev_max = -1e10;

    for (int i = 0; i < genetic.max_gen; ++i) {
      genetic_sampling(pool, initial_pool, cutoffs, query, templ,
                       templ_centered, cd, pdsq, base_clash, templ_overlap,
                       scale, genetic);

      for (int j = genetic.pool_size; j < pool.size(); ++j) {
        minimize_one_conf(pool[j], query, query_centered, templ, templ_centered,
                          nm, simplex, cd, pdsq, base_clash, templ_overlap,
                          scale, genetic, minimize);
      }

      std::nth_element(pool.begin(), pool.begin() + genetic.pool_size - 1,
                       pool.end(), std::greater<>());

      double current_max =
          std::max_element(pool.begin(), pool.begin() + genetic.pool_size)
              ->score();
      if (current_max - prev_max < minimize.ftol && --patience <= 0)
        break;

      prev_max = current_max;
    }

    auto topk = argpartition(pool, max_conf, std::greater<>());
    absl::c_sort(topk, [&](int i, int j) {
      return pool[i].align_score() > pool[j].align_score();
    });

    std::vector<AlignResult> flex_result;
    flex_result.reserve(max_conf);
    for (int i = 0; i < max_conf; ++i) {
      GeneticConf &conf = pool[topk[i]];

      flex_result.push_back(
          { std::move(conf).conf(), conf.rigid(), conf.align_score() });

      AlignResult &result = flex_result.back();
      result.conf.colwise() += templ.cntr();
      result.xform.translation() +=
          templ.cntr() - result.xform.linear() * query.cntr();
    }

    return flex_result;
  }
}  // namespace internal
}  // namespace nuri
