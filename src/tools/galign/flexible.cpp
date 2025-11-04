//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>

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
    struct Invariants {
      Nonnull<const GASamplingArgs *> sampling;
      Nonnull<const GAMinimizeArgs *> minimize;

      Nonnull<const GARigidMolInfo *> query;
      E::Matrix3Xf query_centered;

      Nonnull<const GARigidMolInfo *> templ;
      E::Matrix3Xf templ_centered;

      float base_clash;
      float hetero_scale;
    };

    struct Buffers {
      E::ArrayXXf simplexf;
      E::ArrayXf cutoffs;
    };

    E::Vector3f random_unit() {
      float u = draw_urd(-1.0F, 1.0F), v = std::sqrt(1.0F - u * u),
            t = draw_urd(0.0F, static_cast<float>(constants::kTwoPi));
      return { v * std::cos(t), v * std::sin(t), u };
    }

    E::Vector3f random_translation(float max_trs) {
      return E::Vector3f::NullaryExpr(
          [&] { return draw_urd(-max_trs, max_trs); });
    }

    E::AngleAxisf random_rotation(float max_angle) {
      auto icdf = [max_angle](float p) {
        float q = 2 * p - 1;
        return std::copysign(std::pow(std::abs(q), 1.0F / 3.0F), q) * max_angle;
      };

      E::Vector3f axis = random_unit();
      float angle = icdf(draw_urd(1.0F));
      return E::AngleAxisf(angle, axis);
    }

    float self_clash(const E::Matrix3Xf &pts) {
      float penalty = 0;
      int n = static_cast<int>(pts.cols());

      for (int i = 1; i < n; ++i) {
        E::Vector3f v = pts.col(i);
        for (int j = 0; j < i; ++j) {
          float d = (v - pts.col(j)).squaredNorm();
          if (d < 5.0F)
            penalty += 1.0F + value_if(d < 4.0F, 9.0F);
        }
      }

      return penalty / static_cast<float>(n);
    }

    float flex_score(const E::Matrix3Xf &conf, const Invariants &inv) {
      float aln_score = align_score(*inv.query, conf, *inv.templ,
                                    inv.templ_centered, inv.hetero_scale);

      float clash = self_clash(conf);
      float clash_penalty = nonnegative(clash - inv.base_clash);

      return aln_score - clash_penalty;
    }

    E::Vector3f rotation_vector(const E::Quaternionf &q) {
      E::AngleAxisf aa(q);
      return aa.angle() * aa.axis();
    }

    E::AngleAxisf angle_axis_from_rotvec(const E::Vector3f &rotvec) {
      float angle = rotvec.norm();

      E::Vector3f axis;
      if (angle > 1e-6F) {
        axis = rotvec / angle;
      } else {
        angle = 0.0F;
        axis = E::Vector3f::UnitX();
      }

      return E::AngleAxisf(angle, axis);
    }

    class GeneticConf {
    public:
      GeneticConf(const GARigidMolInfo &query, const E::Matrix3Xf &query_ref)
          : conf_(query_ref), rigid_(E::Isometry3f::Identity()),
            torsion_(E::ArrayXf::Zero(
                static_cast<Eigen::Index>(query.rot_info().size()))) { }

      GeneticConf(const GARigidMolInfo &query, const E::Vector3f &templ_cntr,
                  GAlignResult &&rigid)
          : conf_(std::move(rigid.conf)), rigid_(std::move(rigid.xform)),
            torsion_(E::ArrayXf::Zero(
                static_cast<Eigen::Index>(query.rot_info().size()))),
            score_(rigid.align_score) {
        conf_.colwise() -= templ_cntr;
        rigid_.translation() += rigid_.linear() * query.cntr() - templ_cntr;
      }

      E::Matrix3Xf &conf() & { return conf_; }

      const E::Isometry3f &rigid() const { return rigid_; }

      const E::ArrayXf &torsion() const { return torsion_; }

      float score() const { return score_; }

      void update_from_simplexf(ConstRef<E::ArrayXf> simplexf,
                                const Invariants &inv) {
        E::Translation3f trs(simplexf.head<3>());
        E::AngleAxisf aa =
            angle_axis_from_rotvec(simplexf.segment<3>(3).matrix());

        rigid_ = trs * aa;
        inplace_transform(conf_, rigid_, inv.query_centered);

        for (int i = 0; i < torsion().size(); ++i) {
          torsion_[i] = simplexf[i + 6];
          inv.query->rot_info()[i].rotate(conf_, torsion_[i]);
        }

        score_ = -simplexf[simplexf.size() - 1];
      }

      friend bool operator<(const GeneticConf &lhs, const GeneticConf &rhs) {
        return lhs.score() < rhs.score();
      }

      friend bool operator>(const GeneticConf &lhs, const GeneticConf &rhs) {
        return lhs.score() > rhs.score();
      }

    private:
      E::Matrix3Xf conf_;
      E::Isometry3f rigid_;
      E::ArrayXf torsion_;

      float score_;

      friend class Mutator;
    };

    class Mutator {
    public:
      Mutator(GeneticConf &gconf, const Invariants &inv)
          : gconf_(&gconf), inv_(&inv) { }

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

      Mutator &random(const GASamplingArgs &sampling) {
        int sel = draw_uid(ndim());
        switch (sel) {
        case 0:
          update_trs(random_translation(sampling.max_trs));
          break;
        case 1:
          update_rot(random_rotation(sampling.max_rot).toRotationMatrix());
          break;
        default:
          update_tors(sel - 2, draw_urd(-sampling.max_tors, sampling.max_tors));
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

        gconf_->score_ = flex_score(conf(), *inv_);
      }

    private:
      E::Matrix3Xf &conf() { return gconf_->conf_; }

      void update_trs(const E::Vector3f &delta) {
        delta_.translation() += delta;
        trs_updated_ = true;
      }

      void update_rot(const E::Matrix3f &delta) {
        delta_.linear() = delta * delta_.linear();
        rot_updated_ = true;
      }

      void update_tors(int i, float delta) {
        inv_->query->rot_info()[i].rotate(conf(), delta);
        gconf_->torsion_[i] += delta;
      }

      Nonnull<GeneticConf *> gconf_;

      Nonnull<const Invariants *> inv_;

      E::Isometry3f delta_ = E::Isometry3f::Identity();

      bool trs_updated_ = false;
      bool rot_updated_ = false;
    };

    float exp_cumsum_scores(MutRef<E::ArrayXf> cutoffs,
                            const std::vector<GeneticConf> &confs,
                            const float start = 0.0F) {
      const int n = static_cast<int>(cutoffs.size());
      ABSL_DCHECK_GE(confs.size(), n);

      float total = start;
      for (int i = 0; i < n; ++i)
        cutoffs[i] = (total += std::exp(confs[i].score()));
      return total;
    }

    void fill_initial(std::vector<GeneticConf> &pool, const Invariants &inv) {
      const int pool_size = inv.sampling->pool_size;

      if (pool.size() >= pool_size)
        return;

      E::ArrayXf cutoffs(pool_size);
      exp_cumsum_scores(cutoffs.head(pool.size()), pool);

      for (int i = static_cast<int>(pool.size()); i < pool_size; ++i) {
        const GeneticConf &sel = pool[weighted_select(cutoffs.head(i))];

        GeneticConf &newconf = pool.emplace_back(sel);
        {
          Mutator mut(newconf, inv);
          mut.random(*inv.sampling);
        }

        cutoffs[i] = cutoffs[i - 1] + newconf.score();
      }
    }

    void minimize_one_conf(GeneticConf &gconf, NelderMead<float> &nm,
                           MutRef<E::ArrayXXf> simplex, const Invariants &inv,
                           Buffers &buf) {
      E::Quaternionf q0(gconf.rigid().linear());

      simplex.col(0).head<3>() = gconf.rigid().translation();
      simplex.col(0).segment<3>(3) = rotation_vector(q0);
      simplex.col(0).tail(gconf.torsion().size()) = gconf.torsion();
      simplex.rightCols(nm.n()).colwise() = simplex.col(0);

      for (int i = 0; i < 3; ++i)
        simplex.col(i + 1)[i] += inv.sampling->max_trs;

      for (int i = 0; i < 3; ++i) {
        simplex.col(i + 4).segment<3>(3) = rotation_vector(
            E::AngleAxisf(inv.sampling->max_rot, E::Vector3f::Unit(i)) * q0);
      }

      for (int i = 0; i < gconf.torsion().size(); ++i)
        simplex.col(i + 7)[i + 6] += inv.sampling->max_tors;

      auto eval_func = [&](ConstRef<E::ArrayXf> x) {
        E::Translation3f trs(x.head<3>());
        E::AngleAxisf aa = angle_axis_from_rotvec(x.segment<3>(3).matrix());

        E::Isometry3f xform = trs * aa;
        inplace_transform(gconf.conf(), xform, inv.query_centered);

        for (int i = 0; i < gconf.torsion().size(); ++i)
          inv.query->rot_info()[i].rotate(gconf.conf(), x[i + 6]);

        float score = flex_score(gconf.conf(), inv);
        return -score;
      };

      auto [_, idx] = nm.minimize(eval_func, inv.minimize->max_iters,
                                  inv.minimize->ftol, inv.minimize->alpha,
                                  inv.minimize->gamma, inv.minimize->rho,
                                  inv.minimize->sigma);

      gconf.update_from_simplexf(buf.simplexf.col(idx), inv);
    }

    void genetic_sampling(std::vector<GeneticConf> &pool_sample,
                          const std::vector<GeneticConf> &initial_pool,
                          const Invariants &inv, Buffers &buf) {
      const int pool_size = inv.sampling->pool_size;
      const float initial_sum = buf.cutoffs[pool_size - 1];

      auto prev_cutoffs = buf.cutoffs.tail(pool_size);
      exp_cumsum_scores(prev_cutoffs, pool_sample, initial_sum);

      for (int i = pool_size; i < pool_sample.size(); ++i) {
        const GeneticConf &seed =
            pool_sample[weighted_select(prev_cutoffs, initial_sum)];

        auto sel = weighted_select(buf.cutoffs);
        const GeneticConf &other =
            sel < pool_size ? initial_pool[sel] : pool_sample[sel - pool_size];

        GeneticConf &newconf = pool_sample[i] = seed;

        Mutator mut(newconf, inv);
        mut.crossover(other);
        for (int j = 0; j < inv.sampling->mut_cnt; ++j)
          if (draw_urd(1.0F) <= inv.sampling->mut_prob)
            mut.random(*inv.sampling);
      }
    }
  }  // namespace

  std::vector<GAlignResult>
  flexible_galign_impl(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                       int max_conf, float scale,
                       const GASamplingArgs &sampling,
                       const GAMinimizeArgs &minimize) {
    if (query.rot_info().empty())
      return rigid_galign_impl(query, templ, max_conf, scale,
                               sampling.rigid_min_msd);

    std::vector rigid_result = rigid_galign_impl(
        query, templ, sampling.rigid_max_conf, scale, sampling.rigid_min_msd);

    const int ndimp1 = 6 + static_cast<int>(query.rot_info().size()) + 1;

    Buffers buf {
      E::ArrayXXf(ndimp1, ndimp1),
      E::ArrayXf(sampling.pool_size * 2),
    };

    const Invariants inv {
      &sampling,
      &minimize,
      &query,
      query.ref().colwise() - query.cntr(),
      &templ,
      templ.ref().colwise() - templ.cntr(),
      self_clash(query.ref()),
      scale,
    };

    NelderMead<float> nm(buf.simplexf);
    auto simplex = buf.simplexf.topRows(nm.n());

    std::vector<GeneticConf> pool;
    pool.reserve(sampling.pool_size + sampling.sample_size);

    for (GAlignResult &r: rigid_result)
      pool.push_back(GeneticConf(query, templ.cntr(), std::move(r)));

    fill_initial(pool, inv);

    for (GeneticConf &conf: pool)
      minimize_one_conf(conf, nm, simplex, inv, buf);

    const std::vector<GeneticConf> initial_pool(pool);
    E::ArrayXf cutoffs(sampling.pool_size * 2);
    exp_cumsum_scores(cutoffs.head(sampling.pool_size), initial_pool);

    pool.resize(sampling.pool_size + sampling.sample_size,
                GeneticConf(query, inv.query_centered));

    int patience = sampling.patience;
    float prev_max = -1e10F;

    for (int i = 0; i < sampling.max_gen; ++i) {
      genetic_sampling(pool, initial_pool, inv, buf);

      for (int j = sampling.pool_size; j < pool.size(); ++j) {
        minimize_one_conf(pool[j], nm, simplex, inv, buf);
      }

      std::nth_element(pool.begin(), pool.begin() + sampling.pool_size - 1,
                       pool.end(), std::greater<>());

      float current_max =
          std::max_element(pool.begin(), pool.begin() + sampling.pool_size)
              ->score();
      if (current_max - prev_max < minimize.ftol && --patience <= 0)
        break;

      prev_max = current_max;
    }

    auto topk = argpartition(pool, max_conf, std::greater<>());

    std::vector<GAlignResult> flex_result;
    flex_result.reserve(max_conf);
    for (int i = 0; i < max_conf; ++i) {
      GeneticConf &conf = pool[topk[i]];

      float aln_score = align_score(query, conf.conf(), templ,
                                    inv.templ_centered, inv.hetero_scale);
      flex_result.push_back(
          { std::move(conf.conf()), conf.rigid(), aln_score });

      GAlignResult &result = flex_result.back();
      result.conf.colwise() += templ.cntr();
      result.xform.translation() +=
          templ.cntr() - result.xform.linear() * query.cntr();
    }

    return flex_result;
  }
}  // namespace internal
}  // namespace nuri
