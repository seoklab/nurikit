//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
      Matrix3Xd query_centered;

      Nonnull<const GARigidMolInfo *> templ;
      Matrix3Xd templ_centered;

      double base_clash;
      double hetero_scale;
    };

    struct Buffers {
      ArrayXXd simplexf;
      ArrayXd cutoffs;
    };

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

    double self_clash(const Matrix3Xd &pts) {
      double penalty = 0;
      int n = static_cast<int>(pts.cols());

      for (int i = 1; i < n; ++i) {
        Vector3d v = pts.col(i);
        for (int j = 0; j < i; ++j) {
          double d = (v - pts.col(j)).squaredNorm();
          if (d < 5.0)
            penalty += 1.0 + value_if(d < 4.0, 9.0);
        }
      }

      return penalty / n;
    }

    double flex_score(const Matrix3Xd &conf, const Invariants &inv) {
      double aln_score = align_score(*inv.query, conf, *inv.templ,
                                     inv.templ_centered, inv.hetero_scale);

      double clash = self_clash(conf);
      double clash_penalty = nonnegative(clash - inv.base_clash);

      return aln_score - clash_penalty;
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
                static_cast<Eigen::Index>(query.rot_info().size()))) { }

      GeneticConf(const GARigidMolInfo &query, const Vector3d &templ_cntr,
                  AlignResult &&rigid)
          : conf_(std::move(rigid.conf)), rigid_(std::move(rigid.xform)),
            torsion_(ArrayXd::Zero(
                static_cast<Eigen::Index>(query.rot_info().size()))),
            score_(rigid.align_score) {
        conf_.colwise() -= templ_cntr;
        rigid_.translation() += rigid_.linear() * query.cntr() - templ_cntr;
      }

      Matrix3Xd &conf() & { return conf_; }

      const Isometry3d &rigid() const { return rigid_; }

      const ArrayXd &torsion() const { return torsion_; }

      double score() const { return score_; }

      void update_from_simplexf(ConstRef<ArrayXd> simplexf,
                                const Invariants &inv) {
        Translation3d trs(simplexf.head<3>());
        AngleAxisd aa = angle_axis_from_rotvec(simplexf.segment<3>(3).matrix());

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
      Matrix3Xd conf_;
      Isometry3d rigid_;
      ArrayXd torsion_;

      double score_;

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
        inv_->query->rot_info()[i].rotate(conf(), delta);
        gconf_->torsion_[i] += delta;
      }

      Nonnull<GeneticConf *> gconf_;

      Nonnull<const Invariants *> inv_;

      Isometry3d delta_ = Isometry3d::Identity();

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

    void fill_initial(std::vector<GeneticConf> &pool, const Invariants &inv) {
      const int pool_size = inv.sampling->pool_size;

      if (pool.size() >= pool_size)
        return;

      ArrayXd cutoffs(pool_size);
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

    void minimize_one_conf(GeneticConf &gconf, NelderMead &nm,
                           MutRef<ArrayXXd> simplex, const Invariants &inv,
                           Buffers &buf) {
      Quaterniond q0(gconf.rigid().linear());

      simplex.col(0).head<3>() = gconf.rigid().translation();
      simplex.col(0).segment<3>(3) = rotation_vector(q0);
      simplex.col(0).tail(gconf.torsion().size()) = gconf.torsion();
      simplex.rightCols(nm.n()).colwise() = simplex.col(0);

      for (int i = 0; i < 3; ++i)
        simplex.col(i + 1)[i] += inv.sampling->max_trs;

      for (int i = 0; i < 3; ++i) {
        simplex.col(i + 4).segment<3>(3) = rotation_vector(
            AngleAxisd(inv.sampling->max_rot, Vector3d::Unit(i)) * q0);
      }

      for (int i = 0; i < gconf.torsion().size(); ++i)
        simplex.col(i + 7)[i + 6] += inv.sampling->max_tors;

      auto eval_func = [&](ConstRef<ArrayXd> x) {
        Translation3d trs(x.head<3>());
        AngleAxisd aa = angle_axis_from_rotvec(x.segment<3>(3).matrix());

        Isometry3d xform = trs * aa;
        inplace_transform(gconf.conf(), xform, inv.query_centered);

        for (int i = 0; i < gconf.torsion().size(); ++i)
          inv.query->rot_info()[i].rotate(gconf.conf(), x[i + 6]);

        return -flex_score(gconf.conf(), inv);
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
      const double initial_sum = buf.cutoffs[pool_size - 1];

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
          if (draw_urd(0.0, 1.0) <= inv.sampling->mut_prob)
            mut.random(*inv.sampling);
      }
    }
  }  // namespace

  std::vector<AlignResult>
  flexible_galign_impl(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                       int max_conf, double scale,
                       const GASamplingArgs &sampling,
                       const GAMinimizeArgs &minimize) {
    if (query.rot_info().empty())
      return rigid_galign_impl(query, templ, max_conf, scale,
                               sampling.rigid_min_msd);

    std::vector rigid_result = rigid_galign_impl(
        query, templ, sampling.rigid_max_conf, scale, sampling.rigid_min_msd);

    const int ndimp1 = 6 + static_cast<int>(query.rot_info().size()) + 1;

    Buffers buf {
      ArrayXXd(ndimp1, ndimp1),
      ArrayXd(sampling.pool_size * 2),
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

    NelderMead nm(buf.simplexf);
    auto simplex = buf.simplexf.topRows(nm.n());

    std::vector<GeneticConf> pool;
    pool.reserve(sampling.pool_size + sampling.sample_size);

    for (AlignResult &r: rigid_result)
      pool.push_back(GeneticConf(query, templ.cntr(), std::move(r)));

    fill_initial(pool, inv);

    for (GeneticConf &conf: pool)
      minimize_one_conf(conf, nm, simplex, inv, buf);

    const std::vector<GeneticConf> initial_pool(pool);
    ArrayXd cutoffs(sampling.pool_size * 2);
    exp_cumsum_scores(cutoffs.head(sampling.pool_size), initial_pool);

    pool.resize(sampling.pool_size + sampling.sample_size,
                GeneticConf(query, inv.query_centered));

    int patience = sampling.patience;
    double prev_max = -1e10;

    for (int i = 0; i < sampling.max_gen; ++i) {
      genetic_sampling(pool, initial_pool, inv, buf);

      for (int j = sampling.pool_size; j < pool.size(); ++j) {
        minimize_one_conf(pool[j], nm, simplex, inv, buf);
      }

      std::nth_element(pool.begin(), pool.begin() + sampling.pool_size - 1,
                       pool.end(), std::greater<>());

      double current_max =
          std::max_element(pool.begin(), pool.begin() + sampling.pool_size)
              ->score();
      if (current_max - prev_max < minimize.ftol && --patience <= 0)
        break;

      prev_max = current_max;
    }

    auto topk = argpartition(pool, max_conf, std::greater<>());

    std::vector<AlignResult> flex_result;
    flex_result.reserve(max_conf);
    for (int i = 0; i < max_conf; ++i) {
      GeneticConf &conf = pool[topk[i]];

      double aln_score = align_score(query, conf.conf(), templ,
                                     inv.templ_centered, inv.hetero_scale);
      flex_result.push_back(
          { std::move(conf.conf()), conf.rigid(), aln_score });

      AlignResult &result = flex_result.back();
      result.conf.colwise() += templ.cntr();
      result.xform.translation() +=
          templ.cntr() - result.xform.linear() * query.cntr();
    }

    return flex_result;
  }
}  // namespace internal
}  // namespace nuri
