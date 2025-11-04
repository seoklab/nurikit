//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TOOLS_GALIGN_H_
#define NURI_TOOLS_GALIGN_H_

//! @cond
#include <vector>

#include <absl/base/attributes.h>
#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  class GARotationInfo {
  public:
    static std::vector<GARotationInfo> from(const Molecule &mol,
                                            const E::Matrix3Xf &ref);

    E::Matrix3Xf &rotate(E::Matrix3Xf &pts, float angle) const;

    int origin() const { return origin_; }
    int ref() const { return ref_; }
    float normalizer() const { return normalizer_; }
    const ArrayXi &moving() const { return moving_; }

  private:
    int origin_;
    int ref_;
    float normalizer_;
    ArrayXi moving_;
  };
}  // namespace internal

class GARigidMolInfo {
public:
  GARigidMolInfo(const Molecule &mol, const Matrix3Xd &ref,
                 float vdw_scale = 0.8F, float hetero_scale = 0.7F,
                 int dcut = 6);

  const Molecule &mol() const { return *mol_; }

  const E::Matrix3Xf &ref() const { return ref_; }

  const E::Vector3f &cntr() const { return cntr_; }

  float vdw_scale() const { return vdw_scale_; }

  float hetero_scale() const { return hetero_scale_; }

  int dcut() const { return static_cast<int>(neighbor_vec_.rows()); }

  const std::vector<internal::GARotationInfo> &rot_info() const {
    return rot_info_;
  }

  const ArrayXi &atom_types() const { return atom_types_; }

  auto vdw_radii() const { return vdw_rads_vols_.col(0); }

  auto vdw_vols() const { return vdw_rads_vols_.col(1); }

  const E::ArrayXXf &dists() const { return dists_; }

  const E::MatrixXf &nv() const { return neighbor_vec_; }

  float overlap() const { return overlap_; }

  int n() const { return mol().num_atoms(); }

private:
  internal::Nonnull<const Molecule *> mol_;
  E::Matrix3Xf ref_;
  E::Vector3f cntr_;
  float vdw_scale_;
  float hetero_scale_;

  std::vector<internal::GARotationInfo> rot_info_;

  ArrayXi atom_types_;
  E::ArrayX2f vdw_rads_vols_;

  E::ArrayXXf dists_;
  E::MatrixXf neighbor_vec_;
  float overlap_;
};

struct GAlignResult {
  E::Matrix3Xf conf;
  E::Isometry3f xform = E::Isometry3f::Identity();
  float align_score = -1.0F;
};

struct GASamplingArgs {
  float max_trs = 2.5F;
  float max_rot = deg2rad(120.0F);
  float max_tors = max_rot;

  float rigid_min_msd = 9.0F;
  int rigid_max_conf = 4;

  int pool_size = 10;
  int sample_size = 30;
  int max_gen = 50;
  int patience = 5;

  int mut_cnt = 5;
  float mut_prob = 0.5F;
};

struct GAMinimizeArgs {
  float alpha = 1.0F;
  float gamma = 2.0F;
  float rho = 0.5F;
  float sigma = 0.5F;

  float ftol = 1e-2F;
  int max_iters = 300;
};

namespace internal {
  ABSL_ATTRIBUTE_PURE_FUNCTION ABSL_ATTRIBUTE_HOT extern float
  shape_overlap(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                const E::ArrayXXf &dists, float scale);

  ABSL_ATTRIBUTE_PURE_FUNCTION ABSL_ATTRIBUTE_HOT extern float
  shape_overlap(const GARigidMolInfo &query, const E::Matrix3Xf &qconf,
                const GARigidMolInfo &templ, const E::Matrix3Xf &tconf,
                float scale);

  inline float align_score(const GARigidMolInfo &query,
                           const E::Matrix3Xf &qconf,
                           const GARigidMolInfo &templ,
                           const E::Matrix3Xf &tconf, float scale) {
    return shape_overlap(query, qconf, templ, tconf, scale) / templ.overlap();
  }

  std::vector<GAlignResult>
  rigid_galign_impl(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                    int max_conf = 1, float scale = 0.7F, float min_msd = 9.0F);

  std::vector<GAlignResult>
  flexible_galign_impl(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                       int max_conf = 1, float scale = 0.7F,
                       const GASamplingArgs &sampling = {},
                       const GAMinimizeArgs &minimize = {});
}  // namespace internal

extern std::vector<GAlignResult> galign(const Molecule &mol,
                                        const Matrix3Xd &seed,
                                        const GARigidMolInfo &templ,
                                        bool flexible = true, int max_conf = 1,
                                        const GASamplingArgs &sampling = {},
                                        const GAMinimizeArgs &minimize = {});
}  // namespace nuri

#endif /* NURI_TOOLS_GALIGN_H_ */
