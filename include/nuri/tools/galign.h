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
                                            const Matrix3Xd &ref);

    Matrix3Xd &rotate(Matrix3Xd &pts, double angle) const;

    int origin() const { return origin_; }
    int ref() const { return ref_; }
    double normalizer() const { return normalizer_; }
    const ArrayXi &moving() const { return moving_; }

  private:
    int origin_;
    int ref_;
    double normalizer_;
    ArrayXi moving_;
  };

  class GARigidMolInfo {
  public:
    GARigidMolInfo(const Molecule &mol, const Matrix3Xd &ref,
                   double vdw_scale = 0.8, double hetero_scale = 0.7,
                   int dcut = 6);

    const Molecule &mol() const { return *mol_; }

    const Matrix3Xd &ref() const { return *ref_; }

    const Vector3d &cntr() const { return cntr_; }

    const std::vector<GARotationInfo> &rot_info() const { return rot_info_; }

    const ArrayXi &atom_types() const { return atom_types_; }

    auto vdw_radii() const { return vdw_rads_vols_.col(0); }

    auto vdw_vols() const { return vdw_rads_vols_.col(1); }

    const ArrayXXd &dists() const { return dists_; }

    const MatrixXd &nv() const { return neighbor_vec_; }

    double overlap() const { return overlap_; }

    int n() const { return mol().num_atoms(); }

  private:
    Nonnull<const Molecule *> mol_;
    Nonnull<const Matrix3Xd *> ref_;
    Vector3d cntr_;

    std::vector<GARotationInfo> rot_info_;

    ArrayXi atom_types_;
    ArrayX2d vdw_rads_vols_;

    ArrayXXd dists_;
    MatrixXd neighbor_vec_;
    double overlap_;
  };

  ABSL_ATTRIBUTE_PURE_FUNCTION ABSL_ATTRIBUTE_HOT extern double
  shape_overlap_impl(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                     const ArrayXXd &dists, double scale);

  inline double align_score_impl(const GARigidMolInfo &query,
                                 const GARigidMolInfo &templ,
                                 const ArrayXXd &dists, double scale) {
    return shape_overlap_impl(query, templ, dists, scale) / templ.overlap();
  }

  struct AlignResult {
    Matrix3Xd conf;
    Isometry3d xform = Isometry3d::Identity();
    double align_score = -1.0;
  };

  std::vector<AlignResult>
  rigid_galign_impl(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                    int max_conf = 1, double scale = 0.7, double min_msd = 9.0);

  struct GASamplingArgs {
    double max_trs = 2.5;
    double max_rot = deg2rad(120);
    double max_tors = max_rot;
    int pool_size = 10;
    int sample_size = 30;
    int max_gen = 50;
    int patience = 5;

    int mut_cnt = 5;
    double mut_prob = 0.5;
  };

  struct GAMinimizeArgs {
    double alpha = 1.0;
    double gamma = 2.0;
    double rho = 0.5;
    double sigma = 0.5;

    double ftol = 1e-2;
    int max_iters = 300;
  };

  std::vector<AlignResult>
  flexible_galign_impl(const GARigidMolInfo &query, const GARigidMolInfo &templ,
                       int max_conf = 1, double scale = 0.7,
                       const GASamplingArgs &genetic = {},
                       const GAMinimizeArgs &minimize = {},
                       int rigid_max_conf = 4, double rigid_min_msd = 9.0);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TOOLS_GALIGN_H_ */
