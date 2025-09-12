//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TOOLS_GALIGN_H_
#define NURI_TOOLS_GALIGN_H_

//! @cond
#include <vector>

#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"
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

  class GAMoleculeInfo {
  public:
    GAMoleculeInfo(const Molecule &mol, const Matrix3Xd &ref,
                   double vdw_scale = 0.8);

    const Molecule &mol() const { return *mol_; }

    const Matrix3Xd &ref() const { return *ref_; }

    const std::vector<GARotationInfo> &rot_info() const { return rot_info_; }

    const ArrayXi &atom_types() const { return atom_types_; }

    auto vdw_radii() const { return vdw_rads_vols_.col(0); }

    auto vdw_vols() const { return vdw_rads_vols_.col(1); }

    int n() const { return mol().num_atoms(); }

  private:
    Nonnull<const Molecule *> mol_;
    Nonnull<const Matrix3Xd *> ref_;

    std::vector<GARotationInfo> rot_info_;

    ArrayXi atom_types_;
    ArrayX2d vdw_rads_vols_;
  };

  class GADistanceFeature {
  public:
    GADistanceFeature(int n, double scale = 0.7, int dcut = 6);

    GADistanceFeature(const GAMoleculeInfo &mol, const Matrix3Xd &pts,
                      double scale = 0.7, int dcut = 6);

    GADistanceFeature &update(const GAMoleculeInfo &mol,
                              const Matrix3Xd &pts) noexcept;

    Eigen::Index n() const { return neighbor_vec_.cols(); }

    Eigen::Index dcut() const { return neighbor_vec_.rows(); }

    const ArrayXXd &dists() const { return dists_; }

    const MatrixXd &nv() const { return neighbor_vec_; }

    double overlap() const { return overlap_; }

    double scale() const { return scale_; }

  private:
    ArrayXXd dists_;
    MatrixXd neighbor_vec_;
    double overlap_;
    double scale_;
  };

  double shape_overlap_impl(const GAMoleculeInfo &query,
                            const GAMoleculeInfo &templ, const ArrayXXd &dists,
                            double scale);

  struct AlignResult {
    Matrix3Xd conf;
    Isometry3d xform = Isometry3d::Identity();
    double align_score = -1.0;
  };

  std::vector<AlignResult>
  rigid_galign_impl(const GAMoleculeInfo &query, const GADistanceFeature &qfeat,
                    const GAMoleculeInfo &templ, const GADistanceFeature &tfeat,
                    int max_conf = 1, double min_msd = 9.0);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TOOLS_GALIGN_H_ */
