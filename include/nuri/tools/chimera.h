//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TOOLS_CHIMERA_H_
#define NURI_TOOLS_CHIMERA_H_

#include <Eigen/Dense>

#include "nuri/eigen_config.h"

namespace nuri {
struct MmResult {
  Isometry3d xform;
  ArrayXi sel;
  double msd;
};

/**
 * @brief Match-Maker algorithm for structural alignment of two 3D point clouds.
 *
 * Iteratively estimates a rigid transform by least-squares fitting and rejects
 * outliers ("violations") whose post-fit pairwise distance exceeds a cutoff.
 * In each iteration, the number of points rejected is capped by both ratio
 * limits (the effective cap is the smaller of the two). The process repeats
 * until no violations remain.
 *
 * @param query The query points.
 * @param templ The template points.
 * @param cutoff Distance cutoff in angstroms. A point pair is counted as a
 *        violation if its post-fit distance exceeds this threshold.
 * @param global_ratio Maximum fraction of currently considered aligned points
 *        that may be excluded as outliers in a single iteration.
 * @param viol_ratio Maximum fraction of currently violating points that may be
 *        excluded in a single iteration.
 * @return Alignment result: a rigid-body transform that maps query into
 *         template coordinates, the indices of inlier points, and the mean
 *         squared deviation (msd) of inliers after alignment. If least-squares
 *         fitting fails, msd is set to a negative value and the remaining
 *         fields are in a valid but unspecified state.
 * @note The two point clouds must have the same number of points, and are
 *       assumed to be in correspondence by index.
 */
extern MmResult match_maker(ConstRef<Matrix3Xd> query,
                            ConstRef<Matrix3Xd> templ, double cutoff = 2.0,
                            double global_ratio = 0.1, double viol_ratio = 0.5);
}  // namespace nuri

#endif /* NURI_TOOLS_CHIMERA_H_ */
