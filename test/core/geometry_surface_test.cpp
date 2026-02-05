//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <vector>

#include <absl/algorithm/container.h>
#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "test_utils.h"
#include "nuri/core/geometry.h"

namespace nuri {
namespace {
struct Poi {
  int i, j, k;
  double omega, h;
  Vector3d u, b;
};

struct Torus {
  int i, j;
  double r;
  Vector3d u, t;

  std::vector<int> bids;
};

std::vector<std::vector<std::pair<int, double>>>
find_all_neighbors(const OCTree &oct, const ArrayXd &radii, const double eps) {
  std::vector<std::vector<std::pair<int, double>>> nbrs(radii.size());

  const double maxr = radii.maxCoeff();
  std::vector<int> js;
  std::vector<double> dsqs;
  for (int i = 0; i < oct.pts().cols(); ++i) {
    Vector3d pi = oct.pts().col(i);
    const double ri = radii[i];
    oct.find_neighbors_d(pi, maxr + ri, js, dsqs);

    for (int ji = 0; ji < js.size(); ++ji) {
      const int j = js[ji];
      const double dsq = dsqs[ji];
      const double rsum = ri + radii[j], rdiff = ri - radii[j];
      if (i == j || dsq > rsum * rsum || dsq < rdiff * rdiff + eps)
        continue;

      nbrs[i].push_back({ j, dsq });
    }

    absl::c_sort(nbrs[i], [](const std::pair<int, double> &a,
                             const std::pair<int, double> &b) {
      return a.first < b.first;
    });
  }

  return nbrs;
}

void find_i_lm(const std::vector<std::vector<std::pair<int, double>>> &nbrs,
               const Matrix3Xd &pts, const ArrayXd &radii, const ArrayXd &rsq,
               const double eps) {
  std::vector<Poi> pois;
  std::vector<Torus> tori;

  // Upper triangle map (i, j) -> torus index
  ArrayXXi tmap = ArrayXXi::Constant(radii.size() - 1, radii.size(), -1);

  for (int k = 1; k < radii.size(); ++k) {
    Vector3d ak = pts.col(k);
    double rk = radii[k], rsqk = rsq[k];

    auto end = std::find_if(nbrs[k].rbegin(), nbrs[k].rend(),
                            [&](const std::pair<int, double> &p) {
                              return p.first < k;
                            })
                   .base();
    for (auto jt = nbrs[k].begin(); jt < end; ++jt) {
      const int j = jt->first;
      const double dsqjk = jt->second;
      ABSL_DCHECK_EQ(tmap(j, k), -1);

      Vector3d aj = pts.col(j);
      double rj = radii[j];
      double rsum = rj + rk, rdiff = rj - rk;
      ABSL_DCHECK_LE(dsqjk, rsum * rsum);
      ABSL_DCHECK_GE(dsqjk, rdiff * rdiff);
      ABSL_DCHECK_GE(dsqjk, eps);

      const double dsqinv = 1 / dsqjk;

      tmap(j, k) = static_cast<int>(tori.size());
      Torus &tjk = tori.emplace_back();
      tjk.i = j;
      tjk.j = k;
      tjk.r = std::sqrt(
          0.25 * ((rsum * rsum - dsqjk) * (dsqjk - rdiff * rdiff) * dsqinv));
      tjk.u = std::sqrt(dsqinv) * (ak - aj);
      tjk.t = 0.5 * ((aj + ak) + (rsq[j] - rsqk) * dsqinv * (ak - aj));

      for (auto it = nbrs[k].begin(); it < jt; ++it) {
        const int i = it->first;
        const int itij = tmap(i, j);
        if (itij < 0)
          continue;

        Torus &tij = tori[itij];
        const int itik = tmap(i, k);
        ABSL_DCHECK_NE(itik, -1);
        Torus &tik = tori[itik];

        Vector3d uijk = tij.u.cross(tik.u);
        double cos_ijk = tij.u.dot(tik.u);
        double sin_ijk = uijk.norm();
        if (sin_ijk < eps)
          continue;

        const double scale = 1 / sin_ijk;
        uijk *= scale;
        Vector3d utb = uijk.cross(tij.u);
        Vector3d bijk = tij.t + tik.u.dot(tik.t - tij.t) * scale * utb;
        double hsq = rsq[i] - (bijk - pts.col(i)).squaredNorm();
        if (hsq <= 0) {
          if ((tij.t - ak).squaredNorm() <= rsqk - tij.r * tij.r) {
            tmap(i, j) = -1;
            break;
          }
          continue;
        }

        const int ibp = static_cast<int>(pois.size());
        tij.bids.push_back(ibp);
        tik.bids.push_back(ibp);
        tjk.bids.push_back(ibp);

        Poi &bp = pois.emplace_back();
        bp.i = i;
        bp.j = j;
        bp.k = k;
        bp.omega = std::atan2(sin_ijk, cos_ijk);
        bp.h = std::sqrt(hsq);
        bp.u = uijk;
        bp.b = bijk;
      }
    }
  }
}

TEST(SurfaceTest, Playground) {
  // std::vector<Vector3d> pois;
  // std::vector<Array3i> ijks;
  // Array3i ijk(0, 1, 2);

  // Matrix3d pts;
  // Vector3d rsq = Vector3d::Constant(1.0);

  // // One point intersection of three spheres
  // pts.transpose() << -1.5, 1.0, 2.0,  //
  //     0.5, 1.0, 2.0,                  //
  //     -0.5, 2.0, 2.0;

  // ASSERT_TRUE(find_sphere_intersections(pois, ijks, pts, rsq, ijk, 1e-12));
  // ASSERT_EQ(pois.size(), 1);
  // NURI_EXPECT_EIGEN_EQ(pois[0], Vector3d(-0.5, 1.0, 2.0));

  // // Two point intersection of three spheres
  // const double sqrt3 = std::sqrt(3.0);
  // pois.clear();
  // pts.transpose() << -1.0, 1.0, 2.0,  //
  //     0.0, 1.0, 2.0,                  //
  //     -0.5, 1.5, 2.0;
  // ASSERT_TRUE(find_sphere_intersections(pois, ijks, pts, rsq, ijk, 1e-12));
  // ASSERT_EQ(pois.size(), 2);
  // NURI_EXPECT_EIGEN_EQ(pois[0], Vector3d(-0.5, 1.0, 2.0 + sqrt3 / 2));
  // NURI_EXPECT_EIGEN_EQ(pois[1], Vector3d(-0.5, 1.0, 2.0 - sqrt3 / 2));

  // // No intersection
  // pois.clear();
  // rsq[1] = 3;
  // EXPECT_TRUE(find_sphere_intersections(pois, ijks, pts, rsq, ijk, 1e-12));
  // EXPECT_EQ(pois.size(), 0);

  // // Circle intersection (cannot pick points)
  // pois.clear();
  // pts.transpose() << -1.0, 1.0, 2.0,  //
  //     -2.0, 1.0, 2.0,                 //
  //     0.0, 1.0, 2.0;
  // EXPECT_FALSE(find_sphere_intersections(pois, ijks, pts, rsq, ijk, 1e-12));
  // EXPECT_EQ(pois.size(), 0);

  // auto ret = stack(ijks);
  // std::cout << ret.transpose() << "\n";
}
}  // namespace
}  // namespace nuri
