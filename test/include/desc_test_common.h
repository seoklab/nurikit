//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_TEST_DESC_DESC_TEST_COMMON_H_
#define NURI_TEST_DESC_DESC_TEST_COMMON_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/desc/surface.h"
#include "nuri/fmt/parse_result.h"
#include "nuri/fmt/pdb.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
/**
 * The one probe within `radius` of `x`, -1 if none or several.
 */
inline int probe_at(const Matrix3Xd &pos, const Vector3d &x,
                    const double radius = 1e-9) {
  int found = -1;
  for (int p = 0; p < pos.cols(); ++p) {
    if ((pos.col(p) - x).norm() > radius)
      continue;
    if (found >= 0)
      return -1;
    found = p;
  }
  return found;
}

/**
 * The departure-tangent hemispheres of face `p`: the leading caps of its row
 * with `cosa == 0`, whose axes are the tangents.
 */
inline auto hemispheres(const SurfaceGeometry &s, const int p) {
  const CapTable &caps = s.concave().caps;
  const int beg = caps.off[p];
  int n = 0;
  while (n < caps.off.degree(p) && caps.cosa[beg + n] == 0)
    ++n;
  return caps.axis.middleCols(beg, n);
}

/**
 * Circle of the original atoms `a`, `b`, -1 if none.
 */
inline int circle_of(const SurfaceGeometry &s, const int a, const int b) {
  const auto [lo, hi] = nuri::minmax(a, b);
  const std::vector<SasCircle> &circles = s.toroidal().circles;
  for (int q = 0; q < static_cast<int>(circles.size()); ++q) {
    const SasCircle &c = circles[q];
    const auto [ci, cj] = nuri::minmax(s.order()[c.i], s.order()[c.j]);
    if (ci == lo && cj == hi)
      return q;
  }
  return -1;
}

inline Matrix3Xd star(const int n_ring, const double polar_deg,
                      const double reach) {
  const double polar = polar_deg * constants::kPi / 180;
  Matrix3Xd pts(3, n_ring + 1);
  pts.col(0) << 0, 0, reach;
  for (int k = 0; k < n_ring; ++k) {
    const double az = constants::kTwoPi * k / n_ring;
    pts.col(k + 1) << reach * std::sin(polar) * std::cos(az),
        reach * std::sin(polar) * std::sin(az), reach * std::cos(polar);
  }
  return pts;
}

struct Balls {
  Matrix3Xd pts;
  ArrayXd sar;
};

/**
 * Three spheres on the z axis through one circle, with radii and offsets
 * whose rounded heights are exactly consistent (searched offline); the one in
 * the middle has no surface.
 */
inline Balls shared_circle_pts(const int mid) {
  constexpr std::array<double, 3> r { 3.5, 3.25, 3.375 },
      z { -1.421875, 0.578125, 1.078125 };
  Balls b { Matrix3Xd(3, 3), ArrayXd(3) };
  for (int q = 0; q < 3; ++q) {
    const int src = (q + 1 - mid + 3) % 3;
    b.pts.col(q) << 0, 0, z[src];
    b.sar[q] = r[src];
  }
  return b;
}

struct RigidMotion {
  Matrix3d rot;
  Vector3d t;
};

inline RigidMotion rigid_motion(const int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  const Vector3d ax(nd(rng), nd(rng), nd(rng));
  const Matrix3d rot = AngleAxisd(nd(rng), ax.normalized()).toRotationMatrix();
  const Vector3d t(nd(rng), nd(rng), nd(rng));
  return { rot, 3.0 * t };
}

inline Matrix3Xd rigid(const Matrix3Xd &pts, const int seed) {
  const auto [rot, t] = rigid_motion(seed);
  return (rot * pts).colwise() + t;
}

/**
 * The exactly tangent pair `(0, 1)` and spheres 2, 3 through the touching
 * point `(1.5, 0, 0)`, four spheres through one point; the roots of faces
 * `{0, 2, 3}` and `{1, 2, 3}` there are linked only by the circle of the
 * pair `(2, 3)`, which carries no ring when both are occluders.
 */
inline Balls occluder_shell_pts() {
  Matrix3Xd pts(3, 6);
  pts << 0, 3, 1.5, 1.5, 1.5, 1.5,  //
      0, 0, 1.5, 0, -1.5, -3.5,     //
      0, 0, 0, 1.5, -1.5, -3.5;
  return { pts, ArrayXd::Constant(6, 1.5) };
}

/**
 * Two exactly tangent spheres and a third through the tangency point, the
 * circle of zero length with two coincident cut points.
 */
inline Balls tangent_through_pts() {
  Matrix3Xd pts(3, 3);
  pts.col(0) << 0, 0, 0;
  pts.col(1) << 3, 0, 0;
  pts.col(2) << 1.5, 1.5, 0;
  return { pts, ArrayXd::Constant(3, 1.5) };
}

/**
 * Four coplanar spheres at the corners of a square, through the two points
 * `(0, 0, +-0.5)`, and an apex sphere covering the upper one.
 */
inline Balls square_apex() {
  Matrix3Xd pts(3, 5);
  pts << 1, 1, -1, -1, 0,  //
      1, -1, 1, -1, 0,     //
      0, 0, 0, 0, 1.6;
  return { pts, ArrayXd::Constant(5, 1.5) };
}

inline Balls three_spheres_pts(const double rp) {
  Matrix3Xd pts(3, 3);
  pts << 0, 4.8, 2.6,  //
      0, 0.2, 4.16,    //
      0, 0.3, -0.4;
  ArrayXd sar(3);
  sar << 1.5 + rp, 1.9 + rp, 1.7 + rp;
  return { pts, sar };
}

struct NearTangentFace {
  Balls balls;
  double zc;
};

/**
 * Spheres 1, 2 overlap by 1e-4 in a circle of radius 0.014 that sphere 0
 * cuts nearly tangentially, so the face's two roots `(0, +-h, zc)` sit `h`
 * on either side of the plane of the three centers with nearly coplanar
 * departure tangents; sphere 3 supplies cutters within `2 rp` of them.
 */
inline NearTangentFace near_tangent_face(const double h) {
  const double r = 2.0, d = 4.0 - 1e-4, zs = 3.0;
  const double rc = std::sqrt(r * r - d * d / 4),
               zc = std::sqrt(rc * rc - h * h);
  Matrix3Xd pts(3, 4);
  pts.col(0) << 0, 0, zs;
  pts.col(1) << -d / 2, 0, 0;
  pts.col(2) << d / 2, 0, 0;
  pts.col(3) << 1.5, 1.5, 1.0;
  ArrayXd sar(4);
  sar << std::sqrt(h * h + (zs - zc) * (zs - zc)), r, r, 2.2;
  return { Balls { pts, sar }, zc };
}

/**
 * Every relabeling of the atoms: the result must not depend on the index
 * order the perturbation uses.
 */
template <class F>
void for_each_index_order(const Matrix3Xd &pts, const ArrayXd &sar,
                          const F &f) {
  const int n = static_cast<int>(sar.size());
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  do {
    Matrix3Xd p(3, n);
    ArrayXd r(n);
    for (int i = 0; i < n; ++i) {
      p.col(i) = pts.col(order[i]);
      r[i] = sar[order[i]];
    }
    f(p, r, order);
  } while (std::next_permutation(order.begin(), order.end()));
}

struct Protein {
  Matrix3Xd pts;
  ArrayXd radii;
};

/**
 * Heavy atoms of the first model, restricted to `chain` unless it is `'\0'`.
 */
inline Protein heavy_atoms(const char *name, const char chain = '\0') {
  std::ifstream ifs(std::filesystem::path(NURI_TEST_DATA_DIR) / name);
  EXPECT_TRUE(ifs) << name;
  PDBReader reader(ifs);
  PDBRecord record;
  EXPECT_TRUE(reader.getnext(record));
  ParseResult<PDBModel> model = read_pdb_model(record.text());
  EXPECT_TRUE(model) << model.error_msg();

  const auto &atoms = model->atoms();
  const int n = static_cast<int>(atoms.size());
  Protein out { Matrix3Xd(3, n), ArrayXd(n) };
  int w = 0;
  for (int i = 0; i < n; ++i) {
    const PDBAtom &a = atoms[i];
    if (a.hetero() || a.element().atomic_number() <= 1
        || (chain != '\0' && a.rid().chain_id != chain))
      continue;
    out.pts.col(w) = model->major_conf().col(i);
    out.radii[w] = a.element().vdw_radius();
    ++w;
  }
  out.pts.conservativeResize(3, w);
  out.radii.conservativeResize(w);
  return out;
}
}  // namespace internal
}  // namespace nuri

#endif /* NURI_TEST_DESC_DESC_TEST_COMMON_H_ */
