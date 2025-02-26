//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/nullability.h>
#include <absl/base/optimization.h>
#include <absl/cleanup/cleanup.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/algo/optim.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  namespace {
    double xh_length_approx(const AtomData &heavy) {
      return heavy.element().covalent_radius() + kPt[1].covalent_radius();
    }

    Vector3d opposite_unit_nth(Molecule::Atom atom, const Matrix3Xd &conf,
                               const int n) {
      return safe_normalized(conf.col(atom.id())
                             - conf.col(atom[n].dst().id()));
    }

    Vector3d mean_unit_of_first_n(Molecule::Atom atom, const Matrix3Xd &conf,
                                  const int n) {
      Vector3d mean = safe_normalized(
          (conf(Eigen::all, as_index(atom)).leftCols(n).colwise()
           - conf.col(atom.id()))
              .rowwise()
              .sum());
      return mean;
    }

    // Place H atom(s) on the z-axis. zl must have length of X-H bond.
    void place_initial_z(Molecule::Atom atom, Matrix3Xd &conf, Vector3d z,
                         const int begin, const int end) {
      ABSL_ASSUME(begin >= 0);
      ABSL_ASSUME(end - begin <= 2);
      const int degree = atom.degree();
      ABSL_ASSUME(end <= degree);

      for (int i = begin; i < end; ++i) {
        conf.col(atom[i].dst().id()) = conf.col(atom.id()) + z;
        z = -z;
      }
    }

    void place_initial_z(Molecule::Atom atom, Matrix3Xd &conf,
                         const Vector3d &z) {
      place_initial_z(atom, conf, z,
                      atom.degree() - atom.data().implicit_hydrogens(),
                      atom.degree());
    }

    // Place H atom(s) on the plane defined by the heavy atom and the two x- and
    // y-components. Each component must be scaled by the angle between the
    // X-H bond and the corresponding axis vector, and the length of the X-H
    // bond.
    void place_initial_xc_yc(Molecule::Atom atom, Matrix3Xd &conf, Vector3d xc,
                             const Vector3d &yc, const int begin,
                             const int end) {
      ABSL_ASSUME(begin >= 0);
      ABSL_ASSUME(end - begin <= 2);
      const int degree = atom.degree();
      ABSL_ASSUME(end <= degree);

      for (int i = begin; i < end; ++i) {
        conf.col(atom[i].dst().id()) = conf.col(atom.id()) + xc + yc;
        xc = -xc;
      }
    }

    void place_initial_xc_yc(Molecule::Atom atom, Matrix3Xd &conf,
                             const Vector3d &xc, const Vector3d &yc) {
      place_initial_xc_yc(atom, conf, xc, yc,
                          atom.degree() - atom.data().implicit_hydrogens(),
                          atom.degree());
    }

    bool place_initial_single_fixed(Molecule::Atom atom, Matrix3Xd &conf,
                                    const double xhd) {
      Vector3d mean = mean_unit_of_first_n(atom, conf, atom.degree() - 1);
      place_initial_z(atom, conf, -xhd * mean);
      return true;
    }

    void place_initial_terminal(Molecule::Atom atom, Matrix3Xd &conf) {
      ABSL_DCHECK_EQ(atom.degree(), 1);
      place_initial_z(atom, conf,
                      Vector3d::UnitZ() * xh_length_approx(atom.data()));
    }

    bool place_initial_sp(Molecule::Atom atom, Matrix3Xd &conf) {
      ABSL_DCHECK_GT(atom.degree(), 1);

      const double xhd = xh_length_approx(atom.data());

      Vector3d z;
      switch (atom.data().implicit_hydrogens()) {
      case 1:
        z = opposite_unit_nth(atom, conf, 0);
        break;
      case 2:
        z = Vector3d::UnitZ();
        break;
      default:
        ABSL_UNREACHABLE();
      }

      place_initial_z(atom, conf, z * xhd);
      return atom.data().implicit_hydrogens() == 1;
    }

    auto find_transitive_neighbor_sp2(Molecule::Atom atom, const int h_begin) {
      auto dst = atom[0].dst();

      if (dst.data().hybridization() != constants::kSP2)
        return dst.end();

      return absl::c_find_if(dst, [&](Molecule::Neighbor nei) {
        return nei.dst().id() != atom.id() && nei.dst().id() < h_begin;
      });
    }

    bool place_initial_sp2(Molecule::Atom atom, Matrix3Xd &conf,
                           const int h_begin) {
      ABSL_DCHECK_GT(atom.degree(), 1);

      const double xhd = xh_length_approx(atom.data());

      if (atom.degree() == 3 && atom.data().implicit_hydrogens() == 1)
        return place_initial_single_fixed(atom, conf, xhd);

      Vector3d ux, uy;
      bool h_fixed = false;

      // Possible combination of (degree, implicit H):
      // - (3, 3), (2, 2): free, just assign any position
      // - (3, 2), (2, 1): need a reference plane; if alpha atom is sp2 and beta
      //                   atom exists, place H on the same plane
      if (atom.degree() == atom.data().implicit_hydrogens()) {
        ux = Vector3d::UnitX();
        uy = Vector3d::UnitY();
      } else if (auto nit = find_transitive_neighbor_sp2(atom, h_begin);
                 !nit.end()) {
        h_fixed = atom.degree() == 3;

        uy = opposite_unit_nth(atom, conf, 0);

        Vector3d z =
            uy.cross(conf.col(nit->dst().id()) - conf.col(atom[0].dst().id()));
        ux = safe_normalized(uy.cross(z));
      } else {
        uy = opposite_unit_nth(atom, conf, 0);
        ux = any_perpendicular(uy);
      }

      place_initial_xc_yc(atom, conf, xhd * constants::kCos30 * ux,
                          xhd * constants::kCos60 * uy);
      return h_fixed;
    }

    constexpr double
        kSqrt2 =
            1.4142135623730950488016887242096980785696718753769480731766797380,
        kSqrt3 =
            1.7320508075688772935274463415058723669428052538103806280558069794;

    bool place_initial_sp3(Molecule::Atom atom, Matrix3Xd &conf) {
      const double xhd = xh_length_approx(atom.data());
      const int nh = atom.data().implicit_hydrogens(),
                nfixed = atom.degree() - nh;

      switch (nfixed) {
      case 3:
        return place_initial_single_fixed(atom, conf, xhd);
      case 2: {
        ABSL_DCHECK_GE(atom.degree(), 3);

        Vector3d v = conf.col(atom[0].dst().id()) - conf.col(atom.id()),
                 w = conf.col(atom[1].dst().id()) - conf.col(atom.id());
        Vector3d ux = safe_normalized(v.cross(w)), uy = -safe_normalized(v + w);
        place_initial_xc_yc(atom, conf, xhd * constants::kCos36 * ux,
                            xhd * constants::kCos54 * uy);
        return atom.degree() == 4;
      }
      default:
        break;
      }

      ABSL_DCHECK_LE(nfixed, 1);

      // Given a circumcenter and a vertex of a regular tetrahedron, find the
      // other three vertices.
      // See: https://math.stackexchange.com/a/97112.
      Matrix3d axes;
      if (ABSL_PREDICT_TRUE(nfixed == 1)) {
        axes.col(2) = opposite_unit_nth(atom, conf, 0);
      } else {
        axes.col(2) = Vector3d::UnitZ();
      }
      axes.col(0) = any_perpendicular(axes.col(2));
      axes.col(1) = kSqrt3 * axes.col(2).cross(axes.col(0));

      axes *= xhd / 3;
      axes.leftCols(2) *= kSqrt2;

      Matrix3d vecs;
      vecs.col(0) = axes.col(2) + 2 * axes.col(0);
      vecs.col(1) = axes.col(2) - axes.col(0) + axes.col(1);
      vecs.col(2) = axes.col(2) - axes.col(0) - axes.col(1);

      conf(Eigen::all, as_index(atom)).rightCols(nh) =
          (vecs.colwise() + conf.col(atom.id())).leftCols(nh);

      return false;
    }

    std::pair<Matrix3d, Array3i>
    gen_sp3d_axes_counts_fixed2(Molecule::Atom atom, const Matrix3Xd &conf) {
      Matrix3d axes;
      Array3i cnts;

      axes.rightCols<2>() =
          -(conf(Eigen::all, as_index(atom)).leftCols<2>().colwise()
            - conf.col(atom.id()));
      safe_colwise_normalize(axes.rightCols<2>());

      // two vectors have angle > 150 degrees, likely to be on the axial
      // positions
      if (-axes.col(1).dot(axes.col(2)) > constants::kCos30) {
        axes.col(1) = any_perpendicular(axes.col(2));
        axes.col(0) = axes.col(1).cross(axes.col(2));

        cnts = { 2, 1, 0 };
      } else {
        axes.col(0) = safe_normalized(axes.col(1).cross(axes.col(2)));

        cnts = { 2, 0, 1 };
      }

      return { axes, cnts };
    }

    std::pair<Matrix3d, Array3i>
    gen_sp3d_axes_counts_fixed3(Molecule::Atom atom, const Matrix3Xd &conf) {
      Array3i cnts;

      Matrix3d vecs = -(conf(Eigen::all, as_index(atom)).leftCols<3>().colwise()
                        - conf.col(atom.id()));
      safe_colwise_normalize(vecs);

      constexpr int selector[3][3] = {
        // vector i
        { 0, 1, 2 },
        // vector j
        { 1, 2, 0 },
        // complement vector k, or
        // index of the "other" cosine similairty; see below
        { 2, 0, 1 },
      };

      Array3d cos_xyz = (vecs(Eigen::all, selector[0]).array()
                         * vecs(Eigen::all, selector[1]).array())
                            .colwise()
                            .sum()
                            .transpose();

      Matrix3d axes;

      int min_idx;
      const double min_cos = cos_xyz.minCoeff(&min_idx);
      const int cmpl = selector[2][min_idx];

      if (-min_cos > constants::kCos30) {
        // two of the vectors are collinear
        // -> we only need xy axis

        axes.col(1) = vecs.col(cmpl);
        axes.col(0) = safe_normalized(
            vecs.col(cmpl).cross(vecs.col(selector[0][min_idx])));

        cnts = { 2, 0, 0 };
      } else {
        // xyz vectors likely form sp2 part of the molecule
        // -> we only need z axis

        // for numerical stability, select any one that does not have the
        // largest angle
        axes.col(2) = safe_normalized(
            vecs.col(selector[0][cmpl]).cross(vecs.col(selector[1][cmpl])));

        cnts = { 0, 0, 2 };
      }

      return { axes, cnts };
    }

    bool place_initial_sp3d(Molecule::Atom atom, Matrix3Xd &conf) {
      const double xhd = xh_length_approx(atom.data());
      const int nh = atom.data().implicit_hydrogens(),
                nfixed = atom.degree() - nh;

      Matrix3d axes;
      // xy, y, z
      Array3i max_cnts;

      switch (nfixed) {
      case 4:
        return place_initial_single_fixed(atom, conf, xhd);
      case 3:
        std::tie(axes, max_cnts) = gen_sp3d_axes_counts_fixed3(atom, conf);
        break;
      case 2:
        std::tie(axes, max_cnts) = gen_sp3d_axes_counts_fixed2(atom, conf);
        break;
      case 1: {
        axes.col(2) = opposite_unit_nth(atom, conf, 0);
        axes.col(1) = any_perpendicular(axes.col(2));
        axes.col(0) = axes.col(1).cross(axes.col(2));

        max_cnts = { 2, 1, 1 };
        break;
      }
      case 0:
        axes = Matrix3d::Identity();
        max_cnts = { 2, 1, 2 };
        break;
      default:
        ABSL_UNREACHABLE();
      }

      axes *= xhd;

      // z_begin, y_begin, xy_begin
      Array3i idxs;
      std::exclusive_scan(std::make_reverse_iterator(max_cnts.end()),
                          std::make_reverse_iterator(max_cnts.begin()),
                          idxs.begin(), nfixed);
      idxs.tail(2) = idxs.tail(2).min(atom.degree());

      place_initial_z(atom, conf, axes.col(2), nfixed, idxs[1]);
      place_initial_z(atom, conf, axes.col(1), idxs[1], idxs[2]);
      place_initial_xc_yc(atom, conf, axes.col(0) * constants::kCos30,
                          -constants::kCos60 * axes.col(1), idxs[2],
                          atom.degree());

      return false;
    }

    std::pair<Matrix3d, Array3i>
    gen_sp3d2_axes_counts_fixed2(Molecule::Atom atom, const Matrix3Xd &conf) {
      Matrix3d axes;
      Array3i cnts;

      axes.rightCols<2>() =
          -(conf(Eigen::all, as_index(atom)).leftCols<2>().colwise()
            - conf.col(atom.id()));
      safe_colwise_normalize(axes.rightCols<2>());

      // two vectors have angle > 135 degrees, too large for another axis vector
      // assume they are the opposite vertices of an octahedron
      if (-axes.col(1).dot(axes.col(2)) > constants::kCos45) {
        axes.col(1) = any_perpendicular(axes.col(2));
        axes.col(0) = axes.col(1).cross(axes.col(2));

        cnts = { 2, 2, 0 };
      } else {
        axes.col(0) = safe_normalized(axes.col(1).cross(axes.col(2)));

        cnts = { 2, 1, 1 };
      }

      return { axes, cnts };
    }

    std::pair<Matrix3d, Array3i>
    gen_sp3d2_axes_counts_fixed3(Molecule::Atom atom, const Matrix3Xd &conf) {
      Matrix3d axes;
      Array3i cnts;

      axes = -(conf(Eigen::all, as_index(atom)).leftCols<3>().colwise()
               - conf.col(atom.id()));
      safe_colwise_normalize(axes);

      Array3d cos_xyz = { axes.col(0).dot(axes.col(2)),
                          axes.col(1).dot(axes.col(2)),
                          axes.col(0).dot(axes.col(1)) };
      int min_axis;
      const double min_cos = cos_xyz.minCoeff(&min_axis);

      if (-min_cos <= constants::kCos45) {
        // xyz vectors can form "reasonable" basis
        cnts = { 1, 1, 1 };
      } else if (min_axis == 2) {
        // x-y are collinear
        axes.col(0) = safe_normalized(axes.col(1).cross(axes.col(2)));
        cnts = { 2, 0, 1 };
      } else {
        // x-z or y-z are collinear
        const int keep = 1 - min_axis;

        axes.col(min_axis) = safe_normalized(axes.col(keep).cross(axes.col(2)));

        cnts[min_axis] = 2;
        cnts[keep] = 1;
        cnts[2] = 0;
      }

      return { axes, cnts };
    }

    using Array6d = Array<double, 6, 1>;

    std::pair<Matrix3d, Array3i>
    gen_sp3d2_axes_counts_fixed4(Molecule::Atom atom, const Matrix3Xd &conf) {
      Array3i cnts;

      Matrix<double, 3, 4> vecs =
          -(conf(Eigen::all, as_index(atom)).leftCols<4>().colwise()
            - conf.col(atom.id()));
      safe_colwise_normalize(vecs);

      constexpr int selector[3][6] = {
        { 2, 2, 2, 1, 1, 0 }, // vector i
        { 3, 1, 0, 3, 0, 3 }, // vector j
        { 4, 5, 3, 2, 0, 1 }, // complement cosine indices
      };

      Array6d cos_xyz = (vecs(Eigen::all, selector[0]).array()
                         * vecs(Eigen::all, selector[1]).array())
                            .colwise()
                            .sum()
                            .transpose();

      Matrix3d axes;

      int min_idx;
      cos_xyz.minCoeff(&min_idx);
      axes.col(0) = vecs.col(selector[0][min_idx]);

      const int compl_idx = selector[2][min_idx];
      const double compl_cos = cos_xyz[compl_idx];

      axes.col(1) = vecs.col(selector[0][compl_idx]);
      if (-compl_cos > constants::kCos45) {
        // other two are colinear
        axes.col(2) = axes.col(0).cross(axes.col(1));
        cnts = { 0, 0, 2 };
      } else {
        // other two form reasonable basis
        axes.col(2) = vecs.col(selector[1][compl_idx]);
        cnts = { 0, 1, 1 };
      }

      return { axes, cnts };
    }

    bool place_initial_sp3d2(Molecule::Atom atom, Matrix3Xd &conf) {
      const double xhd = xh_length_approx(atom.data());
      const int nh = atom.data().implicit_hydrogens(),
                nfixed = atom.degree() - nh;

      Matrix3d axes;
      // x, y, z
      Array3i max_cnts;

      switch (nfixed) {
      case 5:
        return place_initial_single_fixed(atom, conf, xhd);
      case 4:
        std::tie(axes, max_cnts) = gen_sp3d2_axes_counts_fixed4(atom, conf);
        break;
      case 3:
        std::tie(axes, max_cnts) = gen_sp3d2_axes_counts_fixed3(atom, conf);
        break;
      case 2:
        std::tie(axes, max_cnts) = gen_sp3d2_axes_counts_fixed2(atom, conf);
        break;
      case 1:
        axes.col(2) = opposite_unit_nth(atom, conf, 0);
        axes.col(1) = any_perpendicular(axes.col(2));
        axes.col(0) = axes.col(1).cross(axes.col(2));

        max_cnts = { 2, 2, 1 };
        break;
      default:
        axes = Matrix3d::Identity();
        max_cnts = { 2, 2, 2 };
        break;
      }

      axes *= xhd;

      // z_begin, y_begin, x_begin
      Array3i idxs;
      std::exclusive_scan(std::make_reverse_iterator(max_cnts.end()),
                          std::make_reverse_iterator(max_cnts.begin()),
                          idxs.begin(), nfixed);
      idxs.tail(2) = idxs.tail(2).min(atom.degree());

      place_initial_z(atom, conf, axes.col(2), nfixed, idxs[1]);
      place_initial_z(atom, conf, axes.col(1), idxs[1], idxs[2]);
      place_initial_z(atom, conf, axes.col(0), idxs[2], atom.degree());

      return false;
    }
  }  // namespace

  /*
   * We have made all hydrogens explicit but each atom keeps track of the number
   * of previously-implied hydrogens.
   */
  std::pair<std::vector<int>, bool>
  place_trailing_hydrogens_initial(const Molecule &mol, Matrix3Xd &conf,
                                   int h_begin) {
    std::vector<int> free_hs;

    for (int i = 0; i < h_begin; ++i) {
      auto atom = mol.atom(i);
      if (atom.data().implicit_hydrogens() == 0)
        continue;

      int sn = static_cast<int>(atom.data().hybridization()),
          nbep = sn - atom.degree(),
          nfixed = atom.degree() - atom.data().implicit_hydrogens();

      if (atom.degree() == 1) {
        ABSL_LOG_IF(WARNING, sn != constants::kTerminal)
            << "Terminal atom " << atom.id()
            << " has non-terminal hybridization";
        sn = constants::kTerminal;
      } else if (nbep < 0) {
        ABSL_LOG(WARNING) << "Invalid nonbonding electron pair count for atom "
                          << atom.id() << ": " << nbep;
        return { free_hs, false };
      }

      ABSL_DCHECK_LE(atom.degree(), sn);

      bool h_fixed;
      // NOLINTNEXTLINE(*-switch-enum)
      switch (sn) {
      case constants::kTerminal:
        place_initial_terminal(atom, conf);
        h_fixed = false;
        break;
      case constants::kSP:
        h_fixed = place_initial_sp(atom, conf);
        break;
      case constants::kSP2:
        h_fixed = place_initial_sp2(atom, conf, h_begin);
        break;
      case constants::kSP3:
        h_fixed = place_initial_sp3(atom, conf);
        break;
      case constants::kSP3D:
        h_fixed = place_initial_sp3d(atom, conf);
        break;
      case constants::kSP3D2:
        h_fixed = place_initial_sp3d2(atom, conf);
        break;
      default:
        ABSL_LOG(WARNING)
            << "Invalid steric number " << sn << " for atom " << atom.id()
            << " with " << nfixed << " fixed neighbors and "
            << atom.data().implicit_hydrogens() << " implicit H";
        return { free_hs, false };
      }
      if (h_fixed)
        continue;

      for (int h = nfixed; h < atom.degree(); ++h)
        free_hs.push_back(atom[h].dst().id());
    }

    return { free_hs, true };
  }

  namespace {
    class FreeHProxy {
    public:
      FreeHProxy(const Molecule &mol, Matrix3Xd &conf, const Matrix3Xd &current,
                 const std::vector<int> &free_hs)
          : opt_blsq_inv_(free_hs.size()), unbound_near_(free_hs.size()),
            rvdw_sq_(2, mol.size()),  //
            mol_(&mol), conf_(&conf), free_hs_(&free_hs) {
        for (int h = 0; h < free_hs.size(); ++h)
          opt_blsq_inv_[h] = 1 / xh_blsq(current, h);

        ArrayXi free_h_inv = ArrayXi::Constant(mol.num_atoms(), -1);
        for (int h = 0; h < free_hs.size(); ++h)
          free_h_inv[free_hs[h]] = h;

        for (int i = 0; i < mol.size(); ++i) {
          rvdw_sq_(0, i) =
              mol[i].data().element().vdw_radius() + kPt[1].vdw_radius();
        }
        rvdw_sq_.row(0) = rvdw_sq_.row(0) * 0.5;
        rvdw_sq_.row(1) = rvdw_sq_.row(0).square();

        hh_rvdw_ = rvdw_sq_(0, mol.size() - 1);
        hh_rvdw_sq_ = rvdw_sq_(1, mol.size() - 1);

        OCTree octree(conf);
        std::vector<double> dbuf;
        for (int h = 0; h < free_hs.size(); ++h) {
          const int i = free_hs[h];
          octree.find_neighbors_d(conf.col(i), 7.5, fixed_near(h), dbuf);

          const int j = mol[i][0].dst().id();
          erase_if(fixed_near(h), [&](int k) {
            int k_h = free_h_inv[k];
            // prevent double evaluation
            if (k_h > h)
              free_near(h).push_back(k_h);
            return k_h >= 0 || k == j;
          });
        }
      }

      ~FreeHProxy() = default;

      FreeHProxy(const FreeHProxy &) = delete;
      FreeHProxy(FreeHProxy &&) = delete;
      FreeHProxy &operator=(const FreeHProxy &) = delete;
      FreeHProxy &operator=(FreeHProxy &&) = delete;

      const Molecule &mol() const { return *mol_; }

      const Matrix3Xd &conf() const { return *conf_; }

      const ArrayXd &opt_blsq_inv() const { return opt_blsq_inv_; }

      Vector3d xh_vec(ConstRef<Matrix3Xd> pts, int h) const {
        return pts.col(h) - conf().col(mol()[free_hs()[h]][0].dst().id());
      }

      double xh_blsq(ConstRef<Matrix3Xd> pts, int h) const {
        return xh_vec(pts, h).squaredNorm();
      }

      const std::vector<int> &fixed_near(int h) const {
        return unbound_near_[h].first;
      }

      const std::vector<int> &free_near(int h) const {
        return unbound_near_[h].second;
      }

      const std::vector<int> &free_hs() const { return *free_hs_; }

      auto rvdw_sq(int i) const { return rvdw_sq_.col(i); }

      double hh_rvdw() const { return hh_rvdw_; }
      double hh_rvdw_sq() const { return hh_rvdw_sq_; }

    private:
      std::vector<int> &fixed_near(int h) { return unbound_near_[h].first; }

      std::vector<int> &free_near(int h) { return unbound_near_[h].second; }

      /* size: nfree */
      ArrayXd opt_blsq_inv_;
      std::vector<std::pair<std::vector<int>, std::vector<int>>> unbound_near_;

      /* size: natoms */
      Array2Xd rvdw_sq_;
      double hh_rvdw_;
      double hh_rvdw_sq_;

      absl::Nonnull<const Molecule *> mol_;
      absl::Nonnull<Matrix3Xd *> conf_;
      absl::Nonnull<const std::vector<int> *> free_hs_;
    };

    // Havel's distance error function:
    // E3 in Distance Geometry in Molecular Modeling, 1994, Ch.6, p. 311.
    double bond_length_error(const FreeHProxy &proxy, MutRef<Matrix3Xd> &gx,
                             ConstRef<Matrix3Xd> x, int h, double weight) {
      Vector3d diff = proxy.xh_vec(x, h);

      const double dsq = diff.squaredNorm(),
                   optsq_inv = proxy.opt_blsq_inv()[h];

      const double ub_term = nonnegative(dsq * optsq_inv - 1);

      const double lb_inner_inv = 2 / (1 + dsq * optsq_inv);
      const double lb_term = nonnegative(lb_inner_inv - 1);

      diff *= 4 * optsq_inv * ub_term
              - 2 * (lb_inner_inv * lb_inner_inv) * optsq_inv * lb_term;
      gx.col(h) += diff * weight;

      return weight * (ub_term * ub_term + lb_term * lb_term);
    }

    constexpr double
        // (1/0.6)^12
        kCutoffRepulFunc =
            459.39365799778338517351879136159987656202664077479908399991720624,
        // // -2 * (1/0.6)^6
        // kCutoffAttrFunc =
        //     -42.86694101508916323731138545953360768175582990397805212620027435,
        // -12 * (1/0.6)^13
        kCutoffRepulGrad = -12 * kCutoffRepulFunc / 0.6;
    // // 2 * 6 * (1/0.6)^7
    // kCutoffAttrGrad = -6 * kCutoffAttrFunc / 0.6;

    // Linear smoothing, ax + b
    constexpr double                                      //
        kFuncGrad = kCutoffRepulGrad,                     // + kCutoffAttrGrad,
        kFuncItct = kFuncGrad * -0.6 + kCutoffRepulFunc;  // + kCutoffAttrFunc;

    double lj_repulsion_vector_pair(Vector3d &diff, const double rvdw,
                                    const double rvdw_sq) {
      double dsq = diff.squaredNorm(), d = std::sqrt(dsq);

      if (0.6 * rvdw > d) {
        d += 1e-12;

        double grad = kFuncGrad / rvdw;
        diff *= grad / d;

        double fx = grad * d + kFuncItct;
        return fx;
      }

      double dsq_inv = 1 / dsq;
      double d2_term = rvdw_sq * dsq_inv;
      double d6_term = d2_term * d2_term * d2_term;

      double fx = d6_term * d6_term;
      diff *= -12 * fx * dsq_inv;
      return fx;
    }

    double lj_repulsion(const FreeHProxy &proxy, MutRef<Matrix3Xd> &gx,
                        ConstRef<Matrix3Xd> x, int h) {
      double fx = 0;

      for (int j_h: proxy.free_near(h)) {
        Vector3d diff = x.col(h) - x.col(j_h);

        fx +=
            lj_repulsion_vector_pair(diff, proxy.hh_rvdw(), proxy.hh_rvdw_sq());

        gx.col(h) += diff;
        gx.col(j_h) -= diff;
      }

      for (int i: proxy.fixed_near(h)) {
        Vector3d diff = x.col(h) - proxy.conf().col(i);

        fx += lj_repulsion_vector_pair(diff, proxy.rvdw_sq(i)[0],
                                       proxy.rvdw_sq(i)[1]);

        gx.col(h) += diff;
      }

      return fx;
    }

    double hydrogen_minimizer_funcgrad(FreeHProxy &proxy, ArrayXd &gxa,
                                       ConstRef<ArrayXd> xa, double lj_weight,
                                       double bl_weight) {
      gxa.setZero();

      MutRef<Matrix3Xd> gx = gxa.reshaped(3, proxy.free_hs().size()).matrix();
      ConstRef<Matrix3Xd> x = xa.reshaped(3, proxy.free_hs().size()).matrix();

      double lj_sum = 0, bl_sum = 0;
      for (int h = 0; h < gx.cols(); ++h) {
        double lj_err = lj_weight * lj_repulsion(proxy, gx, x, h);
        lj_sum += lj_err;
      }
      gxa *= lj_weight;

      for (int h = 0; h < gx.cols(); ++h) {
        double bl_err = bond_length_error(proxy, gx, x, h, bl_weight);
        bl_sum += bl_err;
      }

      ABSL_DVLOG(3) << "lj_sum: " << lj_sum << ", bl_sum: " << bl_sum;

      return lj_sum + bl_sum;
    }

  }  // namespace

  bool optimize_free_hydrogens(const Molecule &mol, Matrix3Xd &conf,
                               const std::vector<int> &free_hs) {
    if (free_hs.empty())
      return true;

    Matrix3Xd current = conf(Eigen::all, free_hs);
    FreeHProxy h_proxy(mol, conf, current, free_hs);

    ArrayXi nbd = ArrayXi::Zero(current.size());
    Array2Xd bds(2, current.size());
    LBfgsB minimizer(current.reshaped().array(), { nbd, bds }, 5);

    auto result = minimizer.minimize(
        [&](ArrayXd &gx, ConstRef<ArrayXd> x) {
          return hydrogen_minimizer_funcgrad(h_proxy, gx, x, 0.1, 1);
        },
        1e+7, 1e-1, 300, 300);
    if (result.code != LbfgsbResultCode::kSuccess) {
      ABSL_LOG(WARNING) << "Hydrogen optimization failed or terminated "
                           "prematurely; not updating hydrogen coordinates";
      return false;
    }

    result = minimizer.minimize(
        [&](ArrayXd &gx, ConstRef<ArrayXd> x) {
          return hydrogen_minimizer_funcgrad(h_proxy, gx, x, 0.1, 1e+3);
        },
        1e+8, 1e-1, 300, 300);
    if (result.code != LbfgsbResultCode::kSuccess) {
      ABSL_LOG(WARNING) << "Hydrogen optimization failed or terminated "
                           "prematurely; not updating hydrogen coordinates";
      return false;
    }

    conf(Eigen::all, free_hs) = current;
    return true;
  }
}  // namespace internal

bool Molecule::add_hydrogens(const bool update_confs) {
  const int h_begin = size();

  {
    auto mut = mutator();

    for (auto atom: *this) {
      for (int i = 0; i < atom.data().implicit_hydrogens(); ++i) {
        int h = mut.add_atom({ kPt[1], 0, 0, constants::kTerminal });
        mut.add_bond(atom.id(), h, BondData(constants::kSingleBond));
      }
    }
  }

  for (auto &sub: substructures()) {
    const int size = sub.size();
    for (int i = 0; i < size; ++i) {
      for (auto nei: atom(sub.atom_ids()[i])) {
        if (nei.dst().id() < h_begin)
          continue;

        sub.add_atom(nei.dst().id());
      }
    }
  }

  absl::Cleanup update_implicit_hcnt = [this, h_begin] {
    for (int i = 0; i < size() - h_begin; ++i)
      atom(i).data().set_implicit_hydrogens(0);
  };

  if (h_begin == size() || confs().empty() || !update_confs)
    return true;

  for (auto &conf: confs()) {
    auto [free_hs, ok] =
        internal::place_trailing_hydrogens_initial(*this, conf, h_begin);
    if (!ok)
      return false;

    if (!internal::optimize_free_hydrogens(*this, conf, free_hs))
      return false;
  }

  return true;
}
}  // namespace nuri
