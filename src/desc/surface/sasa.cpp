//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <absl/base/optimization.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/geometry.h"
#include "nuri/core/molecule.h"
#include "nuri/desc/surface.h"

namespace nuri {
namespace internal {
  namespace {
    void sr_sasa_direct(ArrayXd &sasa, const ArrayXd &radii,
                        const Matrix3Xd &pts, const Matrix3Xd &probes) {
      const int nprobe = static_cast<int>(probes.cols());

      for (int i = 0; i < radii.size(); ++i) {
        Vector3d cntr = pts.col(i);
        double rsum = radii[i];

        int naccess = 0;
        for (int j = 0; j < nprobe; ++j) {
          Vector3d pt = cntr + probes.col(j) * rsum;

          bool blocked = false;
          for (int k = 0; k < radii.size(); ++k) {
            if (k == i)
              continue;

            double rcut = radii[k];
            double dsq = (pts.col(k) - pt).squaredNorm();
            if (dsq < rcut * rcut) {
              blocked = true;
              break;
            }
          }

          if (!blocked)
            ++naccess;
        }

        sasa[i] = 4 * constants::kPi * rsum * rsum * naccess / nprobe;
      }
    }

    void sr_sasa_tree(ArrayXd &sasa, const ArrayXd &radii,
                      const Matrix3Xd &conf, const Matrix3Xd &probes) {
      const int nprobe = static_cast<int>(probes.cols());

      const OCTree at(conf);

      Matrix3Xd pbuf(probes);
      OCTree pt(pbuf);
      const Vector3d orig_max = pt.max(), orig_len = pt.len();

      const double cutoff = radii.maxCoeff();
      std::vector<std::vector<int>> idxs(nprobe);
      for (int i = 0; i < radii.size(); ++i) {
        const Vector3d cntr = conf.col(i);
        const double rsum = radii[i];

        pbuf = (probes * rsum).colwise() + cntr;
        pt.notify_transform(orig_max * rsum + cntr, orig_len * rsum);

        for (auto &v: idxs)
          v.clear();
        pt.find_neighbors_tree(at, cutoff, idxs);

        int naccess = 0;
        for (int j = 0; j < idxs.size(); ++j) {
          Vector3d pj = pbuf.col(j);

          bool blocked = false;
          for (int k: idxs[j]) {
            if (i == k)
              continue;

            double dsq = (conf.col(k) - pj).squaredNorm();
            if (dsq < radii[k] * radii[k]) {
              blocked = true;
              break;
            }
          }

          if (!blocked)
            ++naccess;
        }

        sasa[i] = 4 * constants::kPi * rsum * rsum * naccess / nprobe;
      }
    }
  }  // namespace

  ArrayXd sr_sasa_impl(const Matrix3Xd &pts, const ArrayXd &radii, int nprobe,
                       SrSasaMethod method) {
    ArrayXd sasa(radii.size());
    Matrix3Xd probes = canonical_fibonacci_lattice(nprobe);

    if (method == internal::SrSasaMethod::kAuto) {
      method = (radii.size() < 100 && nprobe < 100)
                   ? internal::SrSasaMethod::kDirect
                   : internal::SrSasaMethod::kOctree;
    }

    switch (method) {  // NOLINT(clang-diagnostic-switch-enum)
    case internal::SrSasaMethod::kDirect:
      sr_sasa_direct(sasa, radii, pts, probes);
      break;
    case internal::SrSasaMethod::kOctree:
      sr_sasa_tree(sasa, radii, pts, probes);
      break;
    default:
      ABSL_UNREACHABLE();
    }

    return sasa;
  }
}  // namespace internal

ArrayXd shrake_rupley_sasa(const Molecule &mol, const Matrix3Xd &conf,
                           const int nprobe, const double rprobe,
                           internal::SrSasaMethod method) {
  ArrayXd radii(mol.num_atoms());
  for (auto atom: mol)
    radii[atom.id()] = atom.data().element().vdw_radius() + rprobe;

  ArrayXd sasa = internal::sr_sasa_impl(conf, radii, nprobe, method);
  return sasa;
}
}  // namespace nuri
