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
    void sr_sasa_direct(ArrayXd &sasa, const ArrayXd &radii, const ArrayXd &rsq,
                        const Matrix3Xd &pts, const Matrix3Xd &probes) {
      const int nprobe = static_cast<int>(probes.cols());

      for (int i = 0; i < radii.size(); ++i) {
        Vector3d cntr = pts.col(i);
        double rsum = radii[i];

        int naccess = 0;
        for (int p = 0; p < nprobe; ++p) {
          Vector3d probe = cntr + probes.col(p) * rsum;

          bool blocked = false;
          for (int j = 0; j < radii.size(); ++j) {
            if (j == i)
              continue;

            double dsq = (pts.col(j) - probe).squaredNorm();
            if (dsq <= rsq[j]) {
              blocked = true;
              break;
            }
          }

          if (!blocked)
            ++naccess;
        }

        sasa[i] = 4 * constants::kPi * rsq[i] * naccess / nprobe;
      }
    }

    void sr_sasa_tree(ArrayXd &sasa, const ArrayXd &radii, const ArrayXd &rsq,
                      const Matrix3Xd &pts, const Matrix3Xd &probes) {
      const int nprobe = static_cast<int>(probes.cols());

      const OCTree tree(pts);

      std::vector<int> idxs;
      std::vector<double> dsqs;
      const double cutoff = radii.maxCoeff();
      for (int i = 0; i < radii.size(); ++i) {
        const Vector3d cntr = pts.col(i);
        const double rsum = radii[i];

        tree.find_neighbors_d(cntr, rsum + cutoff, idxs, dsqs);
        auto end = std::remove(idxs.begin(), idxs.end(), i);
        ABSL_DCHECK(end == idxs.end() - 1);

        int naccess = 0;
        for (int p = 0; p < probes.cols(); ++p) {
          Vector3d probe = cntr + probes.col(p) * rsum;

          bool blocked = false;
          for (auto it = idxs.begin(); it < idxs.end() - 1; ++it) {
            int j = *it;
            double dsq = (pts.col(j) - probe).squaredNorm();
            if (dsq <= rsq[j]) {
              blocked = true;
              break;
            }
          }

          if (!blocked)
            ++naccess;
        }

        sasa[i] = 4 * constants::kPi * rsq[i] * naccess / nprobe;
      }
    }
  }  // namespace

  ArrayXd sr_sasa_impl(const Matrix3Xd &pts, const ArrayXd &radii, int nprobe,
                       SrSasaMethod method) {
    ArrayXd sasa(radii.size());

    ArrayXd rsq = radii.square();
    Matrix3Xd probes = canonical_fibonacci_lattice(nprobe);

    switch (method) {
    case internal::SrSasaMethod::kDirect:
      sr_sasa_direct(sasa, radii, rsq, pts, probes);
      break;
    case internal::SrSasaMethod::kAuto:
    case internal::SrSasaMethod::kOctree:
      sr_sasa_tree(sasa, radii, rsq, pts, probes);
      break;
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
