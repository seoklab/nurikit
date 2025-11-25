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
namespace {
  void sr_sasa_direct(ArrayXd &sasa, const Molecule &mol, const Matrix3Xd &conf,
                      const Matrix3Xd &probes, const double rprobe) {
    const int nprobe = static_cast<int>(probes.cols());

    for (int i = 0; i < mol.size(); ++i) {
      Vector3d cntr = conf.col(i);
      double rsum = mol[i].data().element().vdw_radius() + rprobe;

      int naccess = 0;
      for (int j = 0; j < nprobe; ++j) {
        Vector3d pt = cntr + probes.col(j) * rsum;

        bool blocked = false;
        for (int k = 0; k < mol.size(); ++k) {
          if (k == i)
            continue;

          double rcut = mol[k].data().element().vdw_radius() + rprobe;
          double dsq = (conf.col(k) - pt).squaredNorm();
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

  void sr_sasa_tree(ArrayXd &sasa, const Molecule &mol, const Matrix3Xd &conf,
                    const Matrix3Xd &probes, const double rprobe) {
    const int nprobe = static_cast<int>(probes.cols());

    const OCTree at(conf);

    Matrix3Xd pbuf(probes);
    OCTree pt(pbuf);
    const Vector3d orig_max = pt.max(), orig_len = pt.len();

    ArrayXd rcut(mol.size());
    for (int i = 0; i < mol.size(); ++i)
      rcut[i] = mol[i].data().element().vdw_radius() + rprobe;
    const double cutoff = rcut.maxCoeff();

    std::vector<std::vector<int>> idxs(nprobe);
    for (int i = 0; i < mol.size(); ++i) {
      const Vector3d cntr = conf.col(i);
      const double rsum = rcut[i];

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
          if (dsq < rcut[k] * rcut[k]) {
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

ArrayXd shrake_rupley_sasa(const Molecule &mol, const Matrix3Xd &conf,
                           const int nprobe, const double rprobe,
                           internal::SrSasaMethod method) {
  ArrayXd sasa(mol.size());
  Matrix3Xd probes = canonical_fibonacci_lattice(nprobe);

  if (method == internal::SrSasaMethod::kAuto) {
    method = (mol.size() < 100 && nprobe < 100)
                 ? internal::SrSasaMethod::kDirect
                 : internal::SrSasaMethod::kOctree;
  }

  switch (method) {  // NOLINT(clang-diagnostic-switch-enum)
  case internal::SrSasaMethod::kDirect:
    sr_sasa_direct(sasa, mol, conf, probes, rprobe);
    break;
  case internal::SrSasaMethod::kOctree:
    sr_sasa_tree(sasa, mol, conf, probes, rprobe);
    break;
  default:
    ABSL_UNREACHABLE();
  }

  return sasa;
}
}  // namespace nuri
