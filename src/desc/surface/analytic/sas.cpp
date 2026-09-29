//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>

#include "nuri/eigen_config.h"
#include "nuri/desc/surface.h"

namespace nuri {
namespace internal {
  namespace {
    std::vector<SasCircle> circles(const SaPrep &sa) {
      const int n_circ = sa.g.offset(sa.n_enum);
      std::vector<SasCircle> result(n_circ);

      for (int i = 0; i < sa.n_enum; ++i) {
        double ri = sa.sar[i];
        Vector3d pi = sa.pts.col(i);

        for (auto it = sa.g.begin(i), ei = sa.g.end(i); it < ei; ++it) {
          const int j = *it, q = sa.g.eid(it);
          const double rj = sa.sar[j];
          const double d = sa.d[q];

          Vector3d axis = (sa.pts.col(j) - pi) / d;
          double a = (d * d + ri * ri - rj * rj) / (2 * d);
          ABSL_DCHECK_GE(ri * ri - a * a, 0);
          double rl = std::sqrt(ri * ri - a * a);
          Vector3d cntr = pi + a * axis;

          result[q] = { axis, cntr, a, rl, i, j };
        }
      }

      return result;
    }

    int icirc(int tag) {
      return tag >> 1;
    }

    int iside(int tag) {
      return tag & 1;
    }

    std::pair<SasCaps, ArrayXi> cap_rows(const SaPrep &sa,
                                         const std::vector<SasCircle> &circ) {
      const int n_enum = sa.n_enum, n_circ = static_cast<int>(circ.size());

      ArrayXi key(2L * sa.g.m());
      for (int q = 0; q < n_circ; ++q) {
        key[2L * q] = circ[q].i;
        // n_enum = drop bucket
        key[2L * q + 1] = nuri::min(circ[q].j, n_enum);
      }

      ArrayXi tags(2L * n_circ), off(n_enum + 1);
      argsort_bucket(tags, off, key.head(2L * n_circ), [&](int p) {
        // side 1 first, then side 0
        return p < n_circ ? 2 * p + 1 : 2 * (p - n_circ);
      });
      const int m = off[n_enum];
      tags.conservativeResize(m);

      key.setConstant(-1);
      for (int p = 0; p < m; ++p)
        key[tags[p]] = p;

      SasCaps caps { CSR(std::move(tags), std::move(off)), Matrix3Xd(3, m),
                     ArrayXd(m), ArrayXd(m) };
      for (int i = 0; i < n_enum; ++i) {
        const double rs = sa.sar[i];
        for (auto it = caps.h.begin(i), ei = caps.h.end(i); it < ei; ++it) {
          const int p = caps.h.eid(it), q = icirc(*it), side = iside(*it);
          const SasCircle &c = circ[q];
          const double d = sa.d[q];

          caps.axis.col(p) = (1 - 2 * side) * c.axis;
          caps.cosa[p] = (c.a + (d - 2 * c.a) * side) / rs;
          caps.sina[p] = c.rl / rs;
        }
      }
      return { std::move(caps), std::move(key) };
    }
  }  // namespace

  SasGeometry build_sas(const SaPrep &sa) {
    std::vector<SasCircle> circ = circles(sa);
    auto [caps, slot_of] = cap_rows(sa, circ);

    return SasGeometry {};
  }
}  // namespace internal
}  // namespace nuri
