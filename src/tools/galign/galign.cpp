//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/tools/galign.h"

namespace nuri {
std::vector<AlignResult>
galign(const Molecule &mol, const Matrix3Xd &seed, const GARigidMolInfo &templ,
       const bool flexible, const int max_conf, const int rigid_max_conf,
       const double rigid_min_msd, const GASamplingArgs &genetic,
       const GAMinimizeArgs &minimize) {
  GARigidMolInfo query(mol, seed, templ.vdw_scale(), templ.hetero_scale(),
                       templ.dcut());

  if (!flexible)
    return internal::rigid_galign_impl(query, templ, max_conf,
                                       templ.hetero_scale(), rigid_min_msd);

  return internal::flexible_galign_impl(query, templ, max_conf,
                                        templ.hetero_scale(), genetic, minimize,
                                        rigid_max_conf, rigid_min_msd);
}
}  // namespace nuri
