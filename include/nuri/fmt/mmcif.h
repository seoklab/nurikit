//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_MMCIF_H_
#define NURI_FMT_MMCIF_H_

/// @cond
#include <vector>
/// @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/cif.h"

namespace nuri {
std::vector<Molecule> mmcif_read_next_block(CifParser &parser);
}

#endif /* NURI_FMT_MMCIF_H_ */
