//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

namespace nuri {
void MoleculeMutator::accept() noexcept {
  if (mol_ == nullptr) {
    return;
  }

  mol_ = nullptr;
}
}  // namespace nuri
