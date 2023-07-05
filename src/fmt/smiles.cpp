//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/smiles.h"

#include <memory>
#include <string>

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
bool SmilesStream::advance() {
  do {
    std::getline(*is_, line_);
  } while (line_.empty() && is_->good());
  return is_->good();
}

const bool SmileStreamFactory::kRegistered =
  MoleculeStreamFactory::register_factory(
    std::make_unique<SmileStreamFactory>(), { "smi", "smiles" });

Molecule read_smiles(std::string_view smiles) {
  return {};
}
}  // namespace nuri
