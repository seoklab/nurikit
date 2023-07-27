//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_SMILES_H_
#define NURI_FMT_SMILES_H_

#include <string>

#include <absl/base/attributes.h>

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
/**
 * @brief Read a single SMILES string and return a molecule.
 *
 * @param smiles the SMILES string to read.
 * @return A molecule. On failure, the returned molecule is empty.
 */
extern Molecule read_smiles(const std::string &smiles);

class SmilesStream: public DefaultStreamImpl<std::string, read_smiles> {
public:
  using DefaultStreamImpl<std::string, read_smiles>::DefaultStreamImpl;
  bool advance() override;
};

class SmilesStreamFactory: public DefaultStreamFactoryImpl<SmilesStream> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};
}  // namespace nuri

#endif /* NURI_FMT_SMILES_H_ */
