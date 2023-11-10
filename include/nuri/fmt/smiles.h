//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_SMILES_H_
#define NURI_FMT_SMILES_H_

#include <string>
#include <vector>

#include <absl/base/attributes.h>

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
/**
 * @brief Read a single SMILES string and return a molecule.
 *
 * @param smi_block the SMILES block to read. Only the first string is used;
 *                  the rest are ignored. This is to support the interface
 *                  of the reader.
 * @return A molecule. On failure, the returned molecule is empty.
 */
extern Molecule read_smiles(const std::vector<std::string> &smi_block);

class SmilesReader: public DefaultReaderImpl<read_smiles> {
public:
  using DefaultReaderImpl<read_smiles>::DefaultReaderImpl;
  std::vector<std::string> next() override;
};

class SmilesReaderFactory: public DefaultReaderFactoryImpl<SmilesReader> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};
}  // namespace nuri

#endif /* NURI_FMT_SMILES_H_ */
