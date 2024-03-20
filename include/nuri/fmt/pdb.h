//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_PDB_H_
#define NURI_FMT_PDB_H_

/// @cond
#include <string>
#include <vector>

#include <absl/base/attributes.h>
/// @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
/**
 * @brief Read a single PDB string and return a molecule.
 *
 * @param pdb the PDB string to read.
 * @return A molecule. On failure, the returned molecule is empty.
 */
extern Molecule read_pdb(const std::vector<std::string> &pdb);

class PDBReader: public DefaultReaderImpl<read_pdb> {
public:
  using DefaultReaderImpl<read_pdb>::DefaultReaderImpl;

  bool getnext(std::vector<std::string> &block) override;

  bool sanitized() const override { return true; }

private:
  std::vector<std::string> header_;
  std::vector<std::string> rfooter_;
};

class PDBReaderFactory: public DefaultReaderFactoryImpl<PDBReader> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};
}  // namespace nuri

#endif /* NURI_FMT_PDB_H_ */
