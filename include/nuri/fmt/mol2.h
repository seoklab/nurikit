//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_MOL2_H_
#define NURI_FMT_MOL2_H_

//! @cond
#include <string>
#include <vector>

#include <absl/base/attributes.h>
//! @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
/**
 * @brief Read a single Mol2 string and return a molecule.
 *
 * @param mol2 the Mol2 string to read.
 * @return A molecule. On failure, the returned molecule is empty.
 */
extern Molecule read_mol2(const std::vector<std::string> &mol2);

class Mol2Reader final: public DefaultReaderImpl<read_mol2> {
public:
  using DefaultReaderImpl<read_mol2>::DefaultReaderImpl;

  bool getnext(std::vector<std::string> &block) override;

  bool bond_valid() const override { return true; }

private:
  bool read_mol_header_ = false;
};

class Mol2ReaderFactory: public DefaultReaderFactoryImpl<Mol2Reader> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};

extern bool write_mol2(std::string &out, const Molecule &mol, int conf = -1,
                       bool write_sub = true);
}  // namespace nuri

#endif /* NURI_FMT_MOL2_H_ */
