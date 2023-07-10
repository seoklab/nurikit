//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_SMILES_H_
#define NURI_FMT_SMILES_H_

#include <istream>
#include <string_view>

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
extern Molecule read_smiles(std::string_view smiles);

class SmilesStream: public MoleculeStream {
public:
  SmilesStream() = default;
  SmilesStream(std::istream &is): is_(&is) { }

  SmilesStream(const SmilesStream &) = delete;
  SmilesStream &operator=(const SmilesStream &) = delete;
  SmilesStream(SmilesStream &&) noexcept = default;
  SmilesStream &operator=(SmilesStream &&) noexcept = default;

  ~SmilesStream() noexcept override = default;

  bool advance() override;
  Molecule current() const override { return read_smiles(line_); }

private:
  std::istream *is_;
  std::string line_;
};

class SmilesStreamFactory: public DefaultStreamFactoryImpl<SmilesStream> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};
}  // namespace nuri

#endif /* NURI_FMT_SMILES_H_ */
