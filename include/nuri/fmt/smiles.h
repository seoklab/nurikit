//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_SMILES_H_
#define NURI_FMT_SMILES_H_

//! @cond
#include <memory>
#include <string>
#include <string_view>

#include <absl/base/attributes.h>
//! @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/fmt/parse_result.h"

namespace nuri {
/**
 * @brief Read a single SMILES string and return a molecule.
 *
 * @return A molecule, or the reason it could not be parsed.
 */
extern ParseResult<Molecule> read_smiles(std::string_view smiles);

using SmilesRecord = TextRecordImpl<std::string, read_smiles>;

class SmilesReader final: public StreamReaderBase {
public:
  using Record = SmilesRecord;

  using StreamReaderBase::StreamReaderBase;

  std::unique_ptr<MoleculeRecord> make_record() const override {
    return std::make_unique<Record>();
  }

  bool bond_valid() const override { return true; }

private:
  bool fill(MoleculeRecord &record) override;
};

class SmilesReaderFactory: public DefaultReaderFactoryImpl<SmilesReader> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};

extern bool write_smiles(std::string &out, const Molecule &mol,
                         bool canonical = false);
}  // namespace nuri

#endif /* NURI_FMT_SMILES_H_ */
