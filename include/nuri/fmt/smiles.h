//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_SMILES_H_
#define NURI_FMT_SMILES_H_

//! @cond
#include <memory>
#include <string>
#include <vector>

#include <absl/base/attributes.h>
//! @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/fmt/parse_result.h"

namespace nuri {
/**
 * @brief Read a single SMILES string and return a molecule.
 *
 * @param smi_block the SMILES block to read. Only the first string is used;
 *                  the rest are ignored. This is to support the interface
 *                  of the reader.
 * @return A molecule, or the reason it could not be parsed.
 */
extern ParseResult<Molecule>
read_smiles(const std::vector<std::string> &smi_block);

using SmilesRecord = TextRecordImpl<std::vector<std::string>, read_smiles>;

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
