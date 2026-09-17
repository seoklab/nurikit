//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_MOL2_H_
#define NURI_FMT_MOL2_H_

//! @cond
#include <memory>
#include <string>

#include <absl/base/attributes.h>
//! @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/fmt/parse_result.h"

namespace nuri {
/**
 * @brief Read a single Mol2 string and return a molecule.
 *
 * @param mol2 the Mol2 string to read.
 * @return A molecule, or the reason it could not be parsed.
 */
extern ParseResult<Molecule> read_mol2(const internal::TextBlock &mol2);

using Mol2Record = TextRecordImpl<internal::TextBlock, read_mol2>;

class Mol2Reader final: public StreamReaderBase {
public:
  using Record = Mol2Record;

  using StreamReaderBase::StreamReaderBase;

  std::unique_ptr<MoleculeRecord> make_record() const override {
    return std::make_unique<Record>();
  }

  bool bond_valid() const override { return true; }

private:
  bool fill(MoleculeRecord &record) override;

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
