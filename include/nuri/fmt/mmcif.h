//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_MMCIF_H_
#define NURI_FMT_MMCIF_H_

//! @cond
#include <istream>
#include <memory>

#include <absl/base/attributes.h>
//! @endcond

#include "nuri/fmt/base.h"
#include "nuri/fmt/cif.h"
#include "nuri/fmt/parse_result.h"

namespace nuri {
ParseResult<MoleculeBatch::Container>
mmcif_load_frame(const internal::CifFrame &frame);

class MmcifRecord final: public MoleculeRecord {
public:
  ParseResult<MoleculeBatch> parse() override;

private:
  friend class MmcifReader;

  ParseResult<internal::CifBlock> block_;
};

class MmcifReader final: public MoleculeReader {
public:
  using Record = MmcifRecord;

  explicit MmcifReader(std::istream &is): parser_(is) { }

  std::unique_ptr<MoleculeRecord> make_record() const override {
    return std::make_unique<Record>();
  }

  bool bond_valid() const override { return false; }

private:
  bool fill(MoleculeRecord &record) override;

  CifParser parser_;
  bool done_ = false;
};

class MmcifReaderFactory: public DefaultReaderFactoryImpl<MmcifReader> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};
}  // namespace nuri

#endif /* NURI_FMT_MMCIF_H_ */
