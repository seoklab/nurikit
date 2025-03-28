//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_MMCIF_H_
#define NURI_FMT_MMCIF_H_

//! @cond
#include <istream>
#include <string>
#include <vector>

#include <absl/base/attributes.h>
//! @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/fmt/cif.h"

namespace nuri {
std::vector<Molecule> mmcif_load_frame(const internal::CifFrame &frame);

std::vector<Molecule> mmcif_read_next_block(CifParser &parser);

class MmcifReader final: public MoleculeReader {
public:
  explicit MmcifReader(std::istream &is): parser_(is) { }

  bool getnext(std::vector<std::string> &block) override;

  Molecule parse(const std::vector<std::string> &block) const override;

  bool bond_valid() const override { return false; }

private:
  CifParser parser_;
  std::vector<Molecule> mols_;
  int next_ = -1;
};

class MmcifReaderFactory: public DefaultReaderFactoryImpl<MmcifReader> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};
}  // namespace nuri

#endif /* NURI_FMT_MMCIF_H_ */
