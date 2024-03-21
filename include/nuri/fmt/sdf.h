//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_SDF_H_
#define NURI_FMT_SDF_H_

/// @cond
#include <string>
#include <vector>

#include <absl/base/attributes.h>
/// @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
/**
 * @brief Read a single sdf string and return a molecule.
 *
 * @param sdf the sdf string to read.
 * @return A molecule. On failure, the returned molecule is empty.
 */
extern Molecule read_sdf(const std::vector<std::string> &sdf);

class SDFReader: public DefaultReaderImpl<read_sdf> {
public:
  using DefaultReaderImpl<read_sdf>::DefaultReaderImpl;

  bool getnext(std::vector<std::string> &block) override;

  bool sanitized() const override { return false; }
};

class SDFReaderFactory: public DefaultReaderFactoryImpl<SDFReader> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};
}  // namespace nuri

#endif /* NURI_FMT_SDF_H_ */
