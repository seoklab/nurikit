//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_SDF_H_
#define NURI_FMT_SDF_H_

/// @cond
#include <string>
#include <vector>
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

class SDFReaderFactory: public DefaultReaderFactoryImpl<SDFReader> { };

enum class SDFVersion {
  kAutomatic,
  kV2000,
  kV3000,
};

/**
 * @brief Write a molecule to an SDF stream.
 *
 * @param out The string to write to.
 * @param mol The molecule to write.
 * @param conf The index of the conformation to write. If negative, all
 *             conformers are written in separate blocks.
 * @return Whether the write was successful.
 */
extern bool write_sdf(std::string &out, const Molecule &mol, int conf = -1,
                      SDFVersion ver = SDFVersion::kAutomatic);
}  // namespace nuri

#endif /* NURI_FMT_SDF_H_ */
