//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_MOL2_H_
#define NURI_FMT_MOL2_H_

#include <vector>

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

class Mol2Stream
    : public DefaultStreamImpl<std::vector<std::string>, read_mol2> {
public:
  using DefaultStreamImpl<std::vector<std::string>,
                          read_mol2>::DefaultStreamImpl;

  bool advance() override;
};

class Mol2StreamFactory: public DefaultStreamFactoryImpl<Mol2Stream> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};
}  // namespace nuri

#endif /* NURI_FMT_MOL2_H_ */
