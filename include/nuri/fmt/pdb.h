//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_PDB_H_
#define NURI_FMT_PDB_H_

//! @cond
#include <ostream>
#include <string>
#include <vector>

#include <absl/base/attributes.h>
//! @endcond

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
/**
 * @brief Read a single PDB string and return a molecule.
 *
 * @param pdb the PDB string to read.
 * @return A molecule. On failure, the returned molecule is empty.
 */
extern Molecule read_pdb(const std::vector<std::string> &pdb);

class PDBReader final: public DefaultReaderImpl<read_pdb> {
public:
  using DefaultReaderImpl<read_pdb>::DefaultReaderImpl;

  bool getnext(std::vector<std::string> &block) override;

  bool bond_valid() const override { return false; }

private:
  std::vector<std::string> header_;
  std::vector<std::string> rfooter_;
  bool has_model_ = false;
};

class PDBReaderFactory: public DefaultReaderFactoryImpl<PDBReader> {
private:
  static const bool kRegistered ABSL_ATTRIBUTE_UNUSED;
};

extern int write_pdb(std::string &out, const Molecule &mol, int model = -1,
                     int conf = -1);

struct PDBResidueId {
  int res_seq;
  char chain_id;
  char ins_code;
};

extern std::ostream &operator<<(std::ostream &os, const PDBResidueId &id);

template <class Hash>
// NOLINTNEXTLINE(*-identifier-naming)
Hash AbslHashValue(Hash h, PDBResidueId id) {
  return Hash::combine(std::move(h), id.res_seq, id.chain_id, id.ins_code);
}

inline bool operator==(PDBResidueId lhs, PDBResidueId rhs) {
  return static_cast<bool>(static_cast<int>(lhs.res_seq == rhs.res_seq)
                           & static_cast<int>(lhs.chain_id == rhs.chain_id)
                           & static_cast<int>(lhs.ins_code == rhs.ins_code));
}
}  // namespace nuri

#endif /* NURI_FMT_PDB_H_ */
