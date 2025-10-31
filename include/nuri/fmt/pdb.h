//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_PDB_H_
#define NURI_FMT_PDB_H_

//! @cond
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include <absl/base/attributes.h>
//! @endcond

#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/utils.h"

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

/**
 * @brief Represents a single site with position and properties.
 */
class PDBAtomSite {
public:
  PDBAtomSite(char altloc, const Vector3d &pos, double occupancy = 1.0,
              double tempfactor = 0.0) noexcept;

  char altloc() const { return altloc_; }

  const Vector3d &pos() const { return pos_; }

  double occupancy() const { return occupancy_; }

  double tempfactor() const { return tempfactor_; }

private:
  char altloc_;
  Vector3d pos_;
  double occupancy_;
  double tempfactor_;
};

/**
 * @brief Represents a PDB atom with (possibly) multiple alternate locations.
 */
class PDBAtom {
public:
  PDBAtom(std::string_view name, const Element &elem,
          std::vector<PDBAtomSite> &&sites, bool hetero) noexcept;

  std::string_view name() const { return name_; }

  const std::vector<PDBAtomSite> &sites() const { return sites_; }

  const Element &element() const { return *elem_; }

  bool hetero() const { return hetero_; }

private:
  std::string name_;
  std::vector<PDBAtomSite> sites_;
  internal::Nonnull<const Element *> elem_;
  bool hetero_;
};

/**
 * @brief Represents a PDB residue containing atoms.
 */
class PDBResidue {
public:
  PDBResidue(PDBResidueId id, std::string_view name,
             std::vector<int> &&atom_idxs) noexcept;

  PDBResidueId id() const { return id_; }

  std::string_view name() const { return name_; }

  const std::vector<int> &atom_idxs() const { return atom_idxs_; }

private:
  PDBResidueId id_;
  std::string name_;

  std::vector<int> atom_idxs_;
};

/**
 * @brief Represents a PDB chain containing residues.
 */
class PDBChain {
public:
  PDBChain(char id, std::vector<int> &&res_idxs) noexcept;

  char id() const { return id_; }

  const std::vector<int> &res_idxs() const { return res_idxs_; }

private:
  char id_;

  std::vector<int> res_idxs_;
};

/**
 * @brief Represents a PDB model containing chains.
 */
class PDBModel {
public:
  PDBModel(std::vector<PDBAtom> &&atoms, std::vector<PDBResidue> &&residues,
           std::vector<PDBChain> &&chains,
           internal::PropertyMap &&props) noexcept;

  const std::vector<PDBChain> &chains() const { return chains_; }

  const std::vector<PDBResidue> &residues() const { return residues_; }

  const std::vector<PDBAtom> &atoms() const { return atoms_; }

  const Matrix3Xd &major_conf() const { return major_conf_; }

  internal::PropertyMap &props() { return props_; }

  const internal::PropertyMap &props() const { return props_; }

private:
  std::vector<PDBAtom> atoms_;
  std::vector<PDBResidue> residues_;
  std::vector<PDBChain> chains_;

  Matrix3Xd major_conf_;

  internal::PropertyMap props_;
};

extern PDBModel read_pdb_model(const std::vector<std::string> &pdb);
}  // namespace nuri

#endif /* NURI_FMT_PDB_H_ */
