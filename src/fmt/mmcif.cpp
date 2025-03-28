//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/mmcif.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/nullability.h>
#include <absl/base/optimization.h>
#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <boost/container/flat_set.hpp>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "fmt_internal.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/fmt/cif.h"
#include "nuri/meta.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
// NOLINTBEGIN(*-identifier-naming,*-unused-function,*-unused-template)
struct ResidueId {
  int seq_id;
  std::string_view asym_id;
  std::string_view ins_code;
};

std::ostream &operator<<(std::ostream &os, const ResidueId &id) {
  return os << id.asym_id << id.seq_id << id.ins_code;
}

template <class Hash>
Hash AbslHashValue(Hash h, const ResidueId &id) {
  return Hash::combine(std::move(h), id.seq_id, id.asym_id, id.ins_code);
}

bool operator==(const ResidueId &lhs, const ResidueId &rhs) {
  return static_cast<bool>(static_cast<int>(lhs.seq_id == rhs.seq_id)
                           & static_cast<int>(lhs.asym_id == rhs.asym_id)
                           & static_cast<int>(lhs.ins_code == rhs.ins_code));
}

struct AtomId {
  ResidueId res;
  std::string_view atom_id;
};

template <class Hash>
Hash AbslHashValue(Hash h, const AtomId &id) {
  return Hash::combine(std::move(h), id.res, id.atom_id);
}

bool operator==(const AtomId &lhs, const AtomId &rhs) {
  return static_cast<bool>(static_cast<int>(lhs.res == rhs.res)
                           & static_cast<int>(lhs.atom_id == rhs.atom_id));
}
// NOLINTEND(*-identifier-naming,*-unused-function,*-unused-template)

class NullableCifColumn {
public:
  static NullableCifColumn null(const internal::CifFrame &frame) {
    return NullableCifColumn(frame);
  }

  static NullableCifColumn from_key(const internal::CifFrame &frame,
                                    std::string_view key) {
    return NullableCifColumn(frame, frame.find(key));
  }

  const internal::CifValue &operator[](int row) const {
    static const internal::CifValue placeholder {};

    if (ABSL_PREDICT_FALSE(!*this))
      return placeholder;

    return frame_->get(tbl_, col_)[row];
  }

  int table() const { return tbl_; }

  std::string_view key() const {
    if (ABSL_PREDICT_FALSE(!*this))
      return "<missing>";

    return frame_->get(tbl_, col_).key();
  }

  operator bool() const { return tbl_ >= 0; }

private:
  NURI_CLANG_ANALYZER_NOLINT explicit NullableCifColumn(
      const internal::CifFrame &frame)
      : frame_(&frame) { }

  NullableCifColumn(const internal::CifFrame &frame, std::pair<int, int> idx)
      : frame_(&frame), tbl_(idx.first), col_(idx.second) {
    ABSL_DCHECK_GE(tbl_ * col_, 0);
  }

  absl::Nonnull<const internal::CifFrame *> frame_;
  int tbl_ = -1;
  int col_;
};

template <auto converter, bool kRequired = true>
class TypedNullableColumn;

template <class T, bool (*converter)(std::string_view, absl::Nonnull<T *>),
          bool kRequired>
class TypedNullableColumn<converter, kRequired> {
public:
  TypedNullableColumn(NullableCifColumn col): col_(col) { }

  T operator[](int row) const {
    if (ABSL_PREDICT_FALSE(!col_[row])) {
      ABSL_LOG_IF(INFO, kRequired)
          << "Missing value in required column " << col_.key();
      return T();
    }

    T val;
    if (ABSL_PREDICT_FALSE(!converter(*col_[row], &val))) {
      ABSL_LOG(INFO) << "Failed to convert value " << *col_[row]
                     << " in column " << col_.key();
      val = T();
    }
    return val;
  }

  int table() const { return col_.table(); }

private:
  NullableCifColumn col_;
};

constexpr std::string_view kSeqIdKeys[2][2] {
  { "_struct_conn.ptnr1_label_seq_id", "_struct_conn.ptnr2_label_seq_id" },
  {  "_struct_conn.ptnr1_auth_seq_id",  "_struct_conn.ptnr2_auth_seq_id" },
};
constexpr std::string_view kAsymIdKeys[2][2] {
  { "_struct_conn.ptnr1_label_asym_id", "_struct_conn.ptnr2_label_asym_id" },
  {  "_struct_conn.ptnr1_auth_asym_id",  "_struct_conn.ptnr2_auth_asym_id" },
};
constexpr std::string_view kInsCodeKeys[2] {
  "_struct_conn.pdbx_ptnr1_PDB_ins_code",
  "_struct_conn.pdbx_ptnr2_PDB_ins_code",
};

class ResidueIndexer {
public:
  static ResidueIndexer atom_site(const internal::CifFrame &frame) {
    NullableCifColumn seq_id =
        NullableCifColumn::from_key(frame, "_atom_site.auth_seq_id");
    NullableCifColumn asym_id =
        NullableCifColumn::from_key(frame, "_atom_site.auth_asym_id");

    if (seq_id && asym_id) {
      return {
        seq_id,
        asym_id,
        NullableCifColumn::from_key(frame, "_atom_site.pdbx_PDB_ins_code"),
        true,
      };
    }

    ABSL_LOG(INFO)
        << "Missing auth_seq_id/auth_asym_id, falling back to "
           "label_seq_id/label_asym_id; insertion code will be ignored";
    return {
      NullableCifColumn::from_key(frame, "_atom_site.label_seq_id"),
      NullableCifColumn::from_key(frame, "_atom_site.label_asym_id"),
      NullableCifColumn::null(frame),
      false,
    };
  }

  static ResidueIndexer struct_conn(const internal::CifFrame &frame,
                                    const bool auth, const int ptnr_idx) {
    return {
      NullableCifColumn::from_key(frame, kSeqIdKeys[auth][ptnr_idx]),
      NullableCifColumn::from_key(frame, kAsymIdKeys[auth][ptnr_idx]),
      auth ? NullableCifColumn::from_key(frame, kInsCodeKeys[ptnr_idx])
           : NullableCifColumn::null(frame),
      auth,
    };
  }

  std::pair<ResidueId, bool> operator[](int row) const {
    ResidueId id;

    auto seq_id = seq_id_[row];
    if (!absl::SimpleAtoi(*seq_id, &id.seq_id)) {
      if (!seq_id->empty()) {
        ABSL_LOG(WARNING) << "Invalid residue sequence number: " << seq_id;
        return { id, false };
      }

      ABSL_LOG(INFO) << "Missing residue sequence number; assuming 0";
      id.seq_id = 0;
    }

    id.asym_id = *asym_id_[row];
    ABSL_LOG_IF(INFO, id.asym_id.empty())
        << "Missing asym_id, assuming empty chain ID";

    id.ins_code = *ins_code_[row];
    return { id, true };
  }

  std::array<int, 3> tables() const {
    return { seq_id_.table(), asym_id_.table(), ins_code_.table() };
  }

  // NOLINTNEXTLINE(*-unused-member-function)
  bool auth() const { return auth_; }

private:
  ResidueIndexer(NullableCifColumn seq_id, NullableCifColumn asym_id,
                 NullableCifColumn ins_code, bool auth)
      : seq_id_(seq_id), asym_id_(asym_id), ins_code_(ins_code), auth_(auth) { }

  NullableCifColumn seq_id_, asym_id_, ins_code_;
  bool auth_;
};

std::pair<AtomId, bool> resolve_atom_id(const ResidueIndexer &res_idx,
                                        const NullableCifColumn &atom_id,
                                        int row) {
  AtomId id;

  bool ok;
  std::tie(id.res, ok) = res_idx[row];
  if (!ok)
    return { id, false };

  id.atom_id = *atom_id[row];
  ABSL_LOG_IF(INFO, id.atom_id.empty())
      << "Missing atom ID; assuming empty atom name";

  return { id, true };
}

class AuthLabelColumn {
public:
  AuthLabelColumn(const internal::CifFrame &frame, std::string_view first,
                  std::string_view second, bool auth_if_first)
      : col_(NullableCifColumn::from_key(frame, first)), auth_(auth_if_first) {
    if (!col_) {
      col_ = NullableCifColumn::from_key(frame, second);
      auth_ = !auth_if_first;
    }
  }

  const internal::CifValue &operator[](int row) const { return col_[row]; }

  const NullableCifColumn &raw() const { return col_; }

  int table() const { return col_.table(); }

  // NOLINTNEXTLINE(*-unused-member-function)
  bool auth() const { return auth_; }

private:
  NullableCifColumn col_;
  bool auth_;
};

class CoordResolver {
public:
  explicit CoordResolver(const internal::CifFrame &frame)
      : x_(NullableCifColumn::from_key(frame, "_atom_site.Cartn_x")),
        y_(NullableCifColumn::from_key(frame, "_atom_site.Cartn_y")),
        z_(NullableCifColumn::from_key(frame, "_atom_site.Cartn_z")) { }

  // NOLINTNEXTLINE(*-unneeded-member-function)
  Vector3d operator[](int row) const { return { x_[row], y_[row], z_[row] }; }

  std::array<int, 3> tables() const {
    return { x_.table(), y_.table(), z_.table() };
  }

private:
  TypedNullableColumn<absl::SimpleAtod> x_, y_, z_;
};

int tables_nrow_min(const internal::CifFrame &frame,
                    const std::initializer_list<int> &tables) {
  int min_size = 0;

  for (int table: tables) {
    if (table < 0)
      continue;

    int rows = static_cast<int>(frame[table].rows());
    if (min_size == 0)
      min_size = rows;
    else
      min_size = nuri::min(min_size, rows);
  }

  return min_size;
}

class MmcifAtomInfo {
public:
  struct AltCmp {
    // NOLINTNEXTLINE(*-unused-member-function)
    bool operator()(const MmcifAtomInfo &lhs, const MmcifAtomInfo &rhs) const {
      return lhs.alt_id() < rhs.alt_id();
    }
  };

  MmcifAtomInfo() = default;

  MmcifAtomInfo(int row, AtomId id, std::string_view alt_id, float occupancy)
      : row_(row), occupancy_(occupancy), id_(id), alt_id_(alt_id) { }

  static MmcifAtomInfo
  from_row(const ResidueIndexer &res_idx, const AuthLabelColumn &atom_id,
           const NullableCifColumn &alt_id,
           const TypedNullableColumn<absl::SimpleAtof, false> &occupancy,
           int row) {
    auto [id, ok] = resolve_atom_id(res_idx, atom_id.raw(), row);
    if (!ok)
      return {};

    return { row, id, *alt_id[row], occupancy[row] };
  }

  int row() const { return row_; }
  operator bool() const { return row_ >= 0; }

  const AtomId &id() const { return id_; }

  std::string_view alt_id() const { return alt_id_; }

  float occupancy() const { return occupancy_; }

private:
  int row_ = -1;
  float occupancy_;

  AtomId id_;
  std::string_view alt_id_;
};

std::string as_key(std::string_view prefix, std::string_view alt_id) {
  std::string key(prefix);
  if (alt_id.empty())
    return key;

  absl::StrAppend(&key, "-", alt_id);
  return key;
}

class MmcifAtomData {
public:
  explicit MmcifAtomData(MmcifAtomInfo first, std::string_view entity_id)
      : data_ { first }, entity_id_(entity_id) { }

  bool add_info(MmcifAtomInfo info) {
    ABSL_DCHECK(static_cast<bool>(info));

    auto [it, first] = data_.insert(info);
    ABSL_LOG_IF(INFO, !first)
        << "Duplicate atom altloc '" << info.alt_id() << "'; ignoring";
    return first;
  }

  AtomData to_standard(
      const NullableCifColumn &id, const NullableCifColumn &type_symbol,
      const TypedNullableColumn<absl::SimpleAtoi<int>, false> &fchg) const {
    AtomData data;

    std::string_view symbol = *type_symbol[first().row()];
    const Element *elem = kPt.find_element(symbol);
    if (elem != nullptr) {
      data.set_element(*elem);
    } else {
      ABSL_LOG(WARNING) << "Invalid element symbol: " << symbol;
    }

    data.set_name(first().id().atom_id).set_formal_charge(fchg[first().row()]);

    data.props().reserve(2 * data_.size() + 1);
    for (const MmcifAtomInfo &info: data_) {
      data.add_prop(as_key("serial", info.alt_id()), *id[info.row()]);
      data.add_prop(as_key("occupancy", info.alt_id()),
                    absl::StrCat(info.occupancy()));
    }

    return data;
  }

  // NOLINTNEXTLINE(*-unused-member-function)
  const std::vector<MmcifAtomInfo> &data() const { return data_.sequence(); }

  const MmcifAtomInfo &first() const { return *data_.begin(); }

  // NOLINTNEXTLINE(*-unused-member-function)
  int major() const {
    auto it =
        std::max_element(data_.begin(), data_.end(),
                         [](const MmcifAtomInfo &a, const MmcifAtomInfo &b) {
                           return a.occupancy() < b.occupancy();
                         });
    return static_cast<int>(it - data_.begin());
  }

  std::string_view entity_id() const { return entity_id_; }

private:
  boost::container::flat_set<MmcifAtomInfo, MmcifAtomInfo::AltCmp,
                             std::vector<MmcifAtomInfo>>
      data_;
  std::string_view entity_id_;
};

struct MmcifResidueInfo {
  ResidueId id;
  std::string_view comp_id;
  std::vector<int> idxs;
};

class MmcifResidueData {
public:
  int prepare_add_atom(MmcifAtomInfo info, std::string_view comp_id,
                       const internal::CifValue &site_id) {
    ABSL_DCHECK(static_cast<bool>(info));

    ResidueId id = info.id().res;
    auto [it, first] = map_.insert({ id, data_.size() });

    if (first) {
      data_.push_back({ it->first, comp_id, {} });
    } else {
      std::string_view orig_resname = data_[it->second].comp_id;
      if (ABSL_PREDICT_FALSE(orig_resname != comp_id)) {
        ABSL_LOG(WARNING)
            << "Residue name mismatch: " << orig_resname << " vs " << comp_id
            << "; ignoring atom with serial number " << site_id;
        return -1;
      }
    }

    return it->second;
  }

  void add_atom_at(int res_idx, int atom_idx) {
    data_[res_idx].idxs.push_back(atom_idx);
  }

  // NOLINTNEXTLINE(*-unused-member-function)
  MmcifResidueInfo &operator[](size_t idx) { return data_[idx]; }

  // NOLINTNEXTLINE(*-unused-member-function)
  size_t size() const { return data_.size(); }

private:
  absl::flat_hash_map<ResidueId, int> map_;
  std::vector<MmcifResidueInfo> data_;
};

constexpr std::string_view kAtomIdKeys[2] {
  "_struct_conn.ptnr1_label_atom_id", "_struct_conn.ptnr2_label_atom_id"
};

class StructConnIndexer {
public:
  StructConnIndexer(const internal::CifFrame &frame,
                    const ResidueIndexer &site_res_idx, int ptnr_idx)
      : res_idx_(
            ResidueIndexer::struct_conn(frame, site_res_idx.auth(), ptnr_idx)),
        atom_id_(NullableCifColumn::from_key(frame, kAtomIdKeys[ptnr_idx])) { }

  std::pair<AtomId, bool> operator[](int row) const {
    return resolve_atom_id(res_idx_, atom_id_, row);
  }

  std::array<int, 4> tables() const {
    return { res_idx_.tables()[0], res_idx_.tables()[1], res_idx_.tables()[2],
             atom_id_.table() };
  }

private:
  ResidueIndexer res_idx_;
  NullableCifColumn atom_id_;
};

enum class StructConnType {
  kCovaleOrDisulf,
  kHydrog,
  kMetalCoord,
};

class StructConn {
public:
  static StructConn from_row(const StructConnIndexer &ptnr1,
                             const StructConnIndexer &ptnr2,
                             const NullableCifColumn &type,
                             const NullableCifColumn &order, int row) {
    auto [src, sok] = ptnr1[row];
    auto [dst, dok] = ptnr2[row];
    if (!sok || !dok)
      return {};

    StructConnType ct = StructConnType::kCovaleOrDisulf;
    std::string type_lower = absl::AsciiStrToLower(*type[row]);
    if (type_lower == "covale" || type_lower == "disulf") {
      // nothing to do
    } else if (type_lower == "hydrog") {
      ct = StructConnType::kHydrog;
    } else if (type_lower == "metalc") {
      ct = StructConnType::kMetalCoord;
    } else {
      ABSL_LOG(WARNING) << "Unknown conn_type_id: " << type[row]
                        << ", assuming covalent/disulfide bond";
    }

    constants::BondOrder bo = constants::kSingleBond;
    if (!order[row]) {
      ABSL_VLOG(1) << "Missing value_order; assuming single bond";
    } else {
      std::string order_lower = absl::AsciiStrToLower(*order[row]);
      if (order_lower == "sing") {
        // nothing to do
      } else if (order_lower == "doub") {
        bo = constants::kDoubleBond;
      } else if (order_lower == "trip") {
        bo = constants::kTripleBond;
      } else if (order_lower == "quad") {
        bo = constants::kQuadrupleBond;
      } else {
        ABSL_LOG(WARNING) << "Unknown value_order: " << order[row]
                          << ", assuming single bond";
      }
    }

    return { src, dst, ct, bo };
  }

  AtomId src() const { return src_; }

  AtomId dst() const { return dst_; }

  StructConnType type() const { return type_; }

  constants::BondOrder order() const { return order_; }

  operator bool() const { return order_ != constants::kOtherBond; }

private:
  StructConn() = default;

  StructConn(AtomId src, AtomId dst, StructConnType type,
             constants::BondOrder order)
      : src_(src), dst_(dst), type_(type), order_(order) {
    ABSL_DCHECK(order != constants::kOtherBond);
  }

  AtomId src_, dst_;
  StructConnType type_;
  constants::BondOrder order_ = constants::kOtherBond;
};

void update_struct_conn(MoleculeMutator &mut,
                        const std::vector<StructConn> &conns,
                        const absl::flat_hash_map<AtomId, int> &aid_map) {
  mut.mol().reserve_bonds(static_cast<int>(conns.size()));

  for (const StructConn &conn: conns) {
    if (conn.type() != StructConnType::kCovaleOrDisulf) {
      ABSL_LOG(INFO) << "Only covalent/disulfide bonds are yet implemented";
      continue;
    }

    auto sit = aid_map.find(conn.src());
    if (sit == aid_map.end()) {
      ABSL_LOG(WARNING) << "Unknown source atom ID " << conn.src().atom_id
                        << " in residue " << conn.src().res;
      continue;
    }

    auto dit = aid_map.find(conn.dst());
    if (dit == aid_map.end()) {
      ABSL_LOG(WARNING) << "Unknown destination atom ID " << conn.dst().atom_id
                        << " in residue " << conn.dst().res;
      continue;
    }

    auto [_, ok] =
        mut.add_bond(sit->second, dit->second, BondData(conn.order()));
    if (!ok)
      ABSL_LOG(WARNING) << "Duplicate bond between atoms " << sit->second
                        << " and " << dit->second;
  }
}

class MmcifModelData {
public:
  explicit MmcifModelData(int model_num): model_num_(model_num) { }

  void add_atom(MmcifAtomInfo info, std::string_view comp_id,
                std::string_view entity_id, const internal::CifValue &id) {
    ABSL_DCHECK(static_cast<bool>(info));

    int ri = residues_.prepare_add_atom(info, comp_id, id);
    if (ri < 0)
      return;

    auto [it, first] = aid_map_.try_emplace(info.id(), atoms_.size());
    const int ai = it->second;

    if (!first) {
      if (entity_id != atoms_[ai].entity_id()) {
        ABSL_LOG(WARNING)
            << "Entity ID mismatch: " << entity_id << " vs "
            << atoms_[ai].entity_id() << "; "
            << "ignoring atom with serial number " << info.id().atom_id;
        return;
      }

      bool new_altloc = atoms_[ai].add_info(info);
      ABSL_LOG_IF(WARNING, !new_altloc)
          << "Duplicate atom " << info.id().atom_id << " of residue "
          << info.id().res << " (" << comp_id << ") with serial number " << id
          << " and altloc '" << info.alt_id() << "'; ignoring";
      return;
    }

    atoms_.push_back(MmcifAtomData(info, entity_id));
    residues_.add_atom_at(ri, ai);
  }

  Molecule
  to_standard(std::string_view name, const NullableCifColumn &site_id,
              const NullableCifColumn &type_symbol,
              const TypedNullableColumn<absl::SimpleAtoi<int>, false> &fchg,
              const CoordResolver &coords,
              const std::vector<StructConn> &conns) && {
    Molecule mol;
    mol.name() = std::string(name);

    mol.reserve(static_cast<int>(atoms_.size()));

    {
      auto mut = mol.mutator();
      for (const auto &atom: atoms_)
        mut.add_atom(atom.to_standard(site_id, type_symbol, fchg));

      update_struct_conn(mut, conns, aid_map_);
    }

    internal::pdb_update_confs<&MmcifAtomInfo::alt_id>(
        mol, atoms_, [&](const MmcifAtomInfo &info, auto &&pos) {
          pos = coords[info.row()];
        });
    internal::pdb_update_substructs(mol, std::move(residues_), atoms_,
                                    &ResidueId::asym_id,
                                    &MmcifResidueInfo::comp_id,
                                    &ResidueId::seq_id, &ResidueId::ins_code,
                                    &MmcifAtomData::entity_id);
    return mol;
  }

  int model_num() const { return model_num_; }

private:
  MmcifResidueData residues_;

  std::vector<MmcifAtomData> atoms_;
  absl::flat_hash_map<AtomId, int> aid_map_;
  int model_num_;
};
}  // namespace

std::vector<Molecule> mmcif_load_frame(const internal::CifFrame &frame) {
  std::vector<Molecule> mols;

  ResidueIndexer res_idx = ResidueIndexer::atom_site(frame);
  CoordResolver coords(frame);

  // prefer label_{comp,atom}_id over auth_{comp,atom}_id
  // this is because auth_atom_id is missing in _struct_conn tables
  AuthLabelColumn comp_id(frame, "_atom_site.label_comp_id",
                          "_atom_site.auth_comp_id", false),
      atom_id(frame, "_atom_site.label_atom_id", "_atom_site.auth_atom_id",
              false);

  NullableCifColumn site_id =
                        NullableCifColumn::from_key(frame, "_atom_site.id"),
                    alt_id = NullableCifColumn::from_key(
                        frame, "_atom_site.label_alt_id"),
                    type_symbol = NullableCifColumn::from_key(
                        frame, "_atom_site.type_symbol"),
                    entity_id = NullableCifColumn::from_key(
                        frame, "_atom_site.label_entity_id");

  TypedNullableColumn<absl::SimpleAtof, false> occupancy =
      NullableCifColumn::from_key(frame, "_atom_site.occupancy");

  TypedNullableColumn<absl::SimpleAtoi<int>, false>
      model_num =
          NullableCifColumn::from_key(frame, "_atom_site.pdbx_PDB_model_num"),
      fchg =
          NullableCifColumn::from_key(frame, "_atom_site.pdbx_formal_charge");

  const int nsite = tables_nrow_min(
      frame,
      { res_idx.tables()[0], res_idx.tables()[1], res_idx.tables()[2],
        coords.tables()[0], coords.tables()[1], coords.tables()[2],
        comp_id.table(), atom_id.table(), alt_id.table(), type_symbol.table(),
        occupancy.table(), model_num.table(), fchg.table() });
  if (nsite == 0) {
    ABSL_LOG(WARNING) << "No atom site entries found";
    return mols;
  }

  std::vector<MmcifModelData> models;
  absl::flat_hash_map<int, int> model_map;
  for (int i = 0; i < nsite; ++i) {
    const internal::CifValue &id = site_id[i];

    auto info = MmcifAtomInfo::from_row(res_idx, atom_id, alt_id, occupancy, i);
    if (!info) {
      ABSL_LOG(WARNING)
          << "Invalid atom info; ignoring atom with serial number " << id;
      return mols;
    }

    const int mid = model_num[i];

    auto [it, first] = model_map.try_emplace(mid, models.size());
    if (first)
      models.push_back(MmcifModelData(mid));

    models[it->second].add_atom(info, *comp_id[i], *entity_id[i], id);
  }

  StructConnIndexer ptnr1(frame, res_idx, 0), ptnr2(frame, res_idx, 1);
  ABSL_LOG_IF(INFO, atom_id.auth())
      << "_struct_conn table always use label_atom_id, but auth_atom_id is "
         "used in atom_site tables. This may cause unresolved bonds";

  NullableCifColumn conn_type = NullableCifColumn::from_key(
                        frame, "_struct_conn.conn_type_id"),
                    conn_order = NullableCifColumn::from_key(
                        frame, "_struct_conn.value_order");

  const int nconn = tables_nrow_min(
      frame, { ptnr1.tables()[0], ptnr1.tables()[1], ptnr1.tables()[2],
               ptnr1.tables()[3], ptnr2.tables()[0], ptnr2.tables()[1],
               ptnr2.tables()[2], ptnr2.tables()[3], conn_type.table(),
               conn_order.table() });

  std::vector<StructConn> conns;
  conns.reserve(nconn);
  for (int i = 0; i < nconn; ++i) {
    StructConn conn =
        StructConn::from_row(ptnr1, ptnr2, conn_type, conn_order, i);
    if (conn) {
      conns.push_back(conn);
    } else {
      ABSL_LOG(INFO) << "Invalid struct_conn entry; ignoring";
    }
  }

  mols.reserve(models.size());
  for (auto &model: models) {
    const int mid = model.model_num();

    mols.emplace_back(std::move(model).to_standard(frame.name(), site_id,
                                                   type_symbol, fchg, coords,
                                                   conns))
        .add_prop("model", absl::StrCat(mid));
  }

  return mols;
}

std::vector<Molecule> mmcif_read_next_block(CifParser &parser) {
  auto block = parser.next();
  if (!block) {
    if (block.type() == internal::CifBlock::Type::kError)
      ABSL_LOG(ERROR) << "Cannot parse cif block: " << block.error_msg();

    return {};
  }

  return mmcif_load_frame(block.data());
}

bool MmcifReader::getnext(std::vector<std::string> &block) {
  block.clear();

  if (mols_.empty()) {
    mols_ = mmcif_read_next_block(parser_);
    next_ = -1;
  }

  return ++next_ < mols_.size();
}

Molecule
MmcifReader::parse(const std::vector<std::string> & /* block */) const {
  ABSL_DCHECK_GE(next_, 0);
  ABSL_DCHECK_LT(next_, mols_.size());

  return mols_[next_];
}

const bool MmcifReaderFactory::kRegistered =
    register_reader_factory<MmcifReaderFactory>({ "cif", "mmcif" });
}  // namespace nuri
