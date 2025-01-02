//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/mmcif.h"

#include <array>
#include <initializer_list>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/casts.h>
#include <absl/base/nullability.h>
#include <absl/base/optimization.h>
#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <Eigen/Dense>

#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/cif.h"
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
  NullableCifColumn(const internal::CifFrame &frame): frame_(&frame) { }

  NullableCifColumn(const internal::CifFrame &frame, std::pair<int, int> idx)
      : frame_(&frame), tbl_(idx.first), col_(idx.second) {
    ABSL_DCHECK_GE(tbl_ * col_, 0);
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

class ResidueIndexer {
public:
  explicit ResidueIndexer(const internal::CifFrame &frame)
      : seq_id_(NullableCifColumn::from_key(frame, "_atom_site.auth_seq_id")),
        asym_id_(NullableCifColumn::from_key(frame, "_atom_site.auth_asym_id")),
        ins_code_(frame), auth_(true) {
    if (seq_id_ && asym_id_) {
      ins_code_ =
          NullableCifColumn::from_key(frame, "_atom_site.pdbx_PDB_ins_code");
      return;
    }

    ABSL_LOG(INFO)
        << "Missing auth_seq_id/auth_asym_id, falling back to "
           "label_seq_id/label_asym_id; insertion code will be ignored";
    seq_id_ = NullableCifColumn::from_key(frame, "_atom_site.label_seq_id");
    asym_id_ = NullableCifColumn::from_key(frame, "_atom_site.label_asym_id");
    auth_ = false;
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
  NullableCifColumn seq_id_, asym_id_, ins_code_;
  bool auth_;
};

class AuthLabelColumn {
public:
  AuthLabelColumn(const internal::CifFrame &frame, std::string_view auth,
                  std::string_view label)
      : col_(NullableCifColumn::from_key(frame, auth)), auth_(true) {
    if (!col_) {
      col_ = NullableCifColumn::from_key(frame, label);
      auth_ = false;
    }
  }

  const internal::CifValue &operator[](int row) const { return col_[row]; }

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

  Vector3d operator[](int row) const { return { x_[row], y_[row], z_[row] }; }

  std::array<int, 3> tables() const {
    return { x_.table(), y_.table(), z_.table() };
  }

private:
  TypedNullableColumn<absl::SimpleAtod> x_, y_, z_;
};

int atom_site_entries(const internal::CifFrame &frame,
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
  MmcifAtomInfo() = default;

  MmcifAtomInfo(int row, AtomId id, std::string_view alt_id, float occupancy)
      : row_(row), occupancy_(occupancy), id_(id), alt_id_(alt_id) { }

  static MmcifAtomInfo
  from_row(const ResidueIndexer &res_idx, const AuthLabelColumn &atom_id,
           const NullableCifColumn &alt_id,
           const TypedNullableColumn<absl::SimpleAtof, false> &occupancy,
           int row) {
    AtomId id;

    bool ok;
    std::tie(id.res, ok) = res_idx[row];
    if (!ok)
      return {};

    id.atom_id = *atom_id[row];
    ABSL_LOG_IF(INFO, id.atom_id.empty())
        << "Missing atom ID; assuming empty atom name";

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
  explicit MmcifAtomData(MmcifAtomInfo first): data_ { first } { }

  bool add_info(MmcifAtomInfo info) {
    ABSL_DCHECK(static_cast<bool>(info));

    auto [it, first] = insert_sorted(
        data_, info, [](const MmcifAtomInfo &a, const MmcifAtomInfo &b) {
          return a.alt_id() < b.alt_id();
        });

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
      ABSL_LOG(WARNING) << "Invalid elem symbol: " << symbol;
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

  const std::vector<MmcifAtomInfo> &data() const { return data_; }

  const MmcifAtomInfo &first() const { return data_.front(); }

  int major_row() const {
    auto it =
        std::max_element(data_.begin(), data_.end(),
                         [](const MmcifAtomInfo &a, const MmcifAtomInfo &b) {
                           return a.occupancy() < b.occupancy();
                         });
    return it->row();
  }

private:
  std::vector<MmcifAtomInfo> data_;
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

  MmcifResidueInfo &operator[](size_t idx) { return data_[idx]; }

  size_t size() const { return data_.size(); }

private:
  absl::flat_hash_map<ResidueId, int> map_;
  std::vector<MmcifResidueInfo> data_;
};

void update_confs(Molecule &mol, const std::vector<MmcifAtomData> &atoms,
                  const CoordResolver &coords) {
  using internal::make_transform_iterator;

  std::vector<std::string_view> s, t;
  std::vector<std::string_view> *alt_ids = &s, *buf = &t;

  for (const MmcifAtomData &md: atoms) {
    if (md.data().size() <= 1)
      continue;

    buf->clear();
    std::set_union(
        alt_ids->cbegin(), alt_ids->cend(),
        make_transform_iterator<&MmcifAtomInfo::alt_id>(md.data().cbegin()),
        make_transform_iterator<&MmcifAtomInfo::alt_id>(md.data().cend()),
        std::back_inserter(*buf));
    std::swap(alt_ids, buf);
  }

  if (alt_ids->empty()) {
    Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd(3, atoms.size()));
    for (int i = 0; i < atoms.size(); ++i)
      conf.col(i) = coords[atoms[i].first().row()];
    return;
  }

  ABSL_DCHECK_GT(alt_ids->size(), 1);

  mol.confs().resize(alt_ids->size());
  for (int i = 0; i < alt_ids->size(); ++i)
    mol.confs()[i].resize(3, static_cast<int>(atoms.size()));

  for (int i = 0; i < atoms.size(); ++i) {
    const MmcifAtomData &md = atoms[i];

    if (md.data().size() == 1) {
      auto coord = mol.confs()[0].col(i);
      coord = coords[md.first().row()];
      for (int j = 1; j < mol.confs().size(); ++j)
        mol.confs()[j].col(i) = coord;
      continue;
    }

    if (md.data().size() == alt_ids->size()) {
      for (int j = 0; j < mol.confs().size(); ++j)
        mol.confs()[j].col(i) = coords[md.data()[j].row()];
      continue;
    }

    Vector3d major_coord = coords[md.major_row()];

    int j = 0;
    for (int k = 0; k < md.data().size(); ++j) {
      ABSL_DCHECK(j < mol.confs().size());

      std::string_view alt_id = (*alt_ids)[k];
      Eigen::Ref<Vector3d> coord = mol.confs()[j].col(i);

      if (md.data()[k].alt_id() != alt_id) {
        coord = major_coord;
        continue;
      }

      coord = coords[md.data()[k].row()];
      ++k;
    }

    for (; j < mol.confs().size(); ++j)
      mol.confs()[j].col(i) = major_coord;
  }
}

constexpr int kChainIdx = 0;

void update_substructs(Molecule &mol, MmcifResidueData &&residues,
                       const std::vector<MmcifAtomData> &atoms) {
  auto &subs = mol.substructures();

  std::vector<std::pair<std::string_view, std::vector<int>>> chains;

  for (int i = 0; i < atoms.size(); ++i) {
    ResidueId id = atoms[i].first().id().res;

    auto [cit, _] = insert_sorted(chains, { id.asym_id, {} },
                                  [](const auto &a, const auto &b) {
                                    return a.first < b.first;
                                  });
    cit->second.push_back(i);
  }

  subs.reserve(residues.size() + chains.size());
  subs.resize(residues.size(), mol.substructure(SubstructCategory::kResidue));
  subs.resize(residues.size() + chains.size(),
              mol.substructure(SubstructCategory::kChain));

  auto sit = subs.begin();
  for (int i = 0; i < residues.size(); ++i, ++sit) {
    auto &data = residues[i];

    sit->update(std::move(data.idxs), {});
    sit->name() = std::string(data.comp_id);
    sit->set_id(data.id.seq_id);
    sit->add_prop("chain", data.id.asym_id);
    if (!data.id.ins_code.empty())
      sit->add_prop("icode", data.id.ins_code);

    ABSL_DCHECK(sit->props()[kChainIdx].first == "chain");
  }

  for (int i = 0; i < chains.size(); ++i, ++sit) {
    auto &[ch, idxs] = chains[i];
    sit->update(std::move(idxs), {});
    sit->name() = std::string(ch);
    sit->set_id(i);
  }
}

class MmcifModelData {
public:
  void add_atom(MmcifAtomInfo info, std::string_view comp_id,
                const internal::CifValue &id) {
    ABSL_DCHECK(static_cast<bool>(info));

    int ri = residues_.prepare_add_atom(info, comp_id, id);
    if (ri < 0)
      return;

    auto [it, first] = aid_map_.try_emplace(info.id(), atoms_.size());
    const int ai = it->second;

    if (!first) {
      bool new_altloc = atoms_[ai].add_info(info);
      ABSL_LOG_IF(WARNING, !new_altloc)
          << "Duplicate atom " << info.id().atom_id << " of residue "
          << info.id().res << " (" << comp_id << ") with serial number " << id
          << " and altloc '" << info.alt_id() << "'; ignoring";
      return;
    }

    atoms_.push_back(MmcifAtomData(info));
    residues_.add_atom_at(ri, ai);
  }

  Molecule
  to_standard(std::string_view name, const NullableCifColumn &site_id,
              const NullableCifColumn &type_symbol,
              const TypedNullableColumn<absl::SimpleAtoi<int>, false> &fchg,
              const CoordResolver &coords) && {
    Molecule mol;
    mol.name() = std::string(name);

    mol.reserve(static_cast<int>(atoms_.size()));

    {
      auto mut = mol.mutator();
      for (const auto &atom: atoms_)
        mut.add_atom(atom.to_standard(site_id, type_symbol, fchg));
    }

    update_confs(mol, atoms_, coords);
    update_substructs(mol, std::move(residues_), atoms_);

    return mol;
  }

private:
  MmcifResidueData residues_;

  std::vector<MmcifAtomData> atoms_;
  absl::flat_hash_map<AtomId, int> aid_map_;
};
}  // namespace

std::vector<Molecule> mmcif_read_next_block(CifParser &parser) {
  std::vector<Molecule> mols;

  auto block = parser.next();
  if (!block) {
    if (block.type() == internal::CifBlock::Type::kError)
      ABSL_LOG(ERROR) << "Cannot parse cif block: " << block.error_msg();

    return mols;
  }

  ResidueIndexer res_idx(block.data());
  CoordResolver coords(block.data());

  AuthLabelColumn comp_id(block.data(), "_atom_site.auth_comp_id",
                          "_atom_site.label_comp_id"),
      atom_id(block.data(), "_atom_site.auth_atom_id",
              "_atom_site.label_atom_id");

  NullableCifColumn site_id = NullableCifColumn::from_key(block.data(),
                                                          "_atom_site.id"),
                    alt_id = NullableCifColumn::from_key(
                        block.data(), "_atom_site.label_alt_id"),
                    type_symbol = NullableCifColumn::from_key(
                        block.data(), "_atom_site.type_symbol");

  TypedNullableColumn<absl::SimpleAtof, false> occupancy =
      NullableCifColumn::from_key(block.data(), "_atom_site.occupancy");

  TypedNullableColumn<absl::SimpleAtoi<int>, false>
      model_num = NullableCifColumn::from_key(block.data(),
                                              "_atom_site.pdbx_PDB_model_num"),
      fchg = NullableCifColumn::from_key(block.data(),
                                         "_atom_site.pdbx_formal_charge");

  const int rows = atom_site_entries(
      block.data(),
      { res_idx.tables()[0], res_idx.tables()[1], res_idx.tables()[2],
        coords.tables()[0], coords.tables()[1], coords.tables()[2],
        comp_id.table(), atom_id.table(), alt_id.table(), type_symbol.table(),
        occupancy.table(), model_num.table(), fchg.table() });
  if (rows == 0) {
    ABSL_LOG(WARNING) << "No atom site entries found";
    return mols;
  }

  absl::flat_hash_map<int, MmcifModelData> models;

  for (int i = 0; i < rows; ++i) {
    const internal::CifValue &id = site_id[i];

    auto info = MmcifAtomInfo::from_row(res_idx, atom_id, alt_id, occupancy, i);
    if (!info) {
      ABSL_LOG(WARNING)
          << "Invalid atom info; ignoring atom with serial number " << id;
      return mols;
    }

    models[model_num[i]].add_atom(info, *comp_id[i], id);
  }

  mols.reserve(models.size());
  for (auto &model: models) {
    mols.emplace_back(
            std::move(model.second)
                .to_standard(block.name(), site_id, type_symbol, fchg, coords))
        .add_prop("model", absl::StrCat(model.first));
  }

  return mols;
}
}  // namespace nuri
