//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/pdb.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <istream>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/charset.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <boost/container/flat_set.hpp>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "fmt_internal.h"
#include "nuri/core/element.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/core/property_map.h"
#include "nuri/fmt/base.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
bool fast_startswith(std::string_view str, std::string_view prefix) {
  ABSL_DCHECK(str.size() >= prefix.size());
  return std::memcmp(str.data(), prefix.data(), prefix.size()) == 0;
}

void pdb_read_footer(std::istream &is, std::string &line,
                     std::vector<std::string> &rfooter) {
  auto save = is.tellg();

  ReversedStream rs(is);
  while (rs.getline(line)
         && (line.size() < 3                   // invalid record, ignore
             || fast_startswith(line, "CON")   // CONECT
             || fast_startswith(line, "MAS")   // MASTER
             || (fast_startswith(line, "END")  // END (and not ENDMDL)
                 && !absl::StartsWith(line, "ENDMDL")))) {
    rfooter.push_back(line);
  }

  is.clear();
  is.seekg(save);
}

bool pdb_next_nomodel(std::istream &is, std::string &line,
                      std::vector<std::string> &header,
                      std::vector<std::string> &rfooter) {
  while (std::getline(is, line)) {
    header.push_back(line);

    if (absl::StartsWith(line, "MODEL")) {
      pdb_read_footer(is, line, rfooter);
      return true;
    }
  }

  return false;
}

bool pdb_next_model(std::istream &is, std::string &line,
                    std::vector<std::string> &block) {
  size_t orig = block.size();

  while (std::getline(is, line)) {
    // Stop if END/ENDMDL/MASTER/CONECT is found
    // (coordinate section is over)
    if (line.size() >= 3
        && (fast_startswith(line, "END") || fast_startswith(line, "MAS")
            || fast_startswith(line, "CON"))) {
      break;
    }

    block.push_back(line);
  }

  return block.size() != orig;
}
}  // namespace

bool PDBReader::getnext(std::vector<std::string> &block) {
  std::string line;
  line.reserve(80);

  if (header_.empty()) {
    bool has_model = pdb_next_nomodel(*is_, line, header_, rfooter_);
    if (!has_model) {
      if (header_.empty()) {
        block.clear();
        return false;
      }

      std::swap(block, header_);
      header_.clear();
      return true;
    }

    block = header_;
    // last line was "MODEL", remove from header
    header_.pop_back();
  } else {
    block = header_;
  }

  if (!pdb_next_model(*is_, line, block)) {
    block.clear();
    return false;
  }

  block.insert(block.end(), rfooter_.rbegin(), rfooter_.rend());
  return true;
}

namespace {
constexpr char safe_at(std::string_view str, size_t pos, char ifbig = ' ') {
  return pos >= str.size() ? ifbig : str[pos];
}

int safe_atoi(std::string_view str, int iferr = 0) {
  int ret;
  if (ABSL_PREDICT_FALSE(!absl::SimpleAtoi(str, &ret))) {
    ABSL_DLOG(WARNING) << "invalid integer: " << str;
    ret = iferr;
  }
  return ret;
}

float safe_atof(std::string_view str, float iferr = 0) {
  float ret;
  if (ABSL_PREDICT_FALSE(!absl::SimpleAtof(str, &ret))) {
    ABSL_DLOG(WARNING) << "invalid float: " << str;
    ret = iferr;
  }
  return ret;
}

double safe_atod(std::string_view str, double iferr = 0) {
  double ret;
  if (ABSL_PREDICT_FALSE(!absl::SimpleAtod(str, &ret))) {
    ABSL_DLOG(WARNING) << "invalid double: " << str;
    ret = iferr;
  }
  return ret;
}

constexpr std::string_view kTitleSection[] = {
  "OBSLTE", "TITLE",  "SPLIT",  "CAVEAT", "COMPND",
  "SOURCE", "KEYWDS", "EXPDTA", "NUMMDL", "MDLTYP",
  "AUTHOR", "REVDAT", "SPRSDE", "JRNL",   "REMARK",
};
// constexpr std::string_view kPrimaryStructSection[] = {  //
//   "DBREF", "SEQADV", "SEQRES", "MODRES"
// };
constexpr std::string_view kHeterogenSection[] = {  //
  // HET conflicts with HETATM if no space is present
  "HET ", "HETNAM", "HETSYN", "FORMUL"
};
constexpr std::string_view kSecStructSection[] = { "HELIX", "SHEET" };
constexpr std::string_view kConnAnnotSection[] = {  //
  "SSBOND", "LINK", "CISPEP"
};
// constexpr std::string_view kMiscSection[] = { "SITE" };
constexpr std::string_view kXtalCrdXformSection[] = {  //
  "CRYST1", "ORIGX", "SCALE", "MTRIX"
};
// constexpr std::string_view kIgnoredRecords[] = {  //
//   "TER", "MASTER", "ENDMDL", "END"
// };

template <size_t N>
constexpr int idx_of(const std::string_view (&arr)[N], std::string_view sec) {
  for (int i = 0; i < N; ++i) {
    if (arr[i] == sec) {
      return i;
    }
  }
  return -1;
}

// NOLINTNEXTLINE(*-macro-usage)
#define NURI_FIND_SECTION_IDX(arr, sec)                                        \
  constexpr int k##sec##Idx = idx_of(arr, #sec);                               \
  static_assert(k##sec##Idx >= 0, #sec " not found")

NURI_FIND_SECTION_IDX(kTitleSection, NUMMDL);
NURI_FIND_SECTION_IDX(kTitleSection, REVDAT);
NURI_FIND_SECTION_IDX(kTitleSection, JRNL);
NURI_FIND_SECTION_IDX(kTitleSection, REMARK);

#undef NURI_FIND_SECTION_IDX

// NOLINTBEGIN(*-identifier-naming,*-unused-function,*-unused-template)
struct ResidueId {
  int seqnum;
  char chain;
  char icode;
};

std::ostream &operator<<(std::ostream &os, const ResidueId &id) {
  os << id.chain << id.seqnum;
  if (id.icode != ' ')
    os << id.icode;
  return os;
}

struct AtomId {
  ResidueId res;
  std::string_view name;
};

template <class Hash>
Hash AbslHashValue(Hash h, ResidueId id) {
  return Hash::combine(std::move(h), id.seqnum, id.chain, id.icode);
}

bool operator==(ResidueId lhs, ResidueId rhs) {
  return static_cast<bool>(static_cast<int>(lhs.seqnum == rhs.seqnum)
                           & static_cast<int>(lhs.chain == rhs.chain)
                           & static_cast<int>(lhs.icode == rhs.icode));
}

template <class Hash>
Hash AbslHashValue(Hash h, const AtomId &id) {
  return Hash::combine(std::move(h), id.res, id.name);
}

bool operator==(const AtomId &lhs, const AtomId &rhs) {
  return static_cast<bool>(static_cast<int>(lhs.res == rhs.res)
                           & static_cast<int>(lhs.name == rhs.name));
}

// NOLINTEND(*-identifier-naming,*-unused-function,*-unused-template)

using Iterator = std::vector<std::string>::const_iterator;

bool is_record(Iterator it, Iterator end, std::string_view rec) {
  return it != end && absl::StartsWith(*it, rec);
}

void skip_record_common(Iterator &it, const Iterator end,
                        std::string_view rec) {
  for (; is_record(it, end, rec); ++it)
    ;
}

void read_title_section_common(Iterator &it, const Iterator end,
                               std::string_view record, std::string &buf,
                               Molecule &mol) {
  std::string rec_name = absl::AsciiStrToLower(record);

  for (; is_record(it, end, record); ++it) {
    std::string_view cont = safe_slice_strip(*it, 8, 10);
    if (cont.empty() && !buf.empty()) {
      mol.add_prop(rec_name, buf);
      buf.clear();
    }
    absl::StrAppend(&buf, safe_slice_rstrip(*it, 10, 80));
  }

  if (!buf.empty()) {
    mol.add_prop(rec_name, buf);
    buf.clear();
  }
}

void read_header_line(std::string_view line, Molecule &mol) {
  mol.add_prop("classification", safe_slice_strip(line, 10, 50));
  mol.add_prop("date", safe_slice_strip(line, 50, 59));
  mol.name() = std::string(safe_slice_strip(line, 62, 66));
}

void read_remark_record(Iterator &it, const Iterator end, std::string &buf,
                        Molecule &mol) {
  std::string_view prev_num, curr_num;

  for (; is_record(it, end, "REMARK"); ++it) {
    curr_num = safe_slice_strip(*it, 6, 10);

    if (prev_num != curr_num) {
      if (!buf.empty()) {
        buf.pop_back();
        mol.add_prop(absl::StrCat("remark-", prev_num), buf);
        buf.clear();
      }

      prev_num = curr_num;
      // As per the spec, first remark line is always empty
      continue;
    }

    absl::StrAppend(&buf, safe_slice_rstrip(*it, 11, 80), "\n");
  }

  if (!buf.empty()) {
    buf.pop_back();
    mol.add_prop(absl::StrCat("remark-", prev_num), buf);
    buf.clear();
  }
}

void read_title_section(Iterator &it, const Iterator end, std::string &buf,
                        Molecule &mol) {
  if (absl::StartsWith(*it, "HEADER"))
    read_header_line(*it++, mol);

  for (int i = 0; i < kNUMMDLIdx; ++i)
    read_title_section_common(it, end, kTitleSection[i], buf, mol);

  skip_record_common(it, end, "NUMMDL");

  for (int i = kNUMMDLIdx + 1; i < kREVDATIdx; ++i)
    read_title_section_common(it, end, kTitleSection[i], buf, mol);

  skip_record_common(it, end, "REVDAT");

  for (int i = kREVDATIdx + 1; i < kJRNLIdx; ++i)
    read_title_section_common(it, end, kTitleSection[i], buf, mol);

  skip_record_common(it, end, "JRNL");
  read_remark_record(it, end, buf, mol);
}

void read_seqres_record(Iterator &it, const Iterator end, std::string &buf,
                        std::vector<std::pair<char, std::string>> &seqres) {
  char prev_chain = ' ', curr_chain;

  for (; is_record(it, end, "SEQRES"); ++it) {
    std::string_view line = *it;
    if (line.size() < 12) {
      ABSL_LOG(INFO) << "Invalid SEQRES record: line too short (" << line.size()
                     << " < 12)";
      continue;
    }

    curr_chain = line[11];
    if (prev_chain != curr_chain) {
      if (!buf.empty()) {
        buf.pop_back();
        seqres.emplace_back(prev_chain, buf);
        buf.clear();
      }
      prev_chain = curr_chain;
    }

    absl::StrAppend(&buf, safe_slice_rstrip(line, 19, 70), " ");
  }

  if (!buf.empty()) {
    buf.pop_back();
    seqres.emplace_back(prev_chain, buf);
    buf.clear();
  }
}

struct Modres {
  ResidueId id;
  std::string_view resname;
  std::string_view stdres;
  std::string_view comment;
};

void read_modres_record(Iterator &it, const Iterator end,
                        std::vector<Modres> &modres) {
  for (; is_record(it, end, "MODRES"); ++it) {
    std::string_view line = *it;
    if (line.size() < 27) {
      ABSL_LOG(INFO) << "Invalid MODRES record: line too short (" << line.size()
                     << " < 27)";
      continue;
    }

    Modres &mod = modres.emplace_back();
    if (!absl::SimpleAtoi(slice(line, 18, 22), &mod.id.seqnum)) {
      ABSL_LOG(INFO)
          << "Invalid MODRES sequence number: " << slice(line, 18, 22);
      modres.pop_back();
      continue;
    }
    mod.id.chain = line[16];
    mod.id.icode = line[22];
    mod.resname = slice_strip(line, 12, 15);
    mod.stdres = slice_strip(line, 24, 27);
    mod.comment = safe_slice_strip(line, 29, 80);
  }
}

void read_primary_struct_section(
    Iterator &it, const Iterator end, std::string &buf,
    std::vector<std::pair<char, std::string>> &seqres,
    std::vector<Modres> &modres) {
  skip_record_common(it, end, "DBREF");
  skip_record_common(it, end, "SEQADV");

  read_seqres_record(it, end, buf, seqres);
  read_modres_record(it, end, modres);
}

void read_heterogen_section(Iterator &it, const Iterator end) {
  for (std::string_view rec: kHeterogenSection)
    skip_record_common(it, end, rec);
}

void read_sec_struct_section(Iterator &it, const Iterator end) {
  for (std::string_view rec: kSecStructSection)
    skip_record_common(it, end, rec);
}

void read_conn_annot_section(Iterator &it, const Iterator end) {
  for (std::string_view rec: kConnAnnotSection)
    skip_record_common(it, end, rec);
}

struct Site {
  std::string_view name;
  std::vector<ResidueId> residues;
};

void read_misc_section(Iterator &it, const Iterator end,
                       std::vector<Site> &sites) {
  std::string_view curr_name;
  Site site;

  for (; is_record(it, end, "SITE"); ++it) {
    std::string_view line = absl::StripTrailingAsciiWhitespace(*it);
    curr_name = safe_slice_strip(line, 11, 14);

    if (site.name != curr_name) {
      if (!site.residues.empty()) {
        sites.push_back(site);
        site.residues.clear();
      }

      site.name = curr_name;
      int numres = nonnegative(safe_atoi(safe_slice(line, 15, 17)));
      site.residues.reserve(numres);
    }

    // id, seqnum, icode
    // 22,  23-27,    27
    // 33,  34-38,    38
    // 44,  45-49,    49
    // 55,  56-60,    60
    for (int j = 22; j < 56; j += 11) {
      // seqnum is at line[j + 1:j + 5], so
      // line length >= j + 2 (need at least single character)
      if (line.size() < j + 2)
        break;

      ResidueId &id = site.residues.emplace_back();
      id.chain = line[j];
      id.seqnum = safe_atoi(slice(line, j + 1, j + 5));
      // Might be stripped out if icode is a space
      id.icode = safe_at(line, j + 5);
    }
  }

  if (!site.residues.empty())
    sites.push_back(std::move(site));
}

void read_xtal_crd_xform_section(Iterator &it, const Iterator end) {
  for (std::string_view rec: kXtalCrdXformSection)
    skip_record_common(it, end, rec);
}

std::string read_model_line(std::string_view line) {
  line.remove_prefix(5);
  return std::string(absl::StripAsciiWhitespace(line));
}

// TODO(jnooree): move to separate file and provide a more general interface
/*
struct PDBAtomInfoTemplate {
  std::string_view name;
  int atomic_number;
  int implicit_hydrogens;
  std::string_view altname {};  // NOLINT(readability-redundant-member-init)
  constants::Hybridization hyb = constants::kSP3;
  int formal_charge = 0;
  bool conjugated = false;
  bool aromatic = false;
  bool chiral = false;
};

struct PDBBondInfoTemplate {
  std::string_view src;
  std::string_view dst;
  constants::BondOrder order = constants::kSingleBond;
  bool conjugated = false;
  bool aromatic = false;
};

struct PDBBondInfo {
  int src;
  int dst;
  BondData data;
};

template <class Iterator, class Comp>
int find_index_if(const Iterator begin, const Iterator end, Comp &&comp) {
  auto it = std::find_if(begin, end, std::forward<Comp>(comp));
  return static_cast<int>(it - begin);
}

template <class Container, class Comp>
int c_find_index_if(const Container &cont, Comp &&comp) {
  return find_index_if(cont.begin(), cont.end(), std::forward<Comp>(comp));
}

template <class T>
constexpr const T &il_at(const std::initializer_list<T> &list,
                         std::size_t idx) noexcept {
  return *(list.begin() + idx);
}

AtomData from_template(const PDBAtomInfoTemplate &templ) {
  AtomData data(kPt[templ.atomic_number],
                templ.implicit_hydrogens, templ.formal_charge, templ.hyb);
  data.set_conjugated(templ.conjugated);
  data.set_aromatic(templ.aromatic);
  data.set_chiral(templ.chiral);
  return data;
}

BondData from_template(const PDBBondInfoTemplate &templ) {
  BondData data(templ.order);
  data.set_conjugated(templ.conjugated);
  data.set_aromatic(templ.aromatic);
  return data;
}

void handle_negative_hcnt(Molecule::MutableAtom atom) {
  int nbe = nonbonding_electrons(atom);
  bool pair = nbe > 1, radical = nbe > 0;
  int fchg_adjust = pair ? 1 : radical ? 0 : -1;
  atom.data().set_formal_charge(atom.data().formal_charge() + fchg_adjust);

  constants::Hybridization newhyb = internal::from_degree(atom.degree(), nbe);
  if (newhyb < constants::kSP3)
    return;

  atom.data().set_hybridization(newhyb);
  // TODO(jnooree): how to handle conjugation/aromaticity?
}

class AminoAcid {
private:
  static int
  find_name_from(const std::initializer_list<PDBAtomInfoTemplate> &atoms,
                 std::string_view name) {
    int idx =
        c_find_index_if(atoms, [name](auto &p) { return p.name == name; });
    ABSL_DCHECK(idx < atoms.size());
    return idx;
  }

public:
  AminoAcid(const std::initializer_list<PDBAtomInfoTemplate> atoms,
            const std::initializer_list<PDBBondInfoTemplate> bonds) noexcept
      : atoms_(atoms.size()), nterm_(find_name_from(atoms, "N")),
        cterm_(find_name_from(atoms, "C")), bonds_(bonds.size()) {
    for (int i = 0; i < atoms.size(); ++i)
      atoms_[i] = { il_at(atoms, i).name, from_template(il_at(atoms, i)) };

    for (const auto &atom: atoms) {
      if (atom.altname.empty())
        continue;

      alt_ids_.emplace_back(atom.altname, find_name_from(atoms, atom.name));
    }

    for (int i = 0; i < bonds.size(); ++i) {
      bonds_[i] = {
        find_name_from(atoms, il_at(bonds, i).src),
        find_name_from(atoms, il_at(bonds, i).dst),
        from_template(il_at(bonds, i)),
      };
    }
  }

  void update_atoms(Molecule &mol, const std::vector<int> &indices) const {
    for (int atom_idx: indices) {
      AtomData &data = mol[atom_idx].data();

      std::string_view name = data.get_name();
      if (name.empty())
        continue;

      auto it =
          absl::c_find_if(atoms_, [&](auto &p) { return p.first == name; });
      if (it == atoms_.end())
        continue;

      const AtomData &templ = it->second;
      data.set_hybridization(templ.hybridization())
          .set_implicit_hydrogens(templ.implicit_hydrogens())
          .set_formal_charge(templ.formal_charge())
          .reset_flags()
          .add_flags(templ.flags());
    }
  }

  std::pair<int, int> add_bonds(MoleculeMutator &mut,
                                const std::vector<int> &indices) const {
    absl::FixedArray<int> idx_map(atoms_.size(), -1);

    for (int atom_idx: indices) {
      std::string_view name = mut.mol().atom(atom_idx).data().get_name();

      int idx =
          c_find_index_if(atoms_, [name](auto &p) { return p.first == name; });
      if (idx < atoms_.size()) {
        idx_map[idx] = atom_idx;
        continue;
      }

      auto it = std::find_if(alt_ids_.begin(), alt_ids_.end(),
                             [name](auto &p) { return p.first == name; });
      if (it == alt_ids_.end()) {
        ABSL_LOG(WARNING) << "unknown atom " << name << " in amino acid";
        continue;
      }

      idx_map[it->second] = atom_idx;
    }

    for (const auto &[src, dst, data]: bonds_) {
      int mol_src = idx_map[src], mol_dst = idx_map[dst];
      if (mol_src < 0 || mol_dst < 0 || ABSL_PREDICT_FALSE(mol_src == mol_dst))
        continue;

      // Hydrogen always at destination atom, can ignore src
      if (atoms_[dst].second.atomic_number() == 1) {
        auto atom = mut.mol().atom(mol_src);

        int new_hcnt = atom.data().implicit_hydrogens() - 1;
        if (new_hcnt < 0) {
          atom.data().set_implicit_hydrogens(0);
          handle_negative_hcnt(atom);
        } else {
          atom.data().set_implicit_hydrogens(new_hcnt);
        }
      }

      auto [_, added] = mut.add_bond(mol_src, mol_dst, data);
      ABSL_LOG_IF(INFO, !added)
          << "duplicate bond between " << mol_src << " and " << mol_dst;
    }

    return { idx_map[nterm_], idx_map[cterm_] };
  }

private:
  std::vector<std::pair<std::string_view, AtomData>> atoms_;
  std::vector<std::pair<std::string_view, int>> alt_ids_;
  int nterm_;
  int cterm_;

  std::vector<PDBBondInfo> bonds_;
};

// clang-format off
const absl::flat_hash_map<std::string_view, AminoAcid> kAAData {
  { "ALA", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 3 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB1", 1, 0, "1HB", constants::kTerminal },
      { "HB2", 1, 0, "2HB", constants::kTerminal },
      { "HB3", 1, 0, "3HB", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "HB1" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "OXT", "HXT" },
    },
  } },
  { "ARG", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 2 },
      { "CD", 6, 2 },
      { "NE", 7, 1, {}, constants::kSP2, 0, true },
      { "CZ", 6, 0, {}, constants::kSP2, 0, true },
      { "NH1", 7, 2, {}, constants::kSP2, 0, true },
      { "NH2", 7, 2, {}, constants::kSP2, 1, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HG2", 1, 0, "1HG", constants::kTerminal },
      { "HG3", 1, 0, "2HG", constants::kTerminal },
      { "HD2", 1, 0, "1HD", constants::kTerminal },
      { "HD3", 1, 0, "2HD", constants::kTerminal },
      { "HE", 1, 0, {}, constants::kTerminal },
      { "HH11", 1, 0, "1HH1", constants::kTerminal },
      { "HH12", 1, 0, "2HH1", constants::kTerminal },
      { "HH21", 1, 0, "1HH2", constants::kTerminal },
      { "HH22", 1, 0, "2HH2", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD" },
      { "CG", "HG2" },
      { "CG", "HG3" },
      { "CD", "NE" },
      { "CD", "HD2" },
      { "CD", "HD3" },
      { "NE", "CZ", constants::kSingleBond, true },
      { "NE", "HE" },
      { "CZ", "NH1", constants::kSingleBond, true },
      { "CZ", "NH2", constants::kDoubleBond, true },
      { "NH1", "HH11" },
      { "NH1", "HH12" },
      { "NH2", "HH21" },
      { "NH2", "HH22" },
      { "OXT", "HXT" },
    },
  } },
  { "ASN", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 0, {}, constants::kSP2, 0, true },
      { "OD1", 8, 0, {}, constants::kTerminal, 0, true },
      { "ND2", 7, 2, {}, constants::kSP2, 0, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HD21", 1, 0, "1HD2", constants::kTerminal },
      { "HD22", 1, 0, "2HD2", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "OD1", constants::kDoubleBond, true },
      { "CG", "ND2", constants::kSingleBond, true },
      { "ND2", "HD21" },
      { "ND2", "HD22" },
      { "OXT", "HXT" },
    },
  } },
  { "ASP", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 0, {}, constants::kSP2, 0, true },
      { "OD1", 8, 0, {}, constants::kTerminal, 0, true },
      { "OD2", 8, 1, {}, constants::kSP2, 0, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "HB1", constants::kTerminal },
      { "HB3", 1, 0, "HB2", constants::kTerminal },
      { "HD2", 1, 0, {}, constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "OD1", constants::kDoubleBond, true },
      { "CG", "OD2", constants::kSingleBond, true },
      { "OD2", "HD2" },
      { "OXT", "HXT" },
    },
  } },
  { "CYS", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "SG", 16, 1 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HG", 1, 0, {}, constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "SG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "SG", "HG" },
      { "OXT", "HXT" },
    },
  } },
  { "GLN", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 2 },
      { "CD", 6, 0, {}, constants::kSP2, 0, true },
      { "OE1", 8, 0, {}, constants::kTerminal, 0, true },
      { "NE2", 7, 2, {}, constants::kSP2, 0, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HG2", 1, 0, "1HG", constants::kTerminal },
      { "HG3", 1, 0, "2HG", constants::kTerminal },
      { "HE21", 1, 0, "1HE2", constants::kTerminal },
      { "HE22", 1, 0, "2HE2", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD" },
      { "CG", "HG2" },
      { "CG", "HG3" },
      { "CD", "OE1", constants::kDoubleBond, true },
      { "CD", "NE2", constants::kSingleBond, true },
      { "NE2", "HE21" },
      { "NE2", "HE22" },
      { "OXT", "HXT" },
    },
  } },
  { "GLU", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 2 },
      { "CD", 6, 0, {}, constants::kSP2, 0, true },
      { "OE1", 8, 0, {}, constants::kTerminal, 0, true },
      { "OE2", 8, 1, {}, constants::kSP2, 0, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "HB1", constants::kTerminal },
      { "HB3", 1, 0, "HB2", constants::kTerminal },
      { "HG2", 1, 0, "HG1", constants::kTerminal },
      { "HG3", 1, 0, "HG2", constants::kTerminal },
      { "HE2", 1, 0, {}, constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD" },
      { "CG", "HG2" },
      { "CG", "HG3" },
      { "CD", "OE1", constants::kDoubleBond, true },
      { "CD", "OE2", constants::kSingleBond, true },
      { "OE2", "HE2" },
      { "OXT", "HXT" },
    },
  } },
  { "GLY", {
    {
      { "N", 7, 2 },
      { "CA", 6, 2 },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA2", 1, 0, "HA1", constants::kTerminal },
      { "HA3", 1, 0, "HA2", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "HA2" },
      { "CA", "HA3" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "OXT", "HXT" },
    },
  } },
  { "HIS", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 0, {}, constants::kSP2, 0, true, true },
      { "ND1", 7, 1, {}, constants::kSP2, 1, true, true },
      { "CD2", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CE1", 6, 1, {}, constants::kSP2, 0, true, true },
      { "NE2", 7, 1, {}, constants::kSP2, 0, true, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HD1", 1, 0, {}, constants::kTerminal },
      { "HD2", 1, 0, {}, constants::kTerminal },
      { "HE1", 1, 0, {}, constants::kTerminal },
      { "HE2", 1, 0, {}, constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "ND1", constants::kAromaticBond, true, true },
      { "CG", "CD2", constants::kAromaticBond, true, true },
      { "ND1", "CE1", constants::kAromaticBond, true, true },
      { "ND1", "HD1" },
      { "CD2", "NE2", constants::kAromaticBond, true, true },
      { "CD2", "HD2" },
      { "CE1", "NE2", constants::kAromaticBond, true, true },
      { "CE1", "HE1" },
      { "NE2", "HE2" },
      { "OXT", "HXT" },
    },
  } },
  { "ILE", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "CG1", 6, 2 },
      { "CG2", 6, 3 },
      { "CD1", 6, 3 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB", 1, 0, {}, constants::kTerminal },
      { "HG12", 1, 0, "1HG1", constants::kTerminal },
      { "HG13", 1, 0, "2HG1", constants::kTerminal },
      { "HG21", 1, 0, "1HG2", constants::kTerminal },
      { "HG22", 1, 0, "2HG2", constants::kTerminal },
      { "HG23", 1, 0, "3HG2", constants::kTerminal },
      { "HD11", 1, 0, "1HD1", constants::kTerminal },
      { "HD12", 1, 0, "2HD1", constants::kTerminal },
      { "HD13", 1, 0, "3HD1", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG1" },
      { "CB", "CG2" },
      { "CB", "HB" },
      { "CG1", "CD1" },
      { "CG1", "HG12" },
      { "CG1", "HG13" },
      { "CG2", "HG21" },
      { "CG2", "HG22" },
      { "CG2", "HG23" },
      { "CD1", "HD11" },
      { "CD1", "HD12" },
      { "CD1", "HD13" },
      { "OXT", "HXT" },
    },
  } },
  { "LEU", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 1 },
      { "CD1", 6, 3 },
      { "CD2", 6, 3 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HG", 1, 0, {}, constants::kTerminal },
      { "HD11", 1, 0, "1HD1", constants::kTerminal },
      { "HD12", 1, 0, "2HD1", constants::kTerminal },
      { "HD13", 1, 0, "3HD1", constants::kTerminal },
      { "HD21", 1, 0, "1HD2", constants::kTerminal },
      { "HD22", 1, 0, "2HD2", constants::kTerminal },
      { "HD23", 1, 0, "3HD2", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD1" },
      { "CG", "CD2" },
      { "CG", "HG" },
      { "CD1", "HD11" },
      { "CD1", "HD12" },
      { "CD1", "HD13" },
      { "CD2", "HD21" },
      { "CD2", "HD22" },
      { "CD2", "HD23" },
      { "OXT", "HXT" },
    },
  } },
  { "LYS", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 2 },
      { "CD", 6, 2 },
      { "CE", 6, 2 },
      { "NZ", 7, 3, {}, constants::kSP3, 1 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HG2", 1, 0, "1HG", constants::kTerminal },
      { "HG3", 1, 0, "2HG", constants::kTerminal },
      { "HD2", 1, 0, "1HD", constants::kTerminal },
      { "HD3", 1, 0, "2HD", constants::kTerminal },
      { "HE2", 1, 0, "1HE", constants::kTerminal },
      { "HE3", 1, 0, "2HE", constants::kTerminal },
      { "HZ1", 1, 0, "1HZ", constants::kTerminal },
      { "HZ2", 1, 0, "2HZ", constants::kTerminal },
      { "HZ3", 1, 0, "3HZ", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD" },
      { "CG", "HG2" },
      { "CG", "HG3" },
      { "CD", "CE" },
      { "CD", "HD2" },
      { "CD", "HD3" },
      { "CE", "NZ" },
      { "CE", "HE2" },
      { "CE", "HE3" },
      { "NZ", "HZ1" },
      { "NZ", "HZ2" },
      { "NZ", "HZ3" },
      { "OXT", "HXT" },
    },
  } },
  { "MET", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 2 },
      { "SD", 16, 0 },
      { "CE", 6, 3 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HG2", 1, 0, "1HG", constants::kTerminal },
      { "HG3", 1, 0, "2HG", constants::kTerminal },
      { "HE1", 1, 0, "1HE", constants::kTerminal },
      { "HE2", 1, 0, "2HE", constants::kTerminal },
      { "HE3", 1, 0, "3HE", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "SD" },
      { "CG", "HG2" },
      { "CG", "HG3" },
      { "SD", "CE" },
      { "CE", "HE1" },
      { "CE", "HE2" },
      { "CE", "HE3" },
      { "OXT", "HXT" },
    },
  } },
  { "PHE", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 0, {}, constants::kSP2, 0, true, true },
      { "CD1", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CD2", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CE1", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CE2", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CZ", 6, 1, {}, constants::kSP2, 0, true, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HD1", 1, 0, {}, constants::kTerminal },
      { "HD2", 1, 0, {}, constants::kTerminal },
      { "HE1", 1, 0, {}, constants::kTerminal },
      { "HE2", 1, 0, {}, constants::kTerminal },
      { "HZ", 1, 0, {}, constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD1", constants::kAromaticBond, true, true },
      { "CG", "CD2", constants::kAromaticBond, true, true },
      { "CD1", "CE1", constants::kAromaticBond, true, true },
      { "CD1", "HD1" },
      { "CD2", "CE2", constants::kAromaticBond, true, true },
      { "CD2", "HD2" },
      { "CE1", "CZ", constants::kAromaticBond, true, true },
      { "CE1", "HE1" },
      { "CE2", "CZ", constants::kAromaticBond, true, true },
      { "CE2", "HE2" },
      { "CZ", "HZ" },
      { "OXT", "HXT" },
    },
  } },
  { "PRO", {
    {
      { "N", 7, 1 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 2 },
      { "CD", 6, 2 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, "HT1", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HG2", 1, 0, "1HG", constants::kTerminal },
      { "HG3", 1, 0, "2HG", constants::kTerminal },
      { "HD2", 1, 0, "1HD", constants::kTerminal },
      { "HD3", 1, 0, "2HD", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "CD" },
      { "N", "H" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD" },
      { "CG", "HG2" },
      { "CG", "HG3" },
      { "CD", "HD2" },
      { "CD", "HD3" },
      { "OXT", "HXT" },
    },
  } },
  { "SER", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "OG", 8, 1 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HG", 1, 0, {}, constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "OG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "OG", "HG" },
      { "OXT", "HXT" },
    },
  } },
  { "THR", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "OG1", 8, 1 },
      { "CG2", 6, 3 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB", 1, 0, {}, constants::kTerminal },
      { "HG1", 1, 0, {}, constants::kTerminal },
      { "HG21", 1, 0, "1HG2", constants::kTerminal },
      { "HG22", 1, 0, "2HG2", constants::kTerminal },
      { "HG23", 1, 0, "3HG2", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "OG1" },
      { "CB", "CG2" },
      { "CB", "HB" },
      { "OG1", "HG1" },
      { "CG2", "HG21" },
      { "CG2", "HG22" },
      { "CG2", "HG23" },
      { "OXT", "HXT" },
    },
  } },
  { "TRP", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 0, {}, constants::kSP2, 0, true, true },
      { "CD1", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CD2", 6, 0, {}, constants::kSP2, 0, true, true },
      { "NE1", 7, 1, {}, constants::kSP2, 0, true, true },
      { "CE2", 6, 0, {}, constants::kSP2, 0, true, true },
      { "CE3", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CZ2", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CZ3", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CH2", 6, 1, {}, constants::kSP2, 0, true, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HD1", 1, 0, {}, constants::kTerminal },
      { "HE1", 1, 0, {}, constants::kTerminal },
      { "HE3", 1, 0, {}, constants::kTerminal },
      { "HZ2", 1, 0, {}, constants::kTerminal },
      { "HZ3", 1, 0, {}, constants::kTerminal },
      { "HH2", 1, 0, {}, constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD1", constants::kAromaticBond, true, true },
      { "CG", "CD2", constants::kAromaticBond, true, true },
      { "CD1", "NE1", constants::kAromaticBond, true, true },
      { "CD1", "HD1" },
      { "CD2", "CE2", constants::kAromaticBond, true, true },
      { "CD2", "CE3", constants::kAromaticBond, true, true },
      { "NE1", "CE2", constants::kAromaticBond, true, true },
      { "NE1", "HE1" },
      { "CE2", "CZ2", constants::kAromaticBond, true, true },
      { "CE3", "CZ3", constants::kAromaticBond, true, true },
      { "CE3", "HE3" },
      { "CZ2", "CH2", constants::kAromaticBond, true, true },
      { "CZ2", "HZ2" },
      { "CZ3", "CH2", constants::kAromaticBond, true, true },
      { "CZ3", "HZ3" },
      { "CH2", "HH2" },
      { "OXT", "HXT" },
    },
  } },
  { "TYR", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 2 },
      { "CG", 6, 0, {}, constants::kSP2, 0, true, true },
      { "CD1", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CD2", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CE1", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CE2", 6, 1, {}, constants::kSP2, 0, true, true },
      { "CZ", 6, 0, {}, constants::kSP2, 0, true, true },
      { "OH", 8, 1, {}, constants::kSP2, 0, true },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB2", 1, 0, "1HB", constants::kTerminal },
      { "HB3", 1, 0, "2HB", constants::kTerminal },
      { "HD1", 1, 0, {}, constants::kTerminal },
      { "HD2", 1, 0, {}, constants::kTerminal },
      { "HE1", 1, 0, {}, constants::kTerminal },
      { "HE2", 1, 0, {}, constants::kTerminal },
      { "HH", 1, 0, {}, constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG" },
      { "CB", "HB2" },
      { "CB", "HB3" },
      { "CG", "CD1", constants::kAromaticBond, true, true },
      { "CG", "CD2", constants::kAromaticBond, true, true },
      { "CD1", "CE1", constants::kAromaticBond, true, true },
      { "CD1", "HD1" },
      { "CD2", "CE2", constants::kAromaticBond, true, true },
      { "CD2", "HD2" },
      { "CE1", "CZ", constants::kAromaticBond, true, true },
      { "CE1", "HE1" },
      { "CE2", "CZ", constants::kAromaticBond, true, true },
      { "CE2", "HE2" },
      { "CZ", "OH", constants::kSingleBond, true },
      { "OH", "HH" },
      { "OXT", "HXT" },
    },
  } },
  { "VAL", {
    {
      { "N", 7, 2 },
      { "CA", 6, 1, {}, constants::kSP3, 0, false, false, true },
      { "C", 6, 0, {}, constants::kSP2, 0, true },
      { "O", 8, 0, {}, constants::kTerminal, 0, true },
      { "CB", 6, 1 },
      { "CG1", 6, 3 },
      { "CG2", 6, 3 },
      { "OXT", 8, 1, {}, constants::kSP2, 0, true },
      { "H", 1, 0, {}, constants::kTerminal },
      { "H2", 1, 0, "HN2", constants::kTerminal },
      { "HA", 1, 0, {}, constants::kTerminal },
      { "HB", 1, 0, {}, constants::kTerminal },
      { "HG11", 1, 0, "1HG1", constants::kTerminal },
      { "HG12", 1, 0, "2HG1", constants::kTerminal },
      { "HG13", 1, 0, "3HG1", constants::kTerminal },
      { "HG21", 1, 0, "1HG2", constants::kTerminal },
      { "HG22", 1, 0, "2HG2", constants::kTerminal },
      { "HG23", 1, 0, "3HG2", constants::kTerminal },
      { "HXT", 1, 0, {}, constants::kTerminal },
    },
    {
      { "N", "CA" },
      { "N", "H" },
      { "N", "H2" },
      { "CA", "C" },
      { "CA", "CB" },
      { "CA", "HA" },
      { "C", "O", constants::kDoubleBond, true },
      { "C", "OXT", constants::kSingleBond, true },
      { "CB", "CG1" },
      { "CB", "CG2" },
      { "CB", "HB" },
      { "CG1", "HG11" },
      { "CG1", "HG12" },
      { "CG1", "HG13" },
      { "CG2", "HG21" },
      { "CG2", "HG22" },
      { "CG2", "HG23" },
      { "OXT", "HXT" },
    },
  } },
};
// clang-format on
*/

std::pair<int, bool> parse_serial(std::string_view line) {
  int serial;
  line = slice(line, 6, 11);
  bool success = absl::SimpleAtoi(line, &serial);
  return std::make_pair(serial, success && serial >= 0);
}

int last_serial(const std::vector<std::string> &pdb) {
  for (auto it = pdb.rbegin(); it != pdb.rend(); ++it) {
    std::string_view line = *it;
    if (line.size() < 7)
      continue;

    if (fast_startswith(line, "ATOM") || fast_startswith(line, "HETATM")) {
      auto [serial, success] = parse_serial(line);
      if (success)
        return serial;
    }
  }

  return 1;
}

std::string_view get_altloc(std::string_view line) {
  return line.substr(16, 1);
}

std::string as_key(std::string_view prefix, std::string_view altloc) {
  std::string key(prefix);
  if (altloc == " ")
    return key;

  absl::StrAppend(&key, "-", altloc);
  return key;
}

class AtomicLine {
public:
  struct AltCmp {
    // NOLINTNEXTLINE(*-unused-member-function)
    bool operator()(const AtomicLine &lhs, const AtomicLine &rhs) const {
      return lhs.altloc() < rhs.altloc();
    }
  };

  AtomicLine(int serial, AtomId id, std::string_view line)
      : serial_(serial), line_(line), id_(id) { }

  static std::optional<AtomicLine> parse(std::string_view line, int serial) {
    // At least 47 characters (for three coordinates) required for useful data
    if (line.size() < 47) {
      ABSL_LOG(WARNING) << "Invalid ATOM/HETATM record: line too short ("
                        << line.size() << " < 47)";
      return std::nullopt;
    }

    AtomId id;
    if (!absl::SimpleAtoi(safe_slice(line, 22, 26), &id.res.seqnum)) {
      ABSL_LOG(WARNING) << "Invalid residue sequence number: "
                        << safe_slice_strip(line, 22, 26);
      return std::nullopt;
    }
    id.res.chain = line[21];
    id.res.icode = line[26];

    id.name = slice_strip(line, 11, 16);
    if (id.name.empty()) {
      ABSL_LOG(WARNING) << "Empty atom name supplied";
      return std::nullopt;
    }

    return AtomicLine(serial, id, line);
  }

  int serial() const { return serial_; }

  const AtomId &id() const { return id_; }

  std::string_view altloc() const { return get_altloc(line_); }

  std::string_view resname() const { return slice_strip(line_, 17, 20); }

  // NOLINTNEXTLINE(*-unneeded-member-function)
  void parse_coords(Eigen::Ref<Vector3d> pos) const {
    for (int i = 0; i < 3; ++i)
      pos[i] = safe_atod(line_.substr(30 + i * 8, 8));
  }

  float occupancy() const { return safe_atof(safe_slice(line_, 54, 60)); }

  std::string_view element() const { return safe_slice_strip(line_, 76, 78); }

  std::string_view line() const { return line_; }

private:
  int serial_;
  std::string_view line_;
  AtomId id_;
};

class PDBAtomData {
public:
  // NOLINTNEXTLINE(*-unused-member-function)
  explicit PDBAtomData(AtomicLine line): data_({ line }) { }

  bool add_line(AtomicLine line) {
    auto [it, first] = data_.insert(line);
    ABSL_LOG_IF(INFO, !first)
        << "Duplicate atom altloc '" << line.altloc() << "'; ignoring";
    return first;
  }

  void add_anisou(std::string_view line) {
    if (line.size() < 29)
      return;

    auto &prop = extra_.emplace_back();
    prop.first = as_key("anisou", get_altloc(line));

    std::string &anisou_data = prop.second;
    // 6 * 7 + 6 delimiters (comma) = 48
    anisou_data.reserve(48);
    for (int i = 28; i < 64; i += 7)
      absl::StrAppend(&anisou_data, safe_slice_strip(line, i, i + 7), ",");
    anisou_data.pop_back();
  }

  AtomData to_standard() {
    AtomData data;

    std::string_view elem_symb = first().element();
    const Element *element = kPt.find_element(elem_symb);
    if (element != nullptr) {
      data.set_element(*element);
    } else if (elem_symb == "D") {
      data.set_element(1);
    } else {
      // TODO(jnooree): extract element from name if symbol is invalid.
      ABSL_LOG(WARNING) << "Invalid element symbol: " << elem_symb;
    }

    internal::PropertyMap::sequence_type props;
    props.reserve(3 * data_.size() + extra_.size() + 1);
    props.emplace_back(internal::kNameKey, first().id().name);
    for (const AtomicLine &l: data_) {
      props.emplace_back(as_key("serial", l.altloc()),
                         absl::StrCat(l.serial()));
      props.emplace_back(as_key("occupancy", l.altloc()),
                         safe_slice_strip(l.line(), 54, 60));
      props.emplace_back(as_key("tempfactor", l.altloc()),
                         safe_slice_strip(l.line(), 60, 66));
    }
    props.insert(props.end(), std::make_move_iterator(extra_.begin()),
                 std::make_move_iterator(extra_.end()));

    data.props().adopt_sequence(std::move(props));

    return data;
  }

  // NOLINTNEXTLINE(*-unused-member-function)
  const std::vector<AtomicLine> &data() const { return data_.sequence(); }

  const AtomicLine &first() const { return *data_.begin(); }

  // NOLINTNEXTLINE(*-unused-member-function)
  int major() const {
    // cannot use max_element here because we don't want to parse the occupancy
    // on every comparison
    int major = 0;
    float max_occ = first().occupancy();
    for (int i = 1; i < data_.size(); ++i) {
      float occ = data_.begin()[i].occupancy();
      if (occ > max_occ) {
        max_occ = occ;
        major = i;
      }
    }
    return major;
  }

private:
  boost::container::flat_set<AtomicLine, AtomicLine::AltCmp,
                             std::vector<AtomicLine>>
      data_;
  std::vector<std::pair<std::string, std::string>> extra_;
};

struct PDBResidueInfo {
  ResidueId id;
  std::string_view resname;
  std::vector<int> idxs;
};

class PDBResidueData {
public:
  int prepare_add_atom(AtomicLine line) {
    ResidueId id = line.id().res;
    auto [it, first] = map_.insert({ id, data_.size() });

    if (first) {
      data_.push_back({ it->first, line.resname(), {} });
    } else {
      std::string_view orig_resname = data_[it->second].resname;
      if (ABSL_PREDICT_FALSE(orig_resname != line.resname())) {
        ABSL_LOG(WARNING)
            << "Residue name mismatch: " << orig_resname << " vs "
            << line.resname() << "; ignoring atom with serial number "
            << line.serial();
        return -1;
      }
    }

    return it->second;
  }

  void add_atom_at(int res_idx, int atom_idx) {
    data_[res_idx].idxs.push_back(atom_idx);
  }

  // NOLINTNEXTLINE(*-unused-member-function)
  PDBResidueInfo &operator[](size_t idx) { return data_[idx]; }

  // NOLINTNEXTLINE(*-unused-member-function)
  size_t size() const { return data_.size(); }

private:
  absl::flat_hash_map<ResidueId, int> map_;
  std::vector<PDBResidueInfo> data_;
};

void read_anisou_line(std::string_view line, const int serial,
                      std::vector<PDBAtomData> &atom_data,
                      const internal::CompactMap<int, int> &serial_to_idx) {
  const int *idx = serial_to_idx.find(serial);
  if (idx == nullptr) {
    ABSL_LOG(INFO)
        << "ANISOU record presented before ATOM/HETATM record of atom "
        << serial << "; ignoring";
    return;
  }

  atom_data[*idx].add_anisou(line);
}

/**
 * true -> can continue (including minor errors)
 * false -> parse failure, stop here
 */
bool read_atom_or_hetatom_line(std::string_view line, const int serial,
                               absl::flat_hash_map<AtomId, int> &id_map,
                               internal::CompactMap<int, int> &serial_map,
                               std::vector<PDBAtomData> &data,
                               PDBResidueData &residue_data) {
  auto maybe_atom = AtomicLine::parse(line, serial);
  if (!maybe_atom)
    return false;

  AtomicLine al = *maybe_atom;
  int res_idx = residue_data.prepare_add_atom(al);
  if (res_idx < 0)
    return true;

  auto [it, first_id] = id_map.try_emplace(al.id(), data.size());
  const int idx = it->second;

  auto [_, first_ser] = serial_map.try_emplace(serial, idx);

  if (!first_id) {
    if (ABSL_PREDICT_FALSE(!first_ser)) {
      ABSL_LOG(WARNING)
          << "Duplicate atom serial number " << serial << "; ignoring";
      return true;
    }

    bool new_altloc = data[idx].add_line(al);
    ABSL_LOG_IF(WARNING, !new_altloc)
        << "Duplicate atom " << al.id().name << " of residue " << al.id().res
        << " (" << al.resname() << ") with serial number " << serial
        << " and altloc '" << al.altloc() << "'; ignoring";
    return true;
  }

  data.emplace_back(al);
  residue_data.add_atom_at(res_idx, idx);
  return true;
}

bool read_coord_section(Iterator &it, const Iterator end,
                        std::vector<PDBAtomData> &atom_data,
                        PDBResidueData &residue_data,
                        internal::CompactMap<int, int> &serial_to_idx) {
  absl::flat_hash_map<AtomId, int> atom_id_to_idx;

  for (; it != end; ++it) {
    const std::string_view line = *it;
    if (line.size() < 7) {
      // Some record identifiers are shorter than 6 characters, but lines with
      // identifiers only have no useful data anyways.
      // Useful lines (i.e., ATOM, HETATM, ANISOU) are at least 7 characters
      // long (for the serial number).
      // TER, END records might be skipped here.
      continue;
    }

    std::string_view record = slice(line, 0, 6);
    const bool is_atom = fast_startswith(record, "ATOM"),
               is_hetatom = fast_startswith(record, "HETATM"),
               is_connect = fast_startswith(record, "CONECT"),
               is_anisou = fast_startswith(record, "ANISOU");
    if (is_connect)
      return true;
    if (!is_atom && !is_hetatom && !is_anisou)
      continue;

    auto [serial, success] = parse_serial(line);
    if (ABSL_PREDICT_FALSE(!success)) {
      std::string_view err = slice_strip(line, 6, 11);

      if (is_atom || is_hetatom) {
        ABSL_LOG(WARNING) << "invalid atom serial number: " << err;
        return false;
      }

      ABSL_LOG(INFO) << "invalid ANISOU serial number: " << err << " skipping";
      continue;
    }

    if (is_anisou) {
      read_anisou_line(line, serial, atom_data, serial_to_idx);
      continue;
    }

    bool atom_ok = read_atom_or_hetatom_line(
        line, serial, atom_id_to_idx, serial_to_idx, atom_data, residue_data);
    if (!atom_ok)
      return false;
  }

  return true;
}

void read_connect_line(std::string_view line, const int src,
                       MoleculeMutator &mut,
                       const internal::CompactMap<int, int> &serial_to_idx) {
  // 11-16, 16-21, 21-26, 26-31
  for (int i = 11; i < 27; i += 5) {
    // At least one character required for serial number
    if (line.size() <= i)
      break;

    int serial;
    if (!absl::SimpleAtoi(slice(line, i, i + 5), &serial) || serial < 0) {
      ABSL_LOG(WARNING)
          << "Invalid CONECT serial number: " << slice_strip(line, i, i + 5)
          << " the resulting molecule might be invalid";
      continue;
    }

    const int *dst = serial_to_idx.find(serial);
    if (dst == nullptr || src == *dst) {
      ABSL_LOG(INFO)
          << "invalid CONECT record for atom " << serial << "; ignoring";
      continue;
    }

    auto [_, added] = mut.add_bond(src, *dst, BondData(constants::kSingleBond));
    if (added)
      ABSL_VLOG(1) << "Added bond " << src << " -> " << *dst << " from CONECT";
  }
}

void read_connect_section(Iterator &it, const Iterator end,
                          MoleculeMutator &mut,
                          const internal::CompactMap<int, int> &serial_to_idx) {
  for (; it != end; ++it) {
    std::string_view line = *it;
    if (!absl::StartsWith(line, "CONECT"))
      continue;

    line = absl::StripTrailingAsciiWhitespace(line);
    if (line.size() < 12) {
      // Useful CONECT lines are at least 12 characters long
      // (for two serial numbers).
      continue;
    }

    auto [serial, success] = parse_serial(line);
    if (ABSL_PREDICT_FALSE(!success)) {
      std::string_view err = slice_strip(line, 6, 11);
      ABSL_LOG(WARNING) << "invalid CONECT serial number: " << err
                        << " the resulting molecule might be invalid";
      continue;
    }

    const int *idx = serial_to_idx.find(serial);
    if (idx == nullptr) {
      ABSL_LOG(INFO)
          << "invalid CONECT record for atom " << serial << "; ignoring";
      continue;
    }

    read_connect_line(line, *idx, mut, serial_to_idx);
  }
}

void remove_hbonds(MoleculeMutator &mut) {
  const Matrix3Xd &conf = mut.mol().confs()[0];

  for (auto atom: mut.mol()) {
    if (atom.data().atomic_number() != 1 || atom.degree() <= 1)
      continue;

    int argmin;
    (conf(Eigen::all, as_index(atom)).colwise() - conf.col(atom.id()))
        .colwise()
        .squaredNorm()
        .minCoeff(&argmin);

    for (int i = 0; i < atom.degree(); ++i) {
      if (i != argmin) {
        int src = atom.id(), dst = atom[i].dst().id();
        ABSL_VLOG(1)
            << "Erasing excess bond at hydrogen atom " << src << " -> " << dst;
        mut.mark_bond_erase(src, dst);
      }
    }
  }
}
}  // namespace

Molecule read_pdb(const std::vector<std::string> &pdb) {
  Molecule mol;

  if (ABSL_PREDICT_FALSE(pdb.empty()))
    return mol;

  // Order:
  // 1) Title section
  //    HEADER -> OBSLTE -> TITLE -> SPLIT -> CAVEAT -> COMPND -> SOURCE ->
  //    KEYWDS -> EXPDTA -> NUMMDL -> MDLTYP -> AUTHOR -> REVDAT -> SPRSDE ->
  //    JRNL -> REMARK
  // 2) Primary structure section
  //    DBREF -> SEQADV -> SEQRES -> MODRES
  // 3) Heterogen section
  //    HET -> HETNAM -> HETSYN -> FORMUL
  // 4) Secondary structure section
  //    HELIX -> SHEET
  // 5) Connectivity annotation section
  //    SSBOND -> LINK -> CISPEP
  // 6) Misc section
  //    SITE
  // 7) Crystallographic and coordinate transformation section
  //    CRYST1 -> ORIGXn -> SCALEn -> MTRIXn
  // 8) Coordinate section / connectivity section

  // We ignore TER, MASTER, ENDMDL, and END records.
  // Mixed order of coordinates and MODEL records are allowed; however,
  // ATOM/HETATM records must be presented before other records for the same
  // atom (e.g., ANISOU, CONECT, etc.)

  auto it = pdb.begin();
  const auto end = pdb.end();
  std::string buf;

  read_title_section(it, end, buf, mol);

  std::vector<std::pair<char, std::string>> seqres;
  std::vector<Modres> modres;
  read_primary_struct_section(it, end, buf, seqres, modres);

  read_heterogen_section(it, end);
  read_sec_struct_section(it, end);
  read_conn_annot_section(it, end);

  std::vector<Site> sites;
  read_misc_section(it, end, sites);

  read_xtal_crd_xform_section(it, end);

  if (is_record(it, end, "MODEL")) {
    mol.add_prop("model", read_model_line(*it));
    ++it;
  }

  std::vector<PDBAtomData> atom_data;
  PDBResidueData residue_data;
  internal::CompactMap<int, int> serial_to_idx(last_serial(pdb) + 1);
  bool success =
      read_coord_section(it, end, atom_data, residue_data, serial_to_idx);
  if (!success) {
    std::string_view line;
    if (it != end)
      line = *it;

    ABSL_LOG(ERROR) << "Invalid coordinate section record: " << line;
    return mol;
  }

  {
    auto mut = mol.mutator();
    for (PDBAtomData &pd: atom_data)
      mut.add_atom(pd.to_standard());

    internal::pdb_update_confs<&AtomicLine::altloc>(
        mol, atom_data,
        [](const AtomicLine &line, auto &&pos) { line.parse_coords(pos); });

    read_connect_section(it, end, mut, serial_to_idx);
    if (!mol.bond_empty())
      remove_hbonds(mut);
  }

  internal::pdb_update_substructs(
      mol, std::move(residue_data), atom_data,
      [](const ResidueId &id) { return std::string_view(&id.chain, 1); },
      &PDBResidueInfo::resname, &ResidueId::seqnum,
      [](const ResidueId &id) {
        return id.icode == ' ' ? "" : std::string_view(&id.icode, 1);
      });

  return mol;
}

const bool PDBReaderFactory::kRegistered =
    register_reader_factory<PDBReaderFactory>({ "pdb" });

namespace {
void write_atom_single(std::string &out, bool atom_record, int serial,
                       Molecule::Atom atom, ResidueId id,
                       std::string_view atmname, std::string_view resname,
                       const Vector3d &pos) {
  std::string_view record = atom_record ? "ATOM  " : "HETATM";

  absl::StrAppendFormat(
      &out,
      // clang-format off
//          1         2         3         4         5         6         7         8
// 12345678901234567890123456789012345678901234567890123456789012345678901234567890
// ATOM  NNNNN AAAALRRR CSSSSI   XXXX.XXXYYYY.YYYZZZZ.ZZZOOO.OOTTT.TT          EEGG
"%s%5d %-4s%c%-3s %c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s",
      // clang-format on
      record, serial, atmname, ' ', resname, id.chain, id.seqnum, id.icode,
      pos[0], pos[1], pos[2], 1.0, 0, atom.data().element_symbol());

  int fchg = atom.data().formal_charge();
  if (fchg != 0 && fchg < 10 && fchg > -10) {
    absl::StrAppendFormat(&out, "%d%c", std::abs(fchg), fchg < 0 ? '-' : '+');
  } else {
    ABSL_LOG_IF(INFO, fchg != 0)
        << "Formal charge " << fchg << " exceeds 2 characters; ignoring";
  }

  out.push_back('\n');
}

class PDBResolvedResidue {
public:
  PDBResolvedResidue(ResidueId id, std::string_view resname,
                     std::vector<int> &&atoms, std::vector<std::string> &&names)
      : id_(id), resname_(resname), atoms_(std::move(atoms)),
        atom_names_(std::move(names)) {
    ABSL_DCHECK_NE(id.chain, '\0');

    ABSL_LOG_IF(INFO, resname_.size() > 3)
        << "Residue name '" << resname_ << "' exceeds 3 characters; truncating";
    resname_ = resname_.substr(0, 3);
  }

  ResidueId id() const { return id_; }

  std::string_view name() const { return resname_; }

  int natoms() const { return static_cast<int>(atoms_.size()); }

  int write(std::string &out, bool atom, int serial, const Molecule &mol,
            const Matrix3Xd &conf) const {
    for (int i = 0; i < atoms_.size(); ++i, ++serial)
      write_atom_single(out, atom, serial, mol[atoms_[i]], id_, atom_names_[i],
                        resname_, conf.col(atoms_[i]));
    return serial;
  }

  int write(std::string &out, bool atom, int serial,
            const Molecule &mol) const {
    Vector3d zero = Vector3d::Zero();

    for (int i = 0; i < atoms_.size(); ++i, ++serial)
      write_atom_single(out, atom, serial, mol[atoms_[i]], id_, atom_names_[i],
                        resname_, zero);

    return serial;
  }

private:
  ResidueId id_;
  std::string_view resname_;

  std::vector<int> atoms_;
  std::vector<std::string> atom_names_;
};

std::vector<std::vector<int>> group_atoms(const Molecule &mol) {
  std::vector<std::vector<int>> groups(mol.substructures().size() + 1);

  ArrayXi atom_to_sub = ArrayXi::Zero(mol.size());
  for (int i = 0; i < mol.substructures().size(); ++i) {
    const auto &sub = mol.substructures()[i];
    if (sub.category() != SubstructCategory::kResidue)
      continue;

    for (auto atom: sub) {
      int id = atom.as_parent().id();
      if (atom_to_sub[id] > 0)
        continue;

      atom_to_sub[id] = i + 1;
    }
  }

  for (int i = 0; i < mol.size(); ++i)
    groups[atom_to_sub[i]].push_back(i);

  return groups;
}

template <class PT>
char get_sv_select0_default(PT &props, std::string_view key,
                            char notfound = ' ') {
  auto it = internal::find_key(props, key);
  if (it == props.end() || it->second.empty())
    return notfound;

  ABSL_LOG_IF(INFO, it->second.size() > 1)
      << "Value length " << it->second.size() << " for key '" << key
      << "' exceeds 1";
  return it->second[0];
}

ResidueId generate_rid_sub(const Substructure &sub) {
  ResidueId id { sub.id(), get_sv_select0_default(sub.props(), "chain"),
                 get_sv_select0_default(sub.props(), "icode") };

  if (id.seqnum <= -1000 || id.seqnum >= 10000 || absl::ascii_iscntrl(id.chain)
      || absl::ascii_iscntrl(id.icode)) {
    ABSL_LOG(WARNING)
        << "Invalid residue ID for substructure " << sub.id() << ": " << id;
    id.chain = '\0';
  }

  return id;
}

std::vector<std::string> resolve_atom_names(const Molecule &mol,
                                            const std::vector<int> &atoms,
                                            const ResidueId &rid,
                                            std::string_view resname) {
  std::vector names = make_names_unique(atoms, [&](int i) -> std::string {
    auto atom = mol.atom(atoms[i]);
    std::string_view esym = atom.data().element_symbol();

    std::string name(atom.data().get_name());
    name = name.empty() ? esym : name;

    if (name.size() > 4) {
      absl::StripAsciiWhitespace(&name);

      if (name.size() > 4) {
        ABSL_LOG(INFO)
            << "Atom name '" << name << "' exceeds 4 characters; truncating";
        name.resize(4);
      }
    }

    return name;
  });

  if (absl::c_any_of(names, [](std::string_view name) {
        bool bad = name.size() > 4;
        ABSL_LOG_IF(INFO, bad)
            << "Atom name too long: '" << name << "' (" << name.size() << ")";
        return bad;
      })) {
    ABSL_LOG(WARNING)
        << "Atom name exceeds 4 characters after deduplication for residue "
        << rid << " (" << resname << ")";
    names.clear();
  }

  for (int i = 0; i < atoms.size(); i++) {
    std::string_view esym = mol[atoms[i]].data().element_symbol();
    std::string &name = names[i];
    if (name.size() < 4 && esym.size() == 1
        && absl::StartsWithIgnoreCase(name, esym)) {
      // one-letter atom name starts at col 14
      name.insert(name.begin(), ' ');
    }
  }

  return names;
}

constexpr std::string_view kUsualChainChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           kFallbackChainChars =
                               "abcdefghijklmnopqrstuvwxyz0123456789",
                           kExtensionChainChars =
                               " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

char find_first_nonempty(std::string_view trials, absl::CharSet allowed) {
  absl::CharSet candidates = absl::CharSet(trials) & allowed;
  for (char ch: trials) {
    if (candidates.contains(ch))
      return ch;
  }
  return '\0';
}

ResidueId next_chain_residue(const std::vector<PDBResolvedResidue> &residues) {
  if (residues.empty())
    return { 1, 'A', ' ' };

  absl::CharSet current_chains;
  for (const PDBResolvedResidue &res: residues)
    current_chains = current_chains | absl::CharSet::Char(res.id().chain);

  absl::CharSet available = ~current_chains;
  for (std::string_view chain_candidates:
       { kUsualChainChars, kFallbackChainChars, kExtensionChainChars }) {
    char chain = find_first_nonempty(chain_candidates, available);
    if (chain != '\0')
      return { 1, chain, ' ' };
  }

  ABSL_LOG(WARNING) << "No single-letter chain characters available";
  return { 1, '\0', ' ' };
}

std::vector<PDBResolvedResidue> resolve_residues(const Molecule &mol) {
  std::vector<PDBResolvedResidue> residues;

  std::vector<std::vector<int>> sub_to_atoms = group_atoms(mol);

  for (int i = 0; i < mol.substructures().size(); ++i) {
    if (sub_to_atoms[i + 1].empty())
      continue;

    const auto &sub = mol.substructures()[i];
    ResidueId id = generate_rid_sub(sub);
    if (id.chain == '\0') {
      residues.clear();
      return residues;
    }

    std::vector names =
        resolve_atom_names(mol, sub_to_atoms[i + 1], id, sub.name());
    if (names.empty()) {
      residues.clear();
      return residues;
    }

    residues.push_back(
        { id, sub.name(), std::move(sub_to_atoms[i + 1]), std::move(names) });
  }

  if (!sub_to_atoms[0].empty()) {
    ResidueId id = next_chain_residue(residues);
    if (id.chain == '\0') {
      residues.clear();
      return residues;
    }

    std::vector names = resolve_atom_names(mol, sub_to_atoms[0], id, "UNK");
    if (names.empty()) {
      residues.clear();
      return residues;
    }

    residues.push_back(
        { id, "UNK", std::move(sub_to_atoms[0]), std::move(names) });
  }

  return residues;
}

bool can_write_coordinates(const Matrix3Xd &pts) {
  double min = pts.minCoeff(), max = pts.maxCoeff();
  return min >= -999.999 && max <= 9999.999;
}

constexpr std::string_view kNonHetResidues[] {
  "A",   "ALA", "ARG", "ASN", "ASP", "ASX", "C",   "CYS", "DA",
  "DC",  "DG",  "DI",  "DT",  "G",   "GLN", "GLU", "GLX", "GLY",
  "HIS", "I",   "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "PYL",
  "SEC", "SER", "T",   "THR", "TRP", "TYR", "VAL",
};

template <bool kIs3D>
int write_pdb_single_conf(std::string &out, const Molecule &mol,
                          const std::vector<PDBResolvedResidue> &residues,
                          int serial, const int model, int conf) {
  if (kIs3D && !can_write_coordinates(mol.confs()[conf])) {
    ABSL_LOG(ERROR) << "Coordinates of conformation " << conf
                    << " out of PDB range [-999.999, 9999.999]";
    return -1;
  }

  if (model >= 10000) {
    ABSL_LOG(ERROR) << "Model number " << model << " > 9999";
    return -1;
  }

  if (model >= 0)
    absl::StrAppendFormat(&out, "MODEL     %4d\n", model);

  for (const PDBResolvedResidue &res: residues) {
    if constexpr (kIs3D) {
      serial = res.write(out,
                         absl::c_binary_search(kNonHetResidues, res.name()),
                         serial, mol, mol.confs()[conf]);
    } else {
      serial = res.write(
          out, absl::c_binary_search(kNonHetResidues, res.name()), serial, mol);
    }

    ABSL_DCHECK_GE(serial, -9999);
    ABSL_DCHECK_LE(serial, 99999);
  }

  if (model >= 0)
    absl::StrAppend(&out, "ENDMDL\n");

  return model + 1;
}
}  // namespace

int write_pdb(std::string &out, const Molecule &mol, int model, int conf) {
  const std::vector residues = resolve_residues(mol);
  if (residues.empty()) {
    ABSL_LOG(ERROR) << "Failed to resolve PDB residues";
    return -1;
  }

  int max_serial = absl::c_accumulate(
      residues, 0, [](int acc, const PDBResolvedResidue &res) {
        return acc + res.natoms();
      });
  int start = nuri::min(1, 100000 - max_serial);
  if (start <= -10000) {
    ABSL_LOG(ERROR) << "Too many atoms for PDB format: " << max_serial;
    return -1;
  }

  if (!mol.is_3d())
    return write_pdb_single_conf<false>(out, mol, residues, start, model, -1);

  if (conf >= 0)
    return write_pdb_single_conf<true>(out, mol, residues, start, model, conf);

  if (model < 0 && mol.confs().size() > 1)
    model = 1;
  for (int i = 0; i < mol.confs().size(); ++i) {
    model = write_pdb_single_conf<true>(out, mol, residues, start, model, i);
    if (model < 0)
      return -1;
  }

  return model;
}
}  // namespace nuri
