//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/mol2.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>
#include <boost/container/container_fwd.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/fusion/include/std_tuple.hpp>
#include <boost/optional.hpp>
#include <boost/spirit/home/x3.hpp>

#include "nuri/eigen_config.h"
#include "fmt_internal.h"
#include "nuri/algo/guess.h"
#include "nuri/core/element.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/core/property_map.h"
#include "nuri/fmt/base.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
constexpr std::string_view kCommentIndicator = "****";

bool advance_header_unread(std::istream &is, std::vector<std::string> &block,
                           bool &read_mol_header) {
  bool first = true;

  std::string line;
  while (std::getline(is, line)) {
    if (absl::StartsWith(line, "#")) {
      continue;
    }

    if (absl::StartsWith(line, "@<TRIPOS>MOLECULE")) {
      if (first) {
        first = false;
      } else {
        read_mol_header = true;
        break;
      }
    }

    block.push_back(std::move(line));
  }

  return !block.empty();
}

void advance_header_read(std::istream &is, std::vector<std::string> &block,
                         bool &read_mol_header) {
  std::string line;

  block.push_back("@<TRIPOS>MOLECULE");

  while (std::getline(is, line)) {
    if (absl::StartsWith(line, "@<TRIPOS>MOLECULE")) {
      return;
    }

    if (absl::StartsWith(line, "#")) {
      continue;
    }

    block.push_back(std::move(line));
  }

  read_mol_header = false;
}
}  // namespace

bool Mol2Reader::getnext(std::vector<std::string> &block) {
  block.clear();

  if (read_mol_header_) {
    advance_header_read(*is_, block, read_mol_header_);
    return true;
  }

  return advance_header_unread(*is_, block, read_mol_header_);
}

const bool Mol2ReaderFactory::kRegistered =
    register_reader_factory<Mol2ReaderFactory>({ "mol2" });

namespace {
namespace x3 = boost::spirit::x3;

using Iter = std::vector<std::string>::const_iterator;

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
constexpr auto mol_nums_line = *x3::omit[x3::blank] >> x3::uint_
                               >> -(+x3::omit[x3::blank] >> x3::uint_)
                               >> x3::omit[x3::space | x3::eoi];
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

bool mol2_block_end(const Iter it, const Iter end) {
  return it == end || absl::StartsWith(*it, "@");
}

void parse_mol_block(Molecule &mol, Iter &it, const Iter end) {
  if (mol2_block_end(++it, end))
    return;

  mol.name() = *it == kCommentIndicator ? "" : *it;
  if (mol2_block_end(++it, end))
    return;

  auto lit = it->begin();
  std::pair<unsigned int, boost::optional<unsigned int>> nums;
  bool parser_ok = x3::parse(lit, it->end(), parser::mol_nums_line, nums);
  if (parser_ok) {
    mol.reserve(static_cast<int>(nums.first));
    mol.reserve_bonds(static_cast<int>(nums.second.value_or(0)));
  } else {
    ABSL_LOG(WARNING) << "Failed to parse mol block line; this file might be "
                         "incompatible with future versions of NuriKit";
    ABSL_LOG(INFO) << "The line is: " << *it;
  }

  for (; !mol2_block_end(++it, end);)
    ;
}

void atom_data_from_subtype(AtomData &data, int i, std::string_view subtype,
                            std::vector<int> &ccat) {
  if (subtype == "1") {
    data.set_hybridization(constants::kSP);
  } else if (subtype == "2" || subtype == "pl3") {
    data.set_hybridization(constants::kSP2);
  } else if (subtype == "3" || absl::EqualsIgnoreCase(subtype, "o")
             || absl::EqualsIgnoreCase(subtype, "o2") || subtype == "th") {
    data.set_hybridization(constants::kSP3);
  } else if (subtype == "4") {
    // Quaternary ammonium salt
    data.set_hybridization(constants::kSP3);
    data.set_formal_charge(1);
  } else if (subtype == "ar") {
    data.set_hybridization(constants::kSP2);
    data.set_aromatic(true);
    data.set_conjugated(true);
  } else if (subtype == "cat") {
    ccat.push_back(i);
    data.set_hybridization(constants::kSP2);
    data.set_formal_charge(1);
  } else if (subtype == "am") {
    data.set_hybridization(constants::kSP2);
    data.set_conjugated(true);
  } else if (subtype == "co2") {
    data.set_hybridization(constants::kTerminal);
  } else if (subtype == "oh") {
    data.set_hybridization(constants::kSP3D2);
  } else {
    ABSL_LOG(WARNING) << "Unimplemented atom subtype: " << subtype;
  }
}

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
constexpr auto atom_line = *x3::omit[x3::blank]         //
                           >> uint_trailing_blanks      //
                           >> nonblank_trailing_blanks  //
                           >> x3::repeat(3)[double_trailing_blanks]
                           >> +x3::alpha >> -('.' >> +x3::alnum)  //
                           >> -(+x3::omit[x3::blank]              //
                                >> uint_trailing_blanks           //
                                >> -(nonblank_trailing_blanks     //
                                     >> -x3::double_))
                           >> x3::omit[+x3::space | x3::eoi];
using AtomLine = std::tuple<
    unsigned int, std::string, absl::InlinedVector<double, 3>, std::string,
    boost::optional<std::string>,
    boost::optional<std::pair<
        unsigned int,
        boost::optional<std::pair<std::string, boost::optional<double>>>>>>;
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

void process_optional_attrs(
    Molecule::MutableAtom atom,
    absl::flat_hash_map<unsigned int, std::pair<std::vector<int>, std::string>>
        &substructs,
    boost::optional<std::pair<
        unsigned int,
        boost::optional<std::pair<std::string, boost::optional<double>>>>>
        &attrs) {
  if (!attrs) {
    return;
  }

  std::pair<std::vector<int>, std::string> &substruct =
      substructs[attrs->first];
  substruct.first.push_back(atom.id());
  if (!attrs->second) {
    return;
  }

  substruct.second = std::move(attrs->second->first);
  if (!attrs->second->second) {
    return;
  }

  atom.data().set_partial_charge(*attrs->second->second);
}

std::pair<bool, bool> parse_atom_block(
    MoleculeMutator &mutator, std::vector<Vector3d> &pos,
    std::vector<int> &ccat,
    absl::flat_hash_map<unsigned int, std::pair<std::vector<int>, std::string>>
        &substructs,
    Iter &it, const Iter end) {
  parser::AtomLine tokens;
  bool has_hydrogen = false;

  while (!mol2_block_end(++it, end)) {
    if (std::all_of(it->begin(), it->end(), absl::ascii_isblank)) {
      ABSL_LOG(INFO) << "Skipping blank line";
      continue;
    }

    std::get<1>(tokens).clear();
    std::get<2>(tokens).clear();
    std::get<3>(tokens).clear();

    auto lit = it->begin();
    if (!x3::parse(lit, it->end(), parser::atom_line, tokens)) {
      ABSL_LOG(WARNING) << "Failed to parse atom line";
      ABSL_LOG(INFO) << "The line is: " << *it;
      return { false, false };
    }

    pos.push_back(Vector3d(std::get<2>(tokens).data()));

    std::string_view atom_sym = std::get<3>(tokens);
    const Element *elem = kPt.find_element(atom_sym);
    if (elem == nullptr) {
      std::string sym_upper = absl::AsciiStrToUpper(atom_sym);
      elem = kPt.find_element(sym_upper);
      if (elem == nullptr) {
        if (sym_upper == "LP") {
          ABSL_LOG(INFO) << "Lone pair support not implemented yet";
          return { false, false };
        }

        if (sym_upper != "ANY") {
          ABSL_LOG(INFO) << "Cannot find element " << atom_sym
                         << "; check mol2 file consistency";
          return { false, false };
        }

        elem = &kPt[0];
      }
    }

    has_hydrogen |= elem->atomic_number() == 1;

    AtomData data(*elem);
    auto &optional_subtype = std::get<4>(tokens);
    if (optional_subtype) {
      atom_data_from_subtype(data, mutator.mol().size(), *optional_subtype,
                             ccat);
    }

    int idx = mutator.add_atom(data);

    auto &optional_attrs = std::get<5>(tokens);
    process_optional_attrs(mutator.mol().atom(idx), substructs, optional_attrs);

    ABSL_LOG_IF(WARNING, lit != it->end())
        << "Ignoring extra tokens in atom line";
  }

  return { true, has_hydrogen };
}

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
const struct bond_type_: public x3::symbols<BondData> {
  bond_type_() {
    add                                                                //
        ("1", BondData(constants::kSingleBond))                        //
        ("2", BondData(constants::kDoubleBond))                        //
        ("3", BondData(constants::kTripleBond))                        //
        ("am", BondData(constants::kSingleBond).set_conjugated(true))  //
        ("ar",
         BondData(constants::kAromaticBond)
             .add_flags(BondFlags::kConjugated | BondFlags::kAromatic))  //
        ("du", BondData(constants::kSingleBond));
  }
} bond_type;

const auto bond_line = *x3::omit[x3::blank]  //
                       >> +x3::omit[x3::digit] >> +x3::omit[x3::blank]
                       >> x3::repeat(2)[uint_trailing_blanks]  //
                       >> bond_type                            //
                       >> x3::omit[+x3::space | x3::eoi];
using BondLine = std::tuple<absl::InlinedVector<unsigned int, 2>, BondData>;
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

bool parse_bond_block(MoleculeMutator &mutator, Iter &it, const Iter end) {
  parser::BondLine tokens;

  while (!mol2_block_end(++it, end)) {
    if (std::all_of(it->begin(), it->end(), absl::ascii_isblank)) {
      ABSL_LOG(INFO) << "Skipping blank line";
      continue;
    }

    std::get<0>(tokens).clear();

    auto lit = it->begin();
    if (!x3::parse(lit, it->end(), parser::bond_line, tokens)) {
      ABSL_LOG(WARNING) << "Failed to parse bond line";
      ABSL_LOG(INFO) << "The line is: " << *it;
      return false;
    }

    const auto &ids = std::get<0>(tokens);
    int mol_ids[2];

    for (int i = 0; i < 2; ++i) {
      mol_ids[i] = static_cast<int>(ids[i]) - 1;
      if (mol_ids[i] >= mutator.mol().num_atoms() || mol_ids[i] < 0) {
        ABSL_LOG(WARNING) << "Atom index " << ids[i]
                          << " out of range; check mol2 file consistency";
        return false;
      }
    }

    if (mol_ids[0] == mol_ids[1]) {
      ABSL_LOG(WARNING) << "Failed to add self-bond to atom " << ids[0];
      return false;
    }

    auto [_, success] =
        mutator.add_bond(mol_ids[0], mol_ids[1], std::get<1>(tokens));
    if (!success) {
      ABSL_LOG(WARNING) << "Failed to add bond " << ids[0] << " -> " << ids[1]
                        << "; check mol2 file consistency";
      return false;
    }

    ABSL_LOG_IF(WARNING, lit != it->end())
        << "Ignoring extra tokens in bond line";
  }

  return true;
}

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
constexpr auto unity_atom_attr_line = x3::uint_ >> +x3::omit[x3::blank]
                                      >> x3::uint_
                                      >> x3::omit[+x3::space | x3::eoi];
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

std::pair<bool, bool> parse_atom_attr_block(Molecule &mol, Iter &it,
                                            const Iter end) {
  bool has_fcharge = false;
  absl::InlinedVector<int, 2> ids;

  for (++it; !mol2_block_end(it, end);) {
    if (std::all_of(it->begin(), it->end(), absl::ascii_isblank)) {
      ABSL_LOG(INFO) << "Skipping blank line";
      ++it;
      continue;
    }

    ids.clear();

    auto lit = it->begin();
    if (!x3::parse(lit, it->end(), parser::unity_atom_attr_line, ids)) {
      ABSL_LOG(WARNING) << "Failed to parse atom attribute line";
      ABSL_LOG(INFO) << "The line is: " << *it;
      return { false, false };
    }

    ABSL_DCHECK(ids.size() == 2);
    ABSL_LOG_IF(INFO, lit != it->end())
        << "Ignoring extra tokens in atom attribute line";

    --ids[0];
    if (ids[0] >= mol.num_atoms() || ids[0] < 0) {
      ABSL_LOG(WARNING) << "Atom index " << ids[0]
                        << " out of range; check mol2 file consistency";
      return { false, false };
    }

    for (int i = 0; !mol2_block_end(++it, end) && i < ids[1]; ++i) {
      std::pair<std::string_view, std::string_view> tokens =
          absl::StrSplit(*it, ' ', absl::SkipEmpty());

      if (tokens.first != "charge") {
        mol.atom(ids[0]).data().add_prop(std::string(tokens.first),
                                         std::string(tokens.second));
        continue;
      }

      int fcharge;
      if (!absl::SimpleAtoi(tokens.second, &fcharge)) {
        ABSL_LOG(WARNING)
            << "Failed to parse formal charge; continuing without charge";
        continue;
      }

      has_fcharge = true;
      mol.atom(ids[0]).data().set_formal_charge(fcharge);
    }
  }

  return { true, has_fcharge };
}

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
const auto substructure_line = *x3::omit[x3::blank]         //
                               >> uint_trailing_blanks      //
                               >> nonblank_trailing_blanks  //
                               >> +x3::omit[x3::digit]
                               >> x3::omit[+x3::space | x3::eoi];
using SubstructureLine = std::pair<unsigned int, std::string>;
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

bool parse_substructure_block(Molecule &mol, Iter &it, const Iter end) {
  parser::SubstructureLine data;

  while (!mol2_block_end(++it, end)) {
    if (std::all_of(it->begin(), it->end(), absl::ascii_isblank)) {
      ABSL_LOG(INFO) << "Skipping blank line";
      continue;
    }

    data.second.clear();

    auto lit = it->begin();

    if (!x3::parse(lit, it->end(), parser::substructure_line, data)) {
      ABSL_LOG(WARNING) << "Failed to parse substructure line";
      ABSL_LOG(INFO) << "The line is: " << *it;
      return false;
    }

    Substructure &sub = mol.substructures().emplace_back(mol.substructure());
    sub.set_id(static_cast<int>(data.first));
    sub.name() = std::move(data.second);
  }

  return true;
}

// Some mol2 files set bond type of conjugated bonds to aromatic; just make
// them non-aromatic, conjugated bonds
void fix_aromatic_bonds(Molecule &mol) {
  for (auto atom: mol) {
    if (atom.data().is_ring_atom() || atom.degree() < 2)
      continue;

    const int aromatic_count = std::accumulate(
        atom.begin(), atom.end(), 0, [](int sum, Molecule::Neighbor nei) {
          return sum
                 + static_cast<int>(nei.edge_data().order()
                                    == constants::kAromaticBond);
        });
    if (aromatic_count < 2)
      continue;

    // Set one bond to double and the others to single
    constants::BondOrder order = constants::kDoubleBond;
    for (auto nei: atom) {
      if (nei.edge_data().order() != constants::kAromaticBond)
        continue;

      nei.edge_data().set_order(order).del_flags(BondFlags::kAromatic);
      order = constants::kSingleBond;
    }
  }
}

void fix_guadinium(Molecule &mol, const std::vector<int> &ccat) {
  for (int i: ccat) {
    auto atom = mol.atom(i);
    if (atom.data().atomic_number() != 6 || atom.degree() < 2
        || std::any_of(atom.begin(), atom.end(), [](Molecule::Neighbor nei) {
             return nei.dst().data().atomic_number() != 7;
           })) {
      continue;
    }

    bool any_double = false;
    atom.data().set_formal_charge(0);
    for (auto nei: atom) {
      if (nei.edge_data().order() == constants::kDoubleBond) {
        nei.dst().data().set_formal_charge(1);
        any_double = true;
        break;
      }
    }

    if (!any_double) {
      // Why last? Because rdkit does it.
      auto last_lowest_degree = *std::min_element(
          std::make_reverse_iterator(atom.end()),
          std::make_reverse_iterator(atom.begin()),
          [](Molecule::Neighbor lhs, Molecule::Neighbor rhs) {
            return count_heavy(lhs.dst()) < count_heavy(rhs.dst());
          });

      last_lowest_degree.dst().data().set_formal_charge(1);
      last_lowest_degree.edge_data().order() = constants::kDoubleBond;
    }
  }
}
}  // namespace

Molecule read_mol2(const std::vector<std::string> &mol2) {
  Molecule mol;
  std::vector<Vector3d> pos;
  std::vector<int> ccat;
  absl::flat_hash_map<unsigned int, std::pair<std::vector<int>, std::string>>
      substructs;
  bool success = true, has_hydrogen = false, has_fcharge = false;
  bool atom_parsed = false;

  auto it = mol2.begin();
  for (; it != mol2.end() && success; ++it) {
    if (absl::StartsWith(*it, "@<TRIPOS>MOLECULE")) {
      parse_mol_block(mol, it, mol2.end());
      break;
    }
  }

  {
    auto mutator = mol.mutator();

    for (; it != mol2.end() && success;) {
      if (absl::StartsWith(*it, "@<TRIPOS>ATOM")) {
        atom_parsed = true;
        std::tie(success, has_hydrogen) =
            parse_atom_block(mutator, pos, ccat, substructs, it, mol2.end());
      } else if (absl::StartsWith(*it, "@<TRIPOS>BOND")) {
        success = parse_bond_block(mutator, it, mol2.end());
      } else if (absl::StartsWith(*it, "@<TRIPOS>UNITY_ATOM_ATTR")) {
        if (!atom_parsed) {
          success = false;
          ABSL_LOG(WARNING)
              << "UNITY_ATOM_ATTR block must come after ATOM block";
          break;
        }
        std::tie(success, has_fcharge) =
            parse_atom_attr_block(mol, it, mol2.end());
      } else if (absl::StartsWith(*it, "@<TRIPOS>SUBSTRUCTURE")) {
        success = parse_substructure_block(mol, it, mol2.end());
      } else {
        ABSL_LOG_IF(WARNING, absl::StartsWith(*it, "@"))
            << "Unimplemented mol2 block: " << *it;
        ++it;
      }
    }
  }

  if (!success) {
    ABSL_LOG(ERROR) << "Failed to parse mol2 block";
    mol.clear();
    return mol;
  }

  fix_aromatic_bonds(mol);

  if (!has_fcharge)
    fix_guadinium(mol, ccat);

  if (!has_hydrogen && !has_fcharge) {
    guess_fcharge_hydrogens_2d(mol);
  } else if (!has_hydrogen) {
    guess_hydrogens_2d(mol);
  } else if (!has_fcharge) {
    guess_fcharge_2d(mol);
  }

  for (auto atom: mol) {
    const int degree = all_neighbors(atom);
    if (degree <= 1) {
      atom.data().set_hybridization(
          static_cast<constants::Hybridization>(degree));
    }
  }

  mol.confs().push_back(stack(pos));

  // Only add substructures actually mentioned in the SUBSTRUCTURE block
  for (auto &[_, data]: substructs) {
    for (Substructure &sub: mol.substructures()) {
      if (sub.name() == data.second) {
        sub.update_atoms(internal::IndexSet(
            boost::container::ordered_unique_range, std::move(data.first)));
      }
    }
  }

  return mol;
}

namespace {
int width_of_size(int size) {
  ABSL_DCHECK_GE(size, 0);
  return log_base10(static_cast<unsigned int>(size)) + 1;
}

struct SubstructInfo {
  std::vector<int> sub_of_atom;
  std::vector<int> root_of_sub;
  std::vector<int> sub_ids;
  int num_used_subs;
};

SubstructInfo resolve_substructs(const Molecule &mol) {
  SubstructInfo info;
  info.sub_of_atom.resize(mol.size(), 0);
  if (mol.substructures().empty()) {
    info.root_of_sub.push_back(0);
    info.sub_ids.push_back(0);
    info.num_used_subs = 1;
    return info;
  }

  info.root_of_sub.resize(mol.substructures().size() + 1, -1);

  for (int i = 0; i < mol.substructures().size(); ++i) {
    const auto &sub = mol.substructures()[i];
    int &sub_root = info.root_of_sub[i + 1];

    for (auto atom: sub) {
      int id = atom.as_parent().id();
      if (info.sub_of_atom[id] > 0)
        continue;

      info.sub_of_atom[id] = i + 1;
      if (sub_root < 0)
        sub_root = id;
    }
  }

  auto it = absl::c_find(info.sub_of_atom, 0);
  if (it == info.sub_of_atom.end()) {
    info.root_of_sub[0] = -1;
  } else {
    info.root_of_sub[0] = static_cast<int>(it - info.sub_of_atom.begin());
  }

  info.sub_ids.resize(mol.substructures().size() + 1, -1);
  info.sub_ids[0] = -1;
  info.num_used_subs = 0;

  for (int i = 0; i < info.sub_ids.size(); ++i) {
    if (info.root_of_sub[i] < 0)
      continue;

    info.sub_ids[i] = info.num_used_subs++;
  }

  return info;
}

std::string_view sybyl_subtype_hyb(constants::Hybridization hyb) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (hyb) {
  case constants::kSP:
    return ".1";
  case constants::kSP2:
    return ".2";
  case constants::kSP3:
    return ".3";
  default:
    return "";
  }
}

std::string_view carbon_sybyl_subtype(Molecule::Atom atom) {
  if (atom.data().formal_charge() == 1)
    return ".cat";

  if (atom.data().is_aromatic())
    return ".ar";

  return sybyl_subtype_hyb(atom.data().hybridization());
}

bool is_carbonyl_carbon(Molecule::Atom atom) {
  return absl::c_any_of(atom, [](Molecule::Neighbor nei) {
    return nei.dst().data().atomic_number() == 8
           && nei.edge_data().order() == constants::kDoubleBond
           && nei.edge_data().is_conjugated();
  });
}

bool is_amide_nitrogen(Molecule::Atom atom) {
  bool maybe_amide = false;

  for (auto nei: atom) {
    if (nei.edge_data().order() != constants::kSingleBond)
      return false;

    if (nei.dst().data().atomic_number() != 6)
      continue;

    maybe_amide |= is_carbonyl_carbon(nei.dst());
  }

  return maybe_amide;
}

std::string_view nitrogen_sybyl_subtype(Molecule::Atom atom) {
  if (atom.data().is_aromatic())
    return ".ar";

  int num_total_neighbors = all_neighbors(atom);

  if (num_total_neighbors == 4 && atom.data().formal_charge() == 1)
    return ".4";

  if (is_amide_nitrogen(atom))
    return ".am";

  if (atom.data().hybridization() == constants::kSP2)
    return num_total_neighbors == 3 ? ".pl3" : ".2";

  return sybyl_subtype_hyb(atom.data().hybridization());
}

std::string_view oxygen_sybyl_subtype(Molecule::Atom atom) {
  if (atom.degree() == 1 && atom.data().implicit_hydrogens() == 0) {
    auto nei = atom[0];

    int terminal_oxygens = 0;
    bool has_oxo = false;

    for (auto mei: nei.dst()) {
      if (mei.dst().data().atomic_number() == 8
          && all_neighbors(mei.dst()) == 1) {
        ++terminal_oxygens;
        if (mei.edge_data().order() == constants::kDoubleBond)
          has_oxo = true;
      }
    }

    if (terminal_oxygens >= 2 && has_oxo)
      return ".co2";
  }

  return sybyl_subtype_hyb(atom.data().hybridization());
}

std::string_view sulfur_sybyl_subtype(Molecule::Atom atom) {
  int oxygens = absl::c_count_if(atom, [](Molecule::Neighbor nei) {
    return nei.dst().data().atomic_number() == 8;
  });

  if (oxygens == 1)
    return ".O";
  if (oxygens == 2)
    return ".O2";

  return sybyl_subtype_hyb(atom.data().hybridization());
}

std::vector<std::string> sybyl_atom_types(const Molecule &mol) {
  std::vector<std::string> types(mol.size());

  for (auto atom: mol) {
    types[atom.id()] = atom.data().element_symbol();
    switch (atom.data().atomic_number()) {
    case 6:
      types[atom.id()] += carbon_sybyl_subtype(atom);
      break;
    case 7:
      types[atom.id()] += nitrogen_sybyl_subtype(atom);
      break;
    case 8:
      types[atom.id()] += oxygen_sybyl_subtype(atom);
      break;
    case 15:
      types[atom.id()] += sybyl_subtype_hyb(atom.data().hybridization());
      break;
    case 16:
      types[atom.id()] += sulfur_sybyl_subtype(atom);
      break;
    default:
      break;
    }
  }

  return types;
}

template <class C>
int max_size_of(const C &sized) {
  auto it = absl::c_max_element(sized, [](auto &&lhs, auto &&rhs) {
    return lhs.size() < rhs.size();
  });

  if (it != sized.end())
    return static_cast<int>(it->size());

  return 0;
}

int double_width(double d) {
  return static_cast<int>(absl::StrFormat("%.3f", d).size());
}

int max_width_conf(const Matrix3Xd &coords) {
  double max = coords.maxCoeff(), min = coords.minCoeff();
  return nuri::max(double_width(max), double_width(min));
}

Array4i measure_col_widths_atom(const Molecule &mol,
                                const std::vector<std::string> &atom_names,
                                int subs_id_width, int subs_name_width) {
  Array4i cols;

  cols[0] = max_size_of(atom_names);
  cols[1] = subs_id_width;
  cols[2] = subs_name_width;

  cols[3] = 5;  // 0.000
  for (auto atom: mol) {
    // NOLINTNEXTLINE(clang-diagnostic-float-equal)
    if (atom.data().partial_charge() == 0)
      continue;

    int width = double_width(atom.data().partial_charge());
    cols[3] = std::max(cols[3], width);
  }

  return cols;
}

template <bool is_3d>
void write_atoms(std::string &out, const Molecule &mol, const int conf,
                 const int atom_id_width, const Array4i &extra_widths,
                 const std::vector<std::string> &atom_names,
                 const std::vector<std::string> &atom_types,
                 const SubstructInfo &subs_info,
                 const std::vector<std::string> &sub_names) {
  int coords_width;
  if constexpr (is_3d) {
    const Matrix3Xd &coords = mol.confs()[conf];
    coords_width = max_width_conf(coords);
  } else {
    // 0.000
    coords_width = 5;
  }

  Vector3d pos;
  if constexpr (!is_3d)
    pos.setZero();

  absl::StrAppend(&out, "@<TRIPOS>ATOM\n");

  for (auto atom: mol) {
    if constexpr (is_3d)
      pos = mol.confs()[conf].col(atom.id());

    int sub_idx = subs_info.sub_of_atom[atom.id()];
    // NOLINTNEXTLINE(clang-diagnostic-used-but-marked-unused)
    ABSL_DCHECK_GE(sub_idx, 0);
    ABSL_DCHECK_LT(sub_idx, subs_info.sub_ids.size());

    int sub_id = subs_info.sub_ids[sub_idx];
    // NOLINTNEXTLINE(clang-diagnostic-used-but-marked-unused)
    ABSL_DCHECK_GE(sub_id, 0);

    ABSL_DCHECK_LT(sub_idx, sub_names.size());
    std::string_view sub_name = sub_names[sub_idx];
    ABSL_DCHECK(!sub_name.empty());

    absl::StrAppendFormat(
        &out, "%*d %-*s %*.3f %*.3f %*.3f %-5s %*d %-*s %*.3f\n",             //
        atom_id_width, atom.id() + 1,                                         //
        extra_widths[0], atom_names[atom.id()],                               //
        coords_width, pos.x(), coords_width, pos.y(), coords_width, pos.z(),  //
        atom_types[atom.id()],                                                //
        extra_widths[1], sub_id + 1,                                          //
        extra_widths[2], sub_name,                                            //
        extra_widths[3], atom.data().partial_charge());
  }

  std::string buf;
  bool need_attr_header = true;
  for (auto atom: mol) {
    int num_props = 0;

    if (atom.data().formal_charge() != 0) {
      absl::StrAppendFormat(&buf, "charge %d\n", atom.data().formal_charge());
      ++num_props;
    }

    for (auto &[key, value]: atom.data().props()) {
      if (key == internal::kNameKey || key.empty())
        continue;

      absl::StrAppendFormat(&buf, "%s %s\n", internal::ascii_safe(key),
                            internal::ascii_safe(value));
      ++num_props;
    }

    if (num_props == 0)
      continue;

    if (need_attr_header) {
      absl::StrAppend(&out, "@<TRIPOS>UNITY_ATOM_ATTR\n");
      need_attr_header = false;
    }

    absl::StrAppendFormat(&out, "%*d %d\n%s",  //
                          atom_id_width, atom.id() + 1, num_props, buf);
    buf.clear();
  }
}

bool is_amide_bond(Molecule::Bond bond) {
  if (bond.data().order() != constants::kSingleBond
      || !bond.data().is_conjugated())
    return false;

  auto src = bond.src(), dst = bond.dst();
  if (src.data().atomic_number() == 7 && dst.data().atomic_number() == 6) {
    std::swap(src, dst);
  } else if (src.data().atomic_number() != 6
             || dst.data().atomic_number() != 7) {
    return false;
  }

  return is_carbonyl_carbon(src) && is_amide_nitrogen(dst);
}

std::string_view mol2_bond_type(Molecule::Bond bond) {
  if (is_amide_bond(bond))
    return "am";

  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (bond.data().order()) {
  case constants::kSingleBond:
    return "1";
  case constants::kDoubleBond:
    return "2";
  case constants::kTripleBond:
    return "3";
  case constants::kAromaticBond:
    return "ar";
  case constants::kOtherBond:
    return "du";
  default:
    return "un";
  }
}

void write_bonds(std::string &out, const Molecule &mol, const int atom_id_width,
                 const int bond_id_width) {
  if (mol.bond_empty())
    return;

  absl::StrAppend(&out, "@<TRIPOS>BOND\n");

  for (auto bond: mol.bonds()) {
    absl::StrAppendFormat(&out, "%*d %*d %*d %s\n",            //
                          bond_id_width, bond.id() + 1,        //
                          atom_id_width, bond.src().id() + 1,  //
                          atom_id_width, bond.dst().id() + 1,
                          mol2_bond_type(bond));
  }
}

void write_substructs(std::string &out, const int atom_id_width,
                      const int sub_id_width, const int sub_name_width,
                      const SubstructInfo &sub_info,
                      const std::vector<std::string> &sub_names) {
  absl::StrAppend(&out, "@<TRIPOS>SUBSTRUCTURE\n");

  for (int i = 0; i < sub_info.sub_ids.size(); ++i) {
    if (sub_info.root_of_sub[i] < 0)
      continue;

    absl::StrAppendFormat(&out, "%*d %-*s %*d\n",                 //
                          sub_id_width, sub_info.sub_ids[i] + 1,  //
                          sub_name_width, sub_names[i],           //
                          atom_id_width, sub_info.root_of_sub[i] + 1);
  }
}

template <bool is_3d>
void write_mol2_single_conf(std::string &out, const Molecule &mol, int conf,
                            const bool write_sub, const int atom_id_width,
                            const Array4i &extra_atom_widths,
                            const int bond_id_width, const int sub_id_width,
                            const int sub_name_width,
                            const std::vector<std::string> &atom_names,
                            const std::vector<std::string> &atom_types,
                            const SubstructInfo &sub_info,
                            const std::vector<std::string> &sub_names) {
  if constexpr (is_3d) {
    // NOLINTNEXTLINE(clang-diagnostic-used-but-marked-unused)
    ABSL_DCHECK_LT(conf, mol.confs().size());
  }

  absl::StrAppendFormat(
      &out,
      // clang-format off
R"mol2(@<TRIPOS>MOLECULE
%s
%d %d %d 0 0
SMALL
NO_CHARGES
%s
%s
)mol2",
      // clang-format on
      mol.name().empty() ? kCommentIndicator
                         : internal::ascii_newline_safe(mol.name()),
      mol.num_atoms(), mol.num_bonds(), sub_info.num_used_subs,
      kCommentIndicator,
      internal::ascii_newline_safe(internal::get_key(mol.props(), "comment")));

  if (mol.empty())
    return;

  write_atoms<is_3d>(out, mol, conf, atom_id_width, extra_atom_widths,
                     atom_names, atom_types, sub_info, sub_names);
  write_bonds(out, mol, atom_id_width, bond_id_width);

  if (write_sub)
    write_substructs(out, atom_id_width, sub_id_width, sub_name_width, sub_info,
                     sub_names);
}
}  // namespace

bool write_mol2(std::string &out, const Molecule &mol, int conf,
                bool write_sub) {
  int atom_id_width = width_of_size(mol.num_atoms()),
      bond_id_width = width_of_size(mol.num_bonds());

  std::vector atom_names = make_names_unique(mol, [&mol](int i) {
    auto atom = mol[i];
    std::string_view name = atom.data().get_name();
    return name.empty() ? atom.data().element_symbol() : name;
  });
  std::vector atom_types = sybyl_atom_types(mol);

  const SubstructInfo subs_info = resolve_substructs(mol);
  std::vector substruct_names = make_names_unique(
      subs_info.root_of_sub, [&mol, &subs_info](int idx) -> std::string {
        int root = subs_info.root_of_sub[idx];
        if (root < 0)
          return {};

        if (idx == 0)
          return "UNK";

        const Substructure &sub = mol.substructures()[idx - 1];
        if (sub.name().empty())
          return "UNK";

        std::string name(internal::get_key(sub.props(), "chain"));
        if (!name.empty()) {
          absl::StrAppend(&name, sub.id(),
                          internal::get_key(sub.props(), "icode"));
        }
        absl::StrAppend(&name, sub.name());
        return name;
      });
  int substruct_id_width = width_of_size(subs_info.num_used_subs),
      substruct_name_width = max_size_of(substruct_names);

  auto extra_atom_col_widths = measure_col_widths_atom(
      mol, atom_names, substruct_id_width, substruct_name_width);

  if (!mol.is_3d()) {
    write_mol2_single_conf<false>(out, mol, -1, write_sub, atom_id_width,
                                  extra_atom_col_widths, bond_id_width,
                                  substruct_id_width, substruct_name_width,
                                  atom_names, atom_types, subs_info,
                                  substruct_names);
  } else if (conf < 0) {
    for (int i = 0; i < mol.confs().size(); ++i) {
      write_mol2_single_conf<true>(out, mol, i, write_sub, atom_id_width,
                                   extra_atom_col_widths, bond_id_width,
                                   substruct_id_width, substruct_name_width,
                                   atom_names, atom_types, subs_info,
                                   substruct_names);
    }
  } else {
    write_mol2_single_conf<true>(out, mol, conf, write_sub, atom_id_width,
                                 extra_atom_col_widths, bond_id_width,
                                 substruct_id_width, substruct_name_width,
                                 atom_names, atom_types, subs_info,
                                 substruct_names);
  }

  return true;
}
}  // namespace nuri
