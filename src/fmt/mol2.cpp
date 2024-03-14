//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/mol2.h"

#include <algorithm>
#include <istream>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/fusion/include/std_pair.hpp>
#include <boost/fusion/include/std_tuple.hpp>
#include <boost/optional.hpp>
#include <boost/spirit/home/x3.hpp>

#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "nuri/eigen_config.h"
#include "nuri/algo/guess.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
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
                               >> -(+x3::omit[x3::blank] >> x3::uint_);
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

bool mol2_block_end(const Iter it, const Iter end) {
  return it == end || absl::StartsWith(*it, "@");
}

void parse_mol_block(Molecule &mol, Iter &it, const Iter end) {
  if (mol2_block_end(++it, end))
    return;

  mol.name() = *it == "*****" ? "" : *it;
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
                         "incompatible with future versions of nurikit";
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
template <class T>
struct TrailingBlanksRuleTag;

template <class T, class Tag = TrailingBlanksRuleTag<T>>
struct TrailingBlanksRule: public x3::rule<Tag, T> {
  using Base = x3::rule<Tag, T>;
  using Base::Base;

  constexpr TrailingBlanksRule(): Base("") { }

  template <class RHS>
  // NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
  constexpr auto operator=(const RHS &rhs) const && {
    return Base::operator=(rhs >> +x3::omit[x3::blank]);
  }
};

constexpr auto nonblank_trailing_blanks =
    TrailingBlanksRule<std::string, struct nonblank_trailing_blanks_tag>() =
        +~x3::blank;

constexpr auto double_trailing_blanks = TrailingBlanksRule<double>() =
    x3::double_;

constexpr auto uint_trailing_blanks = TrailingBlanksRule<unsigned int>() =
    x3::uint_;

constexpr auto atom_line = *x3::omit[x3::blank]         //
                           >> uint_trailing_blanks      //
                           >> nonblank_trailing_blanks  //
                           >> x3::repeat(3)[double_trailing_blanks]
                           >> +x3::alpha >> -('.' >> +x3::alnum)  //
                           >> -(+x3::omit[x3::blank]              //
                                >> uint_trailing_blanks           //
                                >> -(nonblank_trailing_blanks     //
                                     >> -x3::double_));
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
    const Element *elem = PeriodicTable::get().find_element(atom_sym);
    if (elem == nullptr) {
      std::string sym_upper = absl::AsciiStrToUpper(atom_sym);
      elem = PeriodicTable::get().find_element(sym_upper);
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

        elem = &PeriodicTable::get()[0];
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
                       >> *x3::omit[x3::blank];
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
      mol_ids[i] = static_cast<int>(ids[i] - 1);
      if (mol_ids[i] >= mutator.mol().num_atoms()) {
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
                                      >> x3::uint_;
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
    if (ids[0] >= mol.num_atoms()) {
      ABSL_LOG(WARNING) << "Atom index " << ids[0]
                        << " out of range; check mol2 file consistency";
      return { false, false };
    }

    for (int i = 0; !mol2_block_end(++it, end) && i < ids[1]; ++i) {
      std::pair<std::string_view, std::string_view> tokens =
          absl::StrSplit(*it, ' ', absl::SkipEmpty());

      if (tokens.first != "charge") {
        ABSL_LOG(WARNING) << "Unimplemented atom attribute " << tokens.first
                          << "; continuing without attribute";
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
                               >> +x3::omit[x3::digit] >> *x3::omit[x3::blank];
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
      auto last_lowest_degree = std::min_element(
          std::make_reverse_iterator(atom.end()),
          std::make_reverse_iterator(atom.begin()),
          [](Molecule::Neighbor lhs, Molecule::Neighbor rhs) {
            return count_heavy(lhs.dst()) < count_heavy(rhs.dst());
          });

      last_lowest_degree->dst().data().set_formal_charge(1);
      last_lowest_degree->edge_data().order() = constants::kDoubleBond;
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
  for (auto &[_, data]: substructs)
    for (Substructure &sub: mol.find_substructures(data.second))
      sub.update_atoms(std::move(data.first));

  return mol;
}
}  // namespace nuri
