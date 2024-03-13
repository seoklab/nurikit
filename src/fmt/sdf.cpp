//
// Project nurikit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/sdf.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <boost/spirit/home/x3.hpp>

#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
bool SDFReader::getnext(std::vector<std::string> &block) {
  block.clear();

  std::string line;
  while (std::getline(*is_, line)) {
    if (absl::StripTrailingAsciiWhitespace(line) == "$$$$")
      break;

    block.push_back(line);
  }

  return !block.empty();
}

const bool SDFReaderFactory::kRegistered =
    register_reader_factory<SDFReaderFactory>({ "mol", "sdf" });

namespace {
namespace x3 = boost::spirit::x3;

using Iterator = std::vector<std::string>::const_iterator;

// NOLINTBEGIN(*-non-private-member-variables-in-classes)
struct HeaderReadResult {
  static HeaderReadResult failure() { return HeaderReadResult(); }

  static HeaderReadResult success(int version, int natoms, int nbonds) {
    return HeaderReadResult(static_cast<int>(version), static_cast<int>(natoms),
                            static_cast<int>(nbonds));
  }

  int version() const { return version_; }

  int natoms() const { return natoms_; }

  int nbonds() const { return nbonds_; }

  operator bool() const { return version_ >= 0; }

private:
  HeaderReadResult(): version_(-1) { }

  HeaderReadResult(int v, int a, int b): version_(v), natoms_(a), nbonds_(b) { }

  int version_;
  int natoms_;
  int nbonds_;
};
// NOLINTEND(*-non-private-member-variables-in-classes)

HeaderReadResult read_sdf_header(Molecule &mol, Iterator &it,
                                 const Iterator end) {
  // header block (3) + counts line (1)
  if (end - it < 4) {
    ABSL_LOG(WARNING) << "Invalid SDF format: header block is missing";
    return HeaderReadResult::failure();
  }

  mol.name() = absl::StripAsciiWhitespace(*it++);

  if (auto stamp = absl::StripAsciiWhitespace(*it++); !stamp.empty())
    mol.add_prop("stamp", std::string(stamp));

  if (auto comment = absl::StripAsciiWhitespace(*it++); !comment.empty())
    mol.add_prop("comment", std::string(comment));

  std::string_view line = *it;

  auto vpos = line.rfind('V');
  if (vpos == std::string_view::npos) {
    ABSL_LOG(WARNING) << "Invalid SDF format: version is missing";
    return HeaderReadResult::failure();
  }

  int version;
  if (!absl::SimpleAtoi(line.substr(vpos + 1), &version)) {
    ABSL_LOG(WARNING) << "Invalid SDF format: cannot parse version";
    return HeaderReadResult::failure();
  }

  int natoms, nbonds;
  if (!absl::SimpleAtoi(line.substr(0, 3), &natoms)
      || !absl::SimpleAtoi(safe_substr(line, 3, 3), &nbonds)) {
    ABSL_LOG(WARNING) << "Invalid SDF format: cannot parse counts line";
    natoms = nbonds = 0;
  }

  return HeaderReadResult::success(version, natoms, nbonds);
}

void read_sdf_extra(Molecule &mol, Iterator it, const Iterator end,
                    std::string &temp) {
  constexpr auto sdf_data_header = '>' >> +x3::omit[x3::blank] >> +x3::char_;
  constexpr auto sdf_data_field_name = '<' >> *(x3::char_ - '>') >> '>';

  for (; it < end;) {
    std::string_view line = *it;
    temp.clear();

    if (!x3::parse(line.begin(), line.end(), sdf_data_header, temp)) {
      ABSL_LOG(INFO) << "Ignoring unknown line: " << line;
      ++it;
      continue;
    }

    std::string key;
    if (!x3::parse(temp.begin(), temp.end(), sdf_data_field_name, key)) {
      ABSL_LOG(INFO) << "Missing field name in data block: " << line;
      key = std::move(temp);
    }

    temp.clear();
    for (; ++it < end;) {
      line = *it;
      if (line.empty() || line[0] == '>')
        break;

      absl::StrAppend(&temp, line, "\n");
    }

    if (!temp.empty())
      temp.pop_back();

    mol.add_prop(std::move(key), temp);
  }
}

bool parse_bond_data(std::string_view part, BondData &data) {
  unsigned int val;
  if (!absl::SimpleAtoi(part, &val))
    return false;

  switch (val) {
  case 1:
    data = BondData(constants::kSingleBond);
    break;
  case 2:
    data = BondData(constants::kDoubleBond);
    break;
  case 3:
    data = BondData(constants::kTripleBond);
    break;
  case 4:
    data = BondData(constants::kAromaticBond);
    break;
  case 5:
    data = BondData(constants::kSingleBond).add_flags(BondFlags::kConjugated);
    break;
  case 6:
    data = BondData(constants::kSingleBond)
               .add_flags(BondFlags::kConjugated | BondFlags::kAromatic);
    break;
  case 7:
    data = BondData(constants::kDoubleBond)
               .add_flags(BondFlags::kConjugated | BondFlags::kAromatic);
    break;
  case 8:
    data = BondData(constants::kOtherBond);
    break;
  default:
    return false;
  }

  return true;
}

enum class ParseLineResult {
  kAtom,
  kBond,
  kProp,
  kExtra,
  kSkip,
  kError,
};

ParseLineResult try_read_v2000_bond_line(MoleculeMutator &mut,
                                         std::string_view line,
                                         const bool bond_context) {
  if (line.empty())
    return ParseLineResult::kSkip;

  if (line[0] == 'M' || line[0] == '>') {
    ABSL_LOG_IF(WARNING, bond_context)
        << "Inconsistent counts block and bonds block";
    return line[0] == 'M' ? ParseLineResult::kProp : ParseLineResult::kExtra;
  }

  if (line.size() < 7) {
    ABSL_LOG_IF(WARNING, bond_context) << "Line too short for bond line";
    return ParseLineResult::kError;
  }

  unsigned int src, dst;
  if (!absl::SimpleAtoi(line.substr(0, 3), &src)
      || !absl::SimpleAtoi(line.substr(3, 3), &dst)) {
    ABSL_LOG_IF(WARNING, bond_context) << "Failed to parse bond indices";

    return ParseLineResult::kError;
  }

  --src;
  --dst;
  if (src >= mut.mol().num_atoms() || dst >= mut.mol().num_atoms()
      || src == dst) {
    ABSL_LOG_IF(WARNING, bond_context) << "Invalid bond indices";
    return ParseLineResult::kError;
  }

  BondData data;
  if (!parse_bond_data(line.substr(6, 3), data)) {
    ABSL_LOG_IF(WARNING, bond_context) << "Failed to parse bond order";
    return ParseLineResult::kError;
  }

  auto [_, added] = mut.add_bond(static_cast<int>(src), static_cast<int>(dst),
                                 std::move(data));
  ABSL_LOG_IF(WARNING, !added) << "Duplicate bond " << src << " - " << dst;

  return ParseLineResult::kBond;
}

ParseLineResult try_read_v2000_atom_line(MoleculeMutator &mut,
                                         std::string_view line,
                                         const bool atom_context,
                                         std::vector<Vector3d> &coords,
                                         std::string &elem_upper) {
  if (line.empty())
    return ParseLineResult::kSkip;

  auto test_for_other = [&](std::string_view onerror, int lineno) {
    auto ret = try_read_v2000_bond_line(mut, line, false);

    ABSL_LOG_IF(WARNING, atom_context && ret != ParseLineResult::kError)
            .AtLocation(__FILE__, lineno)
        << "Inconsistent counts block and atom block";

    if (ret == ParseLineResult::kError) {
      if (atom_context) {
        ABSL_LOG(WARNING).AtLocation(__FILE__, lineno) << onerror;
        ABSL_LOG(INFO).AtLocation(__FILE__, lineno) << "The line is: " << line;
      } else {
        ABSL_LOG(INFO).AtLocation(__FILE__, lineno)
            << "Ignoring unknown line: " << line;
        ret = ParseLineResult::kSkip;
      }
    }

    return ret;
  };

  if (line.size() < 32)
    return test_for_other("Line too short for atom line", __LINE__);

  Vector3d &pos = coords.emplace_back();
  for (int i = 0; i < 3; ++i) {
    if (!absl::SimpleAtod(line.substr(static_cast<size_t>(i) * 10, 10),
                          &pos[i]))
      return test_for_other("Failed to parse atom position", __LINE__);
  }

  AtomData data;

  elem_upper =
      absl::AsciiStrToUpper(absl::StripAsciiWhitespace(slice(line, 30, 34)));
  const Element *elem = kPt.find_element(elem_upper);
  if (elem == nullptr) {
    if (elem_upper == "D") {
      elem = &kPt[1];
      data.set_isotope(*elem->find_isotope(2));
    } else if (elem_upper == "T") {
      elem = &kPt[1];
      data.set_isotope(*elem->find_isotope(3));
    }
  }

  if (elem != nullptr) {
    data.set_element(*elem);
    mut.add_atom(std::move(data));
    return ParseLineResult::kAtom;
  }

  std::string onerror =
      absl::StrCat("Unknown element: ", slice_strip(line, 30, 34));
  return test_for_other(onerror, __LINE__);
}

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
constexpr auto v2000_property_line =  //
    'M' >> +x3::omit[x3::blank] >> +x3::char_;

constexpr auto v2000_property_values =  //
    *x3::omit[x3::blank] >> x3::int_ % +x3::omit[x3::blank];
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

std::vector<int> read_props_values_common(std::string_view data) {
  std::vector<int> values;

  if (data.size() < 8) {
    ABSL_LOG(INFO) << "Empty data provided";
    return values;
  }

  if (unsigned int hint; absl::SimpleAtoi(data.substr(3, 3), &hint))
    values.reserve(static_cast<size_t>(hint) * 2);

  auto value_data = data.substr(7);
  if (!x3::parse(value_data.begin(), value_data.end(),
                 parser::v2000_property_values, values)) {
    ABSL_LOG(WARNING) << "Failed to parse data: " << data;
  }

  ABSL_LOG_IF(WARNING, values.size() % 2 != 0)
      << "Invalid data: odd number of elements: " << data;
  values.pop_back();

  return values;
}

void read_chg(Molecule &mol, std::string_view data) {
  std::vector<int> chgs = read_props_values_common(data);

  for (int i = 0; i + 1 < chgs.size(); i += 2) {
    unsigned int atom = chgs[i] - 1;
    int chg = chgs[i + 1];

    if (atom >= mol.size()) {
      ABSL_LOG(WARNING) << "Atom index out of range: " << atom;
      continue;
    }

    mol.atom(static_cast<int>(atom)).data().set_formal_charge(chg);
  }
}

void read_iso(Molecule &mol, std::string_view data) {
  std::vector<int> isos = read_props_values_common(data);

  for (int i = 0; i + 1 < isos.size(); i += 2) {
    unsigned int atom = isos[i] - 1;
    int n = isos[i + 1];

    if (atom >= mol.size()) {
      ABSL_LOG(WARNING) << "Atom index out of range: " << atom;
      continue;
    }

    AtomData &ad = mol.atom(static_cast<int>(atom)).data();
    const Isotope *iso = ad.element().find_isotope(n);
    if (iso == nullptr) {
      ABSL_LOG(WARNING) << "Isotope not found: " << n;
      continue;
    }

    ad.set_isotope(*iso);
  }
}

ParseLineResult read_v2000_property_block(Molecule &mol, std::string_view line,
                                          std::string &temp) {
  if (line.empty())
    return ParseLineResult::kSkip;

  if (line[0] == '>')
    return ParseLineResult::kExtra;

  temp.clear();
  if (!x3::parse(line.begin(), line.end(), parser::v2000_property_line, temp)
      || temp.size() < 3) {
    ABSL_LOG(WARNING) << "Invalid property block line";
    return ParseLineResult::kSkip;
  }

  std::string_view parsed = temp;
  auto key = parsed.substr(0, 3);

  if (key == "CHG") {
    read_chg(mol, parsed);
  } else if (key == "ISO") {
    read_iso(mol, parsed);
  } else if (key != "END") {
    ABSL_LOG(INFO) << "Unimplemented property block: " << key;
    mol.add_prop(std::string(key), std::string(parsed.substr(3)));
  }

  return ParseLineResult::kProp;
}

bool read_v2000(Molecule &mol, const HeaderReadResult result, Iterator &it,
                const Iterator end) {
  auto mut = mol.mutator();

  ParseLineResult status = ParseLineResult::kAtom;
  std::vector<Vector3d> coords;
  coords.reserve(result.natoms());

  std::string temp;

  for (; status == ParseLineResult::kAtom && it < end; ++it) {
    status = try_read_v2000_atom_line(
        mut, *it, mol.num_atoms() < result.natoms(), coords, temp);

    if (status == ParseLineResult::kError) {
      ABSL_LOG(ERROR) << "Failed to read atom block";
      return false;
    }

    if (status == ParseLineResult::kSkip) {
      ABSL_LOG(INFO) << "Skipping line: " << *it;
      status = ParseLineResult::kAtom;
    }
  }

  for (int i = mol.num_bonds(); status == ParseLineResult::kBond && it < end;
       ++it) {
    status = try_read_v2000_bond_line(mut, *it, i < result.nbonds());

    if (status == ParseLineResult::kError) {
      ABSL_LOG(ERROR) << "Failed to read bond block";
      return false;
    }

    if (status == ParseLineResult::kSkip) {
      ABSL_LOG(INFO) << "Skipping line: " << *it;
      status = ParseLineResult::kBond;
      continue;
    }

    ++i;
  }

  for (; status == ParseLineResult::kProp && it < end; ++it) {
    status = read_v2000_property_block(mol, *it, temp);

    if (status == ParseLineResult::kSkip) {
      ABSL_LOG(INFO) << "Skipping line: " << *it;
      status = ParseLineResult::kProp;
    }
  }

  read_sdf_extra(mol, it, end, temp);

  mol.confs().emplace_back(stack(coords));

  return true;
}
}  // namespace

Molecule read_sdf(const std::vector<std::string> &sdf) {
  Molecule mol;

  auto it = sdf.begin();
  auto header_result = read_sdf_header(mol, it, sdf.end());
  if (!header_result) {
    ABSL_LOG(ERROR) << "Failed to read SDF header";
    mol.clear();
    return mol;
  }

  if (++it == sdf.end()) {
    ABSL_LOG(ERROR) << "No atom block found";
    mol.clear();
    return mol;
  }

  mol.reserve(header_result.natoms());
  mol.reserve_bonds(header_result.nbonds());

  if (header_result.version() == 2000) {
    read_v2000(mol, header_result, it, sdf.end());
    // TODO(jnooree)
    // } else if (header_result.version() == 3000) {
  } else {
    ABSL_LOG(ERROR) << "Unknown SDF version: " << header_result.version();
    mol.clear();
  }

  return mol;
}
}  // namespace nuri
