//
// Project nurikit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/sdf.h"

#include <cstddef>
#include <stack>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/fusion/include/std_pair.hpp>
#include <boost/fusion/include/std_tuple.hpp>
#include <boost/spirit/home/x3.hpp>

#include <absl/algorithm/container.h>
#include <absl/container/inlined_vector.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/strip.h>

#include "nuri/eigen_config.h"
#include "fmt_internal.h"
#include "nuri/algo/guess.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
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
  // Clang analyzer complains about uninitialized members
#ifdef __clang_analyzer__
  HeaderReadResult(): version_(-1), natoms_(0), nbonds_(0) { }
#else
  HeaderReadResult(): version_(-1) { }
#endif

  HeaderReadResult(int v, int a, int b): version_(v), natoms_(a), nbonds_(b) { }

  int version_;
  int natoms_;
  int nbonds_;
};

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

bool parse_sdf_atom(AtomData &data, std::string_view elem_str) {
  const Element *elem = kPt.find_element(elem_str);
  if (elem == nullptr) {
    if (elem_str == "D") {
      elem = &kPt[1];
      data.set_isotope(*elem->find_isotope(2));
    } else if (elem_str == "T") {
      elem = &kPt[1];
      data.set_isotope(*elem->find_isotope(3));
    }
  }

  if (elem == nullptr) {
    return false;
  }

  data.set_element(*elem);
  return true;
}

bool parse_sdf_bond(BondData &data, unsigned int type) {
  switch (type) {
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

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
constexpr auto
    sdf_data_header_name = x3::rule<struct SDFDataHeaderName, std::string>("") =
        '<' >> +(x3::char_ - (x3::lit('<') | '>')) >> '>';

constexpr auto sdf_data_header_num =
    x3::rule<struct SDFDataHeaderNum, std::string>("") = "DT" >> +x3::digit;

constexpr auto sdf_data_header_key =
    x3::rule<struct SDFDataHeaderKey, std::string>("") =
        sdf_data_header_name | sdf_data_header_num;

constexpr auto sdf_data_header =      //
    '>' >> +(+x3::omit[x3::blank] >>  //
             (sdf_data_header_key | +x3::omit[~x3::blank]));
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

void read_sdf_extra(Molecule &mol, Iterator it, const Iterator end) {
  absl::InlinedVector<std::string, 1> header;
  std::string_view key;
  std::string data;

  for (; it < end; ++it) {
    std::string_view line = *it;

    if (!absl::StartsWith(line, ">")) {
      ABSL_LOG(INFO) << "Skipping unknown line: " << line;
      continue;
    }

    header.clear();
    bool parser_ok =
        x3::parse(line.begin(), line.end(), parser::sdf_data_header, header);

    auto kit = absl::c_find_if(header,
                               [](const std::string &s) { return !s.empty(); });

    if (parser_ok && kit != header.end()) {
      key = *kit;
    } else {
      ABSL_LOG(INFO) << "Unparseable data header: " << line;
      key = absl::StripAsciiWhitespace(line.substr(1));
    }

    data.clear();
    for (; ++it < end;) {
      if (it->empty())
        break;

      absl::StrAppend(&data, *it, "\n");
    }

    if (!data.empty())
      data.pop_back();

    mol.add_prop(std::string(key), data);

    if (it == end)
      break;
  }
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
  if (line[0] == 'M')
    return ParseLineResult::kProp;
  if (line[0] == '>')
    return ParseLineResult::kExtra;

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
  unsigned int bt;
  if (!absl::SimpleAtoi(line.substr(6, 3), &bt)) {
    ABSL_LOG_IF(WARNING, bond_context) << "Failed to parse bond order";
    return ParseLineResult::kError;
  }
  if (!parse_sdf_bond(data, bt)) {
    ABSL_LOG_IF(WARNING, bond_context) << "Invalid bond order: " << bt;
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
    auto ret = try_read_v2000_bond_line(mut, line, !atom_context);

    if (ret == ParseLineResult::kError && atom_context) {
      ABSL_LOG(WARNING).AtLocation(__FILE__, lineno) << onerror;
      ABSL_LOG(INFO).AtLocation(__FILE__, lineno) << "The line is: " << line;
    }

    return ret;
  };

  // clang-format off
/*
0        1         2         3         4         5         6         7
1234567890123456789012345678901234567890123456789012345678901234567890
xxxxx.xxxxyyyyy.yyyyzzzzz.zzzz aaaddcccssshhhbbbvvvHHHrrriiimmmnnneee

aaa: symbol
dd:  mass diff
ccc: charge

(the rest are unimplemented)
*/
  // clang-format on

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
  if (!parse_sdf_atom(data, elem_upper)) {
    std::string onerror =
        absl::StrCat("Unknown element: ", slice_strip(line, 30, 34));
    return test_for_other(onerror, __LINE__);
  }

  if (int mass_diff;  //
      absl::SimpleAtoi(safe_slice(line, 34, 36), &mass_diff)
      && mass_diff != 0) {
    data.set_isotope(data.isotope().mass_number + mass_diff);
  }

  if (int charge;
      absl::SimpleAtoi(safe_slice(line, 36, 39), &charge) && charge != 0) {
    // charge == 4 -> doublet radical, unimplemented
    if (charge >= 1 && charge <= 7)
      data.set_formal_charge(4 - charge);
    else
      ABSL_LOG(WARNING) << "Ignoring unknown charge attribute: " << charge;
  }

  mut.add_atom(std::move(data));

  return ParseLineResult::kAtom;
}

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
constexpr auto v2000_property_values =  //
    *x3::omit[x3::blank] >> x3::int_ % +x3::blank;
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

std::vector<int> read_props_values_common(std::string_view data) {
  std::vector<int> values;

  if (data.size() < 8) {
    ABSL_LOG(WARNING) << "Empty data provided for property values";
    return values;
  }

  unsigned int hint = 0;
  if (absl::SimpleAtoi(data.substr(3, 3), &hint))
    values.reserve(static_cast<size_t>(hint) * 2);

  auto value_data = data.substr(7);
  if (!x3::parse(value_data.begin(), value_data.end(),
                 parser::v2000_property_values, values)) {
    ABSL_LOG(WARNING) << "Failed to parse data: " << data;
  }

  ABSL_LOG_IF(WARNING, values.size() != static_cast<size_t>(hint) * 2)
      << "Inconsistent element count: " << data;
  ABSL_LOG_IF(WARNING, values.size() % 2 != 0)
      << "Invalid data: odd number of elements: " << data;

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

ParseLineResult read_v2000_property_block(Molecule &mol,
                                          std::string_view line) {
  if (line.empty())
    return ParseLineResult::kSkip;

  if (line[0] != 'M')
    return ParseLineResult::kExtra;

  line = absl::StripAsciiWhitespace(line.substr(1));
  if (line.size() < 3) {
    ABSL_LOG(WARNING) << "Invalid property block line";
    return ParseLineResult::kSkip;
  }

  auto key = line.substr(0, 3);
  if (key == "CHG") {
    read_chg(mol, line);
  } else if (key == "ISO") {
    read_iso(mol, line);
  } else if (key != "END") {
    ABSL_LOG(INFO) << "Unimplemented property block: " << key;
    mol.add_prop(std::string(key), std::string(line.substr(3)));
  }

  return ParseLineResult::kProp;
}

bool read_v2000(Molecule &mol, std::vector<Vector3d> &coords,
                const HeaderReadResult metadata, Iterator &it,
                const Iterator end) {
  auto mut = mol.mutator();
  std::string temp;

  mol.reserve(metadata.natoms());
  coords.reserve(metadata.natoms());

  mol.reserve_bonds(metadata.nbonds());

  ParseLineResult status = ParseLineResult::kAtom;
  for (; it < end; ++it) {
    status = try_read_v2000_atom_line(
        mut, *it, mol.num_atoms() < metadata.natoms(), coords, temp);

    if (status == ParseLineResult::kError) {
      ABSL_LOG(ERROR) << "Failed to read atom block";
      return false;
    }

    if (status == ParseLineResult::kSkip) {
      ABSL_LOG(INFO) << "Skipping line: " << *it;
    } else if (status != ParseLineResult::kAtom) {
      break;
    }
  }

  ABSL_LOG_IF(WARNING, mol.num_atoms() != metadata.natoms())
      << "Inconsistent counts block and atoms block";

  // If result == kBond, bond already read in try_read_v2000_atom_line
  // Must increment first to avoid bond duplication
  for (int i = mol.num_bonds();
       status == ParseLineResult::kBond && ++it < end;) {
    status = try_read_v2000_bond_line(mut, *it, i < metadata.nbonds());

    if (status == ParseLineResult::kError) {
      ABSL_LOG(ERROR) << "Failed to read bond block";
      return false;
    }

    if (status == ParseLineResult::kSkip) {
      ABSL_LOG(INFO) << "Skipping line: " << *it;
      continue;
    }

    ++i;
  }

  ABSL_LOG_IF(WARNING, mol.num_bonds() != metadata.nbonds())
      << "Inconsistent counts block and bonds block";

  for (; status == ParseLineResult::kProp && it < end; ++it) {
    status = read_v2000_property_block(mol, *it);

    if (status == ParseLineResult::kExtra)
      break;

    if (status == ParseLineResult::kSkip) {
      ABSL_LOG(INFO) << "Skipping line: " << *it;
      status = ParseLineResult::kProp;
    }
  }

  return true;
}

constexpr const char *kV3000LineHeaderCStr = "M  V30";
constexpr std::string_view kV3000LineHeader = kV3000LineHeaderCStr;

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
constexpr auto v3000_line_header = kV3000LineHeaderCStr >> +x3::omit[x3::blank];

constexpr auto v3000_meta_line_noheader =  //
    *x3::omit[x3::blank] >> nonblank_trailing_blanks >> +~x3::blank;

constexpr auto v3000_begin_block =  //
    v3000_line_header               //
    >> "BEGIN" >> +x3::omit[x3::blank] >> +~x3::blank;
constexpr auto v3000_end_block =  //
    v3000_line_header             //
    >> "END" >> +x3::omit[x3::blank] >> +~x3::blank;

constexpr auto v3000_counts_line =  //
    v3000_line_header               //
    >> "COUNTS" >> x3::repeat(2, x3::inf)[+x3::omit[x3::blank] >> x3::int_];

constexpr auto v3000_atom_optional_params  //
    = x3::rule<struct V3000OptionalArgs,
               std::pair<std::string, std::optional<int>>>("")  //
    = +(x3::char_ - (x3::blank | '=')) >> '='
      >> (x3::int_ | x3::omit['(' >> +~x3::blank % +x3::blank >> ')']);

constexpr auto v3000_atom_line =              //
    v3000_line_header                         //
    >> uint_trailing_blanks                   //
    >> nonblank_trailing_blanks               //
    >> x3::repeat(3)[double_trailing_blanks]  //
    >> x3::int_                               //
    >> *(+x3::omit[x3::blank] >> v3000_atom_optional_params);

using AtomLine =
    std::tuple<unsigned int, std::string, absl::InlinedVector<double, 3>, int,
               std::vector<std::pair<std::string, std::optional<int>>>>;

constexpr auto v3000_bond_line =             //
    v3000_line_header >> x3::omit[x3::int_]  //
    >> x3::repeat(3)[x3::omit[+x3::blank] >> x3::uint_];
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

auto test_continuation(std::string_view line) {
  auto rit = std::find_if(line.rbegin(), line.rend(),
                          [](char c) { return !absl::ascii_isblank(c); });
  return std::make_pair(rit.base(), rit != line.rend() && *rit == '-');
}

class ContinuationReader {
public:
  std::string_view getline(Iterator &it, const Iterator end) {
    ABSL_DCHECK(it < end);

    std::string_view result = *it;

    auto [lit, cont] = test_continuation(result);
    if (!cont)
      return result;

    buffer_ = result.substr(0, lit - result.begin() - 1);

    for (bool done = false; !done && ++it < end;) {
      std::string_view line = absl::StripPrefix(*it, kV3000LineHeader);

      if (std::tie(lit, cont) = test_continuation(line); cont) {
        line = line.substr(0, lit - line.begin() - 1);
      } else {
        done = true;
      }

      absl::StrAppend(&buffer_, line);
    }

    result = buffer_;
    return result;
  }

private:
  std::string buffer_;
};

bool try_read_v3000_header(HeaderReadResult &metadata, Iterator &it,
                           const Iterator end, ContinuationReader &reader) {
  std::string_view line = reader.getline(it, end);

  std::string key;
  if (!x3::parse(line.begin(), line.end(), parser::v3000_begin_block, key)
      || key != "CTAB") {
    ABSL_LOG(WARNING) << "Invalid V3000 connection table";
    ABSL_LOG(INFO) << "The line is: " << line;
    return false;
  }

  if (++it == end) {
    ABSL_LOG(WARNING) << "No counts line found";
    return false;
  }

  line = reader.getline(it, end);
  absl::InlinedVector<unsigned int, 5> counts;
  if (!x3::parse(line.begin(), line.end(), parser::v3000_counts_line, counts)) {
    ABSL_LOG(WARNING) << "Invalid counts line";
    ABSL_LOG(INFO) << "The line is: " << line;
    return false;
  }

  ABSL_DCHECK(counts.size() >= 2);

  metadata = HeaderReadResult::success(3000, static_cast<int>(counts[0]),
                                       static_cast<int>(counts[1]));
  return true;
}

bool try_read_v3000_atom_block(MoleculeMutator &mut,
                               std::vector<Vector3d> &coords,
                               const HeaderReadResult metadata, Iterator &it,
                               const Iterator end, ContinuationReader &reader,
                               std::string &key) {
  if (++it >= end) {
    ABSL_LOG(WARNING) << "Missing atom block";
    return false;
  }

  std::string_view line = reader.getline(it, end);
  if (!x3::parse(line.begin(), line.end(), parser::v3000_begin_block, key)
      || key != "ATOM") {
    ABSL_LOG(WARNING) << "Missing atom block";
    ABSL_LOG(INFO) << "The line is: " << line;
    return false;
  }

  parser::AtomLine parsed;
  for (; ++it < end;) {
    line = reader.getline(it, end);

    std::get<1>(parsed).clear();
    std::get<2>(parsed).clear();
    std::get<4>(parsed).clear();

    if (!x3::parse(line.begin(), line.end(), parser::v3000_atom_line, parsed))
      break;

    AtomData data;
    if (!parse_sdf_atom(data, std::get<1>(parsed)))
      break;

    coords.emplace_back(std::get<2>(parsed)[0], std::get<2>(parsed)[1],
                        std::get<2>(parsed)[2]);

    for (const auto &[prop, value]: std::get<4>(parsed)) {
      if (!value) {
        ABSL_LOG(INFO)
            << "Invalid value or unimplemented atom property: " << prop;
        continue;
      }

      if (prop == "CHG") {
        data.set_formal_charge(*value);
      } else if (prop == "MASS") {
        const Isotope *iso = data.element().find_isotope(*value);
        if (iso == nullptr) {
          ABSL_LOG(WARNING) << "Isotope not found: " << *value;
          continue;
        }
        data.set_isotope(*iso);
      } else {
        ABSL_LOG(INFO) << "Unimplemented atom property: " << prop;
      }
    }

    mut.add_atom(std::move(data));
  }

  if (it == end) {
    ABSL_LOG(WARNING) << "Atom block ended before END ATOM directive";
    return true;
  }

  key.clear();
  if (x3::parse(line.begin(), line.end(), parser::v3000_end_block, key)) {
    ++it;

    if (key != "ATOM") {
      ABSL_LOG(WARNING) << "Invalid atom block";
      ABSL_LOG(INFO) << "The line is: " << line;
      return false;
    }
  }

  ABSL_LOG_IF(WARNING, mut.mol().num_atoms() != metadata.natoms())
      << "Inconsistent counts block and atoms block";

  return true;
}

bool try_read_v3000_bond_block(MoleculeMutator &mut,
                               const HeaderReadResult metadata, Iterator &it,
                               const Iterator end, ContinuationReader &reader) {
  std::string_view line;

  absl::InlinedVector<unsigned int, 3> parsed;
  for (; ++it < end;) {
    line = reader.getline(it, end);

    parsed.clear();
    if (!x3::parse(line.begin(), line.end(), parser::v3000_bond_line, parsed))
      break;

    ABSL_DCHECK(parsed.size() == 3);

    int src = static_cast<int>(--parsed[1]),
        dst = static_cast<int>(--parsed[2]);
    if (src >= mut.mol().num_atoms() || dst >= mut.mol().num_atoms()
        || src == dst) {
      ABSL_LOG(WARNING) << "Invalid bond indices: " << src << " - " << dst;
      return false;
    }

    BondData data;
    if (!parse_sdf_bond(data, parsed[0])) {
      ABSL_LOG(WARNING) << "Invalid bond order: " << parsed[2];
      return false;
    }

    auto [_, added] = mut.add_bond(src, dst, std::move(data));
    ABSL_LOG_IF(WARNING, !added) << "Duplicate bond " << src << " - " << dst;
  }

  if (it == end) {
    ABSL_LOG(WARNING) << "Bond block ended before END BOND directive";
    return true;
  }

  std::string key;
  if (x3::parse(line.begin(), line.end(), parser::v3000_end_block, key)
      && key != "BOND") {
    ABSL_LOG(WARNING) << "Invalid bond block";
    ABSL_LOG(INFO) << "The line is: " << line;
    return false;
  }

  ABSL_LOG_IF(WARNING, mut.mol().num_bonds() != metadata.nbonds())
      << "Inconsistent counts block and bonds block";

  return true;
}

bool try_read_v3000_optionals(MoleculeMutator &mut,
                              const HeaderReadResult metadata, Iterator &it,
                              const Iterator end, ContinuationReader &reader) {
  std::stack<std::string, std::vector<std::string>> tokens;
  tokens.push("CTAB");

  std::pair<std::string, std::string> parsed;

  for (; it < end && !tokens.empty(); ++it) {
    std::string_view line = reader.getline(it, end);
    if (!absl::StartsWith(line, kV3000LineHeader))
      break;

    line = line.substr(kV3000LineHeader.size());
    parsed.first.clear();
    parsed.second.clear();
    if (!x3::parse(line.begin(), line.end(), parser::v3000_meta_line_noheader,
                   parsed))
      continue;

    if (parsed.first == "BEGIN") {
      if (parsed.second != "BOND") {
        tokens.push(parsed.second);
        continue;
      }

      if (!try_read_v3000_bond_block(mut, metadata, it, end, reader)) {
        ABSL_LOG(ERROR) << "Failed to read V3000 bond block";
        return false;
      }

      if (it == end) {
        ABSL_LOG(WARNING) << "Block ended before END directive";
        break;
      }
    } else if (parsed.first == "END") {
      if (tokens.top() != parsed.second) {
        ABSL_LOG(ERROR) << "Mismatched V3000 END directive: " << parsed.second
                        << " != " << tokens.top();
        return false;
      }
      tokens.pop();
    }
  }

  ABSL_LOG_IF(WARNING, !tokens.empty())
      << "V3000 connection table ended before END CTAB directive";

  return true;
}

void skip_v3000_unknowns(Iterator &it, const Iterator end,
                         ContinuationReader &reader, std::string &key) {
  if (it == end)
    return;

  for (; ++it < end;) {
    std::string_view line = reader.getline(it, end);

    if (!absl::StartsWith(line, "M"))
      break;

    key.clear();
    if (absl::StartsWith(line, "M")
        && absl::StripAsciiWhitespace(line.substr(1)) == "END")
      break;

    ABSL_LOG(INFO) << "Ignoring unknown line: " << line;
  }
}

bool read_v3000(Molecule &mol, std::vector<Vector3d> &coords,
                HeaderReadResult metadata, Iterator &it, const Iterator end) {
  ContinuationReader reader;
  std::string key;

  if (!try_read_v3000_header(metadata, it, end, reader)) {
    ABSL_LOG(ERROR) << "Failed to read V3000 header";
    return false;
  }

  mol.reserve(metadata.natoms());
  coords.reserve(metadata.natoms());

  mol.reserve_bonds(metadata.nbonds());

  auto mut = mol.mutator();

  if (!try_read_v3000_atom_block(mut, coords, metadata, it, end, reader, key)) {
    ABSL_LOG(ERROR) << "Failed to read V3000 atom block";
    return false;
  }

  if (it == end) {
    ABSL_LOG(WARNING)
        << "V3000 connection table ended before END CTAB directive";
    return true;
  }

  if (!try_read_v3000_optionals(mut, metadata, it, end, reader)) {
    ABSL_LOG(ERROR) << "Failed to read V3000 optional blocks";
    return false;
  }

  skip_v3000_unknowns(it, end, reader, key);

  return true;
}
}  // namespace

Molecule read_sdf(const std::vector<std::string> &sdf) {
  Molecule mol;
  std::vector<Vector3d> coords;

  auto it = sdf.begin();
  const auto end = sdf.end();

  auto metadata = read_sdf_header(mol, it, end);
  if (!metadata) {
    ABSL_LOG(ERROR) << "Failed to read SDF header";
    mol.clear();
    return mol;
  }

  if (++it == end) {
    ABSL_LOG(ERROR) << "No atom block found";
    mol.clear();
    return mol;
  }

  bool ok = false;
  if (metadata.version() == 2000) {
    ok = read_v2000(mol, coords, metadata, it, end);
  } else if (metadata.version() == 3000) {
    ok = read_v3000(mol, coords, metadata, it, end);
  } else {
    ABSL_LOG(ERROR) << "Unknown SDF version: " << metadata.version();
  }

  if (!ok) {
    mol.clear();
    return mol;
  }

  read_sdf_extra(mol, it, end);

  bool has_hydrogen = absl::c_any_of(mol, [](Molecule::Atom atom) {
    return atom.data().atomic_number() == 1;
  });
  bool has_fcharge = absl::c_any_of(mol, [](Molecule::Atom atom) {
    return atom.data().formal_charge() != 0;
  });

  if (!has_hydrogen && !has_fcharge) {
    guess_fcharge_hydrogens_2d(mol);
  } else if (!has_hydrogen) {
    guess_hydrogens_2d(mol);
  } else if (!has_fcharge) {
    guess_fcharge_2d(mol);
  }

  mol.confs().emplace_back(stack(coords));

  return mol;
}
}  // namespace nuri
