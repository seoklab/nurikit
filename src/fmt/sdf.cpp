//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/sdf.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stack>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/container/inlined_vector.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/charset.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/strip.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/fusion/include/std_tuple.hpp>
#include <boost/spirit/home/x3.hpp>

#include "nuri/eigen_config.h"
#include "fmt_internal.h"
#include "nuri/algo/guess.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/core/property_map.h"
#include "nuri/fmt/base.h"
#include "nuri/meta.h"
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
  static HeaderReadResult failure() { return {}; }

  template <class N1, class N2, class N3>
  static HeaderReadResult success(N1 version, N2 natoms, N3 nbonds) {
    return { static_cast<int>(version), static_cast<int>(natoms),
             static_cast<int>(nbonds) };
  }

  int version() const { return version_; }

  int natoms() const { return natoms_; }

  int nbonds() const { return nbonds_; }

  operator bool() const { return version_ >= 0; }

private:
  // Clang analyzer complains about uninitialized members
  NURI_CLANG_ANALYZER_NOLINT HeaderReadResult(): version_(-1) { }

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

  if (auto stamp = absl::StripTrailingAsciiWhitespace(*it++); !stamp.empty())
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

  unsigned int natoms, nbonds;
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
             (sdf_data_header_key | +x3::omit[~x3::blank]))
    >> x3::omit[x3::space | x3::eoi];
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
    *x3::omit[x3::blank] >> x3::int_ % +x3::blank
    >> x3::omit[x3::space | x3::eoi];
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
    >> "COUNTS" >> x3::repeat(2, x3::inf)[+x3::omit[x3::blank] >> x3::uint_]
    >> x3::omit[x3::space | x3::eoi];

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
    >> *(+x3::omit[x3::blank] >> v3000_atom_optional_params)
    >> x3::omit[x3::space | x3::eoi];

using AtomLine =
    std::tuple<unsigned int, std::string, absl::InlinedVector<double, 3>, int,
               std::vector<std::pair<std::string, std::optional<int>>>>;

constexpr auto v3000_bond_line =             //
    v3000_line_header >> x3::omit[x3::int_]  //
    >> x3::repeat(3)[x3::omit[+x3::blank] >> x3::uint_]
    >> x3::omit[x3::space | x3::eoi];
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

  if (++it >= end) {
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

  metadata = HeaderReadResult::success(3000, counts[0], counts[1]);
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

    int src = static_cast<int>(parsed[1]) - 1,
        dst = static_cast<int>(parsed[2]) - 1;
    if (ABSL_PREDICT_FALSE(src >= mut.mol().num_atoms()
                           || dst >= mut.mol().num_atoms()  //
                           || src == dst || src < 0 || dst < 0)) {
      ABSL_LOG(WARNING) << "Invalid bond indices: " << src << " - " << dst;
      return false;
    }

    BondData data;
    if (!parse_sdf_bond(data, parsed[0])) {
      ABSL_LOG(WARNING) << "Invalid bond order: " << parsed[0];
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

  if (++it >= end) {
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

  if (!has_hydrogen)
    guess_hydrogens_2d(mol);

  mol.confs().emplace_back(stack(coords));

  return mol;
}

namespace {
int sdf_bond_order(constants::BondOrder order) {
  if (order == constants::kAromaticBond)
    return 4;

  return nuri::clamp(static_cast<int>(order), 1, 3);
}

constexpr const absl::CharSet kNewlines("\f\n\r");

std::string sdf_props_safe_key(std::string_view key) {
  std::string result;
  result.reserve(key.size());

  absl::CharSet metachars("<>");

  for (int i = 0; i < key.size(); ++i) {
    char c = key[i];

    if (absl::ascii_isspace(key[i])) {
      c = ' ';
    } else if (metachars.contains(key[i]) || !absl::ascii_isprint(key[i])) {
      c = '?';
    }

    result.push_back(c);
  }

  return result;
}

std::string sdf_props_safe_value(std::string_view value) {
  std::string result;
  result.reserve(value.size());

  bool newline = true;
  for (int i = 0; i < value.size(); ++i) {
    char c = value[i];

    if (newline) {
      newline = kNewlines.contains(c);
      if (newline)
        continue;

      c = c == '$' ? '?' : c;
    } else if (c == '\n') {
      newline = true;
    }

    result.push_back(c);
  }

  if (absl::EndsWith(result, "\n"))
    result.pop_back();

  return result;
}

void sdf_gen_props(std::string &footer, const Molecule &mol) {
  for (const auto &[key, value]: mol.props()) {
    absl::StrAppend(&footer, "> <", sdf_props_safe_key(key), ">\n",
                    sdf_props_safe_value(value), "\n\n");
  }
}

bool v2000_can_write_coords(const Matrix3Xd &coords) {
  bool underflow = (coords.array() <= -1e+4).any(),
       overflow = (coords.array() >= 1e+5).any();
  return !underflow && !overflow;
}

bool v2000_can_write(const Molecule &mol, int conf, bool log_on_fail) {
  if (mol.num_atoms() > 999 || mol.num_bonds() > 999) {
    ABSL_LOG_IF(ERROR, log_on_fail)
        << "V2000 format cannot handle more than 999 atoms or bonds";
    return false;
  }

  if (!mol.is_3d())
    return true;

  bool coords_ok = true;
  if (conf < 0) {
    for (const auto &c: mol.confs()) {
      if (!v2000_can_write_coords(c)) {
        coords_ok = false;
        break;
      }
    }
  } else {
    ABSL_DCHECK_LT(conf, mol.confs().size());
    coords_ok = v2000_can_write_coords(mol.confs()[conf]);
  }

  ABSL_LOG_IF(ERROR, !coords_ok && log_on_fail)
      << "V2000 format cannot handle coordinates outside of (-1e+4, 1e+5)";

  return coords_ok;
}

std::string v2000_name(std::string_view name) {
  std::string safe_name = internal::ascii_safe(name.substr(0, 80));
  if (absl::StartsWith(safe_name, "$"))
    safe_name[0] = '?';
  return safe_name;
}

int v2000_gen_mass_diff(const AtomData &data) {
  const Element &el = data.element();
  const Isotope &major = el.major_isotope();

  int diff = data.isotope().mass_number - major.mass_number;
  if (diff < -3 || diff > 4)
    return 0;
  return diff;
}

int v2000_gen_formal_charge(int fchg) {
  int chg_annot = 4 - fchg;
  if (chg_annot < 1 || chg_annot == 4 || chg_annot > 7)
    return 0;
  return chg_annot;
}

template <bool is_3d>
void v2000_write_atoms(std::string &out, const Molecule &mol, int conf) {
  Vector3d pos;
  if constexpr (!is_3d)
    pos.setZero();

  for (auto atom: mol) {
    if constexpr (is_3d)
      pos = mol.confs()[conf].col(atom.id());

    absl::StrAppendFormat(
        &out,
        // clang-format off
        //                            ssshhhbbbvvvHHHrrriiimmmnnneee
        "%10.4f%10.4f%10.4f %-3s%2d%3d  0  0  0  0  0  0  0  0  0  0\n",
        // clang-format on
        pos.x(), pos.y(), pos.z(), atom.data().element().symbol(),
        v2000_gen_mass_diff(atom.data()),
        v2000_gen_formal_charge(atom.data().formal_charge()));
  }
}

void v2000_write_bonds(std::string &out, const Molecule &mol) {
  for (auto bond: mol.bonds()) {
    absl::StrAppendFormat(  //
        &out,
        // clang-format off
        //        sssxxxrrrccc
        "%3d%3d%3d  0  0  0  0\n",
        // clang-format on
        bond.src().id() + 1, bond.dst().id() + 1,
        sdf_bond_order(bond.data().order()));
  }
}

void v2000_atom_properties_common(std::string &out, std::string_view key,
                                  const std::vector<std::pair<int, int>> &data) {
  for (int i = 0; i < data.size(); i += 8) {
    int cnt = nuri::min(8, static_cast<int>(data.size()) - i);
    absl::StrAppendFormat(&out, "M  %s%3d", key, cnt);
    for (int j = 0; j < cnt; ++j)
      absl::StrAppendFormat(&out, " %3d %3d", data[i + j].first,
                            data[i + j].second);
    out.push_back('\n');
  }
}

template <bool is_3d>
void v2000_write_single_conf(std::string &out, const Molecule &mol, int conf,
                             std::string_view header, std::string_view footer) {
  if constexpr (is_3d) {
    // NOLINTNEXTLINE(clang-diagnostic-used-but-marked-unused)
    ABSL_DCHECK_LT(conf, mol.confs().size());
  }

  absl::StrAppendFormat(&out, "%s\n%3d%3d  0  0  0  0  0  0  0  0999 V2000\n",
                        header, mol.num_atoms(), mol.num_bonds());
  v2000_write_atoms<is_3d>(out, mol, conf);
  v2000_write_bonds(out, mol);

  std::vector<std::pair<int, int>> props;

  for (auto atom: mol) {
    if (atom.data().formal_charge() != 0)
      props.emplace_back(atom.id() + 1, atom.data().formal_charge());
  }
  v2000_atom_properties_common(out, "CHG", props);

  props.clear();
  for (auto atom: mol) {
    if (auto iso = atom.data().explicit_isotope(); iso != nullptr)
      props.emplace_back(atom.id() + 1, iso->mass_number);
  }
  v2000_atom_properties_common(out, "ISO", props);

  absl::StrAppend(&out, footer);
}

void v2000_write_mol(std::string &out, const Molecule &mol, int conf,
                     std::string_view header, std::string_view footer) {
  if (!mol.is_3d()) {
    v2000_write_single_conf<false>(out, mol, -1, header, footer);
  } else if (conf < 0) {
    for (int i = 0; i < mol.confs().size(); ++i) {
      v2000_write_single_conf<true>(out, mol, i, header, footer);
    }
  } else {
    v2000_write_single_conf<true>(out, mol, conf, header, footer);
  }
}

template <bool is_3d>
void v3000_write_atoms(std::string &out, const Molecule &mol, int conf) {
  absl::StrAppend(&out, "M  V30 BEGIN ATOM\n");

  Vector3d pos;
  if constexpr (!is_3d)
    pos.setZero();

  for (auto atom: mol) {
    if constexpr (is_3d)
      pos = mol.confs()[conf].col(atom.id());

    absl::StrAppendFormat(&out,                                           //
                          "M  V30 %d %s %.4f %.4f %.4f 0",                //
                          atom.id() + 1, atom.data().element().symbol(),  //
                          pos.x(), pos.y(), pos.z());

    if (atom.data().formal_charge() != 0)
      absl::StrAppendFormat(&out, " CHG=%d", atom.data().formal_charge());

    if (auto iso = atom.data().explicit_isotope(); iso != nullptr)
      absl::StrAppendFormat(&out, " MASS=%d", iso->mass_number);

    out.push_back('\n');
  }

  absl::StrAppend(&out, "M  V30 END ATOM\n");
}

void v3000_write_bonds(std::string &out, const Molecule &mol) {
  absl::StrAppend(&out, "M  V30 BEGIN BOND\n");

  for (auto bond: mol.bonds()) {
    absl::StrAppendFormat(  //
        &out,
        // clang-format off
        //        sssxxxrrrccc
        "M  V30 %d %d %d %d\n",
        // clang-format on
        bond.id() + 1, sdf_bond_order(bond.data().order()),  //
        bond.src().id() + 1, bond.dst().id() + 1);
  }

  absl::StrAppend(&out, "M  V30 END BOND\n");
}

template <bool is_3d>
void v3000_write_single_conf(std::string &out, const Molecule &mol, int conf,
                             std::string_view header, std::string_view footer) {
  if constexpr (is_3d) {
    // NOLINTNEXTLINE(clang-diagnostic-used-but-marked-unused)
    ABSL_DCHECK_LT(conf, mol.confs().size());
  }

  absl::StrAppendFormat(&out,
                        R"sdf(%s
  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS %d %d 0 0 0
)sdf",
                        header, mol.num_atoms(), mol.num_bonds());

  v3000_write_atoms<is_3d>(out, mol, conf);
  v3000_write_bonds(out, mol);

  absl::StrAppend(&out, "M  V30 END CTAB\n", footer);
}

void v3000_write_mol(std::string &out, const Molecule &mol, int conf,
                     std::string_view header, std::string_view footer) {
  if (!mol.is_3d()) {
    v3000_write_single_conf<false>(out, mol, -1, header, footer);
  } else if (conf < 0) {
    for (int i = 0; i < mol.confs().size(); ++i) {
      v3000_write_single_conf<true>(out, mol, i, header, footer);
    }
  } else {
    v3000_write_single_conf<true>(out, mol, conf, header, footer);
  }
}
}  // namespace

bool write_sdf(std::string &out, const Molecule &mol, int conf,
               SDFVersion ver) {
  bool can_v2000 = false;
  if (ver != SDFVersion::kV3000)
    can_v2000 = v2000_can_write(mol, conf, ver == SDFVersion::kV2000);

  if (ver == SDFVersion::kV2000 && !can_v2000)
    return false;

  if (ver == SDFVersion::kAutomatic)
    ver = can_v2000 ? SDFVersion::kV2000 : SDFVersion::kV3000;

  std::string header = absl::StrFormat(
      "%s\n   NuriKit%s%s\n%s",                                          //
      v2000_name(mol.name()),                                            //
      absl::FormatTime("%m%d%y%H%M", absl::Now(), absl::UTCTimeZone()),  //
      mol.is_3d() ? "3D" : "2D",                                         //
      absl::StrReplaceAll(internal::get_key(mol.props(), "comment"),
                          // clang-format off
                          { { "\n", " " } })
      // clang-format on
  );

  std::string footer = "M  END\n";
  sdf_gen_props(footer, mol);
  absl::StrAppend(&footer, "$$$$\n");

  if (ver == SDFVersion::kV2000) {
    v2000_write_mol(out, mol, conf, header, footer);
  } else if (ver == SDFVersion::kV3000) {  // GCOV_EXCL_BR_LINE
    v3000_write_mol(out, mol, conf, header, footer);
  } else {
    ABSL_UNREACHABLE();  // GCOV_EXCL_LINE
  }

  return true;
}
}  // namespace nuri
