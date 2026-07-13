//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/cif.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/charset.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/x3.hpp>

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
// GCOV_EXCL_START
std::ostream &operator<<(std::ostream &os, CifToken type) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (type) {
  case nuri::internal::CifToken::kEOF:
    os << "EOF";
    break;
  case nuri::internal::CifToken::kError:
    os << "Error";
    break;
  case nuri::internal::CifToken::kData:
    os << "Data";
    break;
  case nuri::internal::CifToken::kLoop:
    os << "Loop";
    break;
  case nuri::internal::CifToken::kGlobal:
    os << "Global";
    break;
  case nuri::internal::CifToken::kSave:
    os << "Save";
    break;
  case nuri::internal::CifToken::kStop:
    os << "Stop";
    break;
  case nuri::internal::CifToken::kTag:
    os << "Tag";
    break;
  case nuri::internal::CifToken::kSimpleValue:
    os << "Value";
    break;
  case nuri::internal::CifToken::kQuotedValue:
    os << "QuotedValue";
    break;
  default:
    os << "Unknown";
    break;
  }

  return os;
}
// GCOV_EXCL_STOP
}  // namespace internal

namespace {
namespace x3 = boost::spirit::x3;

using internal::CifToken;
using internal::SIter;

// NOLINTBEGIN(readability-identifier-naming)
namespace parser {
// terminals
constexpr auto non_blank_char = x3::char_ - (' ' | x3::cntrl);
constexpr auto cif_delim = x3::space | x3::eoi;

constexpr auto single_quote_end = x3::raw['\'' >> cif_delim];
constexpr auto double_quote_end = x3::raw['"' >> cif_delim];

constexpr auto tag = x3::raw['_' >> +non_blank_char] >> x3::omit[cif_delim];

const struct simple_keyword_heading_: public x3::symbols<CifToken> {
  simple_keyword_heading_() {
    add("loop_", CifToken::kLoop)       //
        ("global_", CifToken::kGlobal)  //
        ("stop_", CifToken::kStop)      //
        ;
  }
} simple_keyword_heading;

const auto simple_keyword = x3::no_case[simple_keyword_heading]
                            >> x3::raw[x3::eps];

const struct valued_keyword_heading_: public x3::symbols<CifToken> {
  valued_keyword_heading_() {
    add("data_", CifToken::kData)   //
        ("save_", CifToken::kSave)  //
        ;
  }
} valued_keyword_heading;

const auto valued_keyword = x3::no_case[valued_keyword_heading]
                            >> x3::raw[*non_blank_char];

const auto keyword = (simple_keyword | valued_keyword) >> x3::omit[cif_delim];

// CIF 1.1 numeric grammar (spec productions Number/Numeric). A number may carry
// a standard uncertainty, e.g. 1.234(5). x3::eoi anchors a full-string match.
constexpr auto num_sign = -(x3::char_('+') | x3::char_('-'));
constexpr auto num_uint = +x3::digit;
constexpr auto num_exp = (x3::char_('e') | x3::char_('E')) >> num_sign
                         >> num_uint;
constexpr auto num_integer = num_sign >> num_uint;
constexpr auto num_decimal = num_sign
                             >> (*x3::digit >> x3::char_('.') >> num_uint
                                 | +x3::digit >> x3::char_('.'));
constexpr auto num_float = num_integer >> num_exp | num_decimal >> -num_exp;
constexpr auto numeric = (num_float | num_integer)
                         >> -(x3::char_('(') >> num_uint >> x3::char_(')'))
                         >> x3::eoi;
}  // namespace parser
// NOLINTEND(readability-identifier-naming)

// all control characters + space + reserved characters ($_[])
// ';' will be handled separately
constexpr absl::CharSet kSpecialChars = absl::CharSet::Range(0, 32)
                                        | absl::CharSet::Char(127)
                                        | absl::CharSet("$_[]");

std::string_view as_sv(SIter begin, SIter end) {
  return { &*begin, static_cast<size_t>(end - begin) };
}

std::string_view as_sv(boost::iterator_range<SIter> range) {
  return { &*range.begin(), range.size() };
}
}  // namespace

namespace internal {
namespace {
using QuoteEnd = decltype(parser::single_quote_end);
static_assert(std::is_same_v<QuoteEnd, decltype(parser::double_quote_end)>);

std::pair<std::string_view, CifToken> produce_quoted(CifLexer &lexer,
                                                     const QuoteEnd &parser) {
  for (auto qit = lexer.p() + 1; qit < lexer.end(); ++qit) {
    boost::iterator_range<SIter> quote_match;
    if (!x3::parse(qit, lexer.end(), parser, quote_match))
      continue;

    return lexer.produce(as_sv(lexer.p() + 1, quote_match.begin()),
                         CifToken::kQuotedValue, quote_match.end());
  }

  return lexer.error("Unterminated quote at line ", lexer.row());
}

template <bool kContinuation>
std::pair<std::string_view, CifToken>
produce_text_field_impl(CifLexer &lexer, std::string &buf) {
  const size_t begin_row = lexer.row();

  bool left, cont = kContinuation;
  while ((left = lexer.advance_line<false>())) {
    if (lexer.p() < lexer.end() && lexer.c() == ';')
      break;

    std::string_view sep = "\n";
    if constexpr (kContinuation) {
      if (cont) {
        buf.pop_back();
        sep = "";
      }
    }

    std::string_view line = lexer.line();
    if constexpr (kContinuation) {
      std::string_view stripped = absl::StripTrailingAsciiWhitespace(line);
      cont = absl::EndsWith(stripped, "\\");
      if (cont)
        line = stripped;
    }

    absl::StrAppend(&buf, sep, line);
  }
  if (!left)
    return lexer.error("Unterminated text field (started at line ", begin_row,
                       ")");

  if constexpr (kContinuation) {
    if (cont)
      buf.pop_back();
  }

  auto it = lexer.p() + 1;
  ABSL_LOG_IF(WARNING, it < lexer.end() && absl::ascii_isspace(*it) == 0)
      << "Missing whitespace after text field at line "  //
      << lexer.row() << ":" << lexer.col() + 1;
  return lexer.produce(buf, CifToken::kQuotedValue, it);
}

std::pair<std::string_view, CifToken> produce_text_field(CifLexer &lexer,
                                                         std::string &buf) {
  // Preserve trailing whitespace of the content, but detect the line-folding
  // marker (`\` as the only non-blank character) on the stripped first line.
  std::string_view first = slice(lexer.line(), lexer.p() - lexer.begin() + 1,
                                 lexer.end() - lexer.begin());
  if (absl::StripTrailingAsciiWhitespace(first) == "\\") {
    buf = "\\";
    return produce_text_field_impl<true>(lexer, buf);
  }

  buf = first;
  return produce_text_field_impl<false>(lexer, buf);
}
}  // namespace

std::pair<std::string_view, CifToken> CifLexer::next() {
  while (p() < end() || advance_line<true>()) {
    ABSL_DCHECK(p() < end());

    if (p() == begin() && c() == ';')
      return produce_text_field(*this, buf_);

    it_ = std::find_if_not(p(), end(), absl::ascii_isspace);
    if (it_ == end())
      continue;

    boost::iterator_range<SIter> tag_match;
    if (x3::parse(p(), end(), parser::tag, tag_match))
      return produce(as_sv(tag_match), CifToken::kTag, tag_match.end());

    std::pair<CifToken, boost::iterator_range<SIter>> kw_match;
    if (x3::parse(p(), end(), parser::keyword, kw_match))
      return produce(as_sv(kw_match.second), kw_match.first,
                     kw_match.second.end());

    switch (c()) {
    case '#':
      it_ = end();
      continue;
    case '\'':
      return produce_quoted(*this, parser::single_quote_end);
    case '"':
      return produce_quoted(*this, parser::double_quote_end);
    default:
      break;
    }

    if (kSpecialChars.contains(c())) {
      return error("Unexpected special character at line ", row(), ":", col(),
                   ": ", std::string_view(&*p(), 1));
    }

    auto vit = std::find_if(p(), end(), absl::ascii_isspace);
    return produce(as_sv(p(), vit), CifToken::kSimpleValue, vit);
  }

  return done();
}

bool CifValue::operator==(std::string_view other) const {
  return value_ == other && (type_ == Type::kString || type_ == Type::kGeneric);
}

// GCOV_EXCL_START
std::ostream &operator<<(std::ostream &os, const CifValue &value) {
  using ValueType = CifValue::Type;

  switch (value.type()) {
  case ValueType::kGeneric:
    return os << value.value();
  case ValueType::kString:
    return os << '"' << value.value() << '"';
  case ValueType::kUnknown:
    return os << '?';
  case ValueType::kInapplicable:
    return os << '.';
  case ValueType::kFloatNonfinite:
    return os << "<non-finite: " << value.value() << '>';
  }
  ABSL_UNREACHABLE();
}
// GCOV_EXCL_STOP

CifValue cif_float_nonfinite(double value, bool coerce_nonfinite, bool is_unk) {
  if (!coerce_nonfinite)
    return CifValue::float_nonfinite(absl::StrCat(value));

  if (std::isnan(value))
    return is_unk ? CifValue::unknown() : CifValue::inapplicable();

  return CifValue::generic(
      absl::StrCat(value > 0 ? std::numeric_limits<double>::max()
                             : std::numeric_limits<double>::lowest()));
}

void CifTable::add_data(CifValue &&value) {
  if (rows_.empty() || rows_.back().size() == keys_.size())
    rows_.emplace_back().reserve(keys_.size());

  rows_.back().push_back(std::move(value));
}

namespace {
// A valid CIF data name is ``_`` followed by one or more non-blank printable
// ASCII characters (no whitespace, control, or non-ASCII bytes).
bool is_valid_cif_key(std::string_view key) {
  return key.size() >= 2 && key[0] == '_'
         && absl::c_all_of(key, absl::ascii_isgraph);
}
}  // namespace

std::string CifTable::validate() const {
  absl::flat_hash_set<std::string_view> seen;
  for (const std::string &key: keys_) {
    if (!is_valid_cif_key(key))
      return absl::StrCat("Invalid CIF key: ", key);
    if (!seen.insert(key).second)
      return absl::StrCat("Duplicate CIF key: ", key);
  }

  for (const std::vector<CifValue> &row: rows_) {
    if (row.size() != keys_.size())
      return "Inconsistent table size";
  }

  return "";
}

CifFrame::CifFrame(std::vector<CifTable> &&tables, std::string &&name)
    : name_(std::move(name)), tables_(std::move(tables)) {
  for (int i = 0; i < tables_.size(); ++i) {
    for (int j = 0; j < tables_[i].cols(); ++j) {
      // Keep the first occurrence on duplicate keys; validate() reports it.
      index_.insert({
          tables_[i].keys()[j],
          { i, j },
      });
    }
  }
}

std::string CifFrame::validate(bool recursive) const {
  // Each table is internally consistent (valid, unique keys; one value per
  // key).
  if (recursive) {
    for (const CifTable &table: tables_) {
      std::string err = table.validate();
      if (!err.empty())
        return err;
    }
  }

  // Keys must also be unique across the whole frame.
  for (int i = 0; i < tables_.size(); ++i) {
    for (int j = 0; j < tables_[i].cols(); ++j) {
      auto [iref, jref] = find(tables_[i].keys()[j]);
      if (iref != i || jref != j)
        return absl::StrCat("Duplicate or missing key in index: ",
                            tables_[i].keys()[j]);
    }
  }

  return "";
}

std::string CifBlock::validate(bool recursive) const {
  // The eof/error sentinels carry no data frame to validate.
  if (!*this)
    return "";

  if (recursive) {
    std::string err = frame_.validate();
    if (!err.empty())
      return err;
  }

  absl::flat_hash_set<std::string_view> seen;
  for (const CifFrame &save: save_) {
    if (recursive) {
      std::string err = save.validate();
      if (!err.empty())
        return absl::StrCat(err, " in save frame ", save.name());
    }

    if (save.name().empty())
      return "Save frame with empty name";
    if (!seen.insert(save.name()).second)
      return absl::StrCat("Duplicate save frame name: ", save.name());
  }

  return "";
}

namespace {
using BlockType = CifBlock::Type;

enum class CifParseCtx {
  kNonLoop,

  kLoopTag,
  kLoopValue,
};

template <CifGlobalCtx kGCtx>
std::pair<std::string_view, CifToken>
parse_data_impl(std::vector<CifTable> &tables, CifLexer &lexer,
                const std::string_view name) {
  CifParseCtx ctx = CifParseCtx::kNonLoop;

  while (true) {
    auto [data, type] = lexer.next();

    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (type) {
    case CifToken::kEOF:
    case CifToken::kError:
    case CifToken::kData:
    case CifToken::kGlobal:
      return { data, type };
    case CifToken::kSave:
      if (kGCtx == CifGlobalCtx::kBlock || data.empty())
        return { data, type };

      return lexer.error("Nested save block ", data, " in ", name);
    case CifToken::kStop:
      ABSL_LOG(WARNING) << "Stop tag is unimplemented";
      break;
    case CifToken::kLoop:
      tables.emplace_back();
      ctx = CifParseCtx::kLoopTag;
      break;
    case CifToken::kTag:
      // loop ended with new tag, or block starts with non-loop tag
      if (ctx == CifParseCtx::kLoopValue || tables.empty()) {
        if (ctx == CifParseCtx::kLoopValue)
          ctx = CifParseCtx::kNonLoop;
        tables.emplace_back();
      }
      tables.back().add_key(data);
      break;
    default:
      ABSL_DCHECK(is_value_token(type)) << "Unexpected token: " << type;

      if (tables.empty() || tables.back().keys().empty())
        return lexer.error("Unexpected value token ", data, " in ", name);

      if (ctx == CifParseCtx::kLoopTag)
        ctx = CifParseCtx::kLoopValue;

      tables.back().add_data({ data, type });
      break;
    }
  }
}
}  // namespace

std::pair<std::string_view, CifToken> parse_data(CifGlobalCtx ctx,
                                                 std::vector<CifTable> &tables,
                                                 CifLexer &lexer,
                                                 std::string_view name) {
  if (ctx == CifGlobalCtx::kBlock)
    return parse_data_impl<CifGlobalCtx::kBlock>(tables, lexer, name);
  if (ctx == CifGlobalCtx::kSave)
    return parse_data_impl<CifGlobalCtx::kSave>(tables, lexer, name);

  ABSL_UNREACHABLE();
}

CifBlock next_block(CifParser &parser, CifLexer &lexer, std::string &next_name,
                    BlockType &next_block) {
  std::string name = std::move(next_name);
  BlockType block_type = next_block;

  std::vector<CifTable> tables;
  std::vector<CifFrame> save_frames;
  std::string frame_name;

  CifGlobalCtx global_ctx = CifGlobalCtx::kBlock;
  while (true) {
    if (global_ctx == CifGlobalCtx::kBlock) {
      auto [data, type] =
          parse_data_impl<CifGlobalCtx::kBlock>(tables, lexer, name);
      if (type == CifToken::kError)
        return parser.error(data);
      if (type == CifToken::kEOF || type == CifToken::kData
          || type == CifToken::kGlobal) {
        next_name = data;
        next_block = static_cast<BlockType>(type);
        break;
      }

      ABSL_DCHECK_EQ(type, CifToken::kSave);
      global_ctx = CifGlobalCtx::kSave;
      frame_name = data;
      ABSL_LOG_IF(WARNING, frame_name.empty()) << "Empty save block name";
    } else {
      std::vector<CifTable> save_tables;

      auto [data, type] =
          parse_data_impl<CifGlobalCtx::kSave>(save_tables, lexer, frame_name);
      if (type == CifToken::kError)
        return parser.error(data);
      if (type == CifToken::kEOF || type == CifToken::kData
          || type == CifToken::kGlobal)
        return parser.error("Save block ended without save_ tag");

      ABSL_DCHECK_EQ(type, CifToken::kSave);
      global_ctx = CifGlobalCtx::kBlock;
      save_frames.push_back(
          CifFrame(std::move(save_tables), std::move(frame_name)));
      frame_name.clear();
    }
  }

  return CifBlock(CifFrame(std::move(tables), std::move(name)),
                  std::move(save_frames), block_type);
}
}  // namespace internal

using internal::BlockType;
using internal::CifToken;

CifParser::CifParser(std::istream &is): lexer_(is) {
  while (true) {
    auto [data, type] = lexer_.next();

    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (type) {
    case CifToken::kEOF:
    case CifToken::kError:
    case CifToken::kGlobal:
    case CifToken::kData:
      name_ = data;
      block_ = static_cast<BlockType>(type);
      return;
    default:
      ABSL_LOG(INFO) << "Skipping stray token before data block: " << type;
      break;
    }
  }
}

internal::CifBlock CifParser::next() {
  if (block_ == BlockType::kEOF)
    return internal::CifBlock::eof();

  if (block_ == BlockType::kError)
    return internal::CifBlock::error(name_);

  internal::CifBlock block = internal::next_block(*this, lexer_, name_, block_);

  std::string err = block.validate();
  if (!err.empty())
    return error(err);

  return block;
}

internal::CifBlock CifParser::error(std::string_view reason) {
  name_ = reason;
  block_ = BlockType::kError;
  return internal::CifBlock::error(reason);
}

namespace internal {
namespace {
struct ValueScan {
  bool has_space = false;
  bool has_newline = false;
  // A quote conflicts when followed by whitespace; an interior or trailing
  // quote is safe.
  bool single_conflict = false;
  bool double_conflict = false;
  // A newline is immediately followed by ';': such a line would close a text
  // field early, so the value cannot be serialized.
  bool semicolon_line = false;
};

constexpr absl::CharSet kValueScanChars { "\n'\"" };

ValueScan scan_value(std::string_view s) {
  ValueScan r;

  r.has_space = absl::c_any_of(s, absl::ascii_isspace);
  if (!r.has_space)
    return r;

  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (!kValueScanChars.contains(c))
      continue;

    switch (c) {
    case '\'':
    case '"':
      if (i + 1 < s.size() && absl::ascii_isspace(s[i + 1]))
        (c == '\'' ? r.single_conflict : r.double_conflict) = true;
      break;
    case '\n':
      r.has_newline = true;
      if (i + 1 < s.size() && s[i + 1] == ';')
        r.semicolon_line = true;
      break;
    default:
      ABSL_UNREACHABLE();
    }
  }
  return r;
}

// Leading characters the lexer intercepts before a bare value.
constexpr absl::CharSet kBareLeadingForbidden = kSpecialChars
                                                | absl::CharSet("#'\";");

bool bare_safe(std::string_view s, bool has_space) {
  ABSL_DCHECK(!s.empty());

  if (has_space || kBareLeadingForbidden.contains(s.front()))
    return false;

  // "?" / "." would re-lex as null values; we only consider Generic/String here
  if (s == "?" || s == ".")
    return false;

  auto it = s.begin();
  return !x3::parse(it, s.end(), parser::keyword);
}

bool looks_like_number(std::string_view s) {
  auto it = s.begin();
  return x3::parse(it, s.end(), parser::numeric);
}
}  // namespace

CifValueKind write_cif_value(std::string &out, const CifValue &value) {
  switch (value.type()) {
  case CifValue::Type::kUnknown:
    out.push_back('?');
    return CifValueKind::kInline;
  case CifValue::Type::kInapplicable:
    out.push_back('.');
    return CifValueKind::kInline;
  case CifValue::Type::kFloatNonfinite:
    out = absl::StrCat("Non-finite float value cannot be written: ",
                       value.value());
    return CifValueKind::kError;
  case CifValue::Type::kGeneric:
  case CifValue::Type::kString:
    break;
  }

  std::string_view s = value.value();
  if (s.empty()) {
    out.append("''");
    return CifValueKind::kInline;
  }

  ValueScan sc = scan_value(s);

  if (sc.has_newline || (sc.single_conflict && sc.double_conflict)) {
    if (sc.semicolon_line) {
      out = "CIF value cannot be written: line begins with ';'";
      return CifValueKind::kError;
    }

    // Drop any pending separator/padding whitespace so the leading newline
    // below starts the field at line start.
    absl::StripTrailingAsciiWhitespace(&out);
    absl::StrAppend(&out, "\n;", s, "\n;\n");
    return CifValueKind::kTextField;
  }

  // Keep a numeric-looking string quoted so it stays a string (42 vs '42').
  bool force_quote = value.type() == CifValue::Type::kString
                     && looks_like_number(s);
  if (!force_quote && bare_safe(s, sc.has_space)) {
    out.append(s);
    return CifValueKind::kInline;
  }

  std::string_view q = sc.single_conflict ? "\"" : "'";
  absl::StrAppend(&out, q, s, q);
  return CifValueKind::kInline;
}
}  // namespace internal

namespace {
using internal::CifBlock;
using internal::CifFrame;
using internal::CifTable;
using internal::CifValue;
using internal::CifValueKind;
using internal::write_cif_value;

bool write_cif_pairs(std::string &out, const CifTable &table, bool align) {
  size_t width = 0;
  if (align)
    for (const std::string &key: table.keys())
      width = nuri::max(width, key.size());

  const std::vector<CifValue> &row = table.row(0);
  for (size_t j = 0; j < table.cols(); ++j) {
    const std::string &key = table.keys()[j];
    out.append(key);
    if (align)
      out.append(width - key.size(), ' ');
    out.push_back(' ');

    CifValueKind kind = write_cif_value(out, row[j]);
    if (kind == CifValueKind::kError)
      return false;
    if (kind == CifValueKind::kInline)
      out.push_back('\n');
  }
  return true;
}

bool write_cif_loop_plain(std::string &out, const CifTable &table) {
  for (size_t i = 0; i < table.size(); ++i) {
    const std::vector<CifValue> &row = table.row(i);
    bool end_newline = true;
    for (size_t j = 0; j < table.cols(); ++j) {
      if (!end_newline)
        out.push_back(' ');

      CifValueKind kind = write_cif_value(out, row[j]);
      if (kind == CifValueKind::kError)
        return false;
      end_newline = kind == CifValueKind::kTextField;
    }
    if (!end_newline)
      out.push_back('\n');
  }
  return true;
}

bool write_cif_loop_aligned(std::string &out, const CifTable &table) {
  const int ncol = static_cast<int>(table.cols());
  const int nrow = static_cast<int>(table.size());

  ArrayXb texts = ArrayXb::Zero(static_cast<E::Index>(nrow) * ncol);
  ArrayXi widths = ArrayXi::Zero(ncol);

  std::vector<std::string> cells(texts.size());
  for (int i = 0, k = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j, ++k) {
      std::string &cell = cells[k];
      CifValueKind kind = write_cif_value(cell, table.row(i)[j]);
      if (kind == CifValueKind::kError) {
        out = std::move(cell);
        return false;
      }

      bool is_text = kind == CifValueKind::kTextField;
      texts[k] = is_text;
      if (!is_text)
        widths[j] = nuri::max(widths[j], static_cast<int>(cell.size()));
    }
  }

  for (int i = 0, k = 0; i < nrow; ++i) {
    bool end_newline = true;

    for (int j = 0; j < ncol; ++j, ++k) {
      const std::string &cell = cells[k];
      if (texts[k]) {
        // Strip the trailing newline (at line start) or alignment padding so
        // the text field's own leading newline starts it at line start;
        // same as in write_cif_value.
        absl::StripTrailingAsciiWhitespace(&out);
        out.append(cell);
        end_newline = true;
        continue;
      }

      if (!end_newline)
        out.push_back(' ');
      out.append(cell);
      if (j + 1 < ncol)
        out.append(widths[j] - cell.size(), ' ');
      end_newline = false;
    }

    if (!end_newline)
      out.push_back('\n');
  }
  return true;
}
}  // namespace

bool write_cif_table(std::string &out, const CifTable &table, bool align) {
  if (table.empty())
    return true;

  if (table.size() == 1)
    return write_cif_pairs(out, table, align);

  out.append("loop_\n");
  for (const std::string &key: table.keys())
    absl::StrAppend(&out, key, "\n");

  return align ? write_cif_loop_aligned(out, table)
               : write_cif_loop_plain(out, table);
}

bool write_cif_frame(std::string &out, const CifFrame &frame,
                     CifFrame::Type type, bool align) {
  if (type == CifFrame::Type::kGlobal) {
    out.append("global_\n");
  } else {
    absl::StrAppend(&out, type == CifFrame::Type::kData ? "data_" : "save_",
                    frame.name(), "\n");
  }

  for (const CifTable &table: frame) {
    if (!write_cif_table(out, table, align))
      return false;
  }

  if (type == CifFrame::Type::kSave)
    out.append("save_\n");

  return true;
}

bool write_cif_block(std::string &out, const CifBlock &block, bool align) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (block.type()) {
  case CifBlock::Type::kData:
    if (!write_cif_frame(out, block.data(), CifFrame::Type::kData, align))
      return false;
    break;
  case CifBlock::Type::kGlobal: {
    if (!write_cif_frame(out, block.data(), CifFrame::Type::kGlobal, align))
      return false;
    break;
  }
  default:
    out = "cannot serialize an EOF or error CIF block";
    return false;
  }

  for (const CifFrame &save: block.save_frames()) {
    if (!write_cif_frame(out, save, CifFrame::Type::kSave, align))
      return false;
  }

  return true;
}
}  // namespace nuri
