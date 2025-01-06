//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/cif.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/charset.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/x3.hpp>

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

    std::string_view line = absl::StripTrailingAsciiWhitespace(lexer.line());
    if constexpr (kContinuation)
      cont = absl::EndsWith(line, "\\");

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
  buf = slice_rstrip(lexer.line(), lexer.p() - lexer.begin() + 1,
                     lexer.end() - lexer.begin());
  if (buf == "\\")
    return produce_text_field_impl<true>(lexer, buf);

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

  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (value.type()) {
  case ValueType::kGeneric:
    return os << value.value();
  case ValueType::kString:
    return os << '"' << value.value() << '"';
  case ValueType::kUnknown:
    return os << '?';
  case ValueType::kInapplicable:
    return os << '.';
  default:
    ABSL_UNREACHABLE();
  }
}
// GCOV_EXCL_STOP

void CifTable::add_data(CifValue &&value) {
  if (rows_.empty() || rows_.back().size() == keys_.size())
    rows_.emplace_back().reserve(keys_.size());

  rows_.back().push_back(std::move(value));
}

CifFrame::CifFrame(std::vector<CifTable> &&tables, std::string &&name)
    : name_(std::move(name)), tables_(std::move(tables)) {
  for (int i = 0; i < tables_.size(); ++i) {
    for (int j = 0; j < tables_[i].cols(); ++j) {
      auto [_, ok] = index_.insert({
          tables_[i].keys()[j],
          { i, j },
      });

      ABSL_DCHECK(ok) << "Duplicate key: " << tables_[i].keys()[j];
    }
  }
}

std::string CifFrame::validate() const {
  if (!absl::c_all_of(tables_, [](const CifTable &table) {
        return absl::c_all_of(table, [&](const std::vector<CifValue> &row) {
          return row.size() == table.keys().size();
        });
      })) {
    return "Inconsistent table size";
  }

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

  std::string err = block.data().validate();
  if (!err.empty())
    return error(err);

  for (const auto &frame: block.save_frames()) {
    err = frame.validate();
    if (!err.empty()) {
      absl::StrAppend(&err, " in save block ", frame.name());
      return error(err);
    }
  }

  return block;
}

internal::CifBlock CifParser::error(std::string_view reason) {
  name_ = reason;
  block_ = BlockType::kError;
  return internal::CifBlock::error(reason);
}
}  // namespace nuri
