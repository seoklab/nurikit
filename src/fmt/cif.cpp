//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/cif.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>

#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/charset.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <absl/strings/strip.h>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/x3.hpp>

#include "nuri/utils.h"

namespace nuri {
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

  std::string_view line = buf;

  bool left;
  while ((left = lexer.advance_line<false>())) {
    if (lexer.p() < lexer.end() && lexer.c() == ';')
      break;

    std::string_view sep = "\n";
    if constexpr (kContinuation) {
      if (absl::EndsWith(line, "\\")) {
        buf.pop_back();
        sep = "";
      }
    }

    line = absl::StripTrailingAsciiWhitespace(lexer.line());
    absl::StrAppend(&buf, sep, line);
  }
  if (!left)
    return lexer.error("Unterminated text field (started at line ", begin_row,
                       ")");

  if constexpr (kContinuation) {
    if (absl::EndsWith(line, "\\"))
      buf.pop_back();
  }

  auto it = lexer.p() + 1;
  ABSL_LOG_IF(WARNING, it < lexer.end() && std::isspace(*it) == 0)
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

    it_ = std::find_if_not(p(), end(), ::isspace);
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
      // abseil forbids passing characters to absl::StrAppend
      const char cbuf[] = { c(), '\0' };
      return error("Unexpected special character at line ", row(), ":", col(),
                   ": ", cbuf);
    }

    auto vit = std::find_if(p(), end(), ::isspace);
    return produce(as_sv(p(), vit), CifToken::kSimpleValue, vit);
  }

  return done();
}
}  // namespace internal
}  // namespace nuri
