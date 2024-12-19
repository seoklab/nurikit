//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

/**
 * Some testcases were taken from the BioPython repository.
 * Here follows the full license text from BioPython:
 *
 *
 * Copyright 2017 by Francesco Gastaldello. All rights reserved.
 * Revisions copyright 2017 by Peter Cock.  All rights reserved.
 *
 * Converted by Francesco Gastaldello from an older unit test copyright 2002
 * by Thomas Hamelryck.
 *
 * Copyright (c) 1999-2024, The Biopython Contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "nuri/fmt/cif.h"

#include <string_view>

#include <absl/strings/match.h>

#include <gtest/gtest.h>

#include "nuri/meta.h"

namespace nuri {
namespace internal {
namespace {
constexpr auto str_case_contains =
    overload_cast<std::string_view, std::string_view>(
        absl::StrContainsIgnoreCase);

struct CifLexerTest: ::testing::Test {
  void TearDown() override {
    CifLexer lexer(lines_);

    for (auto [edata, etype]: expected_) {
      auto [data, type] = lexer.next();
      if (type == CifToken::kEOF) {
        FAIL()
            << "Unexpected EOF, cursor: " << lexer.row() << ":" << lexer.col();
      }

      ASSERT_EQ(type, etype) << "Cursor: " << lexer.row() << ":" << lexer.col();

      if (etype == CifToken::kError) {
        EXPECT_PRED2(str_case_contains, data, edata)
            << "Cursor: " << lexer.row() << ":" << lexer.col();

        static_cast<void>(lexer.advance_line<true>());
      } else {
        EXPECT_EQ(data, edata)
            << "Cursor: " << lexer.row() << ":" << lexer.col();
      }
    }

    auto [_, type] = lexer.next();
    EXPECT_EQ(type, CifToken::kEOF)
        << "Cursor: " << lexer.row() << ":" << lexer.col();
  }

  void add_expected_values(const std::vector<std::string_view> &expected) {
    for (std::string_view token: expected)
      expected_.push_back({ token, CifToken::kValue });
  }

  std::vector<std::string> lines_;
  std::vector<std::pair<std::string_view, CifToken>> expected_;
};

TEST_F(CifLexerTest, SplitSimpleLines) {
  lines_ = {
    "foo bar",       "foo bar  ",     "'foo' bar",        "foo \"bar\"",
    "foo 'bar a' b", "foo 'bar'a' b", "foo \"bar' a\" b", "foo '' b",
    "foo bar' b",    "foo bar b'",    "foo b'ar'",        "foo 'b'ar'",
    "foo#bar",       "foo #bar",      "foo# bar",         "#foo bar",
  };

  add_expected_values({
      "foo",     "bar",           //
      "foo",     "bar",           //
      "foo",     "bar",           //
      "foo",     "bar",           //
      "foo",     "bar a",  "b",   //
      "foo",     "bar'a",  "b",   //
      "foo",     "bar' a", "b",   //
      "foo",     "",       "b",   //
      "foo",     "bar'",   "b",   //
      "foo",     "bar",    "b'",  //
      "foo",     "b'ar'",         //
      "foo",     "b'ar",          //
      "foo#bar",                  //
      "foo",                      //
      "foo#",    "bar",           //
  });
}

TEST_F(CifLexerTest, UnmatchedQuotes) {
  lines_ = { "foo 'bar", "foo 'ba'r  ", "foo \"bar'" };

  expected_ = {
    {                "foo", CifToken::kValue }, //
    { "unterminated quote", CifToken::kError }, //
    {                "foo", CifToken::kValue }, //
    { "unterminated quote", CifToken::kError }, //
    {                "foo", CifToken::kValue }, //
    { "unterminated quote", CifToken::kError }, //
  };
}

TEST_F(CifLexerTest, TextField) {
  lines_ = {
    "data_verbatim_test", "_test_value",   ";First line",
    "    Second line",    "Third line   ", ";",
    "data_test",          "_key1",         ";foo bar",
    "; _key2 'value 2'",
  };

  expected_ = {
    {   "verbatim_test",CifToken::kData                        },
    {     "_test_value",   CifToken::kTag },
    { R"text(First line
    Second line
Third line)text",
     CifToken::kValue                    },
    {            "test",  CifToken::kData },
    {           "_key1",   CifToken::kTag },
    {         "foo bar", CifToken::kValue },
    {           "_key2",   CifToken::kTag },
    {         "value 2", CifToken::kValue },
  };
}

TEST_F(CifLexerTest, TextFieldTruncated) {
  lines_ = {
    "data_test _key1",  //
    ";foo bar",         //
    ";# missing space here",
    "_key2 val2",
    "data_test",
    "_key1",
    ";foo bar",
  };

  expected_ = {
    {                    "test",  CifToken::kData },
    {                   "_key1",   CifToken::kTag },
    {                 "foo bar", CifToken::kValue },
    {                   "_key2",   CifToken::kTag },
    {                    "val2", CifToken::kValue },
    {                    "test",  CifToken::kData },
    {                   "_key1",   CifToken::kTag },
    { "unterminated text field", CifToken::kError },
  };
}

TEST_F(CifLexerTest, InlineComment) {
  lines_ = {
    "data_verbatim_test",
    "_test_key_value_1 foo # Ignore this comment",
    "_test_key_value_2 foo#NotIgnored",
    "loop_",
    "_test_loop",
    "a b c d # Ignore this comment",
    "e f g",
    "",
  };

  expected_ = {
    {     "verbatim_test",  CifToken::kData },
    { "_test_key_value_1",   CifToken::kTag },
    {               "foo", CifToken::kValue },
    { "_test_key_value_2",   CifToken::kTag },
    {    "foo#NotIgnored", CifToken::kValue },
    {                  "",  CifToken::kLoop },
    {        "_test_loop",   CifToken::kTag },
    {                 "a", CifToken::kValue },
    {                 "b", CifToken::kValue },
    {                 "c", CifToken::kValue },
    {                 "d", CifToken::kValue },
    {                 "e", CifToken::kValue },
    {                 "f", CifToken::kValue },
    {                 "g", CifToken::kValue },
  };
}
}  // namespace
}  // namespace internal
}  // namespace nuri
