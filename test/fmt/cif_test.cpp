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

#include <initializer_list>
#include <sstream>
#include <string_view>
#include <vector>

#include <absl/strings/match.h>
#include <absl/strings/str_join.h>

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
    CifLexer lexer(data_);

    for (auto [edata, etype]: expected_) {
      auto [data, type] = lexer.next();
      if (type == CifToken::kEOF) {
        FAIL()
            << "Unexpected EOF, cursor: " << lexer.row() << ":" << lexer.col();
      }

      if (etype == CifToken::kValue) {
        ASSERT_TRUE(is_value_token(type))
            << "Cursor: " << lexer.row() << ":" << lexer.col();
      } else {
        ASSERT_EQ(type, etype)
            << "Cursor: " << lexer.row() << ":" << lexer.col();
      }

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

  void set_data(std::initializer_list<std::string_view> lines) {
    data_.str(absl::StrJoin(lines, "\n"));
  }

  void add_expected_values(const std::vector<std::string_view> &expected) {
    for (std::string_view token: expected)
      expected_.push_back({ token, CifToken::kValue });
  }

  std::stringstream data_;
  std::vector<std::pair<std::string_view, CifToken>> expected_;
};

TEST_F(CifLexerTest, SplitSimpleLines) {
  set_data({
      "foo bar",
      "foo bar  ",
      "'foo' bar",
      "foo \"bar\"",
      "foo 'bar a' b",
      "foo 'bar'a' b",
      "foo \"bar' a\" b",
      "foo '' b",
      "foo bar' b",
      "foo bar b'",
      "foo b'ar'",
      "foo 'b'ar'",
      "foo#bar",
      "foo #bar",
      "foo# bar",
      "#foo bar",
  });

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
  set_data({
      "foo 'bar",
      "foo 'ba'r  ",
      "foo \"bar'",
  });

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
  set_data({
      "data_verbatim_test",
      "_test_value",
      ";First line",
      "    Second line",
      "Third line   ",
      ";",
      "data_test",
      "_key1",
      ";foo bar",
      "; _key2 'value 2'",
  });

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
  set_data({
      "data_test _key1",
      ";foo bar",
      ";# missing space here",
      "_key2 val2",
      "data_test",
      "_key1",
      ";foo bar",
  });

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
  set_data({
      "data_verbatim_test",
      "_test_key_value_1 foo # Ignore this comment",
      "_test_key_value_2 foo#NotIgnored",
      "loop_",
      "_test_loop",
      "a b c d # Ignore this comment",
      "e f g",
      "",
  });

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

TEST_F(CifLexerTest, UnderscoreValues) {
  set_data({
      "data_4Q9R",
      "loop_",
      "_pdbx_audit_revision_item.ordinal",
      "_pdbx_audit_revision_item.revision_ordinal",
      "_pdbx_audit_revision_item.data_content_type",
      "_pdbx_audit_revision_item.item",
      "1  5 'Structure model' '_atom_site.B_iso_or_equiv'",
  });

  expected_ = {
    {                                        "4Q9R",  CifToken::kData },
    {                                            "",  CifToken::kLoop },
    {           "_pdbx_audit_revision_item.ordinal",   CifToken::kTag },
    {  "_pdbx_audit_revision_item.revision_ordinal",   CifToken::kTag },
    { "_pdbx_audit_revision_item.data_content_type",   CifToken::kTag },
    {              "_pdbx_audit_revision_item.item",   CifToken::kTag },
    {                                           "1", CifToken::kValue },
    {                                           "5", CifToken::kValue },
    {                             "Structure model", CifToken::kValue },
    {                   "_atom_site.B_iso_or_equiv", CifToken::kValue },
  };
}

TEST_F(CifLexerTest, LineContinuation) {
  set_data({
      "data_znvodata",
      "_chemical_name_systematic                           ",
      ";\\",
      " zinc dihydroxide divan\\",
      "adate dihydrate",
      ";",
      "",
      "_chemical_formula_moiety",
      ";\\",
      "H2 O9 V2 Zn3, 2(H2 O)\\",
      ";",
      "_chemical_formula_sum           'H6 O11 V2 Zn3'",
      "_chemical_formula_weight        480.05",
  });

  expected_ = {
    {                               "znvodata",  CifToken::kData },
    {              "_chemical_name_systematic",   CifToken::kTag },
    { " zinc dihydroxide divanadate dihydrate", CifToken::kValue },
    {               "_chemical_formula_moiety",   CifToken::kTag },
    {                  "H2 O9 V2 Zn3, 2(H2 O)", CifToken::kValue },
    {                  "_chemical_formula_sum",   CifToken::kTag },
    {                          "H6 O11 V2 Zn3", CifToken::kValue },
    {               "_chemical_formula_weight",   CifToken::kTag },
    {                                 "480.05", CifToken::kValue },
  };
}

TEST_F(CifLexerTest, LineContinuationMixed) {
  set_data({
      ";C:\\foldername\\filename",
      ";",
      ";\\",
      "C:\\foldername\\filename",
      ";",
      ";\\",
      "C:\\foldername\\file\\",
      "name",
      ";",
      ";",
      "C:\\foldername\\file\\",
      "name",
      ";",
      ";\\",
      "C:\\foldername\\file\\\\",
      "",
      "name\\",
      "",
      "",
      ";",
  });

  expected_ = {
    { "C:\\foldername\\filename", CifToken::kValue },
    { "C:\\foldername\\filename", CifToken::kValue },
    { "C:\\foldername\\filename", CifToken::kValue },
    {
     R"(
C:\foldername\file\
name)",  //
        CifToken::kValue,
     },
    {
     R"(C:\foldername\file\
name
)",  //
        CifToken::kValue,
     },
  };
}

struct CifParseDataTest: ::testing::Test {
  void TearDown() override {
    CifLexer lexer(data_);

    auto [value, state] =
        parse_data(CifGlobalCtx::kBlock, tables_, lexer, "foo");
    ASSERT_EQ(state, expected_state_) << value;
    if (expected_state_ == CifToken::kError) {
      EXPECT_PRED2(str_case_contains, value, expected_)
          << "Cursor: " << lexer.row() << ":" << lexer.col();
      return;
    }

    ASSERT_EQ(tables_.size(), keys_.size());

    for (int i = 0; i < keys_.size(); ++i) {
      ASSERT_EQ(tables_[i].keys().size(), keys_[i].size()) << i;
      for (int j = 0; j < keys_[i].size(); ++j)
        EXPECT_EQ(tables_[i].keys()[j], keys_[i][j]) << i << ", " << j;

      ASSERT_EQ(tables_[i].data().size(), values_[i].size()) << i;
      for (int j = 0; j < values_[i].size(); ++j) {
        ASSERT_EQ(tables_[i].data()[j].size(), values_[i][j].size())
            << i << ", " << j;
        for (int k = 0; k < values_[i][j].size(); ++k)
          EXPECT_EQ(tables_[i].data()[j][k], values_[i][j][k])
              << i << ", " << j << ", " << k;
      }
    }
  }

  std::stringstream data_;

  std::vector<CifTable> tables_;

  std::string_view expected_;
  CifToken expected_state_ = CifToken::kEOF;

  std::vector<std::vector<std::string_view>> keys_;
  std::vector<std::vector<std::vector<std::string_view>>> values_;
};

TEST_F(CifParseDataTest, ParseNonLoop) {
  data_.str(R"cif(_test_value
;First line
    Second line
Third line
; _key1
;foo bar
; _key2 'value 2'
)cif");

  keys_ = {
    { "_test_value", "_key1", "_key2" },
  };
  values_ = {
    {
     {
            R"text(First line
    Second line
Third line)text",
            "foo bar",
            "value 2",
        },  //
    },
  };
}

TEST_F(CifParseDataTest, ParseSimpleLoop) {
  data_.str(R"cif(loop_
_pdbx_audit_revision_item.ordinal
_pdbx_audit_revision_item.revision_ordinal
_pdbx_audit_revision_item.data_content_type
_pdbx_audit_revision_item.item
1  5 'Structure model' '_atom_site.B_iso_or_equiv'
2  5 'Structure model' '_atom_site.Cartn_x'
3  5 'Structure model' '_atom_site.Cartn_y'
4  5 'Structure model' '_atom_site.Cartn_z'
)cif");

  keys_ = {
    {
     "_pdbx_audit_revision_item.ordinal",            //
        "_pdbx_audit_revision_item.revision_ordinal",   //
        "_pdbx_audit_revision_item.data_content_type",  //
        "_pdbx_audit_revision_item.item",               //
    },
  };
  values_ = {
    {
     { "1", "5", "Structure model", "_atom_site.B_iso_or_equiv" },
     { "2", "5", "Structure model", "_atom_site.Cartn_x" },
     { "3", "5", "Structure model", "_atom_site.Cartn_y" },
     { "4", "5", "Structure model", "_atom_site.Cartn_z" },
     },
  };
}

TEST_F(CifParseDataTest, ParseMixedData) {
  data_.str(R"cif(
_test_key_value_1 foo # Ignore this comment
_test_key_value_2 foo#NotIgnored
loop_
_test_loop
a b c d # Ignore this comment
e f g

loop_
_test_loop2
h i j k

_test_key_value_3 foo
_test_key_value_4 bar
)cif");

  keys_ = {
    { "_test_key_value_1", "_test_key_value_2" },
    { "_test_loop" },
    { "_test_loop2" },
    { "_test_key_value_3", "_test_key_value_4" },
  };
  values_ = {
    { { "foo", "foo#NotIgnored" } },
    { { "a" }, { "b" }, { "c" }, { "d" }, { "e" }, { "f" }, { "g" } },
    { { "h" }, { "i" }, { "j" }, { "k" } },
    { { "foo", "bar" } },
  };
}
}  // namespace
}  // namespace internal
}  // namespace nuri
