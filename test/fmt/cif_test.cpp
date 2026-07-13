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

#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>

#include <gtest/gtest.h>

#include "fmt_test_common.h"
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
    {                              "verbatim_test",  CifToken::kData },
    {                                "_test_value",   CifToken::kTag },
    { "First line\n    Second line\nThird line   ", CifToken::kValue },
    {                                       "test",  CifToken::kData },
    {                                      "_key1",   CifToken::kTag },
    {                                    "foo bar", CifToken::kValue },
    {                                      "_key2",   CifToken::kTag },
    {                                    "value 2", CifToken::kValue },
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

TEST_F(CifLexerTest, LineContinuationKeepsLiteralTrailingWhitespace) {
  set_data({
      ";\\",
      "literal line with trailing ws  ",
      "next\\",
      "joined",
      ";",
  });

  expected_ = {
    { "literal line with trailing ws  \nnextjoined", CifToken::kValue },
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

TEST(CifParseTest, PDB1A8O) {
  std::ifstream ifs(test_data("1a8o.cif"));
  ASSERT_TRUE(ifs) << "Failed to open file: 1a8o.cif";

  CifParser parser(ifs);
  CifBlock block = parser.next();
  ASSERT_TRUE(block) << "Failed to parse";

  const CifFrame &frame = block.data();
  EXPECT_EQ(frame.name(), "1A8O");
  EXPECT_EQ(frame.total_cols(), 574);

  {
    auto [i, j] = frame.find("_entry.id");
    ASSERT_GE(i, 0);
    ASSERT_GE(j, 0);

    EXPECT_EQ(frame.tables()[i][j][0], "1A8O");
  }

  std::vector<std::string_view> mon_ids = {
    "MSE", "ASP", "ILE", "ARG", "GLN", "GLY", "PRO", "LYS", "GLU", "PRO",
    "PHE", "ARG", "ASP", "TYR", "VAL", "ASP", "ARG", "PHE", "TYR", "LYS",
    "THR", "LEU", "ARG", "ALA", "GLU", "GLN", "ALA", "SER", "GLN", "GLU",
    "VAL", "LYS", "ASN", "TRP", "MSE", "THR", "GLU", "THR", "LEU", "LEU",
    "VAL", "GLN", "ASN", "ALA", "ASN", "PRO", "ASP", "CYS", "LYS", "THR",
    "ILE", "LEU", "LYS", "ALA", "LEU", "GLY", "PRO", "GLY", "ALA", "THR",
    "LEU", "GLU", "GLU", "MSE", "MSE", "THR", "ALA", "CYS", "GLN", "GLY",
  };
  std::vector<std::string_view> cartn_x = {
    "19.594", "20.255", "20.351", "19.362", "19.457", "20.022", "21.718",
    "21.424", "21.554", "21.835", "21.947", "21.678", "23.126", "23.098",
    "23.433", "22.749", "22.322", "22.498", "21.220", "20.214", "23.062",
    "24.282", "23.423", "25.429", "21.280", "20.173", "20.766", "21.804",
    "19.444", "18.724", "18.011", "17.416", "16.221", "15.459", "15.824",
    "20.116", "20.613", "20.546", "19.488", "19.837", "20.385", "19.526",
    "18.365", "20.090", "21.675", "21.698", "20.859", "20.729", "20.260",
    "19.435", "20.158", "19.512", "18.993", "20.056", "20.300", "21.486",
    "22.285", "23.286", "24.155", "23.025", "22.117", "21.236", "20.159",
    "19.231", "23.152", "24.037", "23.563", "22.398", "24.086", "25.003",
    "24.858", "23.861", "25.748", "24.459", "24.089", "23.580", "24.111",
    "25.415", "26.116", "25.852", "22.544", "21.960", "22.965", "22.928",
    "20.793", "19.999", "19.234", "20.019", "18.495", "19.286", "18.523",
    "23.861", "24.870", "25.788", "26.158", "25.684", "26.777", "26.215",
    "27.235", "28.136", "28.155", "29.030", "26.137", "26.994", "26.279",
    "26.880", "27.408", "28.345", "28.814", "28.620", "24.992", "24.151",
    "24.025", "24.139", "22.787", "21.629", "21.657", "20.489", "20.571",
    "19.408", "19.450", "18.365", "23.839", "23.720", "24.962", "24.853",
    "23.502", "23.661", "22.120", "26.137", "27.387", "27.511", "27.925",
    "28.595", "28.723", "28.016", "29.545", "27.136", "27.202", "26.238",
    "26.585", "26.850", "27.835", "27.667", "26.352", "25.494", "25.797",
    "24.325", "25.037", "23.984", "24.456", "24.305", "22.761", "21.538",
    "21.301", "20.586", "20.130", "19.415", "19.186", "25.033", "25.526",
    "26.755", "27.015", "25.771", "24.608", "23.508", "24.583", "22.406",
    "23.490", "22.406", "21.326", "27.508", "28.691", "28.183", "28.705",
    "29.455", "30.787", "31.428", "32.618", "33.153", "27.116", "26.508",
    "25.826", "25.827", "25.475", "26.150", "24.741", "25.264", "24.587",
    "25.587", "25.302", "23.789", "22.707", "21.787", "21.910", "26.767",
    "27.806", "28.299", "28.656", "29.006", "28.944", "30.295", "30.744",
    "30.326", "29.441", "30.787", "28.332", "28.789", "27.943", "28.374",
    "28.803", "26.740", "25.833", "25.775", "24.998", "24.425", "24.354",
    "24.816", "24.535", "25.454", "26.601", "26.645", "25.240", "24.885",
    "27.391", "28.884", "29.200", "28.729", "29.998", "24.438", "23.066",
    "23.001", "23.824", "22.370", "22.035", "21.831", "21.174", "20.852",
    "20.917", "19.638", "20.949", "20.315", "18.908", "18.539", "20.262",
    "19.688", "20.414", "21.592", "19.714", "18.136", "16.775", "16.738",
    "15.875", "16.101", "15.478", "14.341", "13.247", "14.542", "17.668",
    "17.730", "18.064", "17.491", "18.754", "18.932", "18.279", "18.971",
    "19.343", "18.126", "17.905", "20.444", "21.777", "22.756", "24.069",
    "24.913", "17.344", "16.136", "15.146", "14.599", "15.468", "16.242",
    "17.164", "15.865", "14.932", "14.017", "14.495", "13.700", "13.904",
    "13.254", "12.332", "13.484", "11.975", "12.666", "14.303", "12.641",
    "14.280", "13.452", "15.793", "16.368", "16.285", "16.053", "17.815",
    "17.939", "17.221", "18.427", "16.438", "16.375", "14.950", "14.778",
    "16.869", "18.228", "16.791", "13.947", "12.529", "12.045", "11.151",
    "11.625", "11.950", "11.054", "11.086", "10.326", "12.589", "12.177",
    "13.076", "12.888", "11.978", "13.202", "10.883", "14.054", "14.963",
    "15.702", "15.846", "15.935", "15.286", "16.327", "14.580", "16.162",
    "16.876", "15.961", "16.391", "17.402", "18.238", "19.553", "18.506",
    "14.695", "13.703", "13.270", "13.262", "12.460", "11.372", "12.854",
    "12.954", "12.503", "13.541", "13.184", "12.008", "10.830", "10.505",
    "10.626", "10.093", "14.820", "15.887", "16.443", "17.416", "17.014",
    "16.627", "15.451", "17.619", "15.830", "16.248", "15.758", "14.809",
    "15.689", "16.404", "16.005", "14.639", "14.122", "17.109", "17.396",
    "16.559", "18.588", "14.018", "12.706", "12.516", "11.536", "12.617",
    "13.288", "14.522", "13.454", "13.383", "13.351", "12.406", "14.564",
    "14.482", "13.353", "15.552", "14.378", "14.488", "13.443", "12.968",
    "15.902", "16.144", "13.061", "12.087", "10.746", "10.157", "11.879",
    "11.014", "11.003", "10.171", "10.269", "10.273", "9.002",  "9.101",
    "8.227",  "8.612",  "8.611",  "7.224",  "10.191", "10.458", "10.518",
    "9.916",  "11.791", "11.677", "12.184", "12.967", "11.222", "11.377",
    "10.082", "9.885",  "12.416", "13.824", "14.764", "14.287", "9.214",
    "7.937",  "7.048",  "6.294",  "7.230",  "7.828",  "7.618",  "8.090",
    "7.916",  "7.189",  "6.419",  "6.871",  "6.391",  "6.449",  "7.815",
    "8.305",  "7.481",  "7.371",  "9.788",  "10.832", "12.217", "10.789",
    "6.886",  "6.080",  "6.922",  "8.149",  "6.294",  "7.024",  "7.912",
    "7.680",  "5.901",  "4.734",  "4.839",  "8.952",  "9.861",  "10.886",
    "11.642", "10.910", "11.884", "13.285", "13.524", "11.599", "14.199",
    "15.563", "16.391", "16.022", "16.290", "16.498", "15.473", "17.509",
    "18.426", "18.875", "19.012", "19.645", "20.773", "20.264", "21.920",
    "19.082", "19.510", "18.471", "18.816", "19.784", "21.035", "20.954",
    "19.902", "21.955", "17.199", "16.109", "16.001", "15.690", "14.787",
    "14.776", "13.539", "13.220", "12.888", "16.301", "16.274", "17.413",
    "17.209", "16.429", "15.284", "15.332", "13.844", "18.606", "19.764",
    "19.548", "19.922", "21.047", "21.507", "23.105", "22.645", "18.915",
    "18.636", "17.640", "17.807", "18.050", "18.998", "17.730", "16.631",
    "15.593", "16.104", "15.685", "14.486", "17.033", "17.572", "18.985",
    "19.634", "17.525", "15.855", "19.451", "20.802", "21.001", "20.066",
    "21.152", "20.421", "20.725", "21.768", "19.817", "22.226", "22.536",
    "23.683", "24.328", "23.949", "15.165", "19.774", "22.152", "12.938",
    "23.499", "17.568", "13.544", "15.524", "31.249", "11.999", "14.511",
    "7.439",  "19.303", "17.114", "21.867", "17.573", "26.151", "20.974",
    "20.796", "28.370", "29.565", "21.248", "25.744", "8.691",  "30.789",
    "30.905", "28.623", "24.935", "23.462", "9.924",  "28.729", "13.579",
    "23.652", "25.631", "17.799", "23.547", "16.363", "24.125", "33.063",
    "29.209", "10.391", "12.221", "18.997", "16.360", "27.915", "28.158",
    "21.975", "27.069", "30.148", "21.196", "8.864",  "13.228", "18.577",
    "20.526", "25.758", "7.838",  "20.569", "13.009", "19.229", "17.655",
    "30.445", "9.014",  "3.398",  "31.603", "16.543", "12.037", "7.261",
    "5.607",  "23.532", "30.701", "32.300", "34.351", "9.450",  "29.476",
    "13.681", "26.728", "10.004", "30.553", "23.569", "10.927", "17.983",
    "8.191",  "32.095", "11.520", "13.249", "15.919", "11.187", "16.743",
  };
  std::vector<std::string_view> seqres = {
    R"fasta(GARASVLSGGELDKWEKIRLRPGGKKQYKLKHIVWASRELERFAVNPGLLETSEGCRQILGQLQPSLQTGSEELRSLYNT
IAVLYCVHQRIDVKDTKEALDKIEEEQNKSKKKAQQAAADTGNNSQVSQNYPIVQNLQGQMVHQAISPRTLNAWVKVVEE
KAFSPEVIPMFSALSEGATPQDLNTMLNTVGGHQAAMQMLKETINEEAAEWDRLHPVHAGPIAPGQMREPRGSDIAGTTS
TLQEQIGWMTHNPPIPVGEIYKRWIILGLNKIVRMYSPTSILDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETL
LVQNANPDCKTILKALGPGATLEEMMTACQGVGGPGHKARVLAEAMSQVTNPATIMIQKGNFRNQRKTVKCFNCGKEGHI
AKNCRAPRKKGCWKCGKEGHQMKDCTERQANFLGKIWPSHKGRPGNFLQSRPEPTAPPEESFRFGEETTTPSQKQEPIDK
ELYPLASLRSLFGSDPSSQ)fasta"
  };

  {
    auto [i, j] = frame.find("_entity_poly_seq.mon_id");
    ASSERT_GE(i, 0);
    ASSERT_GE(j, 0);

    auto col = frame[i].col(j);
    ASSERT_EQ(col.size(), mon_ids.size());
    for (int k = 0; k < col.size(); ++k)
      EXPECT_EQ(col[k], mon_ids[k]);
  }

  {
    auto [i, j] = frame.find("_atom_site.Cartn_x");
    ASSERT_GE(i, 0);
    ASSERT_GE(j, 0);

    auto col = frame[i].col(j);
    ASSERT_EQ(col.size(), cartn_x.size());
    for (int k = 0; k < col.size(); ++k)
      EXPECT_EQ(col[k], cartn_x[k]);
  }

  {
    auto [i, j] = frame.find("_struct_ref.pdbx_seq_one_letter_code");
    ASSERT_GE(i, 0);
    ASSERT_GE(j, 0);

    auto col = frame[i].col(j);
    ASSERT_EQ(col.size(), seqres.size());
    for (int k = 0; k < col.size(); ++k)
      EXPECT_EQ(col[k], seqres[k]);
  }
}

TEST(CifParseTest, CCD) {
  std::ifstream ifs(test_data("components_stdres.cif"));
  ASSERT_TRUE(ifs) << "Failed to open file: 1a8o.cif";

  std::vector<std::string_view> ids {
    "GLY", "ALA", "VAL", "LEU", "ILE", "THR", "SER", "MET", "CYS", "PRO",
    "PHE", "TYR", "TRP", "HIS", "LYS", "ARG", "ASP", "GLU", "ASN", "GLN",
  };

  CifParser parser(ifs);
  for (auto id: ids) {
    CifBlock block = parser.next();
    if (!block)
      FAIL() << "Failed to parse block: " << id;

    const CifFrame &frame = block.data();
    EXPECT_EQ(frame.name(), id);

    {
      auto [i, j] = frame.find("_chem_comp.id");
      ASSERT_GE(i, 0);
      ASSERT_GE(j, 0);
      EXPECT_EQ(frame.tables()[i][j][0], frame.name());
    }
  }

  CifBlock block = parser.next();
  EXPECT_TRUE(block.type() == CifBlock::Type::kEOF);
}

struct ExpectedData {
  std::string_view name;
  CifBlock::Type type;
  int total_cols;
  int num_save;
};

// Figure 1, JCIM 2012, 52 (8), 1901-1906. DOI: 10.1021/ci300074v
TEST(CifParseTest, StarFig1) {
  std::stringstream data;
  data.str(R"cif(
global_
  _compound.trial     4
  _compound.source    FDA
data_synthesis
  _sample.length      5.84
  _sample.shape       'needle'
  _solvent.base       Methanol
  _sample.orientation '[1,0,2]'
global_
  _experiment.source  'ConvBeamEI'
  _experiment.date    2011-06-09
data_experiment
  _images.collected   1289
  _images.refined     894
save_fragment_1
  _molecular.weight   234
  _bond_length.max    2.7
save_
save_fragment_2
  _molecular.weight   23
  _bond_length.max    1.1
  _fragment.parent    fragment_1
save_
data_publication
  _author.details     'A.B. Smith'
  _author.laboratory  'LLNL'
)cif");

  CifParser parser(data);

  std::vector<ExpectedData> expected {
    {            "", CifBlock::Type::kGlobal, 2, 0 },
    {   "synthesis",   CifBlock::Type::kData, 4, 0 },
    {            "", CifBlock::Type::kGlobal, 2, 0 },
    {  "experiment",   CifBlock::Type::kData, 2, 2 },
    { "publication",   CifBlock::Type::kData, 2, 0 },
  };

  for (auto [name, type, cols, saved]: expected) {
    CifBlock block = parser.next();
    if (!block)
      FAIL() << "Failed to parse block: " << name;

    const CifFrame &frame = block.data();
    EXPECT_EQ(frame.name(), name);
    EXPECT_EQ(static_cast<int>(block.type()), static_cast<int>(type));
    EXPECT_EQ(block.data().total_cols(), cols);
    EXPECT_EQ(block.save_frames().size(), saved);
  }
}

TEST(CifParseTest, Erroneous) {
  std::stringstream data;
  data.str(R"cif(
data_publication
  _author.details     'A.B. Smith'
  _author.laboratory  'LLNL
)cif");

  CifParser parser(data);
  CifBlock block = parser.next();

  EXPECT_FALSE(block) << "Block should error";
  EXPECT_EQ(static_cast<int>(block.type()),
            static_cast<int>(CifBlock::Type::kError));
  EXPECT_PRED2(str_case_contains, block.error_msg(), "unterminated quote");
}

// ---------------------------------------------------------------------------
// Writer tests
// ---------------------------------------------------------------------------

std::string write_value(const CifValue &value) {
  std::string out;
  EXPECT_NE(write_cif_value(out, value), CifValueKind::kError);
  return out;
}

// Feed an emitted single value back through the lexer.
CifValue relex_value(std::string_view emitted) {
  std::stringstream ss { std::string(emitted) };
  CifLexer lexer(ss);
  auto [data, type] = lexer.next();
  EXPECT_TRUE(is_value_token(type)) << "not a value token: " << emitted;
  return CifValue(data, type);
}

void expect_roundtrips(const CifValue &value) {
  std::string out;
  ASSERT_NE(write_cif_value(out, value), CifValueKind::kError)
      << "value: " << *value;
  CifValue back = relex_value(out);
  EXPECT_EQ(back.value(), value.value()) << "emitted: [" << out << "]";
  EXPECT_EQ(back.is_null(), value.is_null()) << "emitted: [" << out << "]";
  EXPECT_EQ(static_cast<bool>(back.type() & CifValue::Type::kUnknown),
            static_cast<bool>(value.type() & CifValue::Type::kUnknown))
      << "emitted: [" << out << "]";
}

TEST(CifWriteValueTest, NullValues) {
  EXPECT_EQ(write_value(CifValue::unknown()), "?");
  EXPECT_EQ(write_value(CifValue::inapplicable()), ".");
  // a default-constructed value is unknown ('?'), the default null everywhere
  EXPECT_EQ(write_value(CifValue()), "?");
}

TEST(CifWriteValueTest, BareValues) {
  EXPECT_EQ(write_value(CifValue::generic("1.234")), "1.234");
  EXPECT_EQ(write_value(CifValue::generic("N")), "N");
  EXPECT_EQ(write_value(CifValue::generic("a#b")), "a#b");
  EXPECT_EQ(write_value(CifValue::generic("a'b")), "a'b");
  // "loop" (no underscore) is not a reserved word
  EXPECT_EQ(write_value(CifValue::generic("loop")), "loop");
}

TEST(CifWriteValueTest, ValueSemantics) {
  EXPECT_EQ(write_value(CifValue::generic("42")), "42");
  EXPECT_EQ(write_value(CifValue::string("42")), "'42'");
  EXPECT_EQ(write_value(CifValue::string("-1.5e3")), "'-1.5e3'");
  EXPECT_EQ(write_value(CifValue::string("bare")), "bare");
  EXPECT_EQ(write_value(CifValue::string("N")), "N");

  // CIF 1.1 numbers stay quoted so they re-lex as strings, not numbers
  EXPECT_EQ(write_value(CifValue::string("+12")), "'+12'");
  EXPECT_EQ(write_value(CifValue::string(".5")), "'.5'");
  EXPECT_EQ(write_value(CifValue::string("5.")), "'5.'");
  EXPECT_EQ(write_value(CifValue::string("12e3")), "'12e3'");
  EXPECT_EQ(write_value(CifValue::string("-3.4E-5")), "'-3.4E-5'");
  // standard uncertainty is part of the numeric grammar
  EXPECT_EQ(write_value(CifValue::string("1.234(5)")), "'1.234(5)'");
  // non-CIF-numbers are not forced to quote
  EXPECT_EQ(write_value(CifValue::string("nan")), "nan");
  EXPECT_EQ(write_value(CifValue::string("inf")), "inf");
  EXPECT_EQ(write_value(CifValue::string("0x1f")), "0x1f");
  EXPECT_EQ(write_value(CifValue::string("1.2.3")), "1.2.3");
  EXPECT_EQ(write_value(CifValue::string("+")), "+");

  EXPECT_EQ(write_value(cif_value(42)), "42");
  EXPECT_EQ(write_value(cif_value("42")), "'42'");
  EXPECT_EQ(write_value(cif_value("hello")), "hello");
  EXPECT_EQ(write_value(cif_value(1.5)), "1.5");
  EXPECT_EQ(write_value(cif_value(3.14159, 2)), "3.14");
  EXPECT_EQ(write_value(cif_value(7, 4)), "0007");
  EXPECT_EQ(write_value(cif_value(true)), "yes");
  EXPECT_EQ(write_value(cif_value(false)), "no");
  EXPECT_EQ(write_value(cif_value(true, true)), "y");
  EXPECT_EQ(write_value(cif_value(false, true)), "n");

  // zero-padded width includes the sign
  EXPECT_EQ(write_value(cif_value(-7, 4)), "-007");

  // generic overload: non-arithmetic, non-string_view, StrCat-able -> quoted
  CifValue hex = cif_value(absl::Hex(255));
  EXPECT_EQ(hex.type(), CifValue::Type::kString);
  EXPECT_EQ(hex.value(), "ff");
}

TEST(CifWriteValueTest, NullFactory) {
  EXPECT_EQ(write_value(CifValue::null(true)), "?");
  EXPECT_EQ(write_value(CifValue::null(false)), ".");
  EXPECT_TRUE(CifValue::null(true).is_null());
  EXPECT_TRUE(CifValue::null(false).is_null());
}

TEST(CifWriteValueTest, EmptyString) {
  EXPECT_EQ(write_value(CifValue::generic("")), "''");
  EXPECT_EQ(write_value(CifValue::string("")), "''");
}

TEST(CifWriteValueTest, Quoted) {
  EXPECT_EQ(write_value(CifValue::generic("a b")), "'a b'");
  // leading special characters force quoting
  EXPECT_EQ(write_value(CifValue::generic("_tag")), "'_tag'");
  EXPECT_EQ(write_value(CifValue::generic("#c")), "'#c'");
  EXPECT_EQ(write_value(CifValue::generic("$x")), "'$x'");
  EXPECT_EQ(write_value(CifValue::generic("[x")), "'[x'");
  EXPECT_EQ(write_value(CifValue::generic(";x")), "';x'");
  // "?" / "." as real values must be quoted, not emitted as null
  EXPECT_EQ(write_value(CifValue::generic("?")), "'?'");
  EXPECT_EQ(write_value(CifValue::generic(".")), "'.'");
  // reserved words
  EXPECT_EQ(write_value(CifValue::generic("data_x")), "'data_x'");
  EXPECT_EQ(write_value(CifValue::generic("loop_")), "'loop_'");
  EXPECT_EQ(write_value(CifValue::generic("GLOBAL_")), "'GLOBAL_'");
}

TEST(CifWriteValueTest, DoubleQuoted) {
  // contains a single-quote followed by whitespace -> use double quotes
  EXPECT_EQ(write_value(CifValue::generic("a' b")), "\"a' b\"");
  // a double-quote followed by whitespace (no conflicting single quote) ->
  // stays single-quoted
  EXPECT_EQ(write_value(CifValue::generic("a\" b")), "'a\" b'");
}

TEST(CifWriteValueTest, TextField) {
  // embedded newline forces a text field
  EXPECT_EQ(write_value(CifValue::generic("a\nb")), "\n;a\nb\n;\n");
  // both quote styles conflict -> text field
  EXPECT_EQ(write_value(CifValue::generic("a' b\" c")), "\n;a' b\" c\n;\n");
  // trailing whitespace is preserved, not rejected
  EXPECT_EQ(write_value(CifValue::generic("a \nb")), "\n;a \nb\n;\n");
}

TEST(CifWriteValueTest, Unrepresentable) {
  std::string out;
  // subsequent line begins with ';'; the error message replaces `out`
  EXPECT_EQ(write_cif_value(out, CifValue::generic("a\n;b")),
            CifValueKind::kError);
  EXPECT_PRED2(str_case_contains, out, "';'");
}

TEST(CifWriteValueTest, RoundTrip) {
  expect_roundtrips(CifValue::unknown());
  expect_roundtrips(CifValue::inapplicable());
  for (std::string_view s: { "1.234",
                             "N",
                             "a#b",
                             "a'b",
                             "loop",
                             "",
                             "a b",
                             "_tag",
                             "#c",
                             "$x",
                             "[x",
                             ";x",
                             "?",
                             ".",
                             "data_x",
                             "loop_",
                             "a' b",
                             "a\" b",
                             "a' b\" c",
                             "a\nb",
                             "a \nb",
                             "trailing spaces  \nsecond line",
                             "line one\nline two\nline three",
                             "\nleading newline",
                             "trailing newline\n",
                             "'single'",
                             "\"double\"",
                             "  padded  " }) {
    expect_roundtrips(CifValue::generic(s));
  }
}

TEST(CifWriteValueTest, NonfiniteRejectedByDefault) {
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  constexpr double inf = std::numeric_limits<double>::infinity();

  for (double v: { nan, inf, -inf }) {
    CifValue value = cif_value(v);
    EXPECT_EQ(value.type(), CifValue::Type::kFloatNonfinite);
    EXPECT_FALSE(value.is_null());

    std::string out;
    EXPECT_EQ(write_cif_value(out, value), CifValueKind::kError)
        << "value: " << v;
    EXPECT_PRED2(str_case_contains, out, "non-finite");
  }

  // float, not only double
  CifValue f = cif_value(std::numeric_limits<float>::infinity());
  EXPECT_EQ(f.type(), CifValue::Type::kFloatNonfinite);
}

TEST(CifWriteValueTest, NonfiniteCoercedNaN) {
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();

  CifValue inapplicable = cif_value(nan, -1, true, false);
  EXPECT_EQ(inapplicable.type(), CifValue::Type::kInapplicable);
  EXPECT_TRUE(inapplicable.is_null());
  EXPECT_EQ(write_value(inapplicable), ".");

  CifValue unknown = cif_value(nan, -1, true, true);
  EXPECT_EQ(unknown.type(), CifValue::Type::kUnknown);
  EXPECT_TRUE(unknown.is_null());
  EXPECT_EQ(write_value(unknown), "?");

  // default coercion is unknown ('?'), matching the Python binding default
  EXPECT_EQ(write_value(cif_value(nan, -1, true)), "?");
}

TEST(CifWriteValueTest, NonfiniteCoercedInf) {
  constexpr double inf = std::numeric_limits<double>::infinity();

  // +/-Inf -> a sentinel whose magnitude overflows on reparse, so an
  // IEEE-conformant parser reads it back as the original infinity.
  EXPECT_EQ(write_value(cif_value(inf, -1, true)), "8e+88888888");
  EXPECT_EQ(write_value(cif_value(-inf, -1, true)), "-8e+88888888");

  // the sentinel is a valid CIF number and re-lexes verbatim
  expect_roundtrips(cif_value(inf, -1, true));
  expect_roundtrips(cif_value(-inf, -1, true));

  // an abseil parse of the emitted value overflows back to the infinity
  double d;
  ASSERT_TRUE(absl::SimpleAtod(write_value(cif_value(inf, -1, true)), &d));
  EXPECT_TRUE(std::isinf(d));
  EXPECT_GT(d, 0);
  ASSERT_TRUE(absl::SimpleAtod(write_value(cif_value(-inf, -1, true)), &d));
  EXPECT_TRUE(std::isinf(d));
  EXPECT_LT(d, 0);
}

// Collect a frame into an order-independent {key -> column values} map so that
// loop_ / key-value canonicalization does not affect comparison.
using FrameMap =
    absl::flat_hash_map<std::string, std::vector<std::pair<bool, std::string>>>;

FrameMap frame_map(const CifFrame &frame) {
  FrameMap map;
  for (const CifTable &table: frame) {
    for (size_t j = 0; j < table.cols(); ++j) {
      auto &col = map[table.keys()[j]];
      for (size_t i = 0; i < table.size(); ++i) {
        const CifValue &v = table.row(i)[j];
        col.emplace_back(v.is_null(), std::string(v.value()));
      }
    }
  }
  return map;
}

CifBlock reparse(const std::string &cif) {
  std::stringstream ss { cif };
  CifParser parser(ss);
  return parser.next();
}

CifTable
make_table(std::initializer_list<std::string_view> keys,
           std::initializer_list<std::initializer_list<CifValue>> rows) {
  CifTable table;
  for (std::string_view key: keys)
    table.add_key(key);
  for (const auto &row: rows) {
    for (const CifValue &v: row)
      table.add_data(CifValue(v));
  }
  return table;
}

class CifWriteAlignTest: public ::testing::TestWithParam<bool> {
protected:
  static bool align() { return GetParam(); }
};

INSTANTIATE_TEST_SUITE_P(, CifWriteAlignTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &pinfo) {
                           return pinfo.param ? "Aligned" : "Plain";
                         });

TEST_P(CifWriteAlignTest, BuildAndRoundTrip) {
  std::vector<CifTable> tables;
  {
    CifTable t;
    t.add_key("_entry.id");
    t.add_data(CifValue::generic("TEST"));
    tables.push_back(std::move(t));
  }
  {
    CifTable t;
    t.add_key("_atom.id");
    t.add_key("_atom.name");
    t.add_key("_atom.alt");
    for (int i = 1; i <= 3; ++i) {
      t.add_data(CifValue::generic(std::to_string(i)));
      t.add_data(CifValue::string("atom with space"));
      t.add_data(CifValue::unknown());
    }
    tables.push_back(std::move(t));
  }

  CifFrame frame(std::move(tables), "test");
  CifBlock block(std::move(frame), {}, CifBlock::Type::kData);

  std::string out;
  ASSERT_TRUE(write_cif_block(out, block, align()));

  CifBlock back = reparse(out);
  ASSERT_TRUE(back) << out;
  EXPECT_EQ(back.name(), "test");
  EXPECT_EQ(frame_map(block.data()), frame_map(back.data()));
}

TEST(CifWriteBlockTest, TextFieldInPair) {
  CifTable table = make_table({ "_desc" }, { { CifValue::generic("l1\nl2") } });
  std::string out;
  ASSERT_TRUE(write_cif_table(out, table, false));
  EXPECT_EQ(out, "_desc\n;l1\nl2\n;\n");
}

TEST(CifWriteBlockTest, AlignedPairs) {
  CifTable table = make_table(
      {
          "_short", "_a_longer_key"
  },
      { { CifValue::generic("1"), CifValue::generic("2") } });

  std::string plain, aligned;
  ASSERT_TRUE(write_cif_table(plain, table, false));
  ASSERT_TRUE(write_cif_table(aligned, table, true));

  EXPECT_EQ(plain, "_short 1\n_a_longer_key 2\n");
  // values aligned to the widest key (_a_longer_key)
  EXPECT_EQ(aligned, "_short        1\n_a_longer_key 2\n");
}

TEST(CifWriteBlockTest, TextFieldInLoop) {
  // A value after a text field starts on a new line (not on the terminator
  // line), and no blank line is emitted around it.
  CifTable table = make_table(
      {
          "_x", "_y"
  },
      { { CifValue::generic("a\nb"), CifValue::generic("z") },
        { CifValue::generic("p"), CifValue::generic("c\nd") } });
  std::string out;
  ASSERT_TRUE(write_cif_table(out, table, false));
  EXPECT_EQ(out, "loop_\n_x\n_y\n;a\nb\n;\nz\np\n;c\nd\n;\n");
}

TEST(CifWriteBlockTest, Align) {
  CifTable table = make_table(
      {
          "_a", "_b"
  },
      { { CifValue::generic("1"), CifValue::generic("longvalue") },
        { CifValue::generic("2000"), CifValue::generic("x") } });

  std::string plain, aligned;
  ASSERT_TRUE(write_cif_table(plain, table, false));
  ASSERT_TRUE(write_cif_table(aligned, table, true));

  EXPECT_NE(plain, aligned);
  // aligned output pads the first column: "1   " before the second value
  EXPECT_PRED2(str_case_contains, aligned, "1    longvalue");

  std::vector<CifTable> pt, at;
  pt.push_back(make_table(
      {
          "_a", "_b"
  },
      { { CifValue::generic("1"), CifValue::generic("x") } }));
  // reparse both, ensure identical data model
  CifBlock pb = reparse("data_x\n" + plain);
  CifBlock ab = reparse("data_x\n" + aligned);
  ASSERT_TRUE(pb);
  ASSERT_TRUE(ab);
  EXPECT_EQ(frame_map(pb.data()), frame_map(ab.data()));
}

TEST_P(CifWriteAlignTest, SaveFrames) {
  std::vector<CifTable> main_tables;
  {
    CifTable t;
    t.add_key("_data.id");
    t.add_data(CifValue::generic("main"));
    main_tables.push_back(std::move(t));
  }

  std::vector<CifFrame> saves;
  {
    std::vector<CifTable> st;
    CifTable t;
    t.add_key("_frag.weight");
    t.add_data(CifValue::generic("234"));
    st.push_back(std::move(t));
    saves.emplace_back(std::move(st), "fragment_1");
  }

  CifFrame frame(std::move(main_tables), "blk");
  CifBlock block(std::move(frame), std::move(saves), CifBlock::Type::kData);

  std::string out;
  ASSERT_TRUE(write_cif_block(out, block, align()));

  CifBlock back = reparse(out);
  ASSERT_TRUE(back) << out;
  ASSERT_EQ(back.save_frames().size(), 1U);
  EXPECT_EQ(back.save_frames()[0].name(), "fragment_1");
  EXPECT_EQ(frame_map(block.save_frames()[0]),
            frame_map(back.save_frames()[0]));
}

TEST_P(CifWriteAlignTest, PDB1A8ORoundTrip) {
  std::ifstream ifs(test_data("1a8o.cif"));
  ASSERT_TRUE(ifs) << "Failed to open file: 1a8o.cif";

  CifParser parser(ifs);
  CifBlock block = parser.next();
  ASSERT_TRUE(block) << "Failed to parse";

  std::string out;
  ASSERT_TRUE(write_cif_block(out, block, align()));

  CifBlock back = reparse(out);
  ASSERT_TRUE(back) << "Failed to re-parse written CIF";
  EXPECT_EQ(back.name(), "1A8O");
  EXPECT_EQ(frame_map(block.data()), frame_map(back.data()));
}

TEST_P(CifWriteAlignTest, EmptyTable) {
  CifTable table;
  std::string out;
  EXPECT_TRUE(write_cif_table(out, table, align()));
  EXPECT_TRUE(out.empty());
}

TEST_P(CifWriteAlignTest, KeysWithoutRows) {
  // A table with keys but no data rows is semantically empty: it must not emit
  // a loop_ header with zero packets (invalid CIF), just nothing.
  CifTable table;
  table.add_key("_x");
  table.add_key("_y");

  std::string out;
  EXPECT_TRUE(write_cif_table(out, table, align()));
  EXPECT_TRUE(out.empty());
}

TEST(CifWriteBlockTest, AlignedTextFieldInLoop) {
  // Aligned loop with an embedded-newline cell: inline columns are padded to
  // the widest cell, and the text field still starts at line start (any
  // pending padding before it is stripped).
  CifTable table = make_table(
      {
          "_x", "_y"
  },
      { { CifValue::generic("aaaa"), CifValue::generic("z") },
        { CifValue::generic("p"), CifValue::generic("c\nd") } });

  std::string out;
  ASSERT_TRUE(write_cif_table(out, table, true));
  EXPECT_EQ(out, "loop_\n_x\n_y\naaaa z\np\n;c\nd\n;\n");

  CifBlock back = reparse("data_x\n" + out);
  ASSERT_TRUE(back) << out;
  std::vector<CifTable> tv;
  tv.push_back(make_table(
      {
          "_x", "_y"
  },
      { { CifValue::generic("aaaa"), CifValue::generic("z") },
        { CifValue::generic("p"), CifValue::generic("c\nd") } }));
  CifFrame frame(std::move(tv), "x");
  EXPECT_EQ(frame_map(frame), frame_map(back.data()));
}

TEST_P(CifWriteAlignTest, UnrepresentablePropagates) {
  // A value whose next line begins with ';' cannot be serialized; the failure
  // must propagate out of table/frame/block serialization.
  auto bad_table = [] {
    return make_table(
        {
            "_x", "_y"
    },
        { { CifValue::generic("ok"), CifValue::generic("a\n;b") },
          { CifValue::generic("p"), CifValue::generic("q") } });
  };

  {
    CifTable table = bad_table();
    std::string out;
    EXPECT_FALSE(write_cif_table(out, table, align()));
    EXPECT_PRED2(str_case_contains, out, "';'");
  }

  // single-row table -> key-value pairs path
  {
    CifTable pairs = make_table({ "_x" }, { { CifValue::generic("a\n;b") } });
    std::string out;
    EXPECT_FALSE(write_cif_table(out, pairs, align()));
    EXPECT_PRED2(str_case_contains, out, "';'");
  }

  std::vector<CifTable> tv;
  tv.push_back(bad_table());
  CifFrame frame(std::move(tv), "blk");
  std::string fout;
  EXPECT_FALSE(write_cif_frame(fout, frame, CifFrame::Type::kData, align()));
  EXPECT_PRED2(str_case_contains, fout, "';'");

  std::vector<CifTable> btv;
  btv.push_back(bad_table());
  CifBlock block(CifFrame(std::move(btv), "blk"), {}, CifBlock::Type::kData);
  std::string bout;
  EXPECT_FALSE(write_cif_block(bout, block, align()));
  EXPECT_PRED2(str_case_contains, bout, "';'");

  // global block with a valid frame but an unrepresentable value
  std::vector<CifTable> gtv;
  gtv.push_back(bad_table());
  CifBlock gblock(CifFrame(std::move(gtv), ""), {}, CifBlock::Type::kGlobal);
  std::string gout;
  EXPECT_FALSE(write_cif_block(gout, gblock, align()));
  EXPECT_PRED2(str_case_contains, gout, "';'");

  // data block whose save frame carries an unrepresentable value
  std::vector<CifTable> main_tv;
  {
    CifTable t;
    t.add_key("_ok");
    t.add_data(CifValue::generic("1"));
    main_tv.push_back(std::move(t));
  }
  std::vector<CifFrame> saves;
  std::vector<CifTable> stv;
  stv.push_back(bad_table());
  saves.emplace_back(std::move(stv), "save1");
  CifBlock sblock(CifFrame(std::move(main_tv), "blk"), std::move(saves),
                  CifBlock::Type::kData);
  std::string sout;
  EXPECT_FALSE(write_cif_block(sout, sblock, align()));
  EXPECT_PRED2(str_case_contains, sout, "';'");
}

TEST(CifFrameValidateTest, RejectsMalformed) {
  // 2 keys but a single data value -> inconsistent table.
  std::vector<CifTable> itv;
  itv.push_back(make_table({ "_a", "_b" }, { { CifValue::generic("1") } }));
  EXPECT_PRED2(str_case_contains, CifFrame(std::move(itv), "blk").validate(),
               "inconsistent");

  // duplicate key across the frame.
  std::vector<CifTable> dtv;
  dtv.push_back(make_table({ "_a" }, { { CifValue::generic("1") } }));
  dtv.push_back(make_table({ "_a" }, { { CifValue::generic("2") } }));
  EXPECT_PRED2(str_case_contains, CifFrame(std::move(dtv), "blk").validate(),
               "duplicate");

  // malformed key name.
  std::vector<CifTable> ktv;
  ktv.push_back(make_table({ "bad key" }, { { CifValue::generic("1") } }));
  CifFrame kframe(std::move(ktv), "blk");
  EXPECT_PRED2(str_case_contains, kframe.validate(), "invalid cif key");
  // non-recursive skips per-table checks (assumes tables already validated).
  EXPECT_EQ(kframe.validate(false), "");

  // a well-formed frame validates clean.
  std::vector<CifTable> otv;
  otv.push_back(make_table({ "_a.x" }, { { CifValue::generic("1") } }));
  EXPECT_EQ(CifFrame(std::move(otv), "blk").validate(), "");
}

TEST(CifTableValidateTest, RejectsMalformed) {
  EXPECT_PRED2(
      str_case_contains,
      make_table({ "bad key" }, { { CifValue::generic("1") } }).validate(),
      "invalid cif key");

  EXPECT_PRED2(str_case_contains,
               make_table(
                   {
                       "_a", "_a"
  },
                   { { CifValue::generic("1"), CifValue::generic("2") } })
                   .validate(),
               "duplicate");

  EXPECT_PRED2(
      str_case_contains,
      make_table({ "_a", "_b" }, { { CifValue::generic("1") } }).validate(),
      "inconsistent");

  EXPECT_EQ(make_table({ "_a.x" }, { { CifValue::generic("1") } }).validate(),
            "");
}

TEST(CifBlockValidateTest, RejectsMalformed) {
  auto data = [] {
    std::vector<CifTable> tv;
    tv.push_back(make_table({ "_a.x" }, { { CifValue::generic("1") } }));
    return CifFrame(std::move(tv), "blk");
  };
  auto save = [](std::string_view name) {
    std::vector<CifTable> tv;
    tv.push_back(make_table({ "_s.y" }, { { CifValue::generic("2") } }));
    return CifFrame(std::move(tv), std::string(name));
  };
  auto block = [&](std::vector<CifFrame> saves) {
    return CifBlock(data(), std::move(saves), CifBlock::Type::kData);
  };

  std::vector<CifFrame> ok;
  ok.push_back(save("one"));
  ok.push_back(save("two"));
  EXPECT_EQ(block(std::move(ok)).validate(), "");

  std::vector<CifFrame> dup;
  dup.push_back(save("dup"));
  dup.push_back(save("dup"));
  EXPECT_PRED2(str_case_contains, block(std::move(dup)).validate(),
               "duplicate save frame");

  std::vector<CifFrame> unnamed;
  unnamed.push_back(save(""));
  EXPECT_PRED2(str_case_contains, block(std::move(unnamed)).validate(),
               "empty name");

  // A malformed table inside a save frame is reported recursively, tagged with
  // the offending save frame name.
  std::vector<CifFrame> bad_save;
  {
    std::vector<CifTable> tv;
    tv.push_back(
        make_table({ "_s.y", "_s.y" }, { { CifValue::generic("2") } }));
    bad_save.emplace_back(std::move(tv), "sf");
  }
  std::string bad_save_err = block(std::move(bad_save)).validate();
  EXPECT_PRED2(str_case_contains, bad_save_err, "duplicate cif key");
  EXPECT_PRED2(str_case_contains, bad_save_err, "in save frame sf");

  // sentinels carry no data frame.
  EXPECT_EQ(CifBlock::eof().validate(), "");
  EXPECT_EQ(CifBlock::error("boom").validate(), "");
}

TEST_P(CifWriteAlignTest, GlobalBlock) {
  std::vector<CifTable> tv;
  {
    CifTable t;
    t.add_key("_g.x");
    t.add_data(CifValue::generic("1"));
    tv.push_back(std::move(t));
  }
  CifBlock block(CifFrame(std::move(tv), ""), {}, CifBlock::Type::kGlobal);

  std::string out;
  ASSERT_TRUE(write_cif_block(out, block, align()));
  EXPECT_TRUE(absl::StartsWith(out, "global_")) << out;

  CifBlock back = reparse(out);
  ASSERT_TRUE(back) << out;
  EXPECT_EQ(back.type(), CifBlock::Type::kGlobal);
  EXPECT_EQ(frame_map(block.data()), frame_map(back.data()));
}

TEST(CifWriteBlockTest, NonSerializableBlock) {
  std::string out;
  EXPECT_FALSE(write_cif_block(out, CifBlock::eof()));
  EXPECT_PRED2(str_case_contains, out, "cannot serialize");

  out.clear();
  EXPECT_FALSE(write_cif_block(out, CifBlock::error("boom")));
  EXPECT_PRED2(str_case_contains, out, "cannot serialize");
}
}  // namespace
}  // namespace internal
}  // namespace nuri
