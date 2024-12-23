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

#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string_view>
#include <vector>

#include <absl/strings/match.h>
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
}  // namespace
}  // namespace internal
}  // namespace nuri
