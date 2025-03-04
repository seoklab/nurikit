//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/base.h"

#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "nuri/fmt/sdf.h"

namespace nuri {
namespace {
TEST(ReversedStreamTest, HandleEmptyFile) {
  std::istringstream input("");
  ReversedStream reversed(input, '\n', 7);

  std::string line;
  ASSERT_FALSE(reversed.getline(line));
}

TEST(ReversedStreamTest, HandleSingleLineFile) {
  std::istringstream input("single line");
  ReversedStream reversed(input, '\n', 7);

  std::string line;
  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "single line");

  ASSERT_FALSE(reversed.getline(line));
}

TEST(ReversedStreamTest, HandleMultipleNewlines) {
  std::istringstream input("single line\n\n");
  ReversedStream reversed(input, '\n', 7);

  std::string line;
  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "");

  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "single line");

  ASSERT_FALSE(reversed.getline(line));
}

TEST(ReversedStreamTest, ReadBackwardsLines) {
  std::istringstream input("line1\nline2\nline3\nline4");
  ReversedStream reversed(input, '\n', 7);

  std::string line;
  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "line4");

  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "line3");

  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "line2");

  ASSERT_TRUE(reversed.getline(line));
  EXPECT_EQ(line, "line1");

  ASSERT_FALSE(reversed.getline(line));
}

TEST(ReversedStreamTest, ReadBackwardsMixed) {
  std::istringstream iss(" a   bcd ");
  std::string tok;

  std::vector<std::string> forward;
  while (std::getline(iss, tok, ' ')) {
    forward.push_back(tok);
  }

  iss.clear();
  ReversedStream rs(iss, ' ', 2);
  std::vector<std::string> backward;
  while (rs.getline(tok)) {
    backward.push_back(tok);
  }

  for (size_t i = 0; i < forward.size(); ++i) {
    EXPECT_EQ(forward[i], backward[backward.size() - i - 1]);
  }
}

TEST(EscapeTest, EscapeAll) {
  // unicode thumbs up emoji (utf8)
  std::string_view unsafe = " \ta\nb\tc\rd e \xf0\x9f\x91\x8d \n";
  std::string escaped = internal::ascii_safe(unsafe);
  EXPECT_EQ(escaped, "  a_b_c_d_e_????  ");
}

TEST(EscapeTest, EscapeNewlines) {
  // unicode thumbs up emoji (utf8)
  std::string_view unsafe = " \ta\nb\tc\rd e \xf0\x9f\x91\x8d \n";
  std::string escaped = internal::ascii_newline_safe(unsafe);
  EXPECT_EQ(escaped, " \ta b\tc d e ????  ");
}

class DummyReader: public MoleculeReader {
public:
  DummyReader(std::istream & /* is */) { }

  bool getnext(std::vector<std::string> & /* block */) override {
    return false;
  }

  Molecule parse(const std::vector<std::string> & /* block */) const override {
    return {};
  }

  bool bond_valid() const override { return true; }
};

class DummyReaderFactory: public DefaultReaderFactoryImpl<DummyReader> { };

TEST(ReaderFactoryTest, CanFindFactory) {
  // Direct comparison of typeid fails on macOS x86_64.
  // seoklab/nurikit#459
  auto factory = dynamic_cast<const SDFReaderFactory *>(
      MoleculeReaderFactory::find_factory("sdf"));
  EXPECT_NE(factory, nullptr);
}

TEST(ReaderFactoryTest, CanRegisterFactory) {
  const MoleculeReaderFactory *factory =
      MoleculeReaderFactory::find_factory("dummy");
  ASSERT_EQ(factory, nullptr);

  MoleculeReaderFactory::register_factory(
      std::make_unique<DummyReaderFactory>(), { "dummy" });
  auto dummy = dynamic_cast<const DummyReaderFactory *>(
      MoleculeReaderFactory::find_factory("dummy"));
  ASSERT_NE(dummy, nullptr);
  dummy->register_for("also-dummy");

  dummy = dynamic_cast<const DummyReaderFactory *>(
      MoleculeReaderFactory::find_factory("also-dummy"));
  ASSERT_NE(dummy, nullptr);
}
}  // namespace
}  // namespace nuri
