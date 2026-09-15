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

ParseResult<Molecule> stub_parse(const std::vector<std::string> &block) {
  if (block[0] == "1")
    return ParseResult<Molecule>::error("stub failure");

  Molecule mol;
  mol.name() = block[0];
  if (block[0] == "2")
    mol.mutator().add_atom({});
  return ParseResult<Molecule>(std::move(mol));
}

class DummyReader: public MoleculeReader {
public:
  DummyReader(std::istream & /* is */) { }

  std::unique_ptr<MoleculeRecord> make_record() const override {
    return std::make_unique<
        TextRecordImpl<std::vector<std::string>, stub_parse>>();
  }

  bool bond_valid() const override { return true; }

private:
  bool fill(MoleculeRecord & /* record */) override { return false; }
};

class DummyReaderFactory: public DefaultReaderFactoryImpl<DummyReader> { };

class StubReader: public MoleculeReader {
public:
  using Record = TextRecordImpl<std::vector<std::string>, stub_parse>;

  std::unique_ptr<MoleculeRecord> make_record() const override {
    return std::make_unique<Record>();
  }

  bool bond_valid() const override { return true; }

private:
  bool fill(MoleculeRecord &record) override {
    auto &text_record = down_cast<Record &>(record);
    auto &block = text_record.text();
    if (next_ >= 3)
      return false;

    block.assign(1, std::to_string(next_++));
    return true;
  }

  int next_ = 0;
};

TEST(MoleculeStreamTest, ReportsParseStatus) {
  StubReader reader;
  MoleculeStream<> stream = reader.stream();
  const auto &view = stream;

  EXPECT_EQ(view.state().status(), ParseStatus::kEOF);

  ASSERT_TRUE(stream.advance());
  ASSERT_TRUE(stream.state());
  EXPECT_EQ(stream.current().name(), "0");
  EXPECT_TRUE(stream.current().empty());
  EXPECT_EQ(&view.current(), &*view.state());

  ASSERT_TRUE(stream.advance());
  EXPECT_FALSE(stream.state());
  EXPECT_EQ(stream.state().error_msg(), "stub failure");

  ASSERT_TRUE(stream.advance());
  ASSERT_TRUE(stream.state());
  EXPECT_EQ(stream.current().num_atoms(), 1);

  EXPECT_FALSE(stream.advance());
  EXPECT_EQ(view.state().status(), ParseStatus::kEOF);
  EXPECT_FALSE(stream.advance());
  EXPECT_EQ(view.state().status(), ParseStatus::kEOF);
}

TEST(MoleculeStreamTest, RetainsMovedResultsAcrossAdvancement) {
  StubReader reader;
  auto stream = reader.stream();

  ASSERT_TRUE(stream.advance());
  auto first = std::move(stream.state());
  ASSERT_TRUE(first);

  ASSERT_TRUE(stream.advance());
  auto error = std::move(stream.state());
  ASSERT_EQ(error.status(), ParseStatus::kError);

  ASSERT_TRUE(stream.advance());
  auto last = std::move(stream.state());
  ASSERT_TRUE(last);
  EXPECT_FALSE(stream.advance());

  EXPECT_EQ(first->name(), "0");
  EXPECT_EQ(error.error_msg(), "stub failure");
  EXPECT_EQ(last->name(), "2");
  EXPECT_EQ(last->num_atoms(), 1);
}

TEST(MoleculeStreamTest, ExtractKeepsValueOnFailure) {
  StubReader reader;
  MoleculeStream<> stream = reader.stream();

  Molecule mol;
  stream >> mol;
  EXPECT_EQ(mol.name(), "0");

  stream >> mol;
  EXPECT_EQ(mol.name(), "0");

  stream >> mol;
  EXPECT_EQ(mol.name(), "2");

  stream >> mol;
  EXPECT_EQ(mol.name(), "2");
  EXPECT_EQ(mol.num_atoms(), 1);
  EXPECT_EQ(stream.state().status(), ParseStatus::kEOF);
}

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
