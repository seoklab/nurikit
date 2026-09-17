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

TEST(TextBlockTest, PushBackAndIndex) {
  internal::TextBlock block;
  EXPECT_TRUE(block.empty());
  EXPECT_EQ(block.size(), 0);
  EXPECT_EQ(block.end() - block.begin(), 0);

  block.push_back("first");
  block.push_back("");
  block.push_back("third");

  EXPECT_FALSE(block.empty());
  EXPECT_EQ(block.size(), 3);
  EXPECT_EQ(block[0], "first");
  EXPECT_EQ(block[1], "");
  EXPECT_EQ(block[2], "third");
  EXPECT_EQ(block.front(), "first");
  EXPECT_EQ(block.back(), "third");
}

TEST(TextBlockTest, EmptyLineIsALine) {
  internal::TextBlock block;
  block.push_back("");
  EXPECT_FALSE(block.empty());
  EXPECT_EQ(block.size(), 1);
  EXPECT_EQ(block[0], "");
}

TEST(TextBlockTest, InitializerList) {
  internal::TextBlock block { "a", "bc", "" };
  ASSERT_EQ(block.size(), 3);
  EXPECT_EQ(block[1], "bc");
  EXPECT_EQ(block.back(), "");
}

TEST(TextBlockTest, Iterator) {
  internal::TextBlock block { "a", "bc", "def" };
  auto it = block.begin();
  const auto end = block.end();
  EXPECT_EQ(end - it, 3);
  EXPECT_EQ(*it, "a");
  EXPECT_EQ(it[2], "def");

  ++it;
  EXPECT_LT(it, end);
  EXPECT_EQ(*it, "bc");

  it += 2;
  EXPECT_EQ(it, end);

  --it;
  EXPECT_EQ(*it, "def");

  std::vector<std::string_view> lines(block.begin(), block.end());
  EXPECT_EQ(lines, (std::vector<std::string_view> { "a", "bc", "def" }));
}

TEST(TextBlockTest, Append) {
  internal::TextBlock block { "x", "yy" };
  internal::TextBlock other { "", "zzz" };
  block.append(other);
  ASSERT_EQ(block.size(), 4);
  EXPECT_EQ(block[0], "x");
  EXPECT_EQ(block[1], "yy");
  EXPECT_EQ(block[2], "");
  EXPECT_EQ(block[3], "zzz");

  internal::TextBlock empty;
  block.append(empty);
  EXPECT_EQ(block.size(), 4);

  empty.append(block);
  ASSERT_EQ(empty.size(), 4);
  EXPECT_EQ(empty.front(), "x");
  EXPECT_EQ(empty.back(), "zzz");
}

TEST(TextBlockTest, ClearReuse) {
  internal::TextBlock block { "a", "b" };
  block.clear();
  EXPECT_TRUE(block.empty());
  EXPECT_EQ(block.size(), 0);

  block.push_back("c");
  ASSERT_EQ(block.size(), 1);
  EXPECT_EQ(block[0], "c");
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

ParseResult<Molecule> stub_parse(const internal::TextBlock &block) {
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
    return std::make_unique<TextRecordImpl<internal::TextBlock, stub_parse>>();
  }

  bool bond_valid() const override { return true; }

private:
  bool fill(MoleculeRecord & /* record */) override { return false; }
};

class DummyReaderFactory: public DefaultReaderFactoryImpl<DummyReader> { };

class StubReader: public MoleculeReader {
public:
  using Record = TextRecordImpl<internal::TextBlock, stub_parse>;

  std::unique_ptr<MoleculeRecord> make_record() const override {
    return std::make_unique<Record>();
  }

  bool bond_valid() const override { return true; }

private:
  bool fill(MoleculeRecord &record) override {
    auto &text_record = down_cast<Record &>(record);
    auto &block = text_record.text();
    block.clear();
    if (next_ >= 3)
      return false;

    block.push_back(std::to_string(next_++));
    return true;
  }

  int next_ = 0;
};

TEST(MoleculeBatchTest, ReportsParseStatus) {
  StubReader reader;
  auto record = reader.make_record();
  EXPECT_EQ(record->parse().status(), ParseStatus::kEOF);

  ASSERT_TRUE(reader.getnext(*record));
  auto first = record->parse();
  ASSERT_TRUE(first);
  ASSERT_EQ(first->data().size(), 1);
  EXPECT_EQ(first->data()[0].name(), "0");
  EXPECT_TRUE(first->data()[0].empty());

  ASSERT_TRUE(reader.getnext(*record));
  auto error = record->parse();
  ASSERT_EQ(error.status(), ParseStatus::kError);
  EXPECT_EQ(error.error_msg(), "stub failure");

  ASSERT_TRUE(reader.getnext(*record));
  auto last = record->parse();
  ASSERT_TRUE(last);
  ASSERT_EQ(last->data().size(), 1);
  EXPECT_EQ(last->data()[0].num_atoms(), 1);

  EXPECT_FALSE(reader.getnext(*record));
  EXPECT_EQ(record->parse().status(), ParseStatus::kEOF);
  EXPECT_FALSE(reader.getnext(*record));
  EXPECT_EQ(record->parse().status(), ParseStatus::kEOF);

  EXPECT_EQ(first->data()[0].name(), "0");
  EXPECT_EQ(error.error_msg(), "stub failure");
  EXPECT_EQ(last->data()[0].name(), "2");
}

TEST(MoleculeBatchTest, OwnsMovedContainer) {
  MoleculeBatch::Container molecules;
  for (int i = 0; i < 3; ++i) {
    Molecule mol;
    mol.name() = std::to_string(i);
    molecules.push_back(std::move(mol));
  }
  MoleculeBatch batch(std::move(molecules));
  const MoleculeBatch &view = batch;
  ASSERT_EQ(view.data().size(), 3);
  EXPECT_EQ(view.data()[0].name(), "0");
  EXPECT_EQ(view.data()[2].name(), "2");
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
