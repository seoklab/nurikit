//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "nuri/fmt/base.h"
#include "nuri/fmt/mol2.h"
#include "nuri/fmt/pdb.h"
#include "nuri/fmt/sdf.h"
#include "nuri/fmt/smiles.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
constexpr char kInput[] = "first\nempty\nbad\nlast\n";
const std::vector<std::string> kExpected { "first", "error: dummy failure",
                                           "last" };

ParseResult<MoleculeBatch> dummy_parse(const std::string &text) {
  if (text == "bad")
    return ParseResult<MoleculeBatch>::error("dummy failure");
  if (text == "empty")
    return MoleculeBatch(MoleculeBatch::Container {});
  Molecule mol;
  mol.name() = text;
  return MoleculeBatch(std::move(mol));
}

class DummyReader final: public StreamReaderBase {
public:
  using Record = TextRecordImpl<std::string, dummy_parse>;
  using StreamReaderBase::StreamReaderBase;

  std::unique_ptr<MoleculeRecord> make_record() const override {
    return std::make_unique<Record>();
  }

  bool bond_valid() const override { return true; }

private:
  bool fill(MoleculeRecord &record) override {
    auto &text = down_cast<Record &>(record).text();
    text.clear();
    return static_cast<bool>(std::getline(*is_, text));
  }
};

std::vector<std::string> consume(MoleculeRecord &record) {
  std::vector<std::string> results;
  auto result = record.parse();
  if (result.status() == ParseStatus::kError) {
    results.push_back("error: " + std::move(result).error_msg());
  } else if (result) {
    for (const auto &mol: result->data())
      results.push_back(mol.name());
  }
  return results;
}

TEST(MoleculeRecordTest, RetainedAfterReaderDestruction) {
  std::vector<std::unique_ptr<MoleculeRecord>> records;
  {
    std::istringstream is(kInput);
    DummyReader reader(is);
    while (true) {
      auto record = reader.make_record();
      EXPECT_EQ(record->parse().status(), ParseStatus::kEOF);
      if (!reader.getnext(*record))
        break;
      records.push_back(std::move(record));
    }
    auto exhausted = reader.next();
    ASSERT_NE(exhausted, nullptr);
    EXPECT_EQ(exhausted->parse().status(), ParseStatus::kEOF);
  }
  ASSERT_EQ(records.size(), 4);
  std::vector<std::string> actual;
  for (auto it = records.rbegin(); it != records.rend(); ++it) {
    auto results = consume(**it);
    actual.insert(actual.begin(), results.begin(), results.end());
  }
  EXPECT_EQ(actual, kExpected);
}

TEST(MoleculeRecordTest, FillAndReuse) {
  std::istringstream input(kInput);
  DummyReader reader(input);
  DummyReader::Record record;
  EXPECT_EQ(record.parse().status(), ParseStatus::kEOF);
  ASSERT_TRUE(reader.getnext(record));
  EXPECT_EQ(record.text(), "first");

  ASSERT_TRUE(reader.getnext(record));
  auto empty = record.parse();
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->data().empty());
  ASSERT_TRUE(reader.getnext(record));
  auto error = record.parse();
  ASSERT_EQ(error.status(), ParseStatus::kError);
  ASSERT_TRUE(reader.getnext(record));
  EXPECT_EQ(consume(record), (std::vector<std::string> { "last" }));

  record.text() = "unconsumed";
  EXPECT_FALSE(reader.getnext(record));
  EXPECT_TRUE(record.text().empty());
  EXPECT_EQ(record.parse().status(), ParseStatus::kEOF);
  EXPECT_FALSE(reader.getnext(record));
  EXPECT_EQ(record.parse().status(), ParseStatus::kEOF);
}

TEST(MoleculeRecordTest, AllocatingReaderPreservesEmptyBatches) {
  std::istringstream input(kInput);
  DummyReader reader(input);
  std::vector<std::string> results;
  int records = 0;
  while (true) {
    auto record = reader.next();
    ASSERT_NE(record, nullptr);
    auto result = record->parse();
    if (result.status() == ParseStatus::kEOF)
      break;
    ++records;
    if (!result) {
      results.push_back("error: " + std::move(result).error_msg());
    } else {
      for (const auto &mol: result->data())
        results.push_back(mol.name());
    }
  }
  EXPECT_EQ(records, 4);
  EXPECT_EQ(results, kExpected);
}

template <class Reader>
void check_failed_fill() {
  std::istringstream input;
  Reader reader(input);
  typename Reader::Record record;
  for (int i = 0; i < 2; ++i) {
    record.text().push_back({});
    EXPECT_FALSE(reader.getnext(record));
    EXPECT_TRUE(record.text().empty());
    EXPECT_EQ(record.parse().status(), ParseStatus::kEOF);
  }
}

TEST(MoleculeRecordTest, FailedTextFillsClearPreviousContents) {
  check_failed_fill<SmilesReader>();
  check_failed_fill<SDFReader>();
  check_failed_fill<Mol2Reader>();
  check_failed_fill<PDBReader>();
}

struct ParseGate {
  std::mutex mutex;
  std::condition_variable ready;
  int entered = 0;
};

struct ParallelInput {
  ParseGate *gate = nullptr;
  bool empty() const { return gate == nullptr; }
};

ParseResult<Molecule> parallel_parse(const ParallelInput &input) {
  auto &gate = *input.gate;
  std::unique_lock<std::mutex> lock(gate.mutex);
  ++gate.entered;
  gate.ready.notify_all();
  EXPECT_TRUE(gate.ready.wait_for(lock, std::chrono::seconds(5),
                                  [&] { return gate.entered == 2; }));
  return Molecule {};
}

TEST(MoleculeRecordTest, IndependentParsesOverlap) {
  ParseGate gate;
  TextRecordImpl<ParallelInput, parallel_parse> first, second;
  first.text().gate = &gate;
  second.text().gate = &gate;
  auto parse = [](MoleculeRecord &record) {
    auto result = record.parse();
    ASSERT_TRUE(result);
    EXPECT_EQ(result->data().size(), 1);
  };
  std::thread worker([&] { parse(first); });
  parse(second);
  worker.join();
}

TEST(MoleculeRecordTest, WorkerHandoff) {
  using Work = std::pair<std::size_t, std::unique_ptr<MoleculeRecord>>;
  std::deque<Work> queue;
  std::vector<std::vector<std::string>> results;
  std::mutex mutex;
  std::condition_variable ready;
  bool done = false;
  auto worker = [&] {
    while (true) {
      Work work;
      {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&] { return done || !queue.empty(); });
        if (queue.empty())
          return;
        work = std::move(queue.front());
        queue.pop_front();
      }
      ready.notify_all();
      SCOPED_TRACE("worker record consumption");
      SCOPED_TRACE(work.first);
      auto consumed = consume(*work.second);
      {
        std::scoped_lock<std::mutex> lock(mutex);
        results[work.first] = std::move(consumed);
      }
    }
  };
  std::thread first(worker), second(worker);
  {
    std::istringstream is(kInput);
    DummyReader reader(is);
    while (true) {
      auto record = reader.make_record();
      if (!reader.getnext(*record))
        break;
      {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&] { return queue.size() < 2; });
        queue.emplace_back(results.size(), std::move(record));
        results.emplace_back();
      }
      ready.notify_all();
    }
  }
  {
    std::scoped_lock<std::mutex> lock(mutex);
    done = true;
  }
  ready.notify_all();
  first.join();
  second.join();
  std::vector<std::string> actual;
  for (const auto &result: results)
    actual.insert(actual.end(), result.begin(), result.end());
  EXPECT_EQ(actual, kExpected);
}

}  // namespace
}  // namespace nuri
