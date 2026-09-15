//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

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

namespace nuri {
namespace {
constexpr char kInput[] = "first\n\nbad\nlast\n";
const std::vector<std::string> kExpected { "first", "error: dummy failure",
                                           "last" };

ParseResult<Molecule> dummy_parse(const std::string &text) {
  if (text == "bad")
    return ParseResult<Molecule>::error("dummy failure");
  Molecule mol;
  mol.name() = text;
  return mol;
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
    EXPECT_TRUE(text.empty());
    return static_cast<bool>(std::getline(*is_, text));
  }
};

std::vector<std::string> consume(MoleculeRecord &record) {
  std::vector<std::string> results;
  while (true) {
    auto res = record.next();
    if (res.status() == ParseStatus::kEOF)
      break;
    results.push_back(res ? res->name()
                          : "error: " + std::string(res.error_msg()));
  }
  EXPECT_EQ(record.next().status(), ParseStatus::kEOF);
  return results;
}

template <class Reader, class Record>
std::vector<std::string> consume(MoleculeStream<Reader, Record> stream) {
  std::vector<std::string> results;
  while (stream.advance()) {
    results.push_back(
        stream.state() ? stream.current().name()
                       : "error: " + std::string(stream.state().error_msg()));
  }
  EXPECT_FALSE(stream.advance());
  return results;
}

TEST(MoleculeRecordTest, RetainedAfterReaderDestruction) {
  std::vector<std::unique_ptr<MoleculeRecord>> records;
  {
    std::istringstream is(kInput);
    DummyReader reader(is);
    while (true) {
      auto record = reader.make_record();
      EXPECT_EQ(record->next().status(), ParseStatus::kEOF);
      if (!reader.getnext(*record))
        break;
      records.push_back(std::move(record));
    }
    auto exhausted = reader.next();
    ASSERT_NE(exhausted, nullptr);
    EXPECT_EQ(exhausted->next().status(), ParseStatus::kEOF);
    EXPECT_EQ(exhausted->next().status(), ParseStatus::kEOF);
  }
  ASSERT_EQ(records.size(), 4);
  std::vector<std::string> actual;
  for (auto it = records.rbegin(); it != records.rend(); ++it) {
    auto results = consume(**it);
    actual.insert(actual.begin(), results.begin(), results.end());
  }
  EXPECT_EQ(actual, kExpected);
}

TEST(MoleculeRecordTest, ResetAndReuse) {
  std::istringstream input(kInput);
  DummyReader reader(input);
  DummyReader::Record record;
  EXPECT_EQ(record.next().status(), ParseStatus::kEOF);
  ASSERT_TRUE(reader.getnext(record));
  EXPECT_EQ(record.text(), "first");
  record.reset();
  EXPECT_TRUE(record.text().empty());
  EXPECT_EQ(record.next().status(), ParseStatus::kEOF);

  ASSERT_TRUE(reader.getnext(record));
  EXPECT_EQ(record.next().status(), ParseStatus::kEOF);
  ASSERT_TRUE(reader.getnext(record));
  EXPECT_EQ(record.text(), "bad");
  ASSERT_TRUE(reader.getnext(record));
  EXPECT_EQ(consume(record), (std::vector<std::string> { "last" }));

  record.text() = "unconsumed";
  EXPECT_FALSE(reader.getnext(record));
  EXPECT_EQ(record.next().status(), ParseStatus::kEOF);
  EXPECT_FALSE(reader.getnext(record));
  EXPECT_EQ(record.next().status(), ParseStatus::kEOF);
}

TEST(MoleculeRecordTest, TypedStreamSkipsEmptyRecords) {
  std::istringstream input(kInput);
  DummyReader reader(input);
  EXPECT_EQ(consume(MoleculeStream<DummyReader>(reader)), kExpected);
}

TEST(MoleculeRecordTest, PolymorphicStreamSkipsEmptyRecords) {
  std::istringstream input(kInput);
  DummyReader reader(input);
  EXPECT_EQ(consume(reader.stream()), kExpected);
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
