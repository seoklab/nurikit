//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_BASE_H_
#define NURI_FMT_BASE_H_

//! @cond
#include <cstddef>
#include <istream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
//! @endcond

#include "nuri/core/container/dumb_buffer.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/parse_result.h"
#include "nuri/utils.h"

namespace nuri {
class MoleculeReader;
class MoleculeRecord;

template <class Reader = MoleculeReader, class Record = typename Reader::Record>
class MoleculeStream {
public:
  MoleculeStream(Reader &reader)
      : reader_(&reader), record_(down_cast<Record>(reader.make_record())) { }

  MoleculeStream(const MoleculeStream &) = delete;
  MoleculeStream &operator=(const MoleculeStream &) = delete;
  MoleculeStream(MoleculeStream &&) noexcept = default;
  MoleculeStream &operator=(MoleculeStream &&) noexcept = default;
  ~MoleculeStream() noexcept = default;

  /**
   * @brief Advance the stream to the next molecule.
   *
   * @return true if the stream is not at the end, false otherwise.
   * @note If this function returns false, the current result is invalidated.
   */
  ABSL_MUST_USE_RESULT bool advance() {
    ABSL_DCHECK(record_ != nullptr);

    while (true) {
      res_ = record_->next();
      if (res_.status() != ParseStatus::kEOF)
        return true;
      if (!reader_->getnext(*record_))
        return false;
    }
  }

  /**
   * @brief Get the current parse result.
   * @return Reference to the current molecule, parse error, or EOF result.
   * @note The result is EOF before the first advance() and after it returns
   *       false. Each call to advance() replaces the result.
   * @note The result may be moved out. Call advance() before accessing the
   *       moved-from result again.
   */
  ParseResult<Molecule> &state() { return res_; }

  //! @copydoc state()
  const ParseResult<Molecule> &state() const { return res_; }

  /**
   * @brief Get the current molecule.
   * @return Reference to the current molecule.
   * @pre state() must contain a valid molecule, otherwise the behavior is
   *      undefined.
   */
  Molecule &current() { return *res_; }

  /**
   * @brief Get the current molecule.
   * @return Const reference to the current molecule.
   * @pre state() must contain a valid molecule, otherwise the behavior is
   *      undefined.
   */
  const Molecule &current() const { return *res_; }

private:
  Reader *reader_;
  std::unique_ptr<Record> record_;
  ParseResult<Molecule> res_;
};

/**
 * @brief Read the next molecule from the stream.
 * @note \p mol is left unchanged if the stream is at the end or the next block
 *       could not be parsed. This operator cannot report the reason; use
 *       MoleculeStream::state() after advance() to distinguish the two and
 *       obtain the failure reason.
 */
template <class Stream>
Stream &operator>>(Stream &stream, Molecule &mol) {
  if (stream.advance() && stream.state()) {
    mol = std::move(stream.current());
  }
  return stream;
}

class MoleculeRecord {
public:
  MoleculeRecord() = default;
  MoleculeRecord(const MoleculeRecord &) = delete;
  MoleculeRecord &operator=(const MoleculeRecord &) = delete;
  MoleculeRecord(MoleculeRecord &&) noexcept = default;
  MoleculeRecord &operator=(MoleculeRecord &&) noexcept = default;
  virtual ~MoleculeRecord() noexcept = default;

  virtual ParseResult<Molecule> next() = 0;
  virtual void reset() noexcept = 0;
};

template <class T, auto parser>
class TextRecordImpl final: public MoleculeRecord {
public:
  ParseResult<Molecule> next() final {
    if (text_.empty())
      return ParseResult<Molecule>::eof();

    ParseResult<Molecule> res = parser(text_);
    reset();
    return res;
  }

  void reset() noexcept final { text_.clear(); }

  T &text() { return text_; }

  const T &text() const { return text_; }

private:
  T text_;
};

class MoleculeReader {
public:
  using Record = MoleculeRecord;

  MoleculeReader() = default;
  MoleculeReader(const MoleculeReader &) = delete;
  MoleculeReader &operator=(const MoleculeReader &) = delete;
  MoleculeReader(MoleculeReader &&) noexcept = default;
  MoleculeReader &operator=(MoleculeReader &&) noexcept = default;
  virtual ~MoleculeReader() noexcept = default;

  virtual std::unique_ptr<MoleculeRecord> make_record() const = 0;

  /**
   * @brief Advance the reader to the next molecule.
   * @return The next record, or an empty record at the end of the stream.
   */
  std::unique_ptr<MoleculeRecord> next() {
    auto record = make_record();
    ABSL_DCHECK(record != nullptr);
    static_cast<void>(getnext(*record));
    return record;
  }

  /**
   * @brief Advance the reader to the next molecule.
   * @param record The record containing the next molecule(s). If true is
   *               returned, pre-existing contents of the record are discarded.
   *               Otherwise, the record is reset (empty).
   * @return true if the reader has successfully advanced to the next molecule,
   *         false otherwise.
   * @note The name of this method is loosely based on the std::getline()
   *       function due to its similar semantics.
   */
  ABSL_MUST_USE_RESULT bool getnext(MoleculeRecord &record) {
    record.reset();
    if (fill(record))
      return true;
    record.reset();
    return false;
  }

  /**
   * @brief Test whether the reader implementation can provide valid bond
   *        information.
   */
  virtual bool bond_valid() const = 0;

  /**
   * @brief Convert the reader to a stream object.
   */
  MoleculeStream<MoleculeReader> stream() { return { *this }; }

private:
  virtual bool fill(MoleculeRecord &record) = 0;
};

class StreamReaderBase: public MoleculeReader {
public:
  StreamReaderBase(std::istream &is): is_(&is) { }

protected:
  // NOLINTBEGIN(*-non-private-member-variables-in-classes)
  std::istream *is_;
  // NOLINTEND(*-non-private-member-variables-in-classes)
};

class MoleculeReaderFactory {
public:
  MoleculeReaderFactory() = default;
  MoleculeReaderFactory(const MoleculeReaderFactory &) = default;
  MoleculeReaderFactory &operator=(const MoleculeReaderFactory &) = default;
  MoleculeReaderFactory(MoleculeReaderFactory &&) noexcept = default;
  MoleculeReaderFactory &operator=(MoleculeReaderFactory &&) noexcept = default;
  virtual ~MoleculeReaderFactory() noexcept = default;

  /**
   * @brief Create a new reader from the given istream object.
   * @param is The input stream to read from.
   * @return A new reader instance.
   * @note The istream must survive until the returned reader is destructed.
   */
  virtual std::unique_ptr<MoleculeReader>
  from_stream(std::istream &is) const = 0;

  /**
   * @brief Find the factory for the given format name
   * @param name The name of the format to find the factory for.
   * @return A pointer to the factory instance for the given format name, or
   *         nullptr if no factory is registered for the given name.
   */
  static const MoleculeReaderFactory *find_factory(std::string_view name);

  /**
   * @brief Register the factory for the given format name(s).
   * @param factory The factory instance to register.
   * @param names The name(s) of the format to register the factory for.
   * @return Always true.
   * @sa register_for()
   *
   * This function is intended for user-defined factories. If you want to
   * register library-provided factory for custom alias name, register_for()
   * is the right function to use.
   *
   * @note This will always register the factory even if the \p names are empty.
   *       However, such registration is useless; the factory will never be
   *       found by find_factory().
   * @note This function is not thread-safe. Some synchronization mechanism
   *       must be used to call register_*() functions from multiple threads.
   */
  static bool register_factory(std::unique_ptr<MoleculeReaderFactory> factory,
                               const std::vector<std::string> &names);

  /**
   * @brief Register this factory for the given alias name.
   * @param alias An alias name to register the factory for.
   * @sa register_factory()
   *
   * This function is intended for giving an alias name to library-provided
   * factory. If you want to register user-defined factory, register_factory()
   * is the right function to use.
   *
   * @note The instance of the factory must be existing until the end of the
   *       program. The easiest way to achieve this is using the returned
   *       factory instance from find_factory().
   * @note This function is not thread-safe. Some synchronization mechanism
   *       must be used to call register_*() functions from multiple threads.
   */
  void register_for(std::string_view alias) const {
    register_for_name(this, alias);
  }

private:
  static void register_for_name(const MoleculeReaderFactory *factory,
                                std::string_view name);
};

template <class ReaderFactoryImpl>
bool register_reader_factory(const std::vector<std::string> &names) {
  return MoleculeReaderFactory::register_factory(
      std::make_unique<ReaderFactoryImpl>(), names);
}

template <class MoleculeReaderImpl>
class DefaultReaderFactoryImpl: public MoleculeReaderFactory {
public:
  std::unique_ptr<MoleculeReader> from_stream(std::istream &is) const final {
    return std::make_unique<MoleculeReaderImpl>(is);
  }
};

class ReversedStream {
public:
  ReversedStream(std::istream &is, char delim = '\n', std::size_t bufsz = 4096)
      : is_(&is), delim_(delim), buf_(bufsz) {
    reset();
  }

  void reset();

  bool getline(std::string &line);

private:
  void read_block();

  std::istream *is_;
  std::size_t prev_;
  char delim_;
  internal::DumbBuffer<char> buf_;
};

namespace internal {
  /**
   * @brief Replace non-ascii and non-printable characters with '?' and replace
   *        all whitespace characters with '_'.
   *
   * @param str The string to sanitize.
   * @return The sanitized string.
   */
  extern std::string ascii_safe(std::string_view str);

  /**
   * @brief Replace non-ascii and non-printable characters with '?' and replace
   *        all newline characters with ' '.
   *
   * @param str The string to sanitize.
   * @return The sanitized string.
   */
  extern std::string ascii_newline_safe(std::string_view str);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_FMT_BASE_H_ */
