//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_BASE_H_
#define NURI_FMT_BASE_H_

//! @cond
#include <cstddef>
#include <initializer_list>
#include <istream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/base/attributes.h>
#include <absl/container/inlined_vector.h>
#include <absl/log/absl_check.h>
#include <boost/iterator/iterator_facade.hpp>
//! @endcond

#include "nuri/core/container/dumb_buffer.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/parse_result.h"
#include "nuri/iterator.h"

namespace nuri {
class MoleculeBatch {
public:
  using Container = absl::InlinedVector<Molecule, 1>;

  MoleculeBatch(Molecule &&mol) noexcept { data_.emplace_back(std::move(mol)); }

  MoleculeBatch(Container &&mols) noexcept: data_(std::move(mols)) { }

  Container &data() { return data_; }

  const Container &data() const { return data_; }

private:
  Container data_;
};

class MoleculeRecord {
public:
  MoleculeRecord() = default;
  MoleculeRecord(const MoleculeRecord &) = delete;
  MoleculeRecord &operator=(const MoleculeRecord &) = delete;
  MoleculeRecord(MoleculeRecord &&) noexcept = default;
  MoleculeRecord &operator=(MoleculeRecord &&) noexcept = default;
  virtual ~MoleculeRecord() noexcept = default;

  /**
   * @brief Parse the record into an owning batch of molecules.
   * @return A batch (possibly empty), a parse error, or EOF for an empty input
   *         record. A newly constructed record represents EOF.
   * @pre This is the first parse since construction or the latest fill attempt.
   *      Calling parse() again without refilling is undefined behavior, even
   *      after an error, exception, or EOF. Parsing may modify the record.
   * @note The returned batch does not depend on the record or reader lifetime.
   */
  virtual ParseResult<MoleculeBatch> parse() = 0;
};

template <class T, auto parser>
class TextRecordImpl final: public MoleculeRecord {
public:
  ParseResult<MoleculeBatch> parse() override {
    if (text_.empty())
      return ParseResult<MoleculeBatch>::eof();

    return parser(text_).template cast<MoleculeBatch>();
  }

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
    return fill(record);
  }

  /**
   * @brief Test whether the reader implementation can provide valid bond
   *        information.
   */
  virtual bool bond_valid() const = 0;

private:
  /**
   * @brief Replace the record contents with the next input record.
   * @note Clear previous contents before reading, including on early exits.
   *       A false return must leave an empty record, even if reading populated
   *       part of it. Successful text records must contain nonempty input.
   */
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
  class TextBlock {
  public:
    class const_iterator
        : public ProxyIterator<const_iterator, std::string_view,
                               std::random_access_iterator_tag, int> {
    public:
      const_iterator() noexcept = default;

      constexpr const_iterator(const TextBlock &block, int index) noexcept
          : block_(&block), index_(index) { }

    private:
      friend class boost::iterator_core_access;

      std::string_view dereference() const noexcept {
        return (*block_)[index_];
      }

      constexpr bool equal(const const_iterator &other) const noexcept {
        return index_ == other.index_;
      }

      constexpr void increment() noexcept { ++index_; }

      constexpr void decrement() noexcept { --index_; }

      constexpr void advance(int n) noexcept { index_ += n; }

      constexpr int distance_to(const const_iterator &other) const noexcept {
        return other.index_ - index_;
      }

      const TextBlock *block_;
      int index_;
    };

    TextBlock() = default;

    TextBlock(std::initializer_list<std::string_view> lines);

    void push_back(std::string_view line);

    void append(const TextBlock &other);

    void clear() noexcept;

    bool empty() const { return segments_.size() == 1; }

    int size() const { return static_cast<int>(segments_.size()) - 1; }

    std::string_view operator[](int i) const {
      ABSL_DCHECK(i >= 0 && i < size());
      return std::string_view(data_.data() + segments_[i],
                              segments_[i + 1] - segments_[i]);
    }

    std::string_view front() const { return (*this)[0]; }

    std::string_view back() const { return (*this)[size() - 1]; }

    const_iterator begin() const { return { *this, 0 }; }

    const_iterator end() const { return { *this, size() }; }

  private:
    std::string data_;
    std::vector<int> segments_ { 0 };
  };

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
