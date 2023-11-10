//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_BASE_H_
#define NURI_FMT_BASE_H_

#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <absl/base/attributes.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_log.h>

#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
class MoleculeReader;

template <class Reader = MoleculeReader>
class MoleculeStream {
public:
  MoleculeStream(Reader &reader): reader_(&reader) { }

  MoleculeStream(const MoleculeStream &) = delete;
  MoleculeStream &operator=(const MoleculeStream &) = delete;
  MoleculeStream(MoleculeStream &&) noexcept = default;
  MoleculeStream &operator=(MoleculeStream &&) noexcept = default;
  ~MoleculeStream() noexcept = default;

  /**
   * @brief Advance the stream to the next molecule.
   *
   * @return true if the stream is not at the end, false otherwise.
   * @note If this function returns false, the current molecule is not changed.
   */
  ABSL_MUST_USE_RESULT bool advance() {
    block_ = reader_->next();
    if (block_.empty()) {
      return false;
    }

    mol_ = reader_->parse(block_);
    return true;
  }

  Molecule &current() { return mol_; }

  const Molecule &current() const { return mol_; }

private:
  Reader *reader_;
  std::vector<std::string> block_;
  Molecule mol_;
};

template <class Stream>
Stream &operator>>(Stream &stream, Molecule &mol) {
  if (stream.advance()) {
    mol = stream.current();
  }
  return stream;
}

class MoleculeReader {
public:
  MoleculeReader() = default;
  MoleculeReader(const MoleculeReader &) = delete;
  MoleculeReader &operator=(const MoleculeReader &) = delete;
  MoleculeReader(MoleculeReader &&) noexcept = default;
  MoleculeReader &operator=(MoleculeReader &&) noexcept = default;
  virtual ~MoleculeReader() noexcept = default;

  /**
   * @brief Advance the reader to the next molecule.
   * @return The next block containing the next molecule. If the stream is at
   *         the end, an empty block is returned.
   */
  virtual std::vector<std::string> next() = 0;

  /**
   * @brief Parse the current block and return the molecule.
   * @param block The block to parse.
   * @return The current molecule.
   * @note The returned molecule will be empty if the block is empty.
   */
  virtual Molecule parse(const std::vector<std::string> &block) const = 0;

  MoleculeStream<MoleculeReader> stream() { return { *this }; }
};

template <auto parser>
class DefaultReaderImpl: public MoleculeReader {
public:
  DefaultReaderImpl() = default;
  DefaultReaderImpl(std::istream &is): is_(&is) { }

  DefaultReaderImpl(const DefaultReaderImpl &) = delete;
  DefaultReaderImpl &operator=(const DefaultReaderImpl &) = delete;
  DefaultReaderImpl(DefaultReaderImpl &&) noexcept = default;
  DefaultReaderImpl &operator=(DefaultReaderImpl &&) noexcept = default;

  ~DefaultReaderImpl() noexcept override = default;

  Molecule parse(const std::vector<std::string> &block) const final {
    return parser(block);
  }

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

template <class SourceStream, class Reader = MoleculeReader>
class MoleculeReaderWrapper {
public:
  template <class... Args>
  MoleculeReaderWrapper(std::string_view fmt, Args &&...args)
      : is_(std::forward<Args>(args)...) {
    const MoleculeReaderFactory *factory =
        MoleculeReaderFactory::find_factory(fmt);

    if (ABSL_PREDICT_FALSE(factory == nullptr)) {
      ABSL_LOG(WARNING) << "No factory found for " << fmt;
      return;
    }

    reader_ = static_unique_ptr_cast<Reader>(factory->from_stream(is_));
  }

  /**
   * @brief Advance the stream to the next molecule.
   * @return The next block containing the next molecule. If the stream is at
   *         the end, an empty block is returned.
   */
  std::vector<std::string> next() { return reader_->next(); }

  /**
   * @brief Parse the current block and return the molecule.
   * @param block The block to parse.
   * @return The current molecule.
   * @note The returned molecule will be empty if the block is empty.
   */
  Molecule parse(const std::vector<std::string> &block) const {
    return reader_->parse(block);
  }

  MoleculeStream<Reader> stream() { return { *reader_ }; }

  operator bool() const { return is_ && reader_; }

private:
  SourceStream is_;
  std::unique_ptr<Reader> reader_;
};

template <class Reader>
class MoleculeReaderWrapper<std::ifstream, Reader> {
public:
  explicit MoleculeReaderWrapper(const std::filesystem::path &path): is_(path) {
    const std::filesystem::path full_ext = path.extension();
    const std::string_view ext = extension_no_dot(full_ext);

    const MoleculeReaderFactory *factory =
        MoleculeReaderFactory::find_factory(ext);

    if (ABSL_PREDICT_FALSE(factory == nullptr)) {
      ABSL_LOG(WARNING) << "No factory found for " << ext;
      return;
    }

    reader_ = static_unique_ptr_cast<Reader>(factory->from_stream(is_));
  }

  MoleculeReaderWrapper(std::string_view fmt, const std::filesystem::path &path)
      : is_(path) {
    const MoleculeReaderFactory *factory =
        MoleculeReaderFactory::find_factory(fmt);

    if (ABSL_PREDICT_FALSE(factory == nullptr)) {
      ABSL_LOG(WARNING) << "No factory found for " << fmt;
      return;
    }

    reader_ = static_unique_ptr_cast<Reader>(factory->from_stream(is_));
  }

  /**
   * @brief Advance the stream to the next molecule.
   * @return The next block containing the next molecule. If the stream is at
   *         the end, an empty block is returned.
   */
  std::vector<std::string> next() { return reader_->next(); }

  /**
   * @brief Parse the current block and return the molecule.
   * @param block The block to parse.
   * @return The current molecule.
   * @note The returned molecule will be empty if the block is empty.
   */
  Molecule parse(const std::vector<std::string> &block) const {
    return reader_->parse(block);
  }

  MoleculeStream<Reader> stream() { return { *reader_ }; }

  operator bool() const { return is_ && reader_; }

private:
  std::ifstream is_;
  std::unique_ptr<Reader> reader_;
};

template <class Reader = MoleculeReader>
using FileMoleculeReader = MoleculeReaderWrapper<std::ifstream, Reader>;

template <class Reader = MoleculeReader>
using StringMoleculeReader = MoleculeReaderWrapper<std::istringstream, Reader>;
}  // namespace nuri

#endif /* NURI_FMT_BASE_H_ */
