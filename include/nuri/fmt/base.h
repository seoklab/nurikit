//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_BASE_H_
#define NURI_FMT_BASE_H_

#include <filesystem>
#include <fstream>
#include <istream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>

#include <absl/base/attributes.h>
#include <absl/log/absl_log.h>

#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
class MoleculeStream;

template <class Stream = MoleculeStream>
class MoleculeStreamIterator {
  static_assert(!std::is_const_v<Stream>,
                "Stream must not be const-qualified.");

public:
  using iterator_category = std::input_iterator_tag;
  using value_type = Molecule;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

  MoleculeStreamIterator(): stream_(nullptr), end_(true) { }
  MoleculeStreamIterator(Stream &stream)
    : stream_(&stream), end_(!stream_->advance()) { }
  MoleculeStreamIterator(const MoleculeStreamIterator &other) = default;
  MoleculeStreamIterator &
  operator=(const MoleculeStreamIterator &other) = default;
  MoleculeStreamIterator(MoleculeStreamIterator &&other) noexcept = default;
  MoleculeStreamIterator &
  operator=(MoleculeStreamIterator &&other) noexcept = default;
  ~MoleculeStreamIterator() noexcept = default;

  Molecule &operator*() { return mol_; }
  const Molecule &operator*() const { return mol_; }

  Molecule *operator->() { return &mol_; }
  const Molecule *operator->() const { return &mol_; }

  MoleculeStreamIterator &operator++() {
    end_ = !stream_->advance();
    if (!end_) {
      mol_ = stream_->current();
    }
    return *this;
  }

  MoleculeStreamIterator operator++(int) {
    MoleculeStreamIterator tmp(*this);
    ++(*this);
    return tmp;
  }

  bool operator==(const MoleculeStreamIterator &other) const {
    return (end_ && other.end_)
           || (!end_ && !other.end_ && stream_ == other.stream_);
  }

  bool operator!=(const MoleculeStreamIterator &other) const {
    return !(*this == other);
  }

private:
  Stream *stream_;
  Molecule mol_;
  bool end_;
};

class MoleculeStream {
public:
  MoleculeStream() = default;
  MoleculeStream(const MoleculeStream &) = delete;
  MoleculeStream &operator=(const MoleculeStream &) = delete;
  MoleculeStream(MoleculeStream &&) noexcept = default;
  MoleculeStream &operator=(MoleculeStream &&) noexcept = default;
  virtual ~MoleculeStream() noexcept = default;

  /**
   * @brief Advance the stream to the next molecule.
   * @return true if the stream is not at the end, false otherwise.
   */
  ABSL_MUST_USE_RESULT virtual bool advance() = 0;

  /**
   * @brief Get the current molecule.
   * @return The current molecule.
   * @pre Previous call to advance() must return true, otherwise an empty
   *      molecule is returned.
   * @note The returned molecule will be empty if the stream is at the end, or
   *       parsing of the current molecule failed.
   */
  virtual Molecule current() const = 0;

  /**
   * @brief Get the begin iterator of the stream.
   * @return The begin iterator of the stream.
   */
  MoleculeStreamIterator<> begin() { return MoleculeStreamIterator<>(*this); }

  /**
   * @brief Get the past-the-end iterator of the stream.
   * @return The past-the-end iterator of the stream.
   */
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  MoleculeStreamIterator<> end() { return MoleculeStreamIterator<>(); }
};

template <class Stream>
Stream &operator>>(Stream &stream, Molecule &mol) {
  if (stream.advance()) {
    mol = stream.current();
  }
  return stream;
}

template <class Block, Molecule (*parser)(const Block &)>
class DefaultStreamImpl: public MoleculeStream {
public:
  DefaultStreamImpl() = default;
  DefaultStreamImpl(std::istream &is): is_(&is) { }

  DefaultStreamImpl(const DefaultStreamImpl &) = delete;
  DefaultStreamImpl &operator=(const DefaultStreamImpl &) = delete;
  DefaultStreamImpl(DefaultStreamImpl &&) noexcept = default;
  DefaultStreamImpl &operator=(DefaultStreamImpl &&) noexcept = default;

  ~DefaultStreamImpl() noexcept override = default;

  Molecule current() const override { return parser(block_); }

protected:
  // NOLINTBEGIN(*-non-private-member-variables-in-classes)
  std::istream *is_;
  Block block_;
  // NOLINTEND(*-non-private-member-variables-in-classes)
};

class MoleculeStreamFactory {
public:
  MoleculeStreamFactory() = default;
  MoleculeStreamFactory(const MoleculeStreamFactory &) = default;
  MoleculeStreamFactory &operator=(const MoleculeStreamFactory &) = default;
  MoleculeStreamFactory(MoleculeStreamFactory &&) noexcept = default;
  MoleculeStreamFactory &operator=(MoleculeStreamFactory &&) noexcept = default;
  virtual ~MoleculeStreamFactory() noexcept = default;

  /**
   * @brief Create a new stream from the given file.
   * @param is The input stream to read from.
   * @return A new stream instance.
   * @note The input stream must survive until the returned stream is
   *       destructed.
   */
  virtual std::unique_ptr<MoleculeStream>
  from_stream(std::istream &is) const = 0;

  /**
   * @brief Find the factory for the given format name
   * @param name The name of the format to find the factory for.
   * @return A pointer to the factory instance for the given format name, or
   *         nullptr if no factory is registered for the given name.
   */
  static const MoleculeStreamFactory *find_factory(std::string_view name);

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
  static bool register_factory(std::unique_ptr<MoleculeStreamFactory> factory,
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
  static void register_for_name(const MoleculeStreamFactory *factory,
                                std::string_view name);
};

template <class StreamFactoryImpl>
bool register_stream_factory(const std::vector<std::string> &names) {
  return MoleculeStreamFactory::register_factory(
    std::make_unique<StreamFactoryImpl>(), names);
}

template <class MoleculeStreamImpl>
class DefaultStreamFactoryImpl: public MoleculeStreamFactory {
public:
  std::unique_ptr<MoleculeStream> from_stream(std::istream &is) const override {
    return std::make_unique<MoleculeStreamImpl>(is);
  }
};

template <class SourceStream, class Stream = MoleculeStream>
class MoleculeStreamWrapper {
public:
  using iterator = MoleculeStreamIterator<MoleculeStreamWrapper>;

  template <class... Args>
  MoleculeStreamWrapper(std::string_view fmt, Args &&...args)
    : is_(std::forward<Args>(args)...) {
    const MoleculeStreamFactory *factory =
      MoleculeStreamFactory::find_factory(fmt);

    if (ABSL_PREDICT_FALSE(factory == nullptr)) {
      ABSL_LOG(WARNING) << "No factory found for " << fmt;
      return;
    }

    stream_ = static_unique_ptr_cast<Stream>(factory->from_stream(is_));
  }

  /**
   * @brief Advance the stream to the next molecule.
   * @return true if the stream is not at the end, false otherwise.
   */
  ABSL_MUST_USE_RESULT bool advance() { return stream_->advance(); }

  /**
   * @brief Get the current molecule.
   * @return The current molecule.
   * @pre Previous call to advance() must return true, otherwise an empty
   *      molecule is returned.
   */
  Molecule current() const { return stream_->current(); }

  /**
   * @brief Get the begin iterator of the stream.
   * @return The begin iterator of the stream.
   */
  iterator begin() { return iterator(*this); }

  /**
   * @brief Get the past-the-end iterator of the stream.
   * @return The past-the-end iterator of the stream.
   */
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  iterator end() { return iterator(); }

private:
  SourceStream is_;
  std::unique_ptr<Stream> stream_;
};

template <class Stream>
class MoleculeStreamWrapper<std::ifstream, Stream> {
public:
  using iterator = MoleculeStreamIterator<MoleculeStreamWrapper>;

  explicit MoleculeStreamWrapper(const std::filesystem::path &path): is_(path) {
    const std::filesystem::path full_ext = path.extension();
    const std::string_view ext = extension_no_dot(ext);

    const MoleculeStreamFactory *factory =
      MoleculeStreamFactory::find_factory(ext);

    if (ABSL_PREDICT_FALSE(factory == nullptr)) {
      ABSL_LOG(WARNING) << "No factory found for " << ext;
      return;
    }

    stream_ = static_unique_ptr_cast<Stream>(factory->from_stream(is_));
  }

  MoleculeStreamWrapper(std::string_view fmt, const std::filesystem::path &path)
    : is_(path) {
    const MoleculeStreamFactory *factory =
      MoleculeStreamFactory::find_factory(fmt);

    if (ABSL_PREDICT_FALSE(factory == nullptr)) {
      ABSL_LOG(WARNING) << "No factory found for " << fmt;
      return;
    }

    stream_ = static_unique_ptr_cast<Stream>(factory->from_stream(is_));
  }

  /**
   * @brief Advance the stream to the next molecule.
   * @return true if the stream is not at the end, false otherwise.
   */
  ABSL_MUST_USE_RESULT bool advance() { return stream_->advance(); }

  /**
   * @brief Get the current molecule.
   * @return The current molecule.
   * @pre Previous call to advance() must return true, otherwise an empty
   *      molecule is returned.
   */
  Molecule current() const { return stream_->current(); }

  /**
   * @brief Get the begin iterator of the stream.
   * @return The begin iterator of the stream.
   */
  iterator begin() { return iterator(*this); }

  /**
   * @brief Get the past-the-end iterator of the stream.
   * @return The past-the-end iterator of the stream.
   */
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  iterator end() { return iterator(); }

private:
  std::ifstream is_;
  std::unique_ptr<Stream> stream_;
};

template <class Stream = MoleculeStream>
using FileMoleculeStream = MoleculeStreamWrapper<std::ifstream, Stream>;

template <class Stream = MoleculeStream>
using StringMoleculeStream = MoleculeStreamWrapper<std::istringstream, Stream>;
}  // namespace nuri

#endif /* NURI_FMT_BASE_H_ */
