//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_CONTAINER_DUMB_BUFFER_H_
#define NURI_CORE_CONTAINER_DUMB_BUFFER_H_

//! @cond
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <absl/log/absl_check.h>
//! @endcond

#include "nuri/meta.h"

namespace nuri {
namespace internal {
  // NOLINTBEGIN(*-no-malloc,*-owning-memory)
  template <class T>
  class DumbBuffer {
    constexpr static size_t bytes(size_t len) noexcept {
      return len * sizeof(T);
    }

    static T *alloc(size_t len) noexcept {
      void *buf = std::malloc(bytes(len));
      ABSL_QCHECK(buf != nullptr);
      return static_cast<T *>(buf);
    }

    static T *realloc(T *orig, size_t len) noexcept {
      void *buf = std::realloc(static_cast<void *>(orig), bytes(len));
      ABSL_QCHECK(buf != nullptr);
      return static_cast<T *>(buf);
    }

    static void copy(T *dst, const T *src, size_t len) noexcept {
      std::memcpy(static_cast<void *>(dst), static_cast<const void *>(src),
                  bytes(len));
    }

  public:
    static_assert(std::is_same_v<T, remove_cvref_t<T>>,
                  "T must not have cv-qualifiers or reference");
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");
    static_assert(alignof(T) <= alignof(std::max_align_t),
                  "T must have less than or equal alignment to all scalar "
                  "types");

    DumbBuffer(size_t len) noexcept: data_(alloc(len)), len_(len) { }

    DumbBuffer(const DumbBuffer &other) noexcept
        : data_(alloc(other.len_)), len_(other.len_) {
      copy(data_, other.data_, len_);
    }

    DumbBuffer(DumbBuffer &&other) noexcept
        : data_(other.data_), len_(other.len_) {
      other.data_ = nullptr;
      other.len_ = 0;
    }

    ~DumbBuffer() noexcept { std::free(static_cast<void *>(data_)); }

    DumbBuffer &operator=(const DumbBuffer &other) noexcept {
      if (this == &other) {
        return *this;
      }

      resize(other.len_);
      copy(data_, other.data_, len_);
      return *this;
    }

    DumbBuffer &operator=(DumbBuffer &&other) noexcept {
      std::swap(data_, other.data_);
      std::swap(len_, other.len_);
      return *this;
    }

    T &operator[](size_t idx) noexcept { return data_[idx]; }

    const T &operator[](size_t idx) const noexcept { return data_[idx]; }

    T *data() noexcept { return data_; }

    const T *data() const noexcept { return data_; }

    size_t size() const noexcept { return len_; }

    void resize(size_t len) noexcept {
      data_ = realloc(data_, len_);
      len_ = len;
    }

    T *begin() noexcept { return data_; }
    T *end() noexcept { return data_ + len_; }

    const T *cbegin() const noexcept { return data_; }
    const T *cend() const noexcept { return data_ + len_; }
    const T *begin() const noexcept { return cbegin(); }
    const T *end() const noexcept { return cend(); }

  private:
    T *data_;
    size_t len_;
  };
  // NOLINTEND(*-no-malloc,*-owning-memory)
}  // namespace internal
}  // namespace nuri

#endif /* NURI_CORE_CONTAINER_DUMB_BUFFER_H_ */
