//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_CIF_H_
#define NURI_FMT_CIF_H_

/// @cond
#include <cstddef>
#include <istream>
#include <string>
#include <type_traits>

#include <absl/base/attributes.h>
#include <absl/base/nullability.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <absl/strings/str_cat.h>
/// @endcond

namespace nuri {
namespace internal {
  enum class CifToken {
    kEOF,
    kError,

    kData,
    kLoop,
    kGlobal,
    kSave,
    kStop,

    kTag,
    kValue,
  };

  using SIter = std::string::const_iterator;

  class CifLexer {
  public:
    explicit CifLexer(std::istream &is): is_(&is) {
      static_cast<void>(advance_line<true>());
    }

    std::pair<std::string_view, CifToken> next();

    // state observers

    size_t row() const { return row_; }
    size_t col() const { return it_ - begin_ + 1; }

    SIter begin() const { return begin_; }

    SIter p() const { return it_; }

    char c() const {
      ABSL_DCHECK(it_ < end());
      return *it_;
    }

    SIter end() const { return end_; }

    std::string_view line() const { return line_; }

    template <bool kSkipEmpty>
    ABSL_MUST_USE_RESULT bool advance_line() {
      do {
        if (!std::getline(*is_, line_)) {
          // reentrant-safe
          begin_ = it_ = end();
          return false;
        }

        ++row_;
        begin_ = it_ = line_.begin();
        end_ = line_.end();
      } while (kSkipEmpty && begin_ == end_);

      return true;
    }

    // for implementation, do not use

    std::pair<std::string_view, CifToken> produce(std::string_view data,
                                                  CifToken type, SIter after) {
      it_ = after;
      return { data, type };
    }

    template <class... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
    std::pair<std::string_view, CifToken> error(Args &&...args) {
      buf_.clear();
      absl::StrAppend(&buf_, std::forward<Args>(args)...);
      return { buf_, CifToken::kError };
    }

    static std::pair<std::string_view, CifToken> done() {
      return { {}, CifToken::kEOF };
    }

  private:
    absl::Nonnull<std::istream *> is_;
    std::string line_;
    std::string buf_;

    SIter it_;
    SIter begin_;
    SIter end_;
    std::size_t row_ = 0;
  };
}  // namespace internal
}  // namespace nuri

#endif /* NURI_FMT_CIF_H_ */
