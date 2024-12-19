//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_CIF_H_
#define NURI_FMT_CIF_H_

/// @cond
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <absl/base/attributes.h>
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
    explicit CifLexer(const std::vector<std::string> &lines): lines_(&lines) {
      static_cast<void>(advance_line<true>());
    }

    std::pair<std::string_view, CifToken> next();

    // state observers

    size_t row() const { return line_ + 1; }
    size_t col() const { return it_ - begin_ + 1; }

    SIter begin() const { return begin_; }

    SIter p() const { return it_; }

    char c() const {
      ABSL_DCHECK(it_ < end());
      return *it_;
    }

    SIter end() const { return end_; }

    const std::vector<std::string> &lines() const { return *lines_; }

    std::string_view line() const {
      ABSL_DCHECK_LT(line_, lines().size());
      return lines()[line_];
    }

    template <bool kSkipEmpty>
    ABSL_MUST_USE_RESULT bool advance_line() {
      do {
        if (++line_ >= lines().size()) {
          // reentrant-safe
          begin_ = it_ = end();
          return false;
        }

        begin_ = it_ = lines()[line_].begin();
        end_ = lines()[line_].end();
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
    const std::vector<std::string> *lines_;
    std::string buf_;

    SIter it_;
    SIter begin_;
    SIter end_;
    std::int64_t line_ = -1;
  };
}  // namespace internal
}  // namespace nuri

#endif /* NURI_FMT_CIF_H_ */
