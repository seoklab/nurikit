//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_CIF_H_
#define NURI_FMT_CIF_H_

//! @cond
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/attributes.h>
#include <absl/base/nullability.h>
#include <absl/container/btree_map.h>
#include <absl/log/absl_check.h>
#include <absl/strings/str_cat.h>
#include <boost/range/iterator_range.hpp>
//! @endcond

#include "nuri/utils.h"

namespace nuri {
namespace internal {
  enum class CifToken : std::uint32_t {
    kEOF = 0U,
    kError = 1U,

    kData = 2U,
    kLoop = 3U,
    kGlobal = 4U,
    kSave = 5U,
    kStop = 6U,

    // 7-15 reserved for future use

    kTag = 1U << 4,
    kValue = 1U << 5,

    // Flags
    kIsQuoted = 1U << 31,

    // Compound tokens
    kSimpleValue = kValue,
    kQuotedValue = kValue | kIsQuoted,
  };

  constexpr bool is_value_token(CifToken token) {
    return static_cast<bool>(token & CifToken::kValue);
  }

  extern std::ostream &operator<<(std::ostream &os, CifToken type);

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
      // found by fuzzing, argument can point to internal buffer so we need to
      // create a string first then move it
      std::string err = absl::StrCat(std::forward<Args>(args)...);
      buf_ = std::move(err);
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

  class CifValue {
  public:
    enum class Type : std::uint32_t {
      kGeneric = 1U,      // Unquoted literals
      kString = 1U << 1,  // Quoted literals

      // Reserved for future use

      kUnknown = 1U << 30,       // ?
      kInapplicable = 1U << 31,  // .

      kNull = kUnknown | kInapplicable,  // any null value
    };

    CifValue(): type_(Type::kInapplicable) { }

    CifValue(std::string_view value, internal::CifToken type): value_(value) {
      if (type == internal::CifToken::kQuotedValue) {
        type_ = Type::kString;
        return;
      }

      ABSL_DCHECK_EQ(type, internal::CifToken::kSimpleValue);

      if (value == "?") {
        value_.clear();
        type_ = Type::kUnknown;
      } else if (value == ".") {
        value_.clear();
        type_ = Type::kInapplicable;
      }
    }

    std::string_view operator*() const { return value_; }
    std::string_view value() const { return value_; }

    const std::string *operator->() const { return &value_; }

    Type type() const { return type_; }

    constexpr bool is_null() const {
      return static_cast<bool>(type_ & Type::kNull);
    }

    constexpr operator bool() const { return !is_null(); }

    bool operator==(std::string_view other) const;

  private:
    std::string value_;
    Type type_ = Type::kGeneric;
  };

  extern std::ostream &operator<<(std::ostream &os, const CifValue &value);

  class CifTableColumn;

  class CifTable {
  public:
    CifTable() = default;

    const std::vector<std::string> &keys() const { return keys_; }
    void add_key(std::string_view key) { keys_.push_back(std::string { key }); }

    const std::vector<std::vector<CifValue>> &data() const { return rows_; }
    void add_data(CifValue &&value);

    const std::vector<CifValue> &operator[](size_t i) const { return rows_[i]; }
    size_t size() const { return rows_.size(); }

    auto begin() const { return rows_.begin(); }
    auto end() const { return rows_.end(); }

    const std::vector<CifValue> &row(size_t i) const { return rows_[i]; }
    size_t rows() const { return rows_.size(); }

    CifTableColumn col(size_t i) const;
    size_t cols() const { return keys_.size(); }

    bool empty() const { return keys_.empty() || rows_.empty(); }

  private:
    std::vector<std::string> keys_;
    std::vector<std::vector<CifValue>> rows_;
  };

  class CifTableColumn {
  public:
    CifTableColumn(absl::Nonnull<const CifTable *> table, size_t col)
        : table_(table), col_(col) { }

    std::string_view key() const { return table_->keys()[col_]; }
    size_t idx() const { return col_; }

    const CifValue &operator[](size_t i) const { return table_->row(i)[col_]; }
    size_t size() const { return table_->rows(); }

  private:
    absl::Nonnull<const CifTable *> table_;
    size_t col_;
  };

  inline CifTableColumn CifTable::col(size_t i) const {
    return { this, i };
  }

  class CifFrame {
  public:
    CifFrame(std::vector<CifTable> &&tables, std::string &&name);

    CifFrame() noexcept = default;
    CifFrame(CifFrame &&) noexcept = default;
    CifFrame &operator=(CifFrame &&) noexcept = default;
    ~CifFrame() = default;

    // Not copyable due to index
    CifFrame(const CifFrame &) = delete;
    CifFrame &operator=(const CifFrame &) = delete;

    std::string_view name() const { return name_; }

    const std::vector<CifTable> &tables() const { return tables_; }

    const CifTable &operator[](size_t i) const { return tables_[i]; }
    size_t size() const { return tables_.size(); }

    auto begin() const { return tables_.begin(); }
    auto end() const { return tables_.end(); }

    size_t total_cols() const { return index_.size(); }

    auto prefix_search(std::string_view prefix) const {
      return boost::make_iterator_range(index_.lower_bound(prefix),
                                        index_.end());
    }

    std::pair<int, int> find(std::string_view key) const {
      auto it = index_.find(key);
      if (it == index_.end())
        return { -1, -1 };
      return it->second;
    }

    CifTableColumn get(size_t tbl, size_t col) const {
      return tables_[tbl].col(col);
    }

    std::string validate() const;

  private:
    std::string name_;
    std::vector<CifTable> tables_;
    absl::btree_map<std::string_view, std::pair<int, int>> index_;
  };

  class CifBlock {
  public:
    enum class Type : int {
      kEOF = static_cast<int>(CifToken::kEOF),        // sentinel, end of file
      kError = static_cast<int>(CifToken::kError),    // sentinel, error state
      kGlobal = static_cast<int>(CifToken::kGlobal),  // global_
      kData = static_cast<int>(CifToken::kData),      // data_[<name>]
    };

    CifBlock(CifFrame &&frame, std::vector<CifFrame> &&save, Type type) noexcept
        : frame_(std::move(frame)), save_(std::move(save)), type_(type) {
      ABSL_DCHECK(*this);
    }

    static CifBlock eof() noexcept { return {}; }

    static CifBlock error(std::string_view reason) { return { reason }; }
    std::string_view error_msg() const {
      ABSL_DCHECK(type_ == Type::kError);
      return frame_.name();
    }

    std::string_view name() const {
      ABSL_DCHECK(type_ != Type::kError);
      return frame_.name();
    }

    const CifFrame &data() const { return frame_; }

    const std::vector<CifFrame> &save_frames() const { return save_; }

    Type type() const { return type_; }

    constexpr operator bool() const {
      return type_ != Type::kEOF && type_ != Type::kError;
    }

  private:
    CifBlock() noexcept: type_(Type::kEOF) { }

    CifBlock(std::string_view error_msg)
        : frame_({}, std::string { error_msg }), type_(Type::kError) { }

    CifFrame frame_;
    std::vector<CifFrame> save_;
    Type type_;
  };
}  // namespace internal

class CifParser {
public:
  explicit CifParser(std::istream &is);

  /**
   * @brief Parse the next block in the CIF file.
   */
  internal::CifBlock next();

  //! @private
  internal::CifBlock error(std::string_view reason);

private:
  internal::CifLexer lexer_;

  std::string name_;
  internal::CifBlock::Type block_ = internal::CifBlock::Type::kEOF;
};

// Test helpers

namespace internal {
  enum class CifGlobalCtx {
    kBlock,
    kSave,
  };

  extern std::pair<std::string_view, CifToken>
  parse_data(CifGlobalCtx ctx, std::vector<CifTable> &tables, CifLexer &lexer,
             std::string_view name);

  extern CifBlock next_block(CifParser &parser, CifLexer &lexer,
                             std::string &next_name,
                             internal::CifBlock::Type &next_block);
}  // namespace internal
}  // namespace nuri

#endif /* NURI_FMT_CIF_H_ */
