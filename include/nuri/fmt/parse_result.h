//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_PARSE_RESULT_H_
#define NURI_FMT_PARSE_RESULT_H_

//! @cond
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <absl/log/absl_check.h>
#include <absl/strings/str_cat.h>
//! @endcond

namespace nuri {
/**
 * @brief The state of a ParseResult.
 */
enum class ParseStatus : int {
  kEOF = 0,
  kValid = 1,
  kError = 2,
};

/**
 * @brief The result of a parse operation, either a value, an error message, or
 *        a clean end-of-input marker.
 *
 * @tparam T The parsed value type. Must not be std::string.
 *
 * @note A ParseResult is only ever move-constructed or move-assigned from
 *       another ParseResult, so it can never become valueless by exception
 *       unless T's move constructor throws.
 */
template <class T>
class [[nodiscard]] ParseResult {
public:
  static_assert(!std::is_same_v<T, std::string>,
                "ParseResult<std::string> is ambiguous");

  ParseResult() = default;

  // NOLINTNEXTLINE(*-explicit-conversions)
  ParseResult(T &&value) noexcept(std::is_nothrow_move_constructible_v<T>)
      : data_(std::move(value)) { }

  /**
   * @brief Create a result denoting a clean end of input.
   * @note Block parsers never return this; only stream-level parsers do.
   */
  static ParseResult eof() noexcept { return ParseResult(); }

  /**
   * @brief Create a result denoting a parse failure.
   * @param args The reason, concatenated with absl::StrCat(). Must not be
   *        empty.
   */
  template <class... Args>
  static ParseResult error(const Args &...args) {
    return ParseResult(absl::StrCat(args...));
  }

  void reset() noexcept { data_ = std::monostate {}; }

  ParseStatus status() const { return static_cast<ParseStatus>(data_.index()); }

  explicit operator bool() const { return status() == ParseStatus::kValid; }

  /**
   * @brief Get the parsed value.
   * @pre status() == ParseStatus::kValid.
   */
  T &operator*() & {
    ABSL_DCHECK(status() == ParseStatus::kValid);
    return *std::get_if<T>(&data_);
  }

  //! @copydoc operator*()
  const T &operator*() const & {
    ABSL_DCHECK(status() == ParseStatus::kValid);
    return *std::get_if<T>(&data_);
  }

  //! @copydoc operator*()
  T &&operator*() && {
    ABSL_DCHECK(status() == ParseStatus::kValid);
    return std::move(*std::get_if<T>(&data_));
  }

  //! @copydoc operator*()
  T *operator->() { return &**this; }

  //! @copydoc operator*()
  const T *operator->() const { return &**this; }

  /**
   * @brief Get the failure reason.
   * @pre status() == ParseStatus::kError.
   */
  std::string_view error_msg() const {
    ABSL_DCHECK(status() == ParseStatus::kError);
    return *std::get_if<std::string>(&data_);
  }

private:
  explicit ParseResult(std::string &&err): data_(std::move(err)) {
    ABSL_DCHECK(!std::get_if<std::string>(&data_)->empty());
  }

  std::variant<std::monostate, T, std::string> data_;

  template <ParseStatus S, class U>
  constexpr static bool kAlternativeIs = std::is_same_v<
      std::variant_alternative_t<static_cast<std::size_t>(S), decltype(data_)>,
      U>;

  static_assert(kAlternativeIs<ParseStatus::kEOF, std::monostate>
                    && kAlternativeIs<ParseStatus::kValid, T>
                    && kAlternativeIs<ParseStatus::kError, std::string>,
                "ParseStatus values must match data_'s alternative indices");
};
}  // namespace nuri

#endif /* NURI_FMT_PARSE_RESULT_H_ */
