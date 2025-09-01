//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_UTILS_H_
#define NURI_UTILS_H_

//! @cond
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/nullability.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <absl/numeric/bits.h>
#include <absl/strings/ascii.h>
#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"
#include "nuri/meta.h"

namespace nuri {
template <class T, std::enable_if_t<std::is_trivially_copyable_v<T>, int> = 0>
constexpr T min(T a, T b) {
  return std::min(a, b);
}

template <
    class L, class R,
    std::enable_if_t<
        std::is_same_v<internal::remove_cvref_t<L>, internal::remove_cvref_t<R>>
            && !std::is_trivially_copyable_v<internal::remove_cvref_t<L>>
            && std::is_lvalue_reference_v<L> && std::is_lvalue_reference_v<R>,
        int> = 0>
constexpr const std::remove_reference_t<L> &min(L &&a, R &&b) {
  return std::min(std::forward<L>(a), std::forward<R>(b));
}

template <class T, std::enable_if_t<std::is_trivially_copyable_v<T>, int> = 0>
constexpr T max(T a, T b) {
  return std::max(a, b);
}

template <
    class L, class R,
    std::enable_if_t<
        std::is_same_v<internal::remove_cvref_t<L>, internal::remove_cvref_t<R>>
            && !std::is_trivially_copyable_v<internal::remove_cvref_t<L>>
            && std::is_lvalue_reference_v<L> && std::is_lvalue_reference_v<R>,
        int> = 0>
constexpr const std::remove_reference_t<L> &max(L &&a, R &&b) {
  return std::max(std::forward<L>(a), std::forward<R>(b));
}

template <class T, class Comp = std::less<>,
          std::enable_if_t<std::is_trivially_copyable_v<T>, int> = 0>
constexpr std::pair<T, T> minmax(T a, T b, Comp &&comp = {}) {
  return std::minmax(a, b, std::forward<Comp>(comp));
}

template <
    class L, class R, class Comp = std::less<>,
    std::enable_if_t<
        std::is_same_v<internal::remove_cvref_t<L>, internal::remove_cvref_t<R>>
            && !std::is_trivially_copyable_v<internal::remove_cvref_t<L>>
            && std::is_lvalue_reference_v<L> && std::is_lvalue_reference_v<R>,
        int> = 0>
constexpr std::pair<const std::remove_reference_t<L> &,
                    const std::remove_reference_t<L> &>
minmax(L &&a, R &&b, Comp &&comp = {}) {
  return std::minmax(std::forward<L>(a), std::forward<R>(b),
                     std::forward<Comp>(comp));
}

template <class T, std::enable_if_t<std::is_trivially_copyable_v<T>, int> = 0>
constexpr T clamp(T v, T l, T h) {
  return std::clamp(v, l, h);
}

template <
    class T, class L, class H,
    std::enable_if_t<
        std::is_same_v<internal::remove_cvref_t<T>, internal::remove_cvref_t<L>>
            && std::is_same_v<internal::remove_cvref_t<L>,
                              internal::remove_cvref_t<H>>
            && !std::is_trivially_copyable_v<internal::remove_cvref_t<T>>
            && std::is_lvalue_reference_v<T> && std::is_lvalue_reference_v<L>
            && std::is_lvalue_reference_v<H>,
        int> = 0>
constexpr const std::remove_reference_t<T> &clamp(T &&v, L &&l, H &&h) {
  return std::clamp(std::forward<T>(v), std::forward<L>(l), std::forward<H>(h));
}

namespace internal {
  class PowersetStream {
  public:
    explicit PowersetStream(int n): n_(n), r_(0), r_max_(0), state_(0) { }

    PowersetStream &next() {
      if (state_ >= r_max_) {
        ++r_;

        if (ABSL_PREDICT_FALSE(!*this))
          return *this;

        r_max_ |= 1U << (n_ - r_);
        state_ = (1U << r_) - 1;
        return *this;
      }

      constexpr auto nbits = std::numeric_limits<decltype(state_)>::digits;
      unsigned int shifted = state_ << (nbits - n_);

      unsigned int leading_ones = absl::countl_one(shifted);
      unsigned int stripped = (shifted << leading_ones) >> leading_ones;

      int leading_zeros = absl::countl_zero(stripped);
      int next_one_bit = nuri::max(n_ - leading_zeros - 1, 0);

      unsigned int mask = (1U << next_one_bit) - 1;
      state_ = ((1U << (leading_ones + 1)) - 1) << (next_one_bit + 1)
               | (mask & state_);
      return *this;
    }

    unsigned int state() const { return state_; }

    operator bool() const { return r_ <= n_; }

  private:
    int n_;
    int r_;
    unsigned int r_max_;
    unsigned int state_;
  };

  inline PowersetStream &operator>>(PowersetStream &ps, unsigned int &state) {
    ps.next();
    state = ps.state();
    return ps;
  }
}  // namespace internal

//! @privatesection

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr E operator|(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr E &operator|=(E &self, E rhs) {
  return self = self | rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr E operator&(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr E &operator&=(E &self, E rhs) {
  return self = self & rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr E operator^(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) ^ static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr E &operator^=(E &self, E rhs) {
  return self = self ^ rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr E operator~(E val) {
  return static_cast<E>(~static_cast<U>(val));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0,
          std::enable_if_t<std::is_unsigned_v<U>, int> = 0>
constexpr E operator-(E val) {
  return static_cast<E>(-static_cast<U>(val));
}

namespace internal {
  template <class E>
  constexpr bool check_flag(E flags, E flag) {
    // NOLINTNEXTLINE(bugprone-non-zero-enum-to-bool-conversion)
    return static_cast<bool>(flags & flag);
  }

  template <class E, class U = extract_if_enum_t<E>,
            std::enable_if_t<std::is_unsigned_v<U>, U> = 0>
  constexpr E &update_flag(E &flags, bool cond, E flag) {
    E mask = -static_cast<E>(cond);
    flags = (flags & ~flag) | (mask & flag);
    return flags;
  }

  template <class E, class U = extract_if_enum_t<E>,
            std::enable_if_t<std::is_unsigned_v<U>, U> = 0>
  constexpr E &set_flag_if(E &flags, bool cond, E flag) {
    E mask = -static_cast<E>(cond);
    flags |= mask & flag;
    return flags;
  }

  template <class E, class U = extract_if_enum_t<E>,
            std::enable_if_t<std::is_unsigned_v<U>, U> = 0>
  constexpr E &unset_flag_if(E &flags, bool cond, E flag) {
    E mask = -static_cast<E>(cond);
    flags &= ~(mask & flag);
    return flags;
  }

  template <class F>
  int iround(F x) {
    return static_cast<int>(std::lround(x));
  }

  constexpr int negate_if_false(bool cond) {
    int ret = (static_cast<int>(cond) << 1) - 1;
    ABSL_ASSUME(ret == 1 || ret == -1);
    return ret;
  }
}  // namespace internal

template <int N = Eigen::Dynamic, int... Extra>
auto generate_index(Eigen::Index size) {
  Array<int, N, 1, 0, Extra...> result(size);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

template <int N = Eigen::Dynamic, int... Extra, class Container,
          class Comp = std::less<>>
auto argsort(const Container &container, Comp op = {}) {
  auto idxs = generate_index<N, Extra...>(std::size(container));
  std::sort(idxs.begin(), idxs.end(),
            [&](int i, int j) { return op(container[i], container[j]); });
  return idxs;
}

template <int N = Eigen::Dynamic, int... Extra, class Container,
          class Comp = std::less<>>
auto argpartition(const Container &container, int count, Comp op = {}) {
  auto idxs = generate_index<N, Extra...>(std::size(container));
  std::nth_element(idxs.begin(), idxs.begin() + count - 1, idxs.end(),
                   [&](int i, int j) {
                     return op(container[i], container[j]);
                   });
  return idxs;
}

template <class Container>
void mask_to_map(Container &mask) {
  typename Container::value_type idx = 0;
  for (int i = 0; i < mask.size(); ++i) {
    mask[i] = mask[i] ? idx++ : -1;
  }
}

template <typename Derived, typename Base>
std::unique_ptr<Derived>
static_unique_ptr_cast(std::unique_ptr<Base> &&p) noexcept {
  auto d = static_cast<Derived *>(p.release());
  return std::unique_ptr<Derived>(d);
}

inline std::string_view extension_no_dot(const std::filesystem::path &ext) {
  const std::string_view ext_view = ext.native();
  if (ABSL_PREDICT_TRUE(!ext_view.empty())) {
    return ext_view.substr(1);
  }
  return ext_view;
}

constexpr std::string_view slice(std::string_view str, std::size_t begin,
                                 std::size_t end) {
  return str.substr(begin, end - begin);
}

inline std::string_view slice_strip(std::string_view str, std::size_t begin,
                                    std::size_t end) {
  return absl::StripAsciiWhitespace(slice(str, begin, end));
}

inline std::string_view slice_rstrip(std::string_view str, std::size_t begin,
                                     std::size_t end) {
  return absl::StripTrailingAsciiWhitespace(slice(str, begin, end));
}

constexpr std::string_view safe_substr(std::string_view str, size_t begin,
                                       size_t count = std::string_view::npos) {
  if (ABSL_PREDICT_FALSE(begin > str.size()))
    return "";

  return str.substr(begin, count);
}

constexpr std::string_view safe_slice(std::string_view str, size_t begin,
                                      size_t end) {
  if (ABSL_PREDICT_FALSE(begin > str.size()))
    return "";

  return slice(str, begin, end);
}

inline std::string_view safe_slice_strip(std::string_view str, size_t begin,
                                         size_t end) {
  return absl::StripAsciiWhitespace(safe_slice(str, begin, end));
}

inline std::string_view safe_slice_rstrip(std::string_view str, size_t begin,
                                          size_t end) {
  return absl::StripTrailingAsciiWhitespace(safe_slice(str, begin, end));
}

namespace internal {
  template <bool = (sizeof(double) * 3) % alignof(Vector3d) == 0>
  inline void stack_impl(Matrix3Xd &m, const std::vector<Vector3d> &vs) {
    for (int i = 0; i < vs.size(); ++i)
      m.col(i) = vs[i];
  }

  template <>
  inline void stack_impl<true>(Matrix3Xd &m, const std::vector<Vector3d> &vs) {
    ABSL_DCHECK(reinterpret_cast<ptrdiff_t>(vs[0].data()) + sizeof(double) * 3
                == reinterpret_cast<ptrdiff_t>(vs[1].data()))
        << "Bad alignment";
    std::memcpy(m.data(), vs[0].data(), vs.size() * sizeof(double) * 3);
  }
}  // namespace internal

inline Matrix3Xd stack(const std::vector<Vector3d> &vs) {
  if (ABSL_PREDICT_FALSE(vs.empty()))
    return Matrix3Xd(3, 0);

  Matrix3Xd m(3, vs.size());
  internal::stack_impl(m, vs);
  return m;
}

template <class T = int, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
constexpr T value_if(bool cond, T val = 1) {
  return static_cast<int>(cond) * val;
}

template <class Scalar, std::enable_if_t<std::is_arithmetic_v<Scalar>
                                             && (!std::is_integral_v<Scalar>
                                                 || std::is_signed_v<Scalar>),
                                         int> = 0>
constexpr Scalar nonnegative(Scalar x) {
  return nuri::max(x, static_cast<Scalar>(0));
}

template <class UInt,
          std::enable_if_t<std::is_integral_v<UInt> && !std::is_signed_v<UInt>,
                           int> = 0>
constexpr UInt nonnegative(UInt x) {
  return x;
}

template <class UInt,
          std::enable_if_t<std::is_same_v<UInt, unsigned int>, int> = 0>
constexpr int log_base10(UInt x) {
  int lg = (x >= 1000000000)  ? 9
           : (x >= 100000000) ? 8
           : (x >= 10000000)  ? 7
           : (x >= 1000000)   ? 6
           : (x >= 100000)    ? 5
           : (x >= 10000)     ? 4
           : (x >= 1000)      ? 3
           : (x >= 100)       ? 2
           : (x >= 10)        ? 1
                              : 0;
  return lg;
}

template <class Int, std::enable_if_t<std::is_same_v<Int, int>, int> = 0>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
constexpr int log_base10(Int x) {
  int lg = value_if(x < 0);
  lg += log_base10(static_cast<unsigned int>(x < 0 ? -x : x));
  return lg;
}

namespace internal {
#ifdef absl_nullable
  // These 3 are deprecated as of v20250512
  template <class T>
  using Nonnull = T absl_nonnull;
  template <class T>
  using Nullable = T absl_nullable;
  template <class T>
  using NullabilityUnknown = T absl_nullability_unknown;
#else
  using absl::Nonnull;
  using absl::NullabilityUnknown;
  using absl::Nullable;
#endif
}  // namespace internal
}  // namespace nuri

#endif /* NURI_UTILS_H_ */
