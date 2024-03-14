//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_UTILS_H_
#define NURI_UTILS_H_

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/iterator/iterator_facade.hpp>
#include <Eigen/Dense>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <absl/numeric/bits.h>
#include <absl/strings/ascii.h>

#include "nuri/eigen_config.h"

// Introduced in clang 18
// #ifdef __clang_analyzer__
// #define NURI_CLANG_ANALYZER_NOLINT       [[clang::suppress]]
// #define NURI_CLANG_ANALYZER_NOLINT_BEGIN [[clang::suppress]] {
// #define NURI_CLANG_ANALYZER_NOLINT_END   }
// #else
// #define NURI_CLANG_ANALYZER_NOLINT
// #define NURI_CLANG_ANALYZER_NOLINT_BEGIN
// #define NURI_CLANG_ANALYZER_NOLINT_END
// #endif

namespace nuri {
namespace internal {
  // Use of std::underlying_type_t on non-enum types is UB until C++20.
#if __cplusplus >= 202002L
  using std::underlying_type;
  using std::underlying_type_t;

  using std::remove_cvref;
  using std::remove_cvref_t;
#else
  template <class E, bool = std::is_enum_v<E>>
  struct underlying_type { };

  template <class E>
  struct underlying_type<E, false> { };

  template <class E>
  struct underlying_type<E, true> {
    using type = std::underlying_type_t<E>;
  };

  template <class T>
  using underlying_type_t = typename underlying_type<T>::type;

  template <class T>
  struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };

  template <class T>
  using remove_cvref_t = typename remove_cvref<T>::type;
#endif

  template <class To, class From>
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr inline bool is_implicitly_constructible_v =
      std::is_constructible_v<To, From> && std::is_convertible_v<From, To>;

  template <class To, class From>
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr inline bool is_explicitly_constructible_v =
      std::is_constructible_v<To, From> && !std::is_convertible_v<From, To>;

  template <bool is_const, class T>
  struct const_if {
    using type = std::conditional_t<is_const, const T, T>;
  };

  template <bool is_const, class T>
  using const_if_t = typename const_if<is_const, T>::type;

  template <class Iterator, class T, class IfTrue = int>
  using enable_if_compatible_iter_t =
      std::enable_if_t<is_implicitly_constructible_v<
                           T, typename std::iterator_traits<Iterator>::reference>,
                       IfTrue>;

  template <class Iter, class IteratorTag, class IfTrue = int>
  using enable_if_iter_category_t = std::enable_if_t<
      std::is_same_v<typename std::iterator_traits<Iter>::iterator_category,
                     IteratorTag>,
      IfTrue>;

  template <class T, bool = std::is_enum_v<T>>
  struct extract_if_enum { };

  template <class T>
  struct extract_if_enum<T, true> {
    using type = std::underlying_type_t<T>;
  };

  template <class T>
  struct extract_if_enum<T, false> {
    using type = T;
  };

  template <class T>
  using extract_if_enum_t = typename extract_if_enum<T>::type;
}  // namespace internal

namespace internal {
  template <class T, class C = std::less<>, class S = std::vector<T>>
  struct ClearablePQ: public std::priority_queue<T, S, C> {
    using Base = std::priority_queue<T, S, C>;

  public:
    using Base::Base;

    T pop_get() noexcept {
      T v = std::move(this->c.front());
      this->pop();
      return v;
    }

    void clear() noexcept { this->c.clear(); }
  };

  template <class Derived, class RefLike, class Category,
            class Difference = std::ptrdiff_t>
  class ProxyIterator: public boost::iterator_facade<Derived, RefLike, Category,
                                                     RefLike, Difference> { };

  template <class Derived, class RefLike, class Difference>
  class ProxyIterator<Derived, RefLike, std::random_access_iterator_tag,
                      Difference>
      : public boost::iterator_facade<Derived, RefLike,
                                      std::random_access_iterator_tag, RefLike,
                                      Difference> {
    using Parent = typename ProxyIterator::iterator_facade_;

  public:
    // Required to override boost's implementation that returns a proxy
    constexpr typename Parent::reference
    operator[](typename Parent::difference_type n) const noexcept {
      return *(*static_cast<const Derived *>(this) + n);
    }
  };

  template <class Derived, class Iter, class UnaryOp, auto op,
            bool = std::is_member_function_pointer_v<UnaryOp>,
            bool = std::is_class_v<UnaryOp>>
  class TransformIteratorTrampoline;

  template <class Derived, class Traits, class Ref>
  using TransformIteratorBase =
      boost::iterator_facade<Derived, std::remove_reference_t<Ref>,
                             typename Traits::iterator_category, Ref,
                             typename Traits::difference_type>;

  template <class Derived, class Iter, class UnaryOp, UnaryOp op>
  class TransformIteratorTrampoline<Derived, Iter, UnaryOp, op, false, false>
      : public TransformIteratorBase<Derived, std::iterator_traits<Iter>,
                                     decltype(op(*std::declval<Iter>()))> {
  protected:
    using Parent = TransformIteratorTrampoline;
    constexpr static auto dereference_impl(Iter it) { return op(*it); }
  };

  template <class Derived, class Iter, class UnaryOp, UnaryOp op>
  class TransformIteratorTrampoline<Derived, Iter, UnaryOp, op, true, false>
      : public TransformIteratorBase<Derived, std::iterator_traits<Iter>,
                                     decltype((*std::declval<Iter>().*op)())> {
  protected:
    using Parent = TransformIteratorTrampoline;
    constexpr static auto dereference_impl(Iter it) { return (*it.*op)(); }
  };

  template <class Derived, class Iter, class UnaryOp>
  class TransformIteratorTrampoline<Derived, Iter, UnaryOp, nullptr, false, true>
      : public TransformIteratorBase<Derived, std::iterator_traits<Iter>,
                                     decltype(std::declval<UnaryOp>()(
                                         *std::declval<Iter>()))> {
  protected:
    using Parent = TransformIteratorTrampoline;

    TransformIteratorTrampoline(const UnaryOp &op): op_(op) { }

    TransformIteratorTrampoline(UnaryOp &&op): op_(std::move(op)) { }

    constexpr auto dereference_impl(Iter it) const { return op_(*it); }

  private:
    UnaryOp op_;
  };

  template <class Iter, class UnaryOp, auto unaryop = nullptr>
  class TransformIterator
      : public TransformIteratorTrampoline<
            TransformIterator<Iter, UnaryOp, unaryop>, Iter, UnaryOp, unaryop> {
    using Base = typename TransformIterator::Parent;
    using Traits =
        std::iterator_traits<typename TransformIterator::iterator_facade_>;

  public:
    using iterator_category = typename Traits::iterator_category;
    using value_type = typename Traits::value_type;
    using difference_type = typename Traits::difference_type;
    using pointer = typename Traits::pointer;
    using reference = typename Traits::reference;

    constexpr TransformIterator() = default;

    template <class... Args>
    constexpr explicit TransformIterator(Iter it, Args &&...args)
        : Base(std::forward<Args>(args)...), it_(it) { }

    constexpr Iter base() const { return it_; }

  private:
    friend class boost::iterator_core_access;

    constexpr reference dereference() const {
      return Base::dereference_impl(it_);
    }

    constexpr bool equal(TransformIterator rhs) const { return it_ == rhs.it_; }

    constexpr void increment() { ++it_; }
    constexpr void decrement() { --it_; }
    constexpr void advance(difference_type n) { it_ += n; }

    constexpr difference_type distance_to(TransformIterator lhs) const {
      return lhs.it_ - it_;
    }

    Iter it_;
  };

  template <auto unaryop, class Iter>
  constexpr auto make_transform_iterator(Iter it) {
    return TransformIterator<Iter, decltype(unaryop), unaryop>(it);
  }

  template <class UnaryOp, class Iter>
  constexpr auto make_transform_iterator(Iter it, UnaryOp &&op) {
    return TransformIterator<Iter, remove_cvref_t<UnaryOp>>(
        it, std::forward<UnaryOp>(op));
  }

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

  template <class K, class V, V sentinel = -1>
  class CompactMap {
  public:
    static_assert(std::is_convertible_v<K, size_t>,
                  "Key must be an integer-like type");

    using key_type = K;
    using mapped_type = V;
    using value_type = V;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = V &;
    using const_reference = const V &;
    using pointer = V *;
    using const_pointer = const V *;

    CompactMap(size_t cap): data_(cap, sentinel) { }

    template <class... Args>
    std::pair<pointer, bool> try_emplace(key_type key, Args &&...args) {
      handle_resize(key);

      reference v = data_[key];
      bool isnew = v == sentinel;
      if (isnew) {
        v = V(std::forward<Args>(args)...);
      }
      return { &v, isnew };
    }

    pointer find(key_type key) {
      if (key < data_.size()) {
        return data_[key] == sentinel ? nullptr : &data_[key];
      }
      return nullptr;
    }

    const_pointer find(key_type key) const {
      if (key < data_.size()) {
        return data_[key] == sentinel ? nullptr : &data_[key];
      }
      return nullptr;
    }

  private:
    void handle_resize(key_type new_key) {
      if (new_key >= data_.size()) {
        data_.resize(new_key + 1, sentinel);
      }
    }

    std::vector<V> data_;
  };

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
      int next_one_bit = std::max(n_ - leading_zeros - 1, 0);

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

  template <class ZIT>
  struct ZippedIteratorTraits;

  template <template <class...> class ZIT, class Iter, class Jter>
  struct ZippedIteratorTraits<ZIT<Iter, Jter>> {
    using iterator_category = std::common_type_t<
        typename std::iterator_traits<Iter>::iterator_category,
        typename std::iterator_traits<Jter>::iterator_category>;
    using difference_type =
        std::common_type_t<typename std::iterator_traits<Iter>::difference_type,
                           typename std::iterator_traits<Jter>::difference_type>;
    using reference = std::pair<typename std::iterator_traits<Iter>::reference,
                                typename std::iterator_traits<Jter>::reference>;
  };

  template <template <class...> class ZIT, class... Iters>
  struct ZippedIteratorTraits<ZIT<Iters...>> {
    using iterator_category = std::common_type_t<
        typename std::iterator_traits<Iters>::iterator_category...>;
    using difference_type = std::common_type_t<
        typename std::iterator_traits<Iters>::difference_type...>;
    using reference =
        std::tuple<typename std::iterator_traits<Iters>::reference...>;
  };

  template <class Derived>
  class ZippedIteratorBase
      : public ProxyIterator<
            Derived, typename ZippedIteratorTraits<Derived>::reference,
            typename ZippedIteratorTraits<Derived>::iterator_category,
            typename ZippedIteratorTraits<Derived>::difference_type> {
    using Root = typename ZippedIteratorBase::iterator_facade_;
    using Traits = std::iterator_traits<Root>;

  public:
    using iterator_category = typename Traits::iterator_category;
    using value_type = typename Traits::value_type;
    using difference_type = typename Traits::difference_type;
    using pointer = typename Traits::pointer;
    using reference = typename Traits::reference;
  };
}  // namespace internal

template <class... Iters>
class ZippedIterator
    : public internal::ZippedIteratorBase<ZippedIterator<Iters...>> {
  static_assert(sizeof...(Iters) > 1, "At least two iterators are required");

  using Base = internal::ZippedIteratorBase<ZippedIterator>;

public:
  constexpr ZippedIterator() = default;

  constexpr ZippedIterator(Iters... its): its_(its...) { }

  template <size_t N = 0>
  constexpr auto base() const {
    return std::get<N>(its_);
  }

  template <size_t N = sizeof...(Iters), std::enable_if_t<N == 2, int> = 0>
  constexpr auto first() const {
    return base<0>();
  }

  template <size_t N = sizeof...(Iters), std::enable_if_t<N == 2, int> = 0>
  constexpr auto second() const {
    return base<1>();
  }

private:
  friend class boost::iterator_core_access;

  constexpr typename Base::reference dereference() const {
    return std::apply(
        [](auto &...it) { return typename Base::reference(*it...); }, its_);
  }

  constexpr bool equal(const ZippedIterator &rhs) const {
    return base() == rhs.base();
  }

  constexpr void increment() {
    std::apply([](auto &...it) { (static_cast<void>(++it), ...); }, its_);
  }

  constexpr void decrement() {
    std::apply([](auto &...it) { (static_cast<void>(--it), ...); }, its_);
  }

  constexpr void advance(typename Base::difference_type n) {
    std::apply([n](auto &...it) { (static_cast<void>(it += n), ...); }, its_);
  }

  constexpr typename Base::difference_type
  distance_to(const ZippedIterator &lhs) const {
    return lhs.base() - base();
  }

  std::tuple<Iters...> its_;
};

template <class... Iters>
constexpr auto make_zipped_iterator(Iters... iters) {
  return ZippedIterator<Iters...>(iters...);
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E operator|(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E &operator|=(E &self, E rhs) {
  return self = self | rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E operator&(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E &operator&=(E &self, E rhs) {
  return self = self & rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E operator^(E lhs, E rhs) {
  return static_cast<E>(static_cast<U>(lhs) ^ static_cast<U>(rhs));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E &operator^=(E &self, E rhs) {
  return self = self ^ rhs;
}

template <class E, class U = internal::underlying_type_t<E>, U = 0>
constexpr inline E operator~(E val) {
  return static_cast<E>(~static_cast<U>(val));
}

template <class E, class U = internal::underlying_type_t<E>, U = 0,
          std::enable_if_t<std::is_unsigned_v<U>, int> = 0>
constexpr inline E operator-(E val) {
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

  // RDKit-compatible key for name
  constexpr inline std::string_view kNameKey = "_Name";

  template <class PT>
  auto find_key(PT &props, std::string_view key) {
    return absl::c_find_if(props,
                           [key](const auto &p) { return p.first == key; });
  }

  template <class PT>
  auto find_name(PT &props) -> decltype(props.begin()) {
    return find_key(props, kNameKey);
  }

  template <class PT>
  std::string_view get_name(PT &props) {
    auto it = internal::find_name(props);
    if (it == props.end())
      return {};
    return it->second;
  }

  template <class PT>
  void set_name(PT &props, std::string &&name) {
    auto it = find_name(props);
    if (it != props.end()) {
      it->second = std::move(name);
    } else {
      props.emplace_back(kNameKey, std::move(name));
    }
  }

  template <class PT>
  void set_name(PT &props, const char *name) {
    set_name(props, std::string(name));
  }

  template <class PT>
  void set_name(PT &props, std::string_view name) {
    set_name(props, std::string(name));
  }

  constexpr inline int negate_if_false(bool cond) {
    int ret = (static_cast<int>(cond) << 1) - 1;
    ABSL_ASSUME(ret == 1 || ret == -1);
    return ret;
  }
}  // namespace internal

#if __cplusplus >= 202002L
using std::erase;
using std::erase_if;
#else
template <class T, class Alloc, class U>
typename std::vector<T, Alloc>::size_type erase(std::vector<T, Alloc> &c,
                                                const U &value) {
  auto it = std::remove(c.begin(), c.end(), value);
  auto r = std::distance(it, c.end());
  c.erase(it, c.end());
  return r;
}

template <class T, class Alloc, class Pred>
typename std::vector<T, Alloc>::size_type erase_if(std::vector<T, Alloc> &c,
                                                   Pred &&pred) {
  auto it = std::remove_if(c.begin(), c.end(), std::forward<Pred>(pred));
  auto r = std::distance(it, c.end());
  c.erase(it, c.end());
  return r;
}
#endif

template <class T, class Alloc, class Pred>
typename std::vector<T, Alloc>::iterator erase_first(std::vector<T, Alloc> &c,
                                                     Pred &&pred) {
  auto it = std::find_if(c.begin(), c.end(), std::forward<Pred>(pred));
  if (it != c.end()) {
    return c.erase(it);
  }
  return it;
}

template <class Iter, class VT, class Comp,
          internal::enable_if_iter_category_t<
              Iter, std::random_access_iterator_tag> = 0>
Iter find_sorted(Iter begin, Iter end, const VT &value, Comp &&comp) {
  // *it >= value
  auto it = std::lower_bound(begin, end, value, std::forward<Comp>(comp));
  if (it != end && !comp(value, *it)) {
    // value >= *it, i.e., value == *it
    return it;
  }
  return end;
}

template <class Iter, class VT,
          internal::enable_if_iter_category_t<
              Iter, std::random_access_iterator_tag> = 0>
Iter find_sorted(Iter begin, Iter end, const VT &value) {
  return find_sorted(begin, end, value, std::less<>());
}

template <class Container, class Comp,
          internal::enable_if_iter_category_t<
              typename Container::iterator, std::random_access_iterator_tag> = 0>
std::pair<typename Container::iterator, bool>
insert_sorted(Container &c, const typename Container::value_type &value,
              Comp comp) {
  // *it >= value
  auto it = std::lower_bound(c.begin(), c.end(), value, comp);
  if (it != c.end() && !comp(value, *it)) {
    // value >= *it, i.e., value == *it
    return { it, false };
  }
  return { c.insert(it, value), true };
}

template <class Container,
          internal::enable_if_iter_category_t<
              typename Container::iterator, std::random_access_iterator_tag> = 0>
std::pair<typename Container::iterator, bool>
insert_sorted(Container &c, const typename Container::value_type &value) {
  return insert_sorted(c, value, std::less<>());
}

template <class Container, class Comp,
          internal::enable_if_iter_category_t<
              typename Container::iterator, std::random_access_iterator_tag> = 0>
std::pair<typename Container::iterator, bool>
insert_sorted(Container &c, typename Container::value_type &&value, Comp comp) {
  // *it >= value
  auto it = std::lower_bound(c.begin(), c.end(), value, comp);
  if (it != c.end() && !comp(value, *it)) {
    // value >= *it, i.e., value == *it
    return { it, false };
  }
  return { c.insert(it, std::move(value)), true };
}

template <class Container,
          internal::enable_if_iter_category_t<
              typename Container::iterator, std::random_access_iterator_tag> = 0>
std::pair<typename Container::iterator, bool>
insert_sorted(Container &c, typename Container::value_type &&value) {
  return insert_sorted(c, std::move(value), std::less<>());
}

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

constexpr inline std::string_view slice(std::string_view str, std::size_t begin,
                                        std::size_t end) {
  return str.substr(begin, end - begin);
}

inline std::string_view slice_strip(std::string_view str, std::size_t begin,
                                    std::size_t end) {
  return absl::StripAsciiWhitespace(slice(str, begin, end));
}

constexpr std::string_view safe_substr(std::string_view str, size_t begin,
                                       size_t count) {
  if (ABSL_PREDICT_FALSE(begin > str.size()))
    return {};

  return str.substr(begin, count);
}

constexpr std::string_view safe_slice(std::string_view str, size_t begin,
                                      size_t end) {
  if (ABSL_PREDICT_FALSE(begin > str.size()))
    return {};

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

constexpr inline int value_if(bool cond, int val = 1) {
  return static_cast<int>(cond) * val;
}

template <
    class Int,
    std::enable_if_t<std::is_integral_v<Int> && std::is_signed_v<Int>, int> = 0>
constexpr Int nonnegative(Int x) {
  return std::max(x, Int(0));
}

template <class UInt,
          std::enable_if_t<std::is_integral_v<UInt> && !std::is_signed_v<UInt>,
                           int> = 0>
constexpr UInt nonnegative(UInt x) {
  return x;
}
}  // namespace nuri

#endif /* NURI_UTILS_H_ */
