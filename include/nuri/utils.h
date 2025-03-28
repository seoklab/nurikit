//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_UTILS_H_
#define NURI_UTILS_H_

//! @cond
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/log/absl_check.h>
#include <absl/numeric/bits.h>
#include <absl/strings/ascii.h>
#include <boost/iterator/iterator_facade.hpp>
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
  template <class T, class C = std::less<>, class S = std::vector<T>>
  struct ClearablePQ: public std::priority_queue<T, S, C> {
    using Base = std::priority_queue<T, S, C>;

  public:
    using Base::Base;

    T pop_get() noexcept {
      T v = std::move(data().front());
      this->pop();
      return v;
    }

    void clear() noexcept { data().clear(); }

    auto &data() { return this->c; }

    void rebuild() noexcept { absl::c_make_heap(data(), C()); }
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
            bool = std::is_class_v<UnaryOp>>
  class TransformIteratorTrampoline;

  template <class Derived, class Traits, class Ref>
  using TransformIteratorBase =
      boost::iterator_facade<Derived, std::remove_reference_t<Ref>,
                             typename Traits::iterator_category, Ref,
                             typename Traits::difference_type>;

  template <class Derived, class Iter, class UnaryOp, UnaryOp op>
  class TransformIteratorTrampoline<Derived, Iter, UnaryOp, op, false>
      : public TransformIteratorBase<Derived, std::iterator_traits<Iter>,
                                     decltype(std::invoke(
                                         op, *std::declval<Iter>()))> {
  protected:
    using Parent = TransformIteratorTrampoline;
    constexpr static auto dereference_impl(Iter it) {
      return std::invoke(op, *it);
    }
  };

  template <class Derived, class Iter, class UnaryOp>
  class TransformIteratorTrampoline<Derived, Iter, UnaryOp, nullptr, true>
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
}  // namespace nuri

#endif /* NURI_UTILS_H_ */
