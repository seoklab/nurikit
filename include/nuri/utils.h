//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_UTILS_H_
#define NURI_UTILS_H_

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/iterator/iterator_facade.hpp>

#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/strings/ascii.h>

#include "nuri/eigen_config.h"

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
          T, typename std::iterator_traits<Iterator>::reference>>;

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
}  // namespace internal

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
  auto find_name(PT &props) -> decltype(props.begin()) {
    return std::find_if(props.begin(), props.end(),
                        [](const auto &p) { return p.first == kNameKey; });
  }

  template <class PT>
  const std::string *get_name(PT &props) {
    auto it = internal::find_name(props);
    if (it == props.end()) {
      return nullptr;
    }
    return &it->second;
  }

  template <class PT>
  void set_name(PT &props, std::string_view name) {
    auto it = find_name(props);
    if (it != props.end()) {
      it->second = name;
    } else {
      props.emplace_back(kNameKey, name);
    }
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
              Comp &&comp) {
  // *it >= value
  auto it =
      std::lower_bound(c.begin(), c.end(), value, std::forward<Comp>(comp));
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
insert_sorted(Container &c, typename Container::value_type &&value,
              Comp &&comp) {
  // *it >= value
  auto it =
      std::lower_bound(c.begin(), c.end(), value, std::forward<Comp>(comp));
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

inline absl::FixedArray<int> generate_index(int size) {
  absl::FixedArray<int> result(size);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

template <class Container, class Comp>
absl::FixedArray<int> argsort(const Container &container, Comp op) {
  absl::FixedArray<int> idxs = generate_index(container.size());
  std::sort(idxs.begin(), idxs.end(),
            [&](int i, int j) { return op(container[i], container[j]); });
  return idxs;
}

template <class Container, class Comp>
absl::FixedArray<int> argpartition(const Container &container, int count,
                                   Comp op) {
  absl::FixedArray<int> idxs = generate_index(container.size());
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

inline MatrixX3d stack(const std::vector<Vector3d> &vs) {
  MatrixX3d m(vs.size(), 3);
  for (int i = 0; i < vs.size(); ++i) {
    m.row(i) = vs[i];
  }
  return m;
}
}  // namespace nuri

#endif /* NURI_UTILS_H_ */
