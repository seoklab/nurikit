//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_ITERATOR_H_
#define NURI_ITERATOR_H_

//! @cond
#include <cstddef>
#include <functional>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/iterator/iterator_facade.hpp>
//! @endcond

#include "nuri/meta.h"

namespace nuri {
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
}  // namespace nuri

#endif /* NURI_ITERATOR_H_ */
