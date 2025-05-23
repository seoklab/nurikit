//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_CONTAINER_LINEAR_QUEUE_H_
#define NURI_CORE_CONTAINER_LINEAR_QUEUE_H_

//! @cond
#include <utility>

#include <absl/log/absl_check.h>
//! @endcond

namespace nuri {
namespace internal {
  template <class C>
  class LinearQueue {
  public:
    using value_type = typename C::value_type;

    explicit LinearQueue(C &&container) noexcept
        : container_(std::move(container)) {
      clear();
    }

    bool empty() const { return lp_ >= rp_; }

    int size() const { return rp_ - lp_; }

    auto capacity() const { return container_.size(); }

    value_type &front() {
      ABSL_DCHECK_LT(lp_, rp_);
      return container_[lp_];
    }

    const value_type &front() const {
      ABSL_DCHECK_LT(lp_, rp_);
      return container_[lp_];
    }

    value_type &back() {
      ABSL_DCHECK_LT(lp_, rp_);
      return container_[rp_ - 1];
    }

    const value_type &back() const {
      ABSL_DCHECK_LT(lp_, rp_);
      return container_[rp_ - 1];
    }

    void push(const value_type &v) {
      ABSL_DCHECK_LT(rp_, capacity());
      container_[rp_++] = v;
    }

    void push(value_type &&v) {
      ABSL_DCHECK_LT(rp_, capacity());
      container_[rp_++] = std::move(v);
    }

    value_type pop() {
      ABSL_DCHECK_LT(lp_, rp_);
      return std::move(container_[lp_++]);
    }

    void clear() noexcept { lp_ = rp_ = 0; }

    C &container() { return container_; }
    const C &container() const { return container_; }

  private:
    C container_;
    int lp_;
    int rp_;
  };

  template <class C>
  LinearQueue(C &&container) -> LinearQueue<C>;

  template <class C>
  LinearQueue(const LinearQueue<C> &) -> LinearQueue<C>;

  template <class C>
  LinearQueue(LinearQueue<C> &&) -> LinearQueue<C>;
}  // namespace internal
}  // namespace nuri

#endif /* NURI_CORE_CONTAINER_LINEAR_QUEUE_H_ */
