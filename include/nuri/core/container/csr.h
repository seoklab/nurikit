//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_CONTAINER_CSR_H_
#define NURI_CORE_CONTAINER_CSR_H_

#include <type_traits>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
  /**
   * Row boundaries of a flat buffer: row `i` is `[off()[i], off()[i + 1])`.
   */
  template <class Derived>
  class OffsetTableBase {
  public:
    int operator[](int i) const { return derived().off()[i]; }

    int degree(int i) const { return (*this)[i + 1] - (*this)[i]; }

    int max_deg() const {
      int m = 0;
      for (int i = 0; i < size(); ++i)
        m = nuri::max(m, degree(i));
      return m;
    }

    int size() const { return static_cast<int>(derived().off().size()) - 1; }

    int total() const { return (*this)[size()]; }

    void clear_rows() { absl::c_fill(derived().off(), 0); }

  protected:
    OffsetTableBase() = default;

  private:
    Derived &derived() { return static_cast<Derived &>(*this); }

    const Derived &derived() const {
      return static_cast<const Derived &>(*this);
    }
  };

  /**
   * Offset table sized at construction; `off()` is filled by the owner.
   */
  class OffsetTable: public OffsetTableBase<OffsetTable> {
  public:
    OffsetTable(): off_(ArrayXi::Zero(1)) { }

    explicit OffsetTable(int n): off_(n + 1) { }

    explicit OffsetTable(ArrayXi &&off) noexcept: off_(std::move(off)) { }

    ArrayXi &off() { return off_; }

    const ArrayXi &off() const { return off_; }

  private:
    ArrayXi off_;
  };

  /**
   * Offset table that grows a row at a time.
   */
  class GrowableOffsetTable: public OffsetTableBase<GrowableOffsetTable> {
  public:
    GrowableOffsetTable() = default;

    explicit GrowableOffsetTable(int n): off_(n + 1, 0) { }

    explicit GrowableOffsetTable(std::vector<int> &&off) noexcept
        : off_(std::move(off)) { }

    void push_back(int end) { off_.push_back(end); }

    void resize(int n) { off_.resize(n + 1, total()); }

    void reserve(int n) { off_.reserve(n + 1); }

    void clear() { off_.assign(1, 0); }

    std::vector<int> &off() { return off_; }

    const std::vector<int> &off() const { return off_; }

  private:
    std::vector<int> off_ { 0 };
  };

  /**
   * Compressed sparse rows: `adj()` holds the rows back to back, and the
   * position of an entry is its edge id.
   */
  class CSR {
  public:
    using const_iterator = ArrayXi::const_iterator;

    CSR() = default;

    CSR(ArrayXi &&adj, OffsetTable &&off) noexcept
        : adj_(std::move(adj)), tbl_(std::move(off)) { }

    const_iterator begin(int i) const { return adj_.begin() + tbl_[i]; }

    const_iterator end(int i) const { return adj_.begin() + tbl_[i + 1]; }

    int offset(int i) const { return tbl_[i]; }

    int degree(int i) const { return tbl_.degree(i); }

    int max_deg() const { return tbl_.max_deg(); }

    auto nbrs(int i) const { return adj_.segment(tbl_[i], degree(i)); }

    int eid(const_iterator it) const {
      return static_cast<int>(it - adj_.begin());
    }

    int m() const { return static_cast<int>(adj_.size()); }

    int n() const { return tbl_.size(); }

    /**
     * The first `n` rows.
     */
    CSR prefix(int n) const {
      return { ArrayXi(adj_.head(tbl_[n])),
               OffsetTable(ArrayXi(off().head(n + 1))) };
    }

    ArrayXi &adj() { return adj_; }

    const ArrayXi &adj() const { return adj_; }

    ArrayXi &off() { return tbl_.off(); }

    const ArrayXi &off() const { return tbl_.off(); }

  private:
    ArrayXi adj_;
    OffsetTable tbl_;
  };

  using nuri::argsort_bucket;

  /**
   * `Table &` with the constraint, not `OffsetTableBase<Table> &`: the generic
   * `Offset &&` overload is an exact match for a table and would win over a
   * derived-to-base binding, then fail on `off.begin()`.
   */
  template <class Table, class Key, class Map,
            std::enable_if_t<std::is_base_of_v<OffsetTableBase<Table>, Table>,
                             int> = 0>
  void argsort_bucket(ArrayXi &idxs, Table &off, const Key &key,
                      const Map &map) {
    argsort_bucket(idxs, off.off(), key, map);
  }
}  // namespace internal
}  // namespace nuri

#endif /* NURI_CORE_CONTAINER_CSR_H_ */
