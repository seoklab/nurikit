//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/container/csr.h"

#include <vector>

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"

namespace nuri {
namespace internal {
namespace {
template <class Table>
void expect_rows(const Table &tbl, const ArrayXi &off) {
  ASSERT_EQ(tbl.size(), off.size() - 1);
  EXPECT_EQ(tbl.total(), off[off.size() - 1]);

  int max_deg = 0;
  for (int i = 0; i < tbl.size(); ++i) {
    EXPECT_EQ(tbl[i], off[i]) << i;
    EXPECT_EQ(tbl.degree(i), off[i + 1] - off[i]) << i;
    max_deg = nuri::max(max_deg, off[i + 1] - off[i]);
  }
  EXPECT_EQ(tbl.max_deg(), max_deg);
}

TEST(OffsetTableTest, DefaultIsEmpty) {
  const OffsetTable tbl;
  expect_rows(tbl, ArrayXi::Zero(1));
}

TEST(OffsetTableTest, ReadsRows) {
  ArrayXi off(4);
  off << 0, 2, 2, 5;
  const OffsetTable tbl { ArrayXi(off) };
  expect_rows(tbl, off);
}

TEST(OffsetTableTest, ClearsRows) {
  ArrayXi off(4);
  off << 0, 2, 2, 5;
  OffsetTable tbl { ArrayXi(off) };
  tbl.clear_rows();
  expect_rows(tbl, ArrayXi::Zero(4));
}

TEST(OffsetTableTest, ArgsortBucketIsStable) {
  ArrayXi key(5);
  key << 2, 0, 2, 1, 0;

  ArrayXi idxs(5);
  OffsetTable off(3);
  argsort_bucket(idxs, off, key);

  ArrayXi expected_off(4), expected_idxs(5);
  expected_off << 0, 2, 3, 5;
  expected_idxs << 1, 4, 3, 0, 2;
  EXPECT_TRUE((off.off() == expected_off).all()) << off.off().transpose();
  EXPECT_TRUE((idxs == expected_idxs).all()) << idxs.transpose();
}

TEST(OffsetTableTest, ArgsortBucketKeepsEmptyBuckets) {
  ArrayXi key(4);
  key << 2, 0, 2, 0;

  ArrayXi idxs(4);
  OffsetTable off(4);
  argsort_bucket(idxs, off, key);

  ArrayXi expected_off(5), expected_idxs(4);
  expected_off << 0, 2, 2, 4, 4;
  expected_idxs << 1, 3, 0, 2;
  expect_rows(off, expected_off);
  EXPECT_TRUE((idxs == expected_idxs).all()) << idxs.transpose();

  ArrayXi none(0);
  argsort_bucket(none, off, none);
  expect_rows(off, ArrayXi::Zero(5));
}

TEST(OffsetTableTest, ArgsortBucketTakesBlock) {
  ArrayXi key(4);
  key << 2, 0, 2, 0;

  ArrayXi idxs(4);
  OffsetTable off(3);
  argsort_bucket(idxs, off.off().head(3), key);

  ArrayXi expected_start(3), expected_idxs(4);
  expected_start << 0, 2, 2;
  expected_idxs << 1, 3, 0, 2;
  EXPECT_TRUE((off.off().head(3) == expected_start).all())
      << off.off().head(3).transpose();
  EXPECT_TRUE((idxs == expected_idxs).all()) << idxs.transpose();
}

TEST(OffsetTableTest, ArgsortBucketFollowsMap) {
  ArrayXi key(5), order(5);
  key << 2, 0, 2, 1, 0;
  order << 4, 3, 2, 1, 0;

  ArrayXi idxs(5);
  OffsetTable off(3);
  argsort_bucket(idxs, off, key, [&](int p) { return order[p]; });

  ArrayXi expected_idxs(5);
  expected_idxs << 4, 1, 3, 2, 0;
  EXPECT_TRUE((idxs == expected_idxs).all()) << idxs.transpose();
}

TEST(GrowableOffsetTableTest, DefaultIsEmpty) {
  const GrowableOffsetTable tbl;
  expect_rows(tbl, ArrayXi::Zero(1));
}

TEST(GrowableOffsetTableTest, StartsWithEmptyRows) {
  const GrowableOffsetTable tbl(4);
  expect_rows(tbl, ArrayXi::Zero(5));
}

TEST(GrowableOffsetTableTest, GrowsRows) {
  GrowableOffsetTable tbl;
  tbl.reserve(5);
  tbl.push_back(3);
  tbl.push_back(3);
  tbl.push_back(7);

  ArrayXi off(4);
  off << 0, 3, 3, 7;
  expect_rows(tbl, off);

  tbl.resize(5);
  ArrayXi grown(6);
  grown << 0, 3, 3, 7, 7, 7;
  expect_rows(tbl, grown);

  tbl.off()[5] = 9;
  EXPECT_EQ(tbl.degree(4), 2);

  tbl.resize(2);
  expect_rows(tbl, off.head(3));

  tbl.clear();
  expect_rows(tbl, ArrayXi::Zero(1));
}

TEST(GrowableOffsetTableTest, TakesVector) {
  const GrowableOffsetTable tbl {
    std::vector<int> { 0, 2, 2, 5 }
  };
  ArrayXi off(4);
  off << 0, 2, 2, 5;
  expect_rows(tbl, off);
}

TEST(GrowableOffsetTableTest, ClearsRows) {
  GrowableOffsetTable tbl {
    std::vector<int> { 0, 2, 2, 5 }
  };
  tbl.clear_rows();
  expect_rows(tbl, ArrayXi::Zero(4));
}

TEST(GrowableOffsetTableTest, ArgsortBucketFillsRows) {
  ArrayXi key(5);
  key << 2, 0, 2, 1, 0;

  ArrayXi idxs(5);
  GrowableOffsetTable off(3);
  argsort_bucket(idxs, off, key);

  ArrayXi expected_off(4), expected_idxs(5);
  expected_off << 0, 2, 3, 5;
  expected_idxs << 1, 4, 3, 0, 2;
  expect_rows(off, expected_off);
  EXPECT_TRUE((idxs == expected_idxs).all()) << idxs.transpose();
}

TEST(CSRTest, DefaultIsEmpty) {
  const CSR csr;
  EXPECT_EQ(csr.n(), 0);
  EXPECT_EQ(csr.m(), 0);
  EXPECT_EQ(csr.max_deg(), 0);
}

TEST(CSRTest, ReadsRows) {
  ArrayXi adj(6), off(5);
  adj << 1, 2, 0, 0, 3, 2;
  off << 0, 2, 3, 5, 6;
  const CSR csr { ArrayXi(adj), OffsetTable(ArrayXi(off)) };

  EXPECT_EQ(csr.n(), 4);
  EXPECT_EQ(csr.m(), 6);
  EXPECT_EQ(csr.max_deg(), 2);

  EXPECT_EQ(csr.offset(2), 3);
  EXPECT_EQ(csr.degree(3), 1);
  EXPECT_EQ(csr.nbrs(0).size(), 2);
  EXPECT_EQ(csr.nbrs(0)[1], 2);
  EXPECT_EQ(csr.end(1) - csr.begin(1), 1);
  EXPECT_EQ(csr.eid(csr.begin(2)), 3);
  EXPECT_EQ(*csr.begin(2), 0);
}

TEST(CSRTest, ReadsEmptyRow) {
  ArrayXi adj(3), off(4);
  adj << 1, 2, 0;
  off << 0, 2, 2, 3;
  const CSR csr { ArrayXi(adj), OffsetTable(ArrayXi(off)) };

  EXPECT_EQ(csr.degree(1), 0);
  EXPECT_EQ(csr.begin(1), csr.end(1));
  EXPECT_EQ(csr.nbrs(1).size(), 0);
  EXPECT_EQ(csr.offset(1), csr.offset(2));
}

TEST(CSRTest, TakesPrefix) {
  ArrayXi adj(6), off(5);
  adj << 1, 2, 0, 0, 3, 2;
  off << 0, 2, 3, 5, 6;
  const CSR csr { ArrayXi(adj), OffsetTable(ArrayXi(off)) };

  const CSR head = csr.prefix(2);
  EXPECT_EQ(head.n(), 2);
  EXPECT_EQ(head.m(), 3);
  EXPECT_TRUE((head.adj() == adj.head(3)).all()) << head.adj().transpose();
  EXPECT_TRUE((head.off() == off.head(3)).all()) << head.off().transpose();

  const CSR none = csr.prefix(0);
  EXPECT_EQ(none.n(), 0);
  EXPECT_EQ(none.m(), 0);
  ASSERT_EQ(none.off().size(), 1);
  EXPECT_EQ(none.off()[0], 0);
}
}  // namespace
}  // namespace internal
}  // namespace nuri
