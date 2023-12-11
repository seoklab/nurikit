//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_CORE_GEOMETRY_H_
#define NURI_CORE_GEOMETRY_H_

#include <type_traits>
#include <utility>
#include <vector>

#include "nuri/eigen_config.h"

namespace nuri {
namespace internal {
  using Array8i = Array<int, 1, 8>;

  class OCTreeNode {
  public:
    OCTreeNode(Array8i &&children, int nleaf)
        : children_(std::move(children)), nleaf_(nleaf) { }

    const Array8i &children() const { return children_; }

    int child(int i) const { return children_[i]; }

    int operator[](int i) const { return child(i); }

    bool leaf() const { return nleaf_ <= 8; }

    int nleaf() const { return nleaf_; }

  private:
    Array8i children_;  // < 0 -> not exist, >= 0 -> index of child
    int nleaf_;         // <= 8 -> leaf, > 8 -> internal node
  };
}  // namespace internal

class OCTree {
public:
  OCTree(): pts_(nullptr) { }

  OCTree(const MatrixX3d &pts) { rebuild(pts); }

  void rebuild(const MatrixX3d &pts);

  void find_neighbors_k(const Vector3d &pt, int k, std::vector<int> &idxs,
                        std::vector<double> &distsq) const;

  void find_neighbors_d(const Vector3d &pt, double cutoff,
                        std::vector<int> &idxs,
                        std::vector<double> &distsq) const;

  void find_neighbors_kd(const Vector3d &pt, int k, double cutoff,
                         std::vector<int> &idxs,
                         std::vector<double> &distsq) const;

  const MatrixX3d &pts() const { return *pts_; }

  const Vector3d &max() const { return max_; }

  const Vector3d &len() const { return len_; }

  const std::vector<internal::OCTreeNode> &nodes() const { return nodes_; }

  int size() const { return static_cast<int>(nodes_.size()); }

  int root() const { return size() - 1; }

  const internal::OCTreeNode &node(int i) const { return nodes_[i]; }

  const internal::OCTreeNode &operator[](int idx) const { return nodes_[idx]; }

private:
  const MatrixX3d *pts_;
  Vector3d max_;
  Vector3d len_;
  std::vector<internal::OCTreeNode> nodes_;
};

namespace constants {
  extern constexpr inline double kPi =
      3.1415926535897932384626433832795028841971693993751058209749445923078164;

  extern constexpr inline double kTwoPi =
      6.2831853071795864769252867665590057683943387987502116419498891846156328;
}  // namespace constants

template <class DT, std::enable_if_t<std::is_floating_point_v<DT>, int> = 0>
constexpr DT deg2rad(DT deg) {
  return deg * constants::kPi / 180;
}

template <class DT, std::enable_if_t<std::is_integral_v<DT>, int> = 0>
constexpr double deg2rad(DT deg) {
  return deg * constants::kPi / 180;
}
}  // namespace nuri

#endif /* NURI_CORE_GEOMETRY_H_ */
