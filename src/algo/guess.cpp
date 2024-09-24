//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/algo/guess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/inlined_vector.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>

#include "nuri/eigen_config.h"
#include "nuri/algo/rings.h"
#include "nuri/core/element.h"
#include "nuri/core/geometry.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace {
  void reset_atoms(Molecule &mol) {
    for (auto atom: mol) {
      atom.data()
          .set_hybridization(constants::kOtherHyb)
          .set_formal_charge(0)
          .set_implicit_hydrogens(0)
          .del_flags(AtomFlags::kConjugated | AtomFlags::kAromatic);
    }
  }

  void reset_bonds(Molecule &mol) {
    for (auto bond: mol.bonds()) {
      bond.data()
          .set_order(constants::kOtherBond)
          .del_flags(BondFlags::kConjugated | BondFlags::kAromatic);
    }
  }

  // pdist-based solution is faster for about <= 1000 atoms.
  // set 100 for debug builds, for testing purposes.
  constexpr int kSmallLimit =
#ifdef NDEBUG
      1000;
#else
      100;
#endif

  void prepare_conn_search(const Matrix3Xd &pos, ArrayXd &distsq,
                           OCTree &tree) {
    if (pos.cols() <= kSmallLimit) {
      distsq = pdistsq(pos);
    } else {
      tree.rebuild(pos);
    }
  }

  constexpr double rcov_sum_sq(const Element &e, const Element &f,
                               double threshold) {
    double d = e.covalent_radius() + f.covalent_radius() + threshold;
    return d * d;
  }

  void guess_connectivity_small(MoleculeMutator &mut, double threshold,
                                const ArrayXd &distsq) {
    const Molecule &mol = mut.mol();

    const int n = mol.size();
    for (int i = 0, k = 0; i < n - 1; ++i) {
      const Element &e = mol.atom(i).data().element();
      if (e.type() == Element::Type::kMetal)
        continue;

      for (int j = i + 1; j < n; ++j, ++k) {
        const Element &f = mol.atom(j).data().element();
        if (f.type() == Element::Type::kMetal)
          continue;

        if (distsq[k] <= rcov_sum_sq(e, f, threshold))
          mut.add_bond(i, j, {});
      }
    }
  }

  void guess_connectivity_large(MoleculeMutator &mut, const double threshold,
                                const OCTree &oct) {
    std::vector<int> idxs;
    std::vector<double> distsq;

    const Molecule &mol = mut.mol();
    for (auto src: mol) {
      const Element &e = src.data().element();
      if (e.type() == Element::Type::kMetal)
        continue;

      // No atoms will form bonds > 5.0 A.
      oct.find_neighbors_d(oct.pts().col(src.id()), 5.0 + threshold, idxs,
                           distsq);

      for (int i = 0; i < idxs.size(); i++) {
        auto dst = mol.atom(idxs[i]);
        if (src.id() == dst.id())
          continue;

        const Element &f = dst.data().element();
        if (f.type() == Element::Type::kMetal)
          continue;

        if (distsq[i] <= rcov_sum_sq(e, f, threshold))
          mut.add_bond(src.id(), dst.id(), {});
      }
    }
  }

  void remove_excess_bonds_max_n(MoleculeMutator &mut,
                                 Molecule::MutableAtom atom, int n,
                                 const Matrix3Xd &pos) {
    VectorXd dsqs =
        (pos(Eigen::all, as_index(atom)).colwise() - pos.col(atom.id()))
            .colwise()
            .squaredNorm();
    ArrayXi idxs = argpartition(dsqs, n);
    for (int i = n; i < atom.degree(); ++i)
      mut.mark_bond_erase(atom[idxs[i]].eid());
  }

  template <class OnExcess>
  bool no_excess_bonds(const Molecule &mol, OnExcess onexcess) {
    for (auto atom: mol) {
      int max_neighbors;
      switch (atom.data().element().period()) {
      case 0:
        continue;
      case 1:
        max_neighbors = 2 - atom.data().atomic_number();
        break;
      case 2:
        max_neighbors = 4;
        break;
      default:
        max_neighbors = 6;
        break;
      }

      if (atom.degree() > max_neighbors)
        if (!onexcess(atom, max_neighbors))
          return false;
    }

    return true;
  }

  void guess_connectivity(MoleculeMutator &mut, const Matrix3Xd &pos,
                          double threshold, const ArrayXd &dist,
                          const OCTree &oct) {
    if (mut.mol().size() <= kSmallLimit)
      guess_connectivity_small(mut, threshold, dist);
    else
      guess_connectivity_large(mut, threshold, oct);

    no_excess_bonds(mut.mol(), [&](Molecule::Atom atom, int max_neighbors) {
      remove_excess_bonds_max_n(mut, mut.mol()[atom.id()], max_neighbors, pos);
      return true;
    });
  }

  /*
   * Cutoff values taken from:
   *  https://web.archive.org/web/20231205050012/https://www.daylight.com/meetings/mug01/Sayle/m4xbondage.html
   */
  // NOLINTBEGIN(*-identifier-naming)
  constexpr double kCos15 =
      0.9659258262890682867497431997288973676339048390084045504023430763;
  constexpr double kCos115 =
      -0.422618261740699436186978489647730181563129301194864623444415159;
  constexpr double kCos155 =
      -0.906307787036649963242552656754316983267712625175864680871298408;
  constexpr double kTan10_2 =
      0.0874886635259240052220186694349614581194542763681082291452366622;
  constexpr double kTan15_2 =
      0.1316524975873958534715264574097171035928141022232375735535653257;
  constexpr double kTan116_2 =
      1.6003345290410503553267330811833575255040718469227591484115002297;
  constexpr double kTan155_2 =
      4.5107085036620571342899391172547519686713241944553043587162345185;
  // NOLINTEND(*-identifier-naming)

  constants::Hybridization hyb_from_cos(double avg_cos_angle) {
    // Cosine is decreasing in [0, 180] degree.
    //    x <= a --> cos(x) >= cos(a)
    if (avg_cos_angle >= kCos115)
      return constants::kSP3;

    if (avg_cos_angle >= kCos155)
      return constants::kSP2;

    return constants::kSP;
  }

  constants::Hybridization hyb_from_vectors(const Matrix3d &vectors) {
    auto [ssum, csum] = sum_tan2_half(vectors);
    double sum_sin_116_2 = kTan116_2 * csum, sum_sin_155_2 = kTan155_2 * csum;

    if (ssum <= sum_sin_116_2)
      return constants::kSP3;

    if (ssum <= sum_sin_155_2)
      return constants::kSP2;

    return constants::kSP;
  }

  bool torsion_can_sp2(const Matrix3Xd &pos, int a, int b, int c, int d) {
    auto cos = cos_dihedral(pos.col(a), pos.col(b), pos.col(c), pos.col(d));
    return cos >= kCos15 || cos <= -kCos15;
  }

  constants::Hybridization hyb_common(Molecule::Atom atom,
                                      const Matrix3Xd &pos) {
    if (atom.degree() == 2) {
      double cos = cos_angle(pos.col(atom.id()), pos.col(atom[0].dst().id()),
                             pos.col(atom[1].dst().id()));
      constants::Hybridization hyb = hyb_from_cos(cos);

      if (hyb == constants::kSP2) {
        for (int i = 0; i < 2; ++i) {
          auto nei = atom[i];
          if (nei.dst().degree() < 2)
            return constants::kSP2;

          if (all_neighbors(nei.dst()) > 3)
            continue;

          auto oei = atom[1 - i];
          if (absl::c_all_of(nei.dst(), [&](Molecule::Neighbor mei) {
                return mei.dst().id() == atom.id()
                       || torsion_can_sp2(pos, oei.dst().id(), atom.id(),
                                          nei.dst().id(), mei.dst().id());
              })) {
            return constants::kSP2;
          }
        }

        return constants::kSP3;
      }

      return hyb;
    }

    ABSL_DCHECK(atom.degree() == 3);

    Matrix3d vectors = internal::safe_colwise_normalized(
        pos(Eigen::all, as_index(atom)).colwise() - pos.col(atom.id()));
    return hyb_from_vectors(vectors);
  }

  constants::Hybridization hyb_expansion(Molecule::Atom atom) {
    switch (atom.degree()) {
    case 4:
      return constants::kSP3;
    case 5:
      return constants::kSP3D;
    case 6:
      return constants::kSP3D2;
    default:
      return constants::kOtherHyb;
    }
  }

  void hyb_ring(Molecule &mol, const Matrix3Xd &pos,
                const std::vector<int> &ring, double threshold) {
    using MatrixMax36d = Matrix<double, 3, Eigen::Dynamic, 0, 3, 6>;

    const int n = static_cast<int>(ring.size());
    ABSL_DCHECK(n < 7);

    CyclicIndex<1> next(n);
    ArrayXi::ConstMapType rv(ring.data(), n);

    // vectors i -> i+1
    MatrixMax36d vectors = pos(Eigen::all, rv(next)) - pos(Eigen::all, rv);

    // cross product of (i -> i+1) and (i+1 -> i+2), cyclic
    MatrixMax36d cross(3, n);
    for (int i = 0; i < n; ++i)
      cross.col(i) = vectors.col(i).cross(vectors.col(next[i]));
    internal::safe_colwise_normalize(cross);

    auto [ssum, csum] = sum_tan2_half(cross, next);
    if (ssum <= threshold * csum) {
      for (auto id: ring) {
        auto atom = mol.atom(id);
        // If degree >= 3, already covered by average bond angle, because
        // the neighbors will be in the same plane if they're sp2.
        if (atom.degree() < 3)
          atom.data().set_hybridization(constants::kSP2);
      }
    }
  }

  void hyb_antialiasing(Molecule &mol) {
    for (auto atom: mol) {
      AtomData &data = atom.data();
      if (atom.degree() <= 1 || data.hybridization() > constants::kSP2)
        continue;

      bool diff_or_terminal =
          std::none_of(atom.begin(), atom.end(), [&](auto nei) {
            Molecule::Atom dst = nei.dst();
            return dst.data().hybridization() == data.hybridization()
                   || dst.degree() == 1;
          });
      data.set_hybridization(static_cast<constants::Hybridization>(
          data.hybridization() + value_if(diff_or_terminal)));
    }
  }

  void guess_hyb_nonterminal(Molecule &mol, const Matrix3Xd &pos,
                             const std::vector<std::vector<int>> &rings) {
    for (auto atom: mol) {
      if (atom.degree() <= 1 || atom.data().element().period() == 1)
        continue;

      if (atom.degree() <= 3) {
        atom.data().set_hybridization(hyb_common(atom, pos));
        continue;
      }

      switch (atom.data().element().period()) {
      case 1:
        ABSL_UNREACHABLE();
        break;
      case 2:
        atom.data().set_hybridization(constants::kSP3);
        break;
      default:
        atom.data().set_hybridization(hyb_expansion(atom));
        break;
      }
    }

    for (auto &ring: rings) {
      if (ring.size() == 5) {
        hyb_ring(mol, pos, ring, kTan10_2);
      } else if (ring.size() == 6) {
        hyb_ring(mol, pos, ring, kTan15_2);
      }
    }

    hyb_antialiasing(mol);
  }

  bool fg_nitrile(Molecule::MutableAtom atom) {
    auto n = atom[0], m = atom[1];

    auto is_nitrile_nitrogen = [](Molecule::Atom a) {
      return a.data().atomic_number() == 7 && all_neighbors(a) == 1;
    };

    bool is_n = is_nitrile_nitrogen(n.dst()),
         is_m = is_nitrile_nitrogen(m.dst());
    if (is_n == is_m)
      return false;

    if (is_m)
      std::swap(n, m);

    n.edge_data().set_order(constants::kTripleBond);
    n.dst().data().set_hybridization(constants::kTerminal);

    m.edge_data().set_order(constants::kSingleBond);
    return true;
  }

  // * - N = N(+) = N(-) or - C # N
  bool fg_sec(Molecule::MutableAtom atom, ArrayXb &visited) {
    if (atom.data().hybridization() == constants::kSP
        && atom.data().atomic_number() == 6) {
      return fg_nitrile(atom);
    }

    auto n = atom[0], m = atom[1];
    if (absl::c_any_of(std::array { atom, n.dst(), m.dst() },
                       [](Molecule::Atom a) {
                         return a.data().atomic_number() != 7;
                       }))
      return false;

    // must be {2, 1} or {1, 2}, since lower bound is 1 (connected to this atom)
    if (n.dst().degree() + m.dst().degree() != 3)
      return false;

    auto [terminal, internal] = nuri::minmax(n, m, [](auto l, auto r) {
      return l.dst().degree() < r.dst().degree();
    });

    atom.data().set_hybridization(constants::kSP).set_formal_charge(1);

    internal.edge_data().set_order(constants::kDoubleBond);
    internal.dst().data().set_hybridization(constants::kSP);

    terminal.edge_data().set_order(constants::kDoubleBond);
    terminal.dst()
        .data()
        .set_hybridization(constants::kSP)
        .set_formal_charge(-1);

    for (auto i: { atom.id(), n.dst().id(), m.dst().id() })
      visited[i] = true;

    return true;
  }

  // carbon with three heteroatom, double/single/single
  bool fg_guanidinium_like(Molecule::MutableAtom atom, const Vector3d &dsqs,
                           ArrayXb &visited) {
    if (absl::c_any_of(atom, [](auto nei) { return nei.dst().degree() > 3; }))
      return false;

    // Now we favor double bonds on: terminal O > S > N > internal N > S > O
    //                                        0   1   2           21  22  23
    // If the carbon is in a ring, add penalty (10) to ring bonds
    Array3i penalty;
    for (int i = 0; i < atom.degree(); ++i) {
      auto nei = atom[i];
      auto dst = nei.dst();

      switch (dst.data().atomic_number()) {
      case 7:
        penalty[i] = 2 + value_if(dst.degree() > 2, 19);
        break;
      case 8:
        penalty[i] = 0 + value_if(dst.degree() > 1, 23);
        break;
      case 16:
        penalty[i] = 1 + value_if(dst.degree() > 1, 21);
        break;
      default:
        penalty[i] = 100;
        break;
      }

      penalty[i] += value_if(nei.edge_data().is_ring_bond(), 10);
    }

    auto dit =
        std::min_element(make_zipped_iterator(penalty.begin(), dsqs.begin()),
                         make_zipped_iterator(penalty.end(), dsqs.end()));
    int dnei = static_cast<int>(dit.first() - penalty.begin());

    visited[atom.id()] = true;
    atom.data()
        .set_hybridization(constants::kSP2)
        .add_flags(AtomFlags::kConjugated);

    for (int i = 0; i < 3; ++i) {
      auto nei = atom[i];
      auto dst = nei.dst();
      visited[dst.id()] = true;

      dst.data()
          .set_hybridization(constants::kSP2)
          .add_flags(AtomFlags::kConjugated);
      nei.edge_data()
          .set_order(i == dnei ? constants::kDoubleBond
                               : constants::kSingleBond)
          .add_flags(BondFlags::kConjugated);
    }

    return true;
  }

  // 2 <= number of neighbors <= 3
  bool fg_tert_carbon(Molecule::MutableAtom atom, const Matrix3Xd &pos,
                      ArrayXb &visited) {
    using ArrayMax3i = Array<int, Eigen::Dynamic, 1, 0, 3>;
    using VectorMax3d = Matrix<double, Eigen::Dynamic, 1, 0, 3>;

    int ncnt = absl::c_count_if(atom, [](Molecule::Neighbor nei) {
      return nei.dst().data().atomic_number() == 7;
    });
    int ocnt = absl::c_count_if(atom, [](Molecule::Neighbor nei) {
      return nei.dst().data().atomic_number() == 8;
    });
    int scnt = absl::c_count_if(atom, [](Molecule::Neighbor nei) {
      return nei.dst().data().atomic_number() == 16;
    });
    int total = ncnt + ocnt + scnt;

    VectorMax3d dsqs =
        (pos(Eigen::all, as_index(atom)).colwise() - pos.col(atom.id()))
            .colwise()
            .squaredNorm();

    if (total == 3)
      return fg_guanidinium_like(atom, dsqs, visited);

    if (total != 2)
      return false;

    // Now we favor double bonds on:
    //   1) O > S > N
    //   2) terminal > internal
    ArrayMax3i penalty(atom.degree()), max_neighbors(atom.degree());
    for (int i = 0; i < atom.degree(); ++i) {
      auto dst = atom[i].dst();

      int max_allowed_nei;
      switch (dst.data().atomic_number()) {
      case 7:
        penalty[i] = 20;
        max_allowed_nei = 3;
        break;
      case 8:
        penalty[i] = 0;
        max_allowed_nei = 2;
        break;
      case 16:
        penalty[i] = 10;
        max_allowed_nei = 2;
        break;
      default:
        penalty[i] = 100;
        max_allowed_nei = 0;
        break;
      }

      penalty[i] += count_heavy(dst) < max_allowed_nei ? dst.degree() : 100;
      if (dst.data().is_ring_atom())
        penalty[i] += 50;
      max_neighbors[i] = max_allowed_nei;
    }

    if ((penalty >= 100).all())
      return false;

    // Also consider distances here
    auto dit =
        std::min_element(make_zipped_iterator(penalty.begin(), dsqs.begin()),
                         make_zipped_iterator(penalty.end(), dsqs.end()));
    int dnei = static_cast<int>(dit.first() - penalty.begin());

    visited[atom.id()] = true;
    atom.data().set_hybridization(constants::kSP2);

    // Exclude the highest penalty atom if three neighbors exist
    ArrayMax3i ordered = argpartition<Eigen::Dynamic, 3>(penalty, 2);
    for (int i: ordered.head<2>()) {
      auto nei = atom[i];
      auto dst = nei.dst();
      visited[dst.id()] = true;

      int nnei = max_neighbors[i] - value_if(i == dnei);
      dst.data()
          .set_hybridization(constants::kSP2)
          .set_formal_charge(nonnegative(dst.degree() - nnei))
          .set_implicit_hydrogens(nonnegative(nnei - dst.degree()));

      for (auto mei: dst)
        if (mei.edge_data().order() == constants::kOtherBond)
          mei.edge_data().set_order(constants::kSingleBond);
    }

    atom[dnei].edge_data().set_order(constants::kDoubleBond);

    if (ordered.size() == 3) {
      auto nei = atom[ordered[2]];
      nei.edge_data().set_order(constants::kSingleBond);
    }

    return true;
  }

  // -NO2 or -SeOOH
  bool fg_tert_nitrogen_selenium(Molecule::MutableAtom atom,
                                 const Matrix3Xd &pos, ArrayXb &visited,
                                 constants::Hybridization hyb_single,
                                 int fchg_single, int hcnt_single) {
    absl::InlinedVector<Molecule::MutableNeighbor, 3> oxygens;
    for (int i = 0; i < atom.degree(); ++i) {
      auto dst = atom[i].dst();
      if (dst.data().atomic_number() == 8 && dst.degree() == 1)
        oxygens.push_back(atom[i]);
    }

    if (oxygens.size() != 2)
      return false;

    auto [near, far] =
        nuri::minmax(oxygens[0], oxygens[1], [&](auto lhs, auto rhs) {
          return (pos.col(lhs.dst().id()) - pos.col(atom.id())).squaredNorm()
                 < (pos.col(rhs.dst().id()) - pos.col(atom.id())).squaredNorm();
        });

    visited[atom.id()] = true;
    atom.data()
        .set_hybridization(constants::kSP2)
        .set_formal_charge(1)
        .add_flags(AtomFlags::kConjugated);

    visited[near.dst().id()] = true;
    near.dst()
        .data()
        .set_hybridization(constants::kTerminal)
        .set_formal_charge(0)
        .add_flags(AtomFlags::kConjugated);
    near.edge_data()
        .set_order(constants::kDoubleBond)
        .add_flags(BondFlags::kConjugated);

    visited[far.dst().id()] = true;
    far.dst()
        .data()
        .set_hybridization(hyb_single)
        .set_formal_charge(fchg_single)
        .set_implicit_hydrogens(hcnt_single)
        .add_flags(AtomFlags::kConjugated);
    far.edge_data()
        .set_order(constants::kSingleBond)
        .add_flags(BondFlags::kConjugated);

    return true;
  }

  bool fg_tert(Molecule::MutableAtom atom, const Matrix3Xd &pos,
               ArrayXb &visited) {
    if (atom.degree() == 3 && atom.data().hybridization() != constants::kSP2)
      return false;

    switch (atom.data().atomic_number()) {
    case 6:
      return fg_tert_carbon(atom, pos, visited);
    case 7:
      return fg_tert_nitrogen_selenium(atom, pos, visited, constants::kTerminal,
                                       -1, 0);
    case 34:
      return fg_tert_nitrogen_selenium(atom, pos, visited, constants::kSP2, 0,
                                       1);
    default:
      return false;
    }
  }

  // phosphoryl & sulfonyl
  void fg_x_yl(Molecule::MutableAtom atom, const Matrix3Xd &pos,
               ArrayXb &visited) {
    using ArrayMax4d = Array<double, Eigen::Dynamic, 1, 0, 4>;

    const int ocnt = atom.data().atomic_number() - 14;
    if (ocnt > 2 || ocnt < 1)
      return;

    absl::InlinedVector<int, 4> oxygens;
    for (int i = 0; i < atom.degree(); ++i) {
      auto dst = atom[i].dst();
      if (dst.data().atomic_number() == 8 && dst.degree() == 1)
        oxygens.push_back(i);
    }

    if (oxygens.size() < ocnt)
      return;

    ArrayMax4d dsqs = (pos(Eigen::all, oxygens).colwise() - pos.col(atom.id()))
                          .colwise()
                          .squaredNorm();

    absl::FixedArray<int> idxs(oxygens.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    absl::c_nth_element(idxs, idxs.begin() + ocnt - 1,
                        [&](int l, int r) { return dsqs[l] < dsqs[r]; });

    visited[atom.id()] = true;
    atom.data().set_hybridization(constants::kSP3);

    for (auto nei: atom)
      nei.edge_data().set_order(constants::kSingleBond);

    for (int i = 0; i < ocnt; ++i) {
      auto nei = atom[oxygens[idxs[i]]];
      auto dst = nei.dst();
      visited[dst.id()] = true;

      dst.data().set_hybridization(constants::kSP2).set_formal_charge(0);
      nei.edge_data().set_order(constants::kDoubleBond);
    }
  }

  void recognize_fg(Molecule &mol, const Matrix3Xd &pos) {
    ArrayXb visited = ArrayXb::Zero(mol.size());

    for (auto atom: mol) {
      if (visited[atom.id()] || atom.degree() < 2)
        continue;

      if (atom.degree() <= 2 && fg_sec(atom, visited))
        continue;

      if (atom.degree() <= 3 && fg_tert(atom, pos, visited))
        continue;

      if (atom.degree() <= 4)
        fg_x_yl(atom, pos, visited);
    }
  }

  void assign_priority_bonds(Molecule &mol) {
    // Assign confident single bonds
    for (auto atom: mol) {
      if (atom.degree() < 3 || atom.data().hybridization() != constants::kSP3
          || atom.data().element().period() > 2)
        continue;

      for (auto nei: atom)
        nei.edge_data().set_order(constants::kSingleBond);
    }

    // Assign double bonds if no other choice
    for (auto atom: mol) {
      if (atom.data().atomic_number() != 6 || atom.degree() < 3
          || atom.data().hybridization() != constants::kSP2)
        continue;

      int sel = -1;
      for (int i = 0; i < atom.degree(); ++i) {
        auto nei = atom[i];
        if (nei.edge_data().order() > constants::kSingleBond) {
          sel = -1;
          break;
        }

        if (nei.edge_data().order() == constants::kOtherBond) {
          if (sel < 0) {
            sel = i;
          } else {
            sel = -1;
            break;
          }
        }
      }
      if (sel < 0)
        continue;

      auto cnd = atom[sel];
      if (cnd.dst().data().hybridization() <= constants::kSP2)
        cnd.edge_data().set_order(constants::kDoubleBond);
    }
  }

  bool is_aromatic_candidate(Molecule::Atom ring_atom) {
    const AtomData &data = ring_atom.data();
    return 3 <= data.element().valence_electrons()
           && data.element().valence_electrons() <= 6
           && data.hybridization() <= constants::kSP2;
  }

  struct PiElecArgs {
    int pi_e;                // pi electrons
    int endo_double;         // in-ring double bond count
    int exo_double_partner;  // exocyclic double bond partner, < 0 if none
    int exo_single_partner;  // exocyclic single bond partner, < 0 if none
    int fchg;                // required formal charge
    int degree;              // required degree
    int prio;                // priority estimate
  };

  // We will just assume formal charges are incorrect yet
  void count_pi_e_possible(Molecule::Atom atom,
                           absl::InlinedVector<PiElecArgs, 2> &pi_e) {
    ABSL_DCHECK(atom.degree() < 4);

    int exo_neighbor =
        absl::c_find_if(atom,
                        [](Molecule::Neighbor nei) {
                          return !nei.dst().data().is_ring_atom();
                        })
        - atom.begin();

    int exo_double_partner = -1, exo_single_partner = -1;
    if (exo_neighbor < atom.degree()) {
      auto exo = atom[exo_neighbor].dst();

      exo_single_partner = exo_neighbor;
      if (exo.data().atomic_number() == 8 && exo.degree() == 1)
        exo_double_partner = exo_neighbor;
    }

    // 1. all degree 2 could also be considered as degree 3 with one implicit H,
    //    so do it first if it's more common; then,
    // 2. consider specific cases without implicit Hs
    switch (atom.data().element().valence_electrons()) {
    case 3:
      // B, degree 3 (possibly one implicit H)
      pi_e.push_back(
          { 0, 0, -1, exo_single_partner, 0, 3, 1 });  // borazine-like
      if (atom.degree() == 2)
        pi_e.push_back(
            { 1, 1, -1, exo_single_partner, 0, 2, 1 });  // borabenzene
      // Nothing special for explicit degree 3
      break;
    case 4:
      if (atom.degree() == 3 && exo_double_partner >= 0)
        // C=O possible
        pi_e.push_back({ 0, 0, exo_double_partner, -1, 0, 3, 0 });
      // C, degree 3 (possibly one implicit H)
      pi_e.push_back({ 1, 1, -1, exo_single_partner, 0, 3, 1 });  // benzene
      // C with only 2 neighbors is very unlikely to be aromatic
      break;
    case 5:
      // N, degree 3 (possibly one implicit H)
      pi_e.push_back({ 2, 0, -1, exo_single_partner, 0, 3, 1 });  // pyrrole
      if (atom.degree() == 2) {
        pi_e.push_back({ 1, 1, -1, exo_single_partner, 0, 2, 1 });  // pyridine
      } else if (/* atom.degree() == 3 && */ exo_double_partner >= 0) {
        pi_e.push_back({ 1, 1, exo_double_partner, -1, +1, 3, 0 });  // N-oxide
      }
      break;
    case 6:
      if (atom.degree() == 2) {
        // furan
        pi_e.push_back({ 2, 0, -1, exo_single_partner, 0, 2, 1 });
        // furan with C=O(+)-C group
        pi_e.push_back({ 1, 1, -1, exo_single_partner, +1, 2, 0 });
      } else /* if (atom.degree() == 3) */ {
        // furanium ion
        pi_e.push_back({ 2, 0, -1, exo_single_partner, +1, 3, 1 });
      }
      break;
    default:
      ABSL_UNREACHABLE();
    }

    ABSL_DCHECK(pi_e.size() <= 2);
  }

  void test_ring_aromatic(
      Molecule &mol, const std::vector<int> &ring,
      absl::flat_hash_map<int, absl::InlinedVector<PiElecArgs, 2>>
          &pi_e_estimates) {
    absl::FixedArray<absl::InlinedVector<PiElecArgs, 2> *> args(ring.size());
    absl::InlinedVector<int, 7> variable;
    int pie_sum_fixed = 0, db_cnt_fixed = 0, pie_sum_var = 0, db_cnt_var = 0;

    for (int i = 0; i < ring.size(); ++i) {
      auto &pi_e = pi_e_estimates.find(ring[i])->second;
      args[i] = &pi_e;
      if (pi_e.size() == 1) {
        pie_sum_fixed += pi_e[0].pi_e;
        db_cnt_fixed += pi_e[0].endo_double;
      } else {
        pie_sum_var += pi_e[0].pi_e;
        db_cnt_var += pi_e[0].endo_double;
        variable.push_back(i);
      }
    }

    auto is_aromatic = [&]() {
      return (pie_sum_fixed + pie_sum_var) % 4 == 2
             && (db_cnt_fixed + db_cnt_var) % 2 == 0;
    };

    unsigned int selected = 0U;

    if (!is_aromatic()) {
      absl::c_sort(variable, [&](int i, int j) {
        return (*args[i])[1].prio > (*args[j])[1].prio;
      });

      internal::PowersetStream ps(static_cast<int>(variable.size()));
      bool aromatic = false;

      while (!aromatic && ps >> selected) {
        pie_sum_var = db_cnt_var = 0;

        for (int i = 0; i < variable.size(); ++i) {
          const auto &pi_e = *args[variable[i]];
          int idx = static_cast<int>(static_cast<bool>(selected & 1U << i));
          pie_sum_var += pi_e[idx].pi_e;
          db_cnt_var += pi_e[idx].endo_double;
        }

        aromatic = is_aromatic();
      }

      if (!aromatic)
        return;
    }

    for (int i = 0; i < variable.size(); ++i) {
      auto &arg = *args[variable[i]];
      int sel = static_cast<int>(static_cast<bool>(selected & 1U << i));
      arg[0] = arg[sel];
      arg.resize(1);
    }

    for (int i = 0; i < ring.size(); ++i) {
      auto atom = mol.atom(ring[i]);
      auto nei = *mol.find_neighbor(ring[i], ring[(i + 1) % ring.size()]);

      const auto &pi_e = (*args[i])[0];
      atom.data()  //
          .set_formal_charge(pi_e.fchg)
          .set_implicit_hydrogens(pi_e.degree - atom.degree())
          .add_flags(AtomFlags::kConjugated | AtomFlags::kAromatic);
      nei.edge_data()
          .set_order(constants::kAromaticBond)
          .add_flags(BondFlags::kConjugated | BondFlags::kAromatic);

      if (pi_e.exo_single_partner >= 0) {
        auto exo = atom[pi_e.exo_single_partner];
        exo.edge_data().set_order(constants::kSingleBond);
      } else if (pi_e.exo_double_partner >= 0) {
        auto exo = atom[pi_e.exo_double_partner];

        constants::BondOrder order;
        int fchg;
        if (atom.data().atomic_number() == 6) {
          // C == O
          order = constants::kDoubleBond;
          fchg = exo.dst().degree() == 1 ? 0 : 1;
        } else /* if (atom.data().atomic_number() == 7) */ {
          // N+ -- O-
          order = constants::kSingleBond;
          fchg = exo.dst().degree() == 1 ? -1 : 0;
        }

        exo.edge_data().set_order(order).add_flags(BondFlags::kConjugated);
        exo.dst().data().set_formal_charge(fchg).set_hybridization(
            constants::kSP2);
      }
    }
  }

  void find_aromatics(Molecule &mol,
                      const std::vector<std::vector<int>> &rings) {
    absl::flat_hash_map<int, absl::InlinedVector<PiElecArgs, 2>> pi_e_estimates;

    for (auto &ring: mol.ring_groups()) {
      for (int i: ring) {
        auto atom = mol.atom(i);
        if (is_aromatic_candidate(atom))
          count_pi_e_possible(atom, pi_e_estimates[i]);
      }
    }

    for (auto &ring: rings) {
      if (ring.size() < 5 || ring.size() > 7)
        continue;

      if (absl::c_any_of(ring,
                         [&](int i) { return !pi_e_estimates.contains(i); }))
        continue;

      test_ring_aromatic(mol, ring, pi_e_estimates);
    }
  }

  bool find_oxo(Molecule::MutableAtom atom) {
    if (atom.data().hybridization() != constants::kSP2)
      return false;

    auto nit = absl::c_find_if(atom, [](Molecule::Neighbor nei) {
      return nei.dst().data().atomic_number() == 8
             && all_neighbors(nei.dst()) == 1
             && nei.edge_data().order() == constants::kOtherBond;
    });
    if (nit == atom.end())
      return false;

    nit->edge_data().set_order(constants::kDoubleBond);
    return true;
  }

  double double_bond_distsq_approx(int an1, int an2) {
    if (an1 != 6) {
      if (an1 == 7 && an2 == 7)
        return 1.32 * 1.32;
      return -1;
    }

    switch (an2) {
    case 6:  // C=C
      return 1.38 * 1.38;
    case 8:  // C=O
      return 1.28 * 1.28;
    case 16:  // C=S
      return 1.70 * 1.70;
    default:
      return -1;
    }
  }

  double triple_bond_distsq_approx(int an1, int an2) {
    if (an1 != 6)
      return -1;

    switch (an2) {
    case 6:  // C#C
      return 1.25 * 1.25;
    case 7:  // C#N
      return 1.22 * 1.22;
    default:
      return -1;
    }
  }

  std::pair<double, double> multiple_bond_distsq_cutoffs(int an1, int an2) {
    return { double_bond_distsq_approx(an1, an2),
             triple_bond_distsq_approx(an1, an2) };
  }

  template <auto cutoff>
  auto ordered(const AtomData &a, const AtomData &b) {
    auto [an1, an2] = nuri::minmax(a.atomic_number(), b.atomic_number());
    return cutoff(an1, an2);
  }

  void mark_multiple_bonds_all_terminal(Molecule::MutableAtom atom,
                                        const Matrix3Xd &pos) {
    auto nei = atom[0];
    if (nei.edge_data().order() != constants::kOtherBond
        || all_neighbors(nei.dst()) != 1)
      return;

    double distsq =
        (pos.col(nei.dst().id()) - pos.col(atom.id())).squaredNorm();
    auto [double_sq, triple_sq] =
        ordered<multiple_bond_distsq_cutoffs>(atom.data(), nei.dst().data());

    constants::BondOrder order;
    if (distsq <= triple_sq) {
      order = constants::kTripleBond;
    } else if (distsq <= double_sq) {
      order = constants::kDoubleBond;
    } else {
      order = constants::kSingleBond;
    }
    nei.edge_data().set_order(order);
  }

  struct MultipleBondFindParams {
    int idx;
    double distsq;
    double cuttofsq;
  };

  template <auto thres>
  void mark_multiple_bonds_hyb(Molecule::MutableAtom atom, const Matrix3Xd &pos,
                               constants::BondOrder ifshort) {
    absl::InlinedVector<MultipleBondFindParams, 3> candidates;
    for (int i = 0; i < atom.degree(); ++i) {
      auto nei = atom[i];
      if (nei.edge_data().order() != constants::kOtherBond)
        continue;
      if (all_neighbors(nei.dst()) > 1
          && nei.dst().data().hybridization() != atom.data().hybridization())
        continue;
      if (sum_bond_order(nei.dst()) >= internal::common_valence(
              internal::effective_element_or_element(nei.dst())))
        continue;

      double distsq =
                 (pos.col(nei.dst().id()) - pos.col(atom.id())).squaredNorm(),
             cuttofsq = ordered<thres>(atom.data(), nei.dst().data());
      if (distsq <= cuttofsq)
        candidates.push_back({ i, distsq, cuttofsq });
      else
        nei.edge_data().set_order(constants::kSingleBond);
    }

    if (candidates.empty())
      return;

    if (candidates.size() > 1) {
      for (auto &arg: candidates) {
        arg.distsq = std::sqrt(arg.distsq);
        arg.cuttofsq = std::sqrt(arg.cuttofsq);
      }
      absl::c_sort(candidates, [](const auto &lhs, const auto &rhs) {
        return lhs.distsq - lhs.cuttofsq < rhs.distsq - rhs.cuttofsq;
      });

      for (int i = 1; i < candidates.size(); ++i)
        atom[candidates[i].idx].edge_data().set_order(constants::kSingleBond);
    }

    atom[candidates[0].idx].edge_data().set_order(ifshort);
  }

  void find_multiple_bonds(Molecule::MutableAtom atom, const Matrix3Xd &pos) {
    if (atom.degree() == 0)
      return;

    int total_degree = all_neighbors(atom);
    if (total_degree == 1) {
      mark_multiple_bonds_all_terminal(atom, pos);
      return;
    }

    if (atom.data().hybridization() > constants::kSP2
        || absl::c_any_of(atom, [](auto nei) {
             return nei.edge_data().order() >= constants::kDoubleBond;
           }))
      return;

    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (atom.data().hybridization()) {
    case constants::kSP:
      mark_multiple_bonds_hyb<triple_bond_distsq_approx>(
          atom, pos, constants::kTripleBond);
      ABSL_FALLTHROUGH_INTENDED;
    case constants::kSP2:
      mark_multiple_bonds_hyb<double_bond_distsq_approx>(
          atom, pos, constants::kDoubleBond);
      break;
    default:
      break;
    }
  }

  void assign_bond_order_others(Molecule &mol) {
    for (auto bond: mol.bonds()) {
      if (bond.data().order() != constants::kOtherBond)
        continue;
      bond.data().set_order(constants::kSingleBond);
    }
  }

  void assign_bond_orders(Molecule &mol, const Matrix3Xd &pos) {
    for (auto atom: mol) {
      int valence = sum_bond_order(atom),
          common_valence = internal::common_valence(
              internal::effective_element_or_element(atom));
      if (valence >= common_valence)
        continue;

      if (find_oxo(atom))
        continue;

      find_multiple_bonds(atom, pos);
    }

    assign_bond_order_others(mol);
  }

  enum class Conflict {
    kValenceOverflow,
    kHybConflict,
  };

  struct ConflictInfo {
    Conflict why;
    int overflowed;
    const Element *effective;
  };

  class Conflicts {
  public:
    using iterator = absl::flat_hash_map<int, ConflictInfo>::iterator;

    void add(int idx, Conflict why, const Element &effective, int overflowed) {
      data_.insert_or_assign(idx, { why, overflowed, &effective });
    }

    void mark_resolved(iterator it) { data_.erase(it); }

    iterator find(int idx) { return data_.find(idx); }

    bool empty() const { return data_.empty(); }

    auto begin() { return data_.begin(); }

    auto end() { return data_.end(); }

  private:
    absl::flat_hash_map<int, ConflictInfo> data_;
  };

  template <class Overflow>
  void guess_hfh_atom(Molecule::MutableAtom atom, const Element &effective,
                      Conflicts &conflicts, Overflow overflowed) {
    AtomData &data = atom.data();

    int sum_bo = sum_bond_order(atom);
    int cv = internal::common_valence(effective);
    int unused_valence = cv - sum_bo;
    int pred_h = data.implicit_hydrogens() + unused_valence;

    data.set_implicit_hydrogens(nuri::max(pred_h, 0));

    sum_bo = sum_bond_order(atom);
    int nbe = effective.valence_electrons() - sum_bo;
    int n_over = overflowed(pred_h, nbe);
    if (n_over > 0) {
      conflicts.add(atom.id(), Conflict::kValenceOverflow, effective, n_over);
      return;
    }

    int total_degree = all_neighbors(atom);
    if (total_degree <= 1) {
      data.set_hybridization(
          static_cast<constants::Hybridization>(total_degree));
      return;
    }

    int sn = internal::steric_number(total_degree, nbe);
    if (sn == constants::kSP3 && data.is_conjugated() && nbe > 0)
      --sn;

    if (data.hybridization() == constants::kOtherHyb) {
      data.set_hybridization(static_cast<constants::Hybridization>(sn));
    } else if (sn > data.hybridization()) {
      int extra_bonds_required = sn - data.hybridization();
      int extra_bonds_unresolvable =
          extra_bonds_required - data.implicit_hydrogens();

      if (extra_bonds_unresolvable > 0) {
        data.set_hybridization(
            clamp_hyb(data.hybridization() + extra_bonds_unresolvable));
        extra_bonds_required -= extra_bonds_unresolvable;
        if (extra_bonds_required <= 0)
          return;
      }

      conflicts.add(atom.id(), Conflict::kHybConflict, effective,
                    extra_bonds_required);
    } else {
      ABSL_DCHECK(sn == data.hybridization());
    }
  }

  // Unrecoverable failure:
  //   1. Degree overflow, 1st row -> 1, 2nd row -> 4, 3rd or higher -> 6
  //   2. Impossible combination of atomic number and formal charge
  //   3. Bond to metal atom
  //
  // Recoverable errors:
  //   1. Valence overflow, only if degree not overflows. In other words, only
  //      if multiple bond or nonbonding electrons exists on the overflowing
  //      atom.
  //   2. Hybridization mismatch. This might also include unmarked conjugated
  //      atoms.
  bool guess_hyb_fcharge_hydrogens(Molecule &mol, Conflicts &conflicts) {
    auto log_degree_overflow = [](int line, Molecule::Atom atom,
                                  int max_degree) {
      ABSL_LOG(WARNING).AtLocation(__FILE__, line)
          << "Degree overflow for atom " << atom.id() << " ("
          << atom.data().element().symbol() << ").";
      ABSL_LOG(INFO).AtLocation(__FILE__, line)
          << "Max allowed neighbors: " << max_degree
          << ", actual: " << atom.degree();
    };

    for (auto atom: mol) {
      AtomData &data = atom.data();
      if (data.atomic_number() == 0)
        continue;

      if (data.atomic_number() <= 2) {
        int hyb = 2 - data.atomic_number();
        if (atom.degree() > hyb) {
          log_degree_overflow(__LINE__, atom, hyb);
          return false;
        }

        ABSL_DCHECK(data.implicit_hydrogens() == 0);

        data.set_hybridization(static_cast<constants::Hybridization>(
            nuri::max(hyb, atom.degree())));
        continue;
      }

      if (data.element().type() == Element::Type::kMetal) {
        if (atom.degree() != 0) {
          ABSL_LOG(WARNING) << "Bond to metal atom is not allowed.";
          return false;
        }

        data.set_hybridization(constants::kUnbound)
            .set_formal_charge(data.element().valence_electrons())
            .set_implicit_hydrogens(0);
        continue;
      }

      const int max_degree = data.element().period() > 2 ? 6 : 4;
      if (atom.degree() > max_degree) {
        log_degree_overflow(__LINE__, atom, max_degree);
        return false;
      }

      const Element *effective = effective_element(atom);
      if (effective == nullptr) {
        ABSL_LOG(WARNING)
            << "Impossible combination of atomic number and formal charge for "
               "atom "
            << atom.id() << " (" << data.element().symbol() << ").";
        ABSL_LOG(INFO) << "Atomic number: " << data.atomic_number()
                       << ", formal charge: " << data.formal_charge();
        return false;
      }

      if (data.element().period() <= 2) {
        guess_hfh_atom(atom, *effective, conflicts,
                       [](int pred_h, int /* nbe */) { return -pred_h; });
      } else {
        guess_hfh_atom(atom, *effective, conflicts,
                       [](int /* pred_h */, int nbe) { return -nbe; });
      }
    }

    return true;
  }

  bool try_fix_overflow(
      Molecule::MutableAtom atom, Conflicts::iterator &it, Conflicts &conflicts,
      std::vector<std::pair<int, Conflicts::iterator>> &nei_conflict) {
    const ConflictInfo &info = it->second;
    int overflowed = info.overflowed;

    // pre: implicit H count == 0
    // Might be resolved by removing multiple bonds if neighbor is also
    // overflowing, or by updating formal charge (max +-2)
    absl::c_sort(
        nei_conflict, [&](const std::pair<int, Conflicts::iterator> &lhs,
                          const std::pair<int, Conflicts::iterator> &rhs) {
          auto ln = atom[lhs.first], rn = atom[rhs.first];
          return ln.dst().degree() > rn.dst().degree()
                 || (ln.dst().degree() == rn.dst().degree()
                     && ln.edge_data().order() > rn.edge_data().order());
        });

    for (auto [i, jt]: nei_conflict) {
      auto nei = atom[i];
      int nei_req = jt->second.overflowed;
      int available = std::min(overflowed, nei_req);
      auto ord = static_cast<constants::BondOrder>(nei.edge_data().order()
                                                   - available);
      if (ord < constants::kSingleBond)
        continue;

      nei.edge_data().order() = ord;

      jt->second.overflowed -= available;
      if (nei_req == 0)
        conflicts.mark_resolved(jt);

      overflowed -= available;
      if (overflowed == 0) {
        conflicts.mark_resolved(it++);
        return true;
      }
    }

    int fchg = atom.data().formal_charge();
    if (info.effective->period() == 2) {
      // Cannot handle atoms with already the highest available max bond order
      // (group 14 -> 4, other groups can have < 4)
      if (overflowed >= 2 || info.effective->group() == 14)
        return false;

      fchg += info.effective->group() > 14 ? overflowed : -overflowed;
    } else {
      // PCl6-
      fchg -= overflowed;
    }

    if (fchg < -2 || fchg > 2)
      return false;

    const Element *new_effective =
        kPt.find_element(atom.data().atomic_number() - fchg);
    if (new_effective == nullptr
        || new_effective->period() != atom.data().element().period())
      return false;

    atom.data().set_formal_charge(fchg);
    conflicts.mark_resolved(it++);
    return true;
  }

  bool hyb_conflict_can_sp2(const Matrix3Xd &pos, Molecule::Neighbor nei) {
    auto src = nei.src(), dst = nei.dst();
    auto prev = absl::c_find_if(src, [&](Molecule::Neighbor n) {
                  return n.dst().id() != dst.id();
                })->dst();
    auto next = absl::c_find_if(dst, [&](Molecule::Neighbor n) {
                  return n.dst().id() != src.id();
                })->dst();

    return torsion_can_sp2(pos, prev.id(), src.id(), dst.id(), next.id());
  }

  bool try_fix_hyb_conflict(
      const Matrix3Xd &pos, Molecule::MutableAtom atom, Conflicts::iterator &it,
      Conflicts &conflicts,
      std::vector<std::pair<int, Conflicts::iterator>> &nei_conflict) {
    // steric number != hybridization
    // 1. Missing multiple bond, if neighbor also has conflicts (conjugation
    //    already marked)
    // 2. Wrong hybridization, if no neighbor has conflicts

    absl::c_sort(
        nei_conflict, [&](const std::pair<int, Conflicts::iterator> &lhs,
                          const std::pair<int, Conflicts::iterator> &rhs) {
          auto ln = atom[lhs.first], rn = atom[rhs.first];
          return ln.dst().degree() < rn.dst().degree()
                 || (ln.dst().degree() == rn.dst().degree()
                     && ln.edge_data().order() < rn.edge_data().order());
        });

    const ConflictInfo &info = it->second;
    int required = info.overflowed;
    for (auto [i, jt]: nei_conflict) {
      auto nei = atom[i];

      if (atom.degree() > 1 && nei.dst().degree() > 1
          && atom.data().hybridization() == constants::kSP2
          && nei.dst().data().hybridization() == constants::kSP2) {
        if (!hyb_conflict_can_sp2(pos, nei))
          continue;
      }

      int nei_req = jt->second.overflowed;
      int available = std::min(
          { required, nei_req,
            nonnegative(constants::kTripleBond - nei.edge_data().order()) });
      auto ord = static_cast<constants::BondOrder>(nei.edge_data().order()
                                                   + available);

      ABSL_DCHECK(nei.src().data().implicit_hydrogens() >= available);
      ABSL_DCHECK(nei.dst().data().implicit_hydrogens() >= available);

      nei.edge_data().order() = ord;
      nei.src().data().set_implicit_hydrogens(
          nei.src().data().implicit_hydrogens() - available);
      nei.dst().data().set_implicit_hydrogens(
          nei.dst().data().implicit_hydrogens() - available);

      jt->second.overflowed -= available;
      if (nei_req == 0)
        conflicts.mark_resolved(jt);

      required -= available;
      if (required <= 0)
        break;
    }

    if (required > 0) {
      int pred_hyb = atom.data().hybridization() + required;
      if (pred_hyb > constants::kSP3D2)
        return false;

      atom.data().set_hybridization(
          static_cast<constants::Hybridization>(pred_hyb));
    }

    conflicts.mark_resolved(it++);
    return true;
  }

  bool try_fix_conflicts(Molecule &mol, const Matrix3Xd &pos,
                         Conflicts &conflicts) {
    std::vector<std::pair<int, Conflicts::iterator>> nei_conflict;

    for (auto it = conflicts.begin(); it != conflicts.end();) {
      auto atom = mol.atom(it->first);

      nei_conflict.clear();
      for (int i = 0; i < atom.degree(); ++i) {
        auto jt = conflicts.find(atom[i].dst().id());
        if (jt != conflicts.end() && jt->second.why == it->second.why)
          nei_conflict.push_back({ i, jt });
      }

      bool resolved = false;
      switch (it->second.why) {
      case Conflict::kValenceOverflow:
        resolved = try_fix_overflow(atom, it, conflicts, nei_conflict);
        break;
      case Conflict::kHybConflict:
        resolved = try_fix_hyb_conflict(pos, atom, it, conflicts, nei_conflict);
        break;
      }

      if (!resolved)
        ++it;
    }

    return conflicts.empty();
  }

  bool hyb_incorrect_atom_can_conjugate(Molecule::Atom atom) {
    bool any_multiple = absl::c_any_of(atom, [](Molecule::Neighbor nei) {
      return nei.edge_data().order() > constants::kSingleBond;
    });
    int nbe = nonbonding_electrons(atom), sum_bo = sum_bond_order(atom);
    return any_multiple || nbe > 0 || sum_bo < 4;
  }

  bool is_conjugated_candidate(Molecule::Atom atom) {
    return atom.data().is_conjugated()
           || (atom.degree() == 3
               && atom.data().hybridization() <= constants::kSP2)
           || (atom.degree() <= 2 && hyb_incorrect_atom_can_conjugate(atom));
  }

  void find_conjugated_groups_dfs(const Molecule &mol, const Matrix3Xd &pos,
                                  absl::flat_hash_set<int> &candidates,
                                  std::vector<std::vector<int>> &groups,
                                  int cur,
                                  Molecule::const_neighbor_iterator prev_nei) {
    groups.back().push_back(cur);
    candidates.erase(cur);

    auto src = mol.atom(cur);
    for (auto nit = src.begin(); nit != src.end(); ++nit) {
      auto dst = nit->dst();
      if (!candidates.contains(dst.id()))
        continue;

      if (prev_nei.end()) {
        auto rev = dst.find_adjacent(cur);
        bool has_rev = !rev.end();
        ABSL_ASSUME(has_rev);

        prev_nei = rev;
        find_conjugated_groups_dfs(mol, pos, candidates, groups, dst.id(), rev);
        continue;
      }

      if (prev_nei->edge_data().order() == constants::kSingleBond) {
        if (nit->edge_data().order() == constants::kSingleBond
            && src.data().atomic_number() != 0
            && dst.data().atomic_number() != 0) {
          // Single - single bond -> conjugated if curr has lone pair and
          // next doesn't, or vice versa, or any of them is dummy
          const int src_nbe = nonbonding_electrons(src),
                    dst_nbe = nonbonding_electrons(dst);
          if ((src_nbe > 0 && dst_nbe > 0) || (src_nbe <= 0 && dst_nbe <= 0))
            continue;
        }
      } else if (nit->edge_data().order() != constants::kSingleBond
                 && prev_nei->edge_data().order() != constants::kAromaticBond
                 && nit->edge_data().order() != constants::kAromaticBond) {
        // Aromatic - aromatic bond -> conjugated
        // Aromatic - double/triple bond
        //          -> exocyclic C=O like structure (technically conjugated)
        // double/triple - double/triple bond -> allene-like
        continue;
      }

      Molecule::const_neighbor_iterator next_nei =
          std::find_if(dst.begin(), dst.end(), [&](Molecule::Neighbor n) {
            return n.dst().id() != cur;
          });
      if (!next_nei.end()
          && !torsion_can_sp2(pos, prev_nei->src().id(), prev_nei->dst().id(),
                              next_nei->src().id(), next_nei->dst().id())) {
        continue;
      }

      find_conjugated_groups_dfs(mol, pos, candidates, groups, dst.id(), nit);
    }
  }

  void find_conjugated_groups(const Molecule &mol, const Matrix3Xd &pos,
                              absl::flat_hash_set<int> &candidates,
                              std::vector<std::vector<int>> &groups) {
    groups.emplace_back();
    for (auto atom: mol) {
      if (!candidates.contains(atom.id()))
        continue;

      find_conjugated_groups_dfs(mol, pos, candidates, groups, atom.id(),
                                 atom.end());

      if (groups.back().size() < 3) {
        candidates.insert(groups.back().begin(), groups.back().end());
        groups.back().clear();
      } else {
        groups.emplace_back();
      }
    }

    if (groups.back().empty())
      groups.pop_back();
  }

  void mark_conjugated(Molecule &mol,
                       const std::vector<std::vector<int>> &groups,
                       Conflicts &conflicts) {
    for (const std::vector<int> &group: groups) {
      ABSL_DCHECK(group.size() > 2) << "Group size: " << group.size();

      Substructure sub = mol.atom_substructure(group);

      for (auto atom: sub) {
        AtomData &data = atom.data();
        data.add_flags(AtomFlags::kConjugated)
            .set_hybridization(
                nuri::min(constants::kSP2, data.hybridization()));

        auto it = conflicts.find(atom.as_parent().id());
        if (it != conflicts.end() && it->second.why == Conflict::kHybConflict) {
          --it->second.overflowed;
          if (it->second.overflowed <= 0)
            conflicts.mark_resolved(it);
        }
      }

      for (auto bond: sub.bonds())
        bond.data().add_flags(BondFlags::kConjugated);
    }
  }

  void guess_conjugated(Molecule &mol, const Matrix3Xd &pos,
                        Conflicts &conflicts) {
    absl::flat_hash_set<int> candidates;
    std::vector<std::vector<int>> groups;

    for (auto atom: mol)
      if (is_conjugated_candidate(atom))
        candidates.insert(atom.id());

    find_conjugated_groups(mol, pos, candidates, groups);
    mark_conjugated(mol, groups, conflicts);
  }

  bool guess_types_common(Molecule &mol, const Matrix3Xd &pos) {
    Rings rings;
    bool ok;

    std::tie(rings, ok) = find_all_rings(mol, 6);
    if (!ok)
      rings = find_sssr(mol, 6);

    guess_hyb_nonterminal(mol, pos, rings);
    recognize_fg(mol, pos);
    assign_priority_bonds(mol);
    find_aromatics(mol, rings);
    assign_bond_orders(mol, pos);

    Conflicts conflicts;
    guess_hyb_fcharge_hydrogens(mol, conflicts);
    guess_conjugated(mol, pos, conflicts);

    if (!conflicts.empty()) {
      bool success = try_fix_conflicts(mol, pos, conflicts);
      if (!success) {
        ABSL_LOG(WARNING) << "Failed to fix conflicts.";
        return false;
      }
    }

    return true;
  }
}  // namespace

void guess_connectivity(MoleculeMutator &mut, int conf, double threshold) {
  const Molecule &mol = mut.mol();
  ABSL_DCHECK(conf >= 0 && conf < mol.confs().size());

  const Matrix3Xd &pos = mol.confs()[conf];

  // For small molecules
  ArrayXd distsq;
  // For large molecules, e.g. proteins
  OCTree tree;
  prepare_conn_search(pos, distsq, tree);
  guess_connectivity(mut, pos, threshold, distsq, tree);
  mut.finalize();
}

bool guess_everything(MoleculeMutator &mut, int conf, double threshold) {
  Molecule &mol = mut.mol();
  guess_connectivity(mut, conf, threshold);

  reset_atoms(mol);
  reset_bonds(mol);
  return guess_types_common(mol, mol.confs()[conf]);
}

bool guess_all_types(Molecule &mol, int conf) {
  ABSL_DCHECK(conf >= 0 && conf < mol.confs().size());

  if (!no_excess_bonds(mol, [](Molecule::Atom atom, int /* max_neighbors */) {
        ABSL_LOG(WARNING) << "Degree overflow for atom " << atom.id() << " ("
                          << atom.data().element().symbol() << ").";
        return false;
      })) {
    return false;
  }

  reset_atoms(mol);
  reset_bonds(mol);
  return guess_types_common(mol, mol.confs()[conf]);
}
}  // namespace nuri
