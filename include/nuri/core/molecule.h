//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_CORE_MOLECULE_H_
#define NURI_CORE_MOLECULE_H_

//! @cond
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/base/nullability.h>
#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/log/absl_check.h>
#include <boost/container/flat_map.hpp>
#include <Eigen/Dense>
//! @endcond

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/graph.h"
#include "nuri/core/property_map.h"
#include "nuri/meta.h"
#include "nuri/utils.h"

namespace nuri {
namespace constants {
  /**
   * @brief The hybridization state of an atom object.
   */
  enum Hybridization : int {
    kUnbound = 0,   // Unbound
    kTerminal = 1,  // Terminal
    kSP = 2,
    kSP2 = 3,
    kSP3 = 4,
    kSP3D = 5,
    kSP3D2 = 6,
    kOtherHyb = 7,  // Unknown/other
  };

  inline std::ostream &operator<<(std::ostream &os, Hybridization hyb) {
    switch (hyb) {
    case kUnbound:
      return os << "unbound";
    case kTerminal:
      return os << "terminal";
    case kSP:
      return os << "sp";
    case kSP2:
      return os << "sp2";
    case kSP3:
      return os << "sp3";
    case kSP3D:
      return os << "sp3d";
    case kSP3D2:
      return os << "sp3d2";
    case kOtherHyb:
      break;
    }

    return os << "other";
  }

  /**
   * @brief The bond order of a bond object.
   */
  enum BondOrder : int {
    kOtherBond = 0,
    kSingleBond = 1,
    kDoubleBond = 2,
    kTripleBond = 3,
    kQuadrupleBond = 4,
    kAromaticBond = 5,
  };

  inline std::ostream &operator<<(std::ostream &os, BondOrder bo) {
    switch (bo) {
    case kOtherBond:
      break;
    case kSingleBond:
      return os << "single";
    case kDoubleBond:
      return os << "double";
    case kTripleBond:
      return os << "triple";
    case kQuadrupleBond:
      return os << "quadruple";
    case kAromaticBond:
      return os << "aromatic";
    }

    return os << "other";
  }

  constexpr double kBondOrderToDouble[] = { 0.0, 1.0, 2.0, 3.0, 4.0, 1.5 };
}  // namespace constants

inline constants::Hybridization clamp_hyb(int hyb) {
  return nuri::clamp(static_cast<constants::Hybridization>(hyb),
                     constants::kTerminal, constants::kSP3D2);
}

inline constants::BondOrder clamp_ord(int ord) {
  return nuri::clamp(static_cast<constants::BondOrder>(ord),
                     constants::kSingleBond, constants::kTripleBond);
}

enum class AtomFlags : std::uint32_t {
  kAromatic = 0x1,
  kConjugated = 0x2,
  kRing = 0x4,
  kChiral = 0x8,
  kClockWise = 0x10,
};

class AtomData {
public:
  /**
   * @brief Creates a dummy atom with unknown hybridization.
   */
  AtomData(): AtomData(kPt[0]) { }

  AtomData(const Element &element, int implicit_hydrogens = 0,
           int formal_charge = 0,
           constants::Hybridization hyb = constants::kOtherHyb,
           double partial_charge = 0.0, int mass_number = -1,
           bool is_aromatic = false, bool is_in_ring = false,
           bool is_chiral = false, bool is_clockwise = false);

  /**
   * @brief Get the atomic number of the atom.
   * @note This is equivalent to `element().Element::atomic_number()`,
   *       provided for convenience.
   * @return The atomic number of the element.
   */
  int atomic_number() const { return element().atomic_number(); }

  /**
   * @brief Get the atomic weight of the atom.
   * @note This is equivalent to `element().Element::atomic_weight()`,
   *       provided for convenience.
   * @return The atomic weight of the element.
   */
  double atomic_weight() const { return element().atomic_weight(); }

  /**
   * @brief Get the element symbol of the atom.
   * @note This is equivalent to `element().Element::symbol()`, provided for
   *       convenience.
   * @return The symbol of the element.
   */
  std::string_view element_symbol() const { return element().symbol(); }

  /**
   * @brief Get the element name of the atom.
   * @note This is equivalent to `element().Element::name()`, provided for
   *       convenience.
   * @return The name of the element.
   */
  std::string_view element_name() const { return element().name(); }

  /**
   * @brief Change the element of the atom.
   *
   * @param element A reference to the new element object.
   * @return *this.
   */
  AtomData &set_element(const Element &element) {
    element_ = &element;
    return *this;
  }

  /**
   * @brief Set the element of the atom by atomic number.
   *
   * @param atomic_number The atomic number of the new element.
   * @return *this.
   *
   * @note When the atomic number is out of range, the behavior is undefined.
   */
  AtomData &set_element(int atomic_number) {
    set_element(kPt[atomic_number]);
    return *this;
  }

  /**
   * @brief Get the element data of the atom.
   * @return Immutable reference to the element object.
   */
  const Element &element() const noexcept {
    // GCOV_EXCL_START
    ABSL_ASSUME(element_ != nullptr);
    // GCOV_EXCL_STOP
    return *element_;
  }

  /**
   * @brief Get the isotope of the atom.
   * @return A const reference to the isotope object. If any isotope was
   *         explicitly given, returns that isotope. Otherwise, returns the
   *         representative isotope of the element.
   * @sa explicit_isotope()
   */
  const Isotope &isotope() const {
    return ABSL_PREDICT_TRUE(isotope_ == nullptr) ? element().major_isotope()
                                                  : *isotope_;
  }

  /**
   * @brief Set explicit isotope of the atom.
   *
   * @param isotope A reference to the new isotope object.
   * @return *this.
   *
   * @note If the isotope object references an element different from the
   *       element of the atom, the behavior is undefined.
   */
  AtomData &set_isotope(const Isotope &isotope) {
    isotope_ = &isotope;
    return *this;
  }

  /**
   * @brief Set explicit isotope of the atom by mass number.
   *
   * @param mass_number The mass number of the new isotope.
   * @return *this.
   *
   * @note If no isotope with the given mass number is known, explicitly-set
   *       isotope of this atom will be cleared.
   */
  AtomData &set_isotope(int mass_number) {
    isotope_ = element().find_isotope(mass_number);
    return *this;
  }

  /**
   * @brief Get the explicitly set isotope of the atom.
   * @return A pointer to the explicitly set isotope object. If none was
   *         explicitly given, returns `nullptr`. Normally isotope() would
   *         be the preferred method to get the isotope, which returns the
   *         representative isotope of the element if none was explicitly given.
   * @sa isotope()
   */
  absl::Nullable<const Isotope *> explicit_isotope() const { return isotope_; }

  AtomData &set_hybridization(constants::Hybridization hyb) {
    hyb_ = hyb;
    return *this;
  }

  constants::Hybridization hybridization() const { return hyb_; }

  /**
   * @brief Set the number of implicit hydrogen atoms.
   *
   * @param implicit_hydrogens The new number of implicit hydrogen atoms.
   * @return *this.
   *
   * @note If the number of implicit hydrogen atoms is negative, the behavior is
   *       undefined.
   */
  AtomData &set_implicit_hydrogens(int implicit_hydrogens) {
    ABSL_DCHECK(implicit_hydrogens >= 0);
    implicit_hydrogens_ = implicit_hydrogens;
    return *this;
  }

  /**
   * @brief Get the number of implicit hydrogen atoms.
   * @return The number of implicit hydrogen atoms.
   *
   * @note The number of implicit hydrogen atoms is always non-negative.
   */
  int implicit_hydrogens() const { return implicit_hydrogens_; }

  AtomData &set_aromatic(bool is_aromatic) {
    internal::update_flag(flags_, is_aromatic, AtomFlags::kAromatic);
    return *this;
  }

  bool is_aromatic() const {
    return internal::check_flag(flags_, AtomFlags::kAromatic);
  }

  AtomData &set_conjugated(bool is_conjugated) {
    internal::update_flag(flags_, is_conjugated, AtomFlags::kConjugated);
    return *this;
  }

  bool is_conjugated() const {
    return internal::check_flag(flags_, AtomFlags::kConjugated);
  }

  AtomData &set_ring_atom(bool is_ring_atom) {
    internal::update_flag(flags_, is_ring_atom, AtomFlags::kRing);
    return *this;
  }

  bool is_ring_atom() const {
    return internal::check_flag(flags_, AtomFlags::kRing);
  }

  AtomData &set_chiral(bool is_chiral) {
    internal::update_flag(flags_, is_chiral, AtomFlags::kChiral);
    return *this;
  }

  bool is_chiral() const {
    return internal::check_flag(flags_, AtomFlags::kChiral);
  }

  AtomData &set_clockwise(bool is_clockwise) {
    internal::update_flag(flags_, is_clockwise, AtomFlags::kClockWise);
    return *this;
  }

  /**
   * @brief Get handedness of a chiral atom.
   *
   * @pre is_chiral() == `true`, otherwise return value would be meaningless.
   * @return Whether the chiral atom is "clockwise." See stereochemistry
   *         definition of SMILES for more information.
   * @note Unlike the SMILES convention, implicit hydrogen is placed at the
   *       *end* of the neighbor list. This is to ensure that the chirality of
   *       the atom is unchanged when the hydrogen is removed or added.
   */
  bool is_clockwise() const {
    return internal::check_flag(flags_, AtomFlags::kClockWise);
  }

  AtomFlags flags() const { return flags_; }

  AtomData &add_flags(AtomFlags flags) {
    flags_ |= flags;
    return *this;
  }

  AtomData &del_flags(AtomFlags flags) {
    flags_ &= ~flags;
    return *this;
  }

  AtomData &reset_flags() {
    flags_ = static_cast<AtomFlags>(0);
    return *this;
  }

  AtomData &set_partial_charge(double charge) {
    partial_charge_ = charge;
    return *this;
  }

  double partial_charge() const { return partial_charge_; }

  AtomData &set_formal_charge(int charge) {
    formal_charge_ = charge;
    return *this;
  }

  int formal_charge() const { return formal_charge_; }

  std::string_view get_name() const { return internal::get_name(props_); }

  template <class ST>
  AtomData &set_name(ST &&name) {
    internal::set_name(props_, std::forward<ST>(name));
    return *this;
  }

  template <class KT, class VT>
  AtomData &add_prop(KT &&key, VT &&val) {
    internal::set_key(props_, std::forward<KT>(key), std::forward<VT>(val));
    return *this;
  }

  internal::PropertyMap &props() { return props_; }

  const internal::PropertyMap &props() const { return props_; }

private:
  friend bool operator==(const AtomData &lhs, const AtomData &rhs) noexcept;

  const Element *element_;
  int implicit_hydrogens_;
  int formal_charge_;
  constants::Hybridization hyb_;
  AtomFlags flags_;
  double partial_charge_;
  const Isotope *isotope_;
  internal::PropertyMap props_;
};

inline bool operator==(const AtomData &lhs, const AtomData &rhs) noexcept {
  return lhs.element() == rhs.element()
         && lhs.hybridization() == rhs.hybridization()
         && lhs.flags_ == rhs.flags_
         && lhs.formal_charge() == rhs.formal_charge();
}

enum class BondFlags : std::uint32_t {
  kRing = 0x1,
  kAromatic = 0x2,
  kConjugated = 0x4,
  kConfigSpecified = 0x8,
  kTransConfig = 0x10,
};

class BondData {
public:
  BondData(): BondData(constants::kOtherBond) { }

  explicit BondData(constants::BondOrder order)
      : order_(order), flags_(static_cast<BondFlags>(0)) { }

  /**
   * @brief Get the bond order of the bond.
   */
  constants::BondOrder order() const { return order_; }

  /**
   * @brief Get the approximate bond order of the bond.
   * @return Approximate bond order, e.g., 1.5 for an aromatic bond.
   */
  double approx_order() const { return constants::kBondOrderToDouble[order_]; }

  /**
   * @brief Get the read-write reference to bond order.
   */
  constants::BondOrder &order() { return order_; }

  BondData &set_order(constants::BondOrder order) {
    order_ = order;
    return *this;
  }

  /**
   * @brief Test if this bond is rotatable.
   *
   * @return Whether the bond is rotatable. Only non-conjugated single bonds
   *         that are not part of a ring are considered rotatable.
   */
  bool is_rotatable() const {
    return order_ <= constants::kSingleBond
           && !internal::check_flag(flags_,
                                    BondFlags::kConjugated | BondFlags::kRing);
  }

  bool is_ring_bond() const {
    return internal::check_flag(flags_, BondFlags::kRing);
  }

  BondData &set_ring_bond(bool ring) {
    internal::update_flag(flags_, ring, BondFlags::kRing);
    return *this;
  }

  bool is_aromatic() const {
    return internal::check_flag(flags_, BondFlags::kAromatic);
  }

  BondData &set_aromatic(bool aromatic) {
    internal::update_flag(flags_, aromatic, BondFlags::kAromatic);
    return *this;
  }

  bool is_conjugated() const {
    return internal::check_flag(flags_, BondFlags::kConjugated);
  }

  BondData &set_conjugated(bool conj) {
    internal::update_flag(flags_, conj, BondFlags::kConjugated);
    return *this;
  }

  /**
   * @brief Test if the bond configuration is explicitly specified.
   */
  bool has_config() const {
    return internal::check_flag(flags_, BondFlags::kConfigSpecified);
  }

  /**
   * @brief Set whether the bond configuration is explicitly specified.
   */
  BondData &set_config(bool config) {
    internal::update_flag(flags_, config, BondFlags::kConfigSpecified);
    return *this;
  }

  /**
   * @brief Get the cis-trans configuration of the bond.
   * @return Whether the bond is in trans configuration.
   *
   * @pre has_config(), otherwise return value would be meaningless.
   * @note This flag is only meaningful for torsionally restricted bonds, such
   *       as double bonds.
   *
   * For bonds with more than 3 neighboring atoms, "trans" configuration is not
   * a well defined term. In such cases, this will return whether the first two
   * neighbors are on the same side of the bond. For example, in the following
   * structure, the bond between atoms 0 and 1 is considered to be in a cis
   * configuration (assuming the neighbors are ordered in the same way as the
   * atoms).
   *
   * \code{.unparsed}
   *  2       4
   *   \     /
   *    0 = 1
   *   /     \
   *  3       5
   * \endcode
   */
  bool is_trans() const {
    return internal::check_flag(flags_, BondFlags::kTransConfig);
  }

  /**
   * @brief Set cis-trans configuration of the bond.
   * @param trans Whether the bond is in trans configuration or not.
   * @pre has_config()
   */
  BondData &set_trans(bool trans) {
    internal::update_flag(flags_, trans, BondFlags::kTransConfig);
    return *this;
  }

  BondFlags flags() const { return flags_; }

  BondData &add_flags(BondFlags flags) {
    flags_ |= flags;
    return *this;
  }

  BondData &del_flags(BondFlags flags) {
    flags_ &= ~flags;
    return *this;
  }

  BondData &reset_flags() {
    flags_ = static_cast<BondFlags>(0);
    return *this;
  }

  std::string_view get_name() const { return internal::get_name(props_); }

  template <class ST>
  BondData &set_name(ST &&name) {
    internal::set_name(props_, std::forward<ST>(name));
    return *this;
  }

  template <class KT, class VT>
  BondData &add_prop(KT &&key, VT &&val) {
    internal::set_key(props_, std::forward<KT>(key), std::forward<VT>(val));
    return *this;
  }

  internal::PropertyMap &props() { return props_; }

  const internal::PropertyMap &props() const { return props_; }

private:
  constants::BondOrder order_;
  BondFlags flags_;
  internal::PropertyMap props_;
};

class Molecule;
class MoleculeMutator;

enum class SubstructCategory {
  kUnknown,
  kResidue,
  kChain,
};

namespace internal {
  template <bool is_const = false>
  class Substructure {
  public:
    using GraphType = const_if_t<is_const, Graph<AtomData, BondData>>;

    // Should use SubgraphOf<GraphType> here, but Subgraph was used directly for
    // better clangd autocompletion.
    using SubgraphType = Subgraph<AtomData, BondData, is_const>;

    using MutableAtom = typename SubgraphType::NodeRef;
    using Atom = typename SubgraphType::ConstNodeRef;

    using iterator = typename SubgraphType::iterator;
    using const_iterator = typename SubgraphType::const_iterator;
    using atom_iterator = iterator;
    using const_atom_iterator = const_iterator;

    using bond_iterator = typename SubgraphType::edge_iterator;
    using const_bond_iterator = typename SubgraphType::const_edge_iterator;
    using MutableBond = typename SubgraphType::EdgeRef;
    using Bond = typename SubgraphType::ConstEdgeRef;

    using BondsWrapper = typename SubgraphType::EdgesWrapper;
    using ConstBondsWrapper = typename SubgraphType::ConstEdgesWrapper;

    using MutableNeighbor = typename SubgraphType::AdjRef;
    using Neighbor = typename SubgraphType::ConstAdjRef;
    using neighbor_iterator = typename SubgraphType::adjacency_iterator;
    using const_neighbor_iterator =
        typename SubgraphType::const_adjacency_iterator;

    Substructure(const SubgraphType &sub,
                 SubstructCategory cat = SubstructCategory::kUnknown)
        : graph_(sub), cat_(cat) { }

    Substructure(SubgraphType &&sub,
                 SubstructCategory cat = SubstructCategory::kUnknown) noexcept
        : graph_(std::move(sub)), cat_(cat) { }

    Substructure(const SubgraphType &sub, const std::string &name,
                 SubstructCategory cat = SubstructCategory::kUnknown)
        : graph_(sub), name_(name), cat_(cat) { }

    Substructure(SubgraphType &&sub, std::string &&name,
                 SubstructCategory cat = SubstructCategory::kUnknown) noexcept
        : graph_(std::move(sub)), name_(std::move(name)), cat_(cat) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    Substructure(const Substructure<other_const> &other)
        : graph_(other.graph_), name_(other.name_), id_(other.id_),
          cat_(other.cat_), props_(other.props_) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    Substructure(Substructure<other_const> &&other) noexcept
        : graph_(std::move(other.graph_)), name_(std::move(other.name_)),
          id_(other.id_), cat_(other.cat_), props_(std::move(other.props_)) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    Substructure &operator=(const Substructure<other_const> &other) {
      graph_ = other.graph_;
      name_ = other.name_;
      id_ = other.id_;
      cat_ = other.cat_;
      props_ = other.props_;
      return *this;
    }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    Substructure &operator=(Substructure<other_const> &&other) noexcept {
      graph_ = std::move(other.graph_);
      name_ = std::move(other.name_);
      id_ = other.id_;
      cat_ = other.cat_;
      props_ = std::move(other.props_);
      return *this;
    }

    bool empty() const { return graph_.empty(); }
    int size() const { return graph_.size(); }
    int num_atoms() const { return graph_.num_nodes(); }
    int num_bonds() const { return graph_.num_edges(); }
    int count_heavy_atoms() const {
      return absl::c_count_if(graph_, [](Substructure::Atom atom) {
        return atom.data().atomic_number() != 1;
      });
    }

    void clear() noexcept {
      graph_.clear();
      name_.clear();
      id_ = 0;
      cat_ = SubstructCategory::kUnknown;
      props_.clear();
    }

    void clear_atoms() noexcept { graph_.clear(); }

    void update(IndexSet &&atoms, IndexSet &&bonds) {
      graph_.update(std::move(atoms), std::move(bonds));
    }

    void update_atoms(IndexSet &&atoms) noexcept {
      graph_.update_nodes(std::move(atoms));
    }

    void reserve_atoms(int n) { graph_.reserve_nodes(n); }

    void add_atom(int id) { graph_.add_node(id); }

    void add_atoms(const IndexSet &atoms, bool bonds = false) {
      if (bonds) {
        graph_.add_nodes_with_edges(atoms);
      } else {
        graph_.add_nodes(atoms);
      }
    }

    bool contains_atom(int id) const { return graph_.contains_node(id); }
    bool contains_atom(typename GraphType::ConstNodeRef atom) const {
      return graph_.contains_node(atom);
    }

    MutableAtom operator[](int idx) { return graph_[idx]; }
    Atom operator[](int idx) const { return graph_[idx]; }

    MutableAtom atom(int idx) { return graph_.node(idx); }
    Atom atom(int idx) const { return graph_.node(idx); }

    iterator find_atom(int id) { return graph_.find_node(id); }
    iterator find_atom(typename GraphType::ConstNodeRef atom) {
      return graph_.find_node(atom);
    }

    const_iterator find_atom(int id) const { return graph_.find_node(id); }
    const_iterator find_atom(typename GraphType::ConstNodeRef atom) const {
      return graph_.find_node(atom);
    }

    void erase_atom(int idx) { graph_.erase_node(idx); }
    void erase_atom(Atom atom) { graph_.erase_node(atom); }

    void erase_atoms(const_iterator begin, const_iterator end) {
      graph_.erase_nodes(begin, end);
    }

    void erase_atom_of(int id) { graph_.erase_node_of(id); }

    void erase_atom_of(typename GraphType::ConstNodeRef atom) {
      graph_.erase_node_of(atom);
    }

    template <class UnaryPred>
    void erase_atoms_if(UnaryPred &&pred) {
      graph_.erase_nodes_if(std::forward<UnaryPred>(pred));
    }

    iterator begin() { return graph_.begin(); }
    iterator end() { return graph_.end(); }

    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    const_iterator cbegin() const { return graph_.cbegin(); }
    const_iterator cend() const { return graph_.cend(); }

    atom_iterator atom_begin() { return graph_.node_begin(); }
    atom_iterator atom_end() { return graph_.node_end(); }

    const_atom_iterator atom_begin() const { return atom_cbegin(); }
    const_atom_iterator atom_end() const { return atom_cend(); }

    const_atom_iterator atom_cbegin() const { return graph_.node_cbegin(); }
    const_atom_iterator atom_cend() const { return graph_.node_cend(); }

    const std::vector<int> &atom_ids() const { return graph_.node_ids(); }

    BondsWrapper bonds() { return graph_.edges(); }
    ConstBondsWrapper bonds() const { return graph_.edges(); }

    void clear_bonds() noexcept { graph_.clear_edges(); }

    void update_bonds(IndexSet &&bonds) noexcept {
      graph_.update_edges(std::move(bonds));
    }

    void refresh_bonds() { graph_.refresh_edges(); }

    void reserve_bonds(int n) { graph_.reserve_edges(n); }

    void add_bond(int id) { graph_.add_edge(id); }

    void add_bonds(const IndexSet &bonds) { graph_.add_edges(bonds); }

    bool contains_bond(int id) const { return graph_.contains_edge(id); }
    bool contains_bond(typename GraphType::ConstEdgeRef bond) const {
      return graph_.contains_edge(bond);
    }

    MutableBond bond(int idx) { return graph_.edge(idx); }
    Bond bond(int idx) const { return graph_.edge(idx); }

    bond_iterator find_bond(int id) { return graph_.find_edge(id); }
    bond_iterator find_bond(typename GraphType::ConstEdgeRef bond) {
      return graph_.find_edge(bond);
    }

    const_bond_iterator find_bond(int id) const { return graph_.find_edge(id); }
    const_bond_iterator find_bond(typename GraphType::ConstEdgeRef bond) const {
      return graph_.find_edge(bond);
    }

    void erase_bond(int idx) { graph_.erase_edge(idx); }
    void erase_bond(Bond bond) { graph_.erase_edge(bond); }

    void erase_bonds(const_bond_iterator begin, const_bond_iterator end) {
      graph_.erase_edges(begin, end);
    }

    void erase_bond_of(int id) { graph_.erase_edge_of(id); }

    void erase_bond_of(typename GraphType::ConstEdgeRef bond) {
      graph_.erase_edge_of(bond);
    }

    template <class UnaryPred>
    void erase_bonds_if(UnaryPred &&pred) {
      graph_.erase_edges_if(std::forward<UnaryPred>(pred));
    }

    bond_iterator bond_begin() { return graph_.edge_begin(); }
    bond_iterator bond_end() { return graph_.edge_end(); }

    const_bond_iterator bond_begin() const { return bond_cbegin(); }
    const_bond_iterator bond_end() const { return bond_cend(); }

    const_bond_iterator bond_cbegin() const { return graph_.edge_cbegin(); }
    const_bond_iterator bond_cend() const { return graph_.edge_cend(); }

    const std::vector<int> &bond_ids() const { return graph_.edge_ids(); }

    int degree(int id) const { return graph_.degree(id); }

    bond_iterator find_bond(Atom src, Atom dst) {
      return graph_.find_edge(src, dst);
    }

    const_bond_iterator find_bond(Atom src, Atom dst) const {
      return graph_.find_edge(src, dst);
    }

    bond_iterator find_bond(typename GraphType::ConstNodeRef src,
                            typename GraphType::ConstNodeRef dst) {
      return graph_.find_edge(src, dst);
    }

    const_bond_iterator find_bond(typename GraphType::ConstNodeRef src,
                                  typename GraphType::ConstNodeRef dst) const {
      return graph_.find_edge(src, dst);
    }

    neighbor_iterator find_neighbor(Atom src, Atom dst) {
      return graph_.find_adjacent(src, dst);
    }

    const_neighbor_iterator find_neighbor(Atom src, Atom dst) const {
      return graph_.find_adjacent(src, dst);
    }

    neighbor_iterator neighbor_begin(int id) { return graph_.adj_begin(id); }
    neighbor_iterator neighbor_end(int id) { return graph_.adj_end(id); }

    const_neighbor_iterator neighbor_begin(int id) const {
      return graph_.adj_begin(id);
    }
    const_neighbor_iterator neighbor_end(int id) const {
      return graph_.adj_end(id);
    }

    const_neighbor_iterator neighbor_cbegin(int id) const {
      return graph_.adj_cbegin(id);
    }
    const_neighbor_iterator neighbor_cend(int id) const {
      return graph_.adj_cend(id);
    }

    std::string &name() { return name_; }

    const std::string &name() const { return name_; }

    void set_id(int id) { id_ = id; }

    int id() const { return id_; }

    SubstructCategory &category() { return cat_; }

    SubstructCategory category() const { return cat_; }

    template <class KT, class VT>
    void add_prop(KT &&key, VT &&val) {
      internal::set_key(props_, std::forward<KT>(key), std::forward<VT>(val));
    }

    internal::PropertyMap &props() { return props_; }

    const internal::PropertyMap &props() const { return props_; }

  private:
    friend Molecule;
    friend MoleculeMutator;

    void rebind(typename SubgraphType::parent_type &parent) {
      graph_.rebind(parent);
    }

    SubgraphType graph_;

    std::string name_;
    int id_ = 0;
    SubstructCategory cat_;
    internal::PropertyMap props_;
  };

  extern std::pair<std::vector<int>, bool>
  place_trailing_hydrogens_initial(const Molecule &mol, Matrix3Xd &conf,
                                   int h_begin);

  extern bool optimize_free_hydrogens(const Molecule &mol, Matrix3Xd &conf,
                                      const std::vector<int> &free_hs);
}  // namespace internal

using Substructure = internal::Substructure<false>;
using ConstSubstructure = internal::Substructure<true>;

/**
 * @brief Read-only molecule class.
 *
 * The few allowed mutable operations on the molecule are:
 *    - updating atom/bond properties,
 *    - modifing conformers, and
 *    - adding/removing hydrogens.
 *
 * Note that it is the responsibility of the user to ensure the molecule is in a
 * chemically valid state after property modification. For sanitization, use the
 * MoleculeSanitizer class.
 */
class Molecule {
public:
  using GraphType = Graph<AtomData, BondData>;

  using MutableAtom = GraphType::NodeRef;
  using Atom = GraphType::ConstNodeRef;
  using iterator = GraphType::iterator;
  using const_iterator = GraphType::const_iterator;
  using atom_iterator = iterator;
  using const_atom_iterator = const_iterator;

  using MutableBond = GraphType::EdgeRef;
  using Bond = GraphType::ConstEdgeRef;
  using bond_iterator = GraphType::edge_iterator;
  using const_bond_iterator = GraphType::const_edge_iterator;

  using MutableNeighbor = GraphType::AdjRef;
  using Neighbor = GraphType::ConstAdjRef;
  using neighbor_iterator = GraphType::adjacency_iterator;
  using const_neighbor_iterator = GraphType::const_adjacency_iterator;

  friend MoleculeMutator;

  /**
   * @brief Construct an empty Molecule object.
   */
  Molecule() noexcept = default;
  ~Molecule() noexcept = default;

  Molecule(const Molecule &other) noexcept;
  Molecule(Molecule &&other) noexcept;
  Molecule &operator=(const Molecule &other) noexcept;
  Molecule &operator=(Molecule &&other) noexcept;

  /**
   * @brief Construct a Molecule object from a range of atom data.
   * @tparam Iterator The type of the iterator.
   * @param begin The begin iterator of the range, where `AtomData` must be
   *              constructible from value type of \p begin.
   * @param end The past-the-end iterator of the range.
   */
  template <class Iterator,
            class = internal::enable_if_compatible_iter_t<Iterator, AtomData>>
  Molecule(Iterator begin, Iterator end);

  std::string &name() { return name_; }

  const std::string &name() const { return name_; }

  void reserve(int num_atoms) { graph_.reserve(num_atoms); }

  void reserve_bonds(int num_bonds) { graph_.reserve_edges(num_bonds); }

  /**
   * @brief Check if the molecule has any atoms.
   */
  bool empty() const { return graph_.empty(); }

  /**
   * @brief Get the number of atoms in the molecule.
   * @sa num_atoms()
   */
  int size() const { return graph_.size(); }

  /**
   * @brief Get the number of atoms in the molecule.
   * @sa size()
   */
  int num_atoms() const { return graph_.num_nodes(); }

  /**
   * @brief Get the number of heavy atoms in the molecule (i.e., non-hydrogen
   *        atoms).
   * @sa size(), num_atoms()
   * @note Time complexity is O(V).
   */
  int count_heavy_atoms() const {
    return absl::c_count_if(graph_, [](Molecule::Atom a) {
      return a.data().atomic_number() != 1;
    });
  }

  /**
   * @brief Get a mutable atom of the molecule.
   * @param atom_idx Index of the atom to get.
   * @return A mutable view over \p atom_idx -th atom of the molecule.
   * @note If the atom index is out of range, the behavior is undefined. The
   *       returned reference is invalidated when the molecule is modified.
   */
  MutableAtom atom(int atom_idx) { return graph_.node(atom_idx); }

  /**
   * @brief Get an atom of the molecule.
   * @param atom_idx Index of the atom to get.
   * @return A read-only view over \p atom_idx -th atom of the molecule.
   * @note If the atom index is out of range, the behavior is undefined. The
   *       returned reference is invalidated when the molecule is modified.
   */
  Atom atom(int atom_idx) const { return graph_.node(atom_idx); }

  /**
   * @brief Get a mutable atom of the molecule.
   * @param atom_idx Index of the atom to get.
   * @return A mutable view over \p atom_idx -th atom of the molecule.
   * @note If the atom index is out of range, the behavior is undefined. The
   *       returned reference is invalidated when the molecule is modified.
   */
  MutableAtom operator[](int atom_idx) { return graph_[atom_idx]; }

  /**
   * @brief Get an atom of the molecule.
   * @param atom_idx Index of the atom to get.
   * @return A read-only view over \p atom_idx -th atom of the molecule.
   * @note If the atom index is out of range, the behavior is undefined. The
   *       returned reference is invalidated when the molecule is modified.
   */
  Atom operator[](int atom_idx) const { return graph_[atom_idx]; }

  /**
   * @brief The begin iterator of the molecule over atoms.
   * @sa atom_begin()
   */
  iterator begin() { return graph_.begin(); }
  /**
   * @brief The begin iterator of the molecule over atoms.
   * @sa atom_begin()
   */
  const_iterator begin() const { return graph_.begin(); }

  /**
   * @brief The begin iterator of the molecule over atoms.
   * @sa begin()
   */
  iterator atom_begin() { return begin(); }
  /**
   * @brief The begin iterator of the molecule over atoms.
   * @sa begin()
   */
  const_iterator atom_begin() const { return begin(); }

  /**
   * @brief The past-the-end iterator of the molecule over atoms.
   */
  iterator end() { return graph_.end(); }
  /**
   * @brief The past-the-end iterator of the molecule over atoms.
   */
  const_iterator end() const { return graph_.end(); }

  /**
   * @brief The past-the-end iterator of the molecule over atoms.
   */
  iterator atom_end() { return end(); }
  /**
   * @brief The past-the-end iterator of the molecule over atoms.
   */
  const_iterator atom_end() const { return end(); }

  /**
   * @brief Check if the molecule has any bonds.
   */
  bool bond_empty() const { return graph_.edge_empty(); }

  /**
   * @brief Get the number of bonds in the molecule.
   */
  int num_bonds() const { return graph_.num_edges(); }

  /**
   * @brief Get a mutable bond of the molecule.
   * @param bond_idx Index of the bond to get.
   * @return A mutable view over bond \p bond_idx of the molecule.
   * @note If the atom index is out of range, the behavior is undefined. The
   *       returned reference is invalidated when the molecule is modified.
   */
  MutableBond bond(int bond_idx) { return graph_.edge(bond_idx); }

  /**
   * @brief Get a bond of the molecule.
   * @param bond_idx Index of the bond to get.
   * @return A read-only view over bond \p bond_idx of the molecule.
   * @note If the atom index is out of range, the behavior is undefined. The
   *       returned reference is invalidated when the molecule is modified.
   */
  Bond bond(int bond_idx) const { return graph_.edge(bond_idx); }

  /**
   * @brief Get a bond of the molecule.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @return An iterator to the bond between \p src and \p dst of the molecule.
   *         If no such bond exists, the returned iterator is equal to
   *         bond_end().
   */
  bond_iterator find_bond(int src, int dst) {
    return graph_.find_edge(src, dst);
  }

  /**
   * @brief Get a bond of the molecule.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @return An iterator to the bond between \p src and \p dst of the molecule.
   *         If no such bond exists, the returned iterator is equal to
   *         bond_end().
   */
  const_bond_iterator find_bond(int src, int dst) const {
    return graph_.find_edge(src, dst);
  }

  /**
   * @brief Get a bond of the molecule.
   * @param src The source atom of the bond.
   * @param dst The destination atom of the bond.
   * @return An iterator to the bond between \p src and \p dst of the molecule.
   *         If no such bond exists, the returned iterator is equal to
   *         bond_end().
   */
  bond_iterator find_bond(Atom src, Atom dst) {
    return graph_.find_edge(src, dst);
  }

  /**
   * @brief Get a bond of the molecule.
   * @param src The source atom of the bond.
   * @param dst The destination atom of the bond.
   * @return An iterator to the bond between \p src and \p dst of the molecule.
   *         If no such bond exists, the returned iterator is equal to
   *         bond_end().
   */
  const_bond_iterator find_bond(Atom src, Atom dst) const {
    return graph_.find_edge(src, dst);
  }

  /**
   * @brief Get an iterable, modifiable view over bonds of the molecule.
   */
  auto bonds() { return graph_.edges(); }

  /**
   * @brief Get an iterable, non-modifiable view over bonds of the molecule.
   */
  auto bonds() const { return graph_.edges(); }

  /**
   * @brief Get an iterable, non-modifiable view over bonds of the molecule.
   */
  auto cbonds() const { return graph_.edges(); }

  /**
   * @brief The begin iterator of the molecule over bonds.
   */
  bond_iterator bond_begin() { return graph_.edge_begin(); }
  /**
   * @brief The begin iterator of the molecule over bonds.
   */
  const_bond_iterator bond_begin() const { return graph_.edge_begin(); }
  /**
   * @brief The past-the-end iterator of the molecule over bonds.
   */
  bond_iterator bond_end() { return graph_.edge_end(); }
  /**
   * @brief The past-the-end iterator of the molecule over bonds.
   */
  const_bond_iterator bond_end() const { return graph_.edge_end(); }

  /**
   * @brief Get the number of bonds between heavy atoms in the molecule.
   * @sa num_bonds()
   * @note Time complexity is O(E).
   */
  int count_heavy_bonds() const {
    return absl::c_count_if(bonds(), [](Molecule::Bond bond) {
      return bond.src().data().atomic_number() != 1
             && bond.dst().data().atomic_number() != 1;
    });
  }

  /**
   * @brief Get the explicitly connected neighbor atoms of the atom.
   * @param atom Index of the atom.
   * @return The number of bonds connected to the atom.
   * @note If the atom index is out of range, the behavior is undefined.
   * @sa all_neighbors(), count_heavy(), count_hydrogens()
   */
  int num_neighbors(int atom) const { return graph_.degree(atom); }

  /**
   * @brief Find a neighbor of the atom.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @return An iterator to the neighbor wrapper between \p src and \p dst of
   *         the molecule. If no such bond exists, the returned iterator is
   *         equal to \ref neighbor_end() "neighbor_end(\p src)"
   */
  neighbor_iterator find_neighbor(int src, int dst) {
    return graph_.find_adjacent(src, dst);
  }

  /**
   * @brief Find a neighbor of the atom.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @return An iterator to the neighbor wrapper between \p src and \p dst of
   *         the molecule. If no such bond exists, the returned iterator is
   *         equal to \ref neighbor_end() "neighbor_end(\p src)"
   */
  const_neighbor_iterator find_neighbor(int src, int dst) const {
    return graph_.find_adjacent(src, dst);
  }

  /**
   * @brief Find a neighbor of the atom.
   * @param src The source atom of the bond.
   * @param dst The destination atom of the bond.
   * @return An iterator to the neighbor wrapper between \p src and \p dst of
   *         the molecule. If no such bond exists, the returned iterator is
   *         equal to \ref neighbor_end() "neighbor_end(\p src)"
   */
  neighbor_iterator find_neighbor(Atom src, Atom dst) {
    return graph_.find_adjacent(src, dst);
  }

  /**
   * @brief Find a neighbor of the atom.
   * @param src The source atom of the bond.
   * @param dst The destination atom of the bond.
   * @return An iterator to the neighbor wrapper between \p src and \p dst of
   *         the molecule. If no such bond exists, the returned iterator is
   *         equal to \ref neighbor_end() "neighbor_end(\p src)"
   */
  const_neighbor_iterator find_neighbor(Atom src, Atom dst) const {
    return graph_.find_adjacent(src, dst);
  }

  /**
   * @brief The begin iterator of an atom over its neighbors.
   */
  neighbor_iterator neighbor_begin(int atom_idx) {
    return graph_.adj_begin(atom_idx);
  }
  /**
   * @brief The begin iterator of an atom over its neighbors.
   */
  const_neighbor_iterator neighbor_begin(int atom_idx) const {
    return graph_.adj_begin(atom_idx);
  }
  /**
   * @brief The past-the-end iterator of an atom over its neighbors.
   */
  neighbor_iterator neighbor_end(int atom_idx) {
    return graph_.adj_end(atom_idx);
  }
  /**
   * @brief The past-the-end iterator of an atom over its neighbors.
   */
  const_neighbor_iterator neighbor_end(int atom_idx) const {
    return graph_.adj_end(atom_idx);
  }

  /**
   * @brief Get a MoleculeMutator object associated with the molecule.
   * @return The MoleculeMutator object to update this molecule.
   * @sa MoleculeMutator
   */
  MoleculeMutator mutator();

  /**
   * @brief Reset the molecule to an empty state.
   * @note Don't call this method if you have an active MoleculeMutator object.
   *       Call MoleculeMutator::clear() instead.
   *
   * This method effectively resets the molecule to the state of a default
   * constructed molecule. Unlike clear_atoms(), this will also clear name,
   * conformers, substructures, and properties.
   */
  void clear() noexcept;

  /**
   * @brief Clear all atoms and bonds of the molecule.
   * @note Don't call this method if you have an active MoleculeMutator object.
   *       Call MoleculeMutator::clear_atoms() instead.
   *
   * All conformers will be resized to 0. All substructures will contain no
   * atoms.
   */
  void clear_atoms() noexcept;

  /**
   * @brief Clear all bonds of the molecule.
   * @note Don't call this method if you have an active MoleculeMutator object.
   *       Call MoleculeMutator::clear_bonds() instead.
   *
   * Conformers and substructures are not affected.
   */
  void clear_bonds() noexcept;

  /**
   * @brief Make all implicit hydrogens explicit.
   *
   * @param update_confs Whether to update the conformers after adding
   *        hydrogens. Default is `true`.
   * @param optimize Whether to optimize the geometry of the molecule after
   *        initial hydrogen placement. Default is `true`. When this is `false`,
   *        the resulting molecule may have clashes between atoms.
   * @return Whether the operation was successful. Fails only if the molecule
   *         has more than one conformers and the geometry optimization fails.
   *         When this function returns false, the coordinates of the newly
   *         added hydrogens are not guaranteed to be chemically valid.
   */
  ABSL_MUST_USE_RESULT bool add_hydrogens(bool update_confs = true,
                                          bool optimize = true);

  /**
   * @brief Erase all trivial hydrogens from the molecule.
   *
   * Trivial hydrogens must satisfy the following conditions:
   *
   *   1. The hydrogen atom has single neighbor,
   *   2. The neighbor is a heavy atom,
   *   3. The two atoms are connected by a single bond, and
   *   4. The hydrogen atom has no implicit hydrogens.
   */
  void erase_hydrogens();

  /**
   * @brief Check if the molecule has any 3D conformations.
   * @return `true` if the molecule has any 3D conformations, `false` otherwise.
   */
  bool is_3d() const { return !conformers_.empty(); }

  /**
   * @brief Get all atomic coordinates of the conformers.
   * @return Atomic coordinates of the conformers of the molecule.
   * @note If any conformer's column size is not equal to the number of atoms,
   *       the behavior is undefined.
   */
  std::vector<Matrix3Xd> &confs() { return conformers_; }

  /**
   * @brief Get all atomic coordinates of the conformers.
   * @return Atomic coordinates of the conformers of the molecule.
   */
  const std::vector<Matrix3Xd> &confs() const { return conformers_; }

  /**
   * @brief Transform the molecule with the given affine transformation.
   * @param trans The affine transformation to apply.
   */
  void transform(const Eigen::Affine3d &trans) {
    for (Matrix3Xd &m: conformers_)
      m = trans * m;
  }

  /**
   * @brief Transform a conformer of the molecule with the given affine
   *        transformation.
   * @param i The index of the conformer to transform.
   * @param trans The affine transformation to apply.
   * @note The behavior is undefined if the conformer index is out of range.
   */
  void transform(int i, const Eigen::Affine3d &trans) {
    Matrix3Xd &m = conformers_[i];
    m = trans * m;
  }

  /**
   * @brief Calculate the squared distance between two atoms.
   * @param src Index of the source atom.
   * @param dst Index of the destination atom.
   * @param conf Index of the conformer to use for the calculation.
   * @return The squared distance between the two atoms.
   * @note The behavior is undefined if any of the indices are out of range.
   */
  double distsq(int src, int dst, int conf = 0) const;

  /**
   * @brief Calculate the squared distance between two bonded atoms.
   * @param bond The bond between the two atoms.
   * @param conf Index of the conformer to use for the calculation.
   * @return The squared distance between the two atoms.
   * @note The behavior is undefined the bond is not in the molecule, or if the
   *       conformer index is out of range.
   */
  double distsq(Bond bond, int conf = 0) const {
    return distsq(bond.src().id(), bond.dst().id(), conf);
  }

  /**
   * @brief Calculate the distance between two atoms.
   * @param src Index of the source atom.
   * @param dst Index of the destination atom.
   * @param conf Index of the conformer to use for the calculation.
   * @return The distance between the two atoms.
   * @note The behavior is undefined if any of the indices are out of range.
   */
  double distance(int src, int dst, int conf = 0) const {
    return std::sqrt(distsq(src, dst, conf));
  }

  /**
   * @brief Calculate the distance between two bonded atoms.
   * @param bond The bond between the two atoms.
   * @param conf Index of the conformer to use for the calculation.
   * @return The distance between the two atoms.
   * @note The behavior is undefined the bond is not in the molecule, or if the
   *       conformer index is out of range.
   */
  double distance(Bond bond, int conf = 0) const {
    return distance(bond.src().id(), bond.dst().id(), conf);
  }

  /**
   * @brief Calculate the bond lengths of the molecule.
   * @param conf Index of the conformer to use for the calculation.
   * @return An array of bond lengths. The order of the bond lengths is the same
   *         as the order of the bonds returned by bonds().
   * @note The behavior is undefined if \p conf is out of range.
   */
  ArrayXd bond_lengths(int conf = 0) const;

  /**
   * @brief Rotate a bond.
   * @param ref_atom Index of the reference atom.
   * @param pivot_atom Index of the pivot atom.
   * @param angle Angle to rotate (in degrees).
   * @return `true` if the rotation was applied, `false` if the rotation was
   *         not applied.
   * @note Rotatability only considers ring membership. The user is responsible
   *       to check bond order or other constraints.
   *
   * The rotation is applied to all conformers of the molecule.
   *
   * The part of the reference atom is fixed, and the part of the pivot atom
   * will be rotated about the reference atom -> pivot atom axis. Positive angle
   * means counter-clockwise rotation (as in the right-hand rule).
   *
   * The behavior is undefined if any of the indices are out of range, or if the
   * atoms are not connected by a bond.
   */
  bool rotate_bond(int ref_atom, int pivot_atom, double angle);

  /**
   * @brief Rotate a bond.
   * @param bid The index of bond to rotate.
   * @param angle Angle to rotate (in degrees).
   * @return `true` if the rotation was applied, `false` if the rotation was
   *         not applied.
   * @note Rotatability only considers ring membership. The user is responsible
   *       to check bond order or other constraints.
   *
   * The rotation is applied to all conformers of the molecule.
   *
   * The source atom of the bond is fixed, and the destination atom will be
   * rotated about the source atom -> destination atom axis. Positive angle
   * means counter-clockwise rotation (as in the right-hand rule).
   *
   * The behavior is undefined if any of the indices are out of range.
   */
  bool rotate_bond(int bid, double angle);

  /**
   * @brief Rotate a bond of a conformer.
   * @param i The index of the conformer to transform.
   * @param ref_atom Index of the pivot atom.
   * @param pivot_atom Index of the pivot atom.
   * @param angle Angle to rotate (in degrees).
   * @return `true` if the rotation was applied, `false` if the rotation was
   *         not applied.
   * @note Rotatability only considers ring membership. The user is responsible
   *       to check bond order or other constraints.
   *
   * The part of the pivot atom is fixed, and the part of the pivot atom will be
   * rotated about the pivot atom -> pivot atom axis. Positive angle means
   * counter-clockwise rotation (as in the right-hand rule).
   *
   * The behavior is undefined if any of the indices are out of range, or if the
   * atoms are not connected by a bond.
   */
  bool rotate_bond_conf(int i, int ref_atom, int pivot_atom, double angle);

  /**
   * @brief Rotate a bond of a conformer.
   * @param i The index of the conformer to transform.
   * @param bid The index of bond to rotate.
   * @param angle Angle to rotate (in degrees).
   * @return `true` if the rotation was applied, `false` if the rotation was
   *         not applied (e.g. the bond is not rotatable, etc.).
   * @note Rotatability only considers ring membership. The user is responsible
   *       to check bond order or other constraints.
   *
   * The source atom of the bond is fixed, and the destination atom will be
   * rotated about the source atom -> destination atom axis. Positive angle
   * means counter-clockwise rotation (as in the right-hand rule).
   *
   * The behavior is undefined if any of the indices are out of range.
   */
  bool rotate_bond_conf(int i, int bid, double angle);

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @param cat The category of the substructure.
   * @return The new substructure.
   */
  Substructure
  substructure(SubstructCategory cat = SubstructCategory::kUnknown) {
    return { make_subgraph(graph_), cat };
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @param atoms Indices of atoms in the substructure.
   * @param bonds Indices of bonds in the substructure.
   * @param cat The category of the substructure.
   * @return The new substructure.
   */
  Substructure
  substructure(internal::IndexSet &&atoms, internal::IndexSet &&bonds,
               SubstructCategory cat = SubstructCategory::kUnknown) {
    return { make_subgraph(graph_, std::move(atoms), std::move(bonds)), cat };
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   * @param atoms Indices of atoms in the substructure. All bonds between the
   *        atoms will also be included in the substructure.
   * @param cat The category of the substructure.
   */
  Substructure
  atom_substructure(internal::IndexSet &&atoms,
                    SubstructCategory cat = SubstructCategory::kUnknown) {
    return { subgraph_from_nodes(graph_, std::move(atoms)), cat };
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   * @param bonds Indices of bonds in the substructure. All atoms connected by
   *        the bonds will also be included in the substructure.
   * @param cat The category of the substructure.
   */
  Substructure bond_substructure(
      internal::IndexSet &&bonds,
      SubstructCategory cat = SubstructCategory::kUnknown) noexcept {
    return { subgraph_from_edges(graph_, std::move(bonds)), cat };
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @param cat The category of the substructure.
   * @return The new substructure.
   */
  ConstSubstructure
  substructure(SubstructCategory cat = SubstructCategory::kUnknown) const {
    return { Subgraph(graph_), cat };
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @param atoms Indices of atoms in the substructure.
   * @param bonds Indices of bonds in the substructure.
   * @param cat The category of the substructure.
   * @return The new substructure.
   */
  ConstSubstructure
  substructure(internal::IndexSet &&atoms, internal::IndexSet &&bonds,
               SubstructCategory cat = SubstructCategory::kUnknown) const {
    return { make_subgraph(graph_, std::move(atoms), std::move(bonds)), cat };
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   * @param atoms Indices of atoms in the substructure. All bonds between the
   *        atoms will also be included in the substructure.
   * @param cat The category of the substructure.
   */
  ConstSubstructure
  atom_substructure(internal::IndexSet &&atoms,
                    SubstructCategory cat = SubstructCategory::kUnknown) const {
    return { subgraph_from_nodes(graph_, std::move(atoms)), cat };
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   * @param bonds Indices of bonds in the substructure. All atoms connected by
   *        the bonds will also be included in the substructure.
   * @param cat The category of the substructure.
   */
  ConstSubstructure
  bond_substructure(internal::IndexSet &&bonds,
                    SubstructCategory cat = SubstructCategory::kUnknown) const {
    return { subgraph_from_edges(graph_, std::move(bonds)), cat };
  }

  /**
   * @brief Get the substructures.
   *
   * @return A reference to all substructures.
   */
  std::vector<Substructure> &substructures() { return substructs_; }

  /**
   * @brief Get the substructures.
   *
   * @return A const reference to all substructures.
   */
  const std::vector<Substructure> &substructures() const { return substructs_; }

  /**
   * @brief Update topology of the molecule.
   *
   * This method is safe to call multiple times, but will be automatically
   * called from the MoleculeMutator class anyway. Thus, user code normally
   * don't need to call this method.
   *
   * @note This only changes ring atom/bond flags. If the molecule is modified
   *       in a way that changes aromaticity, the aromaticity flags should be
   *       updated manually or by using the MoleculeSanitizer class.
   * @warning All molecule-related methods/functions assume that the molecule
   *          has valid topology data.
   */
  void update_topology();

  /**
   * @brief Get size of SSSR.
   * @return The number of rings in the smallest set of smallest rings.
   */
  int num_sssr() const { return num_bonds() - num_atoms() + num_fragments(); }

  /**
   * @brief Get the ring groups of the molecule, i.e. all rings of a molecule,
   *        merged into groups of rings that share at least one atom.
   * @return List of the ring groups. If a ring group consists of single ring,
   *         the atom indices form a ring in the order of the returned vector.
   */
  const std::vector<std::vector<int>> &ring_groups() const {
    return ring_groups_;
  }

  /**
   * @brief Get number of fragments (aka connected components).
   * @return The number of fragments.
   */
  int num_fragments() const { return num_fragments_; }

  /**
   * @brief Merge other molecule-like object into this molecule.
   * @tparam MoleculeLike Type of the other molecule-like object.
   * @param other The other molecule-like object.
   * @warning The resulting molecule might not be chemically valid. Use the
   *          MolculeSanitizer class to sanitize the molecule if necessary.
   * @note Size of conformers will be updated, but the positions of the added
   *       atoms will not be set. It is the responsibility of the user to set
   *       the positions of the added atoms if necessary.
   * @note This method effectively calls update_topology().
   */
  template <class MoleculeLike>
  void merge(const MoleculeLike &other) {
    graph_.merge(other.graph_);
    update_topology();

    for (Matrix3Xd &conf: conformers_)
      conf.conservativeResize(Eigen::NoChange, size());
  }

  template <class KT, class VT>
  void add_prop(KT &&key, VT &&val) {
    internal::set_key(props_, std::forward<KT>(key), std::forward<VT>(val));
  }

  internal::PropertyMap &props() { return props_; }

  const internal::PropertyMap &props() const { return props_; }

private:
  Molecule(GraphType &&graph, std::vector<Matrix3Xd> &&conformers) noexcept
      : graph_(std::move(graph)), conformers_(std::move(conformers)) { }

  void rebind_substructs() noexcept;

  bool rotate_bond_common(int i, int ref_atom, int pivot_atom, double angle);

  GraphType graph_;
  std::vector<Matrix3Xd> conformers_;
  std::string name_;
  internal::PropertyMap props_;

  std::vector<Substructure> substructs_;

  std::vector<std::vector<int>> ring_groups_;
  int num_fragments_ = 0;
};

/**
 * @brief A class to mutate a molecule.
 *
 * Atom and bond addition is directly applied to the molecule, but atom and bond
 * erasure is delayed until the `finalize()` method is called. This is because
 * the erasure of atoms might change the atom indices of the remaining atoms,
 * and might give unexpected results if the erasure is done before any other
 * mutations. The erasures will be applied in this order:
 *
 * 1. Bond erasures.
 * 2. Atom erasures.
 *
 * If `finalize()` method is not explicitly called, the destructor will
 * automatically call it.
 *
 * Conformers of molecules will resize to the new number of atoms also at the
 * `finalize()` call. Currently, there is no way to specify the new positions of
 * the added atoms. Assign the new positions manually after the mutation if
 * necessary.
 *
 * @note Having multiple instances of this class for the same molecule might
 *       result in an inconsistent state.
 */
class MoleculeMutator {
public:
  /**
   * @brief Construct a new MoleculeMutator object.
   *
   * @param mol The molecule to mutate.
   */
  MoleculeMutator(Molecule &mol)
      : mol_(&mol), prev_num_atoms_(mol.num_atoms()),
        prev_num_bonds_(mol.num_bonds()) { }

  MoleculeMutator() = delete;
  MoleculeMutator(const MoleculeMutator &) = delete;
  MoleculeMutator &operator=(const MoleculeMutator &) = delete;
  MoleculeMutator(MoleculeMutator &&) noexcept = default;
  MoleculeMutator &operator=(MoleculeMutator &&) noexcept = default;

  ~MoleculeMutator() noexcept { finalize(); }

  /**
   * @brief Add an atom to the molecule.
   * @param atom The data of the atom to add.
   * @return The index of the added atom.
   */
  int add_atom(const AtomData &atom) { return mol().graph_.add_node(atom); }

  /**
   * @brief Add an atom to the molecule.
   * @param atom The data of the atom to add.
   * @return The index of the added atom.
   */
  int add_atom(AtomData &&atom) noexcept {
    return mol().graph_.add_node(std::move(atom));
  }

  /**
   * @brief Add an atom to the molecule.
   * @param atom An atom to add.
   * @return The index of the added atom.
   */
  int add_atom(const Molecule::Atom &atom) { return add_atom(atom.data()); }

  /**
   * @brief Mark an atom to be erased.
   * @param atom_idx Index of the atom to erase, after all additions.
   * @note The behavior is undefined if the atom index is out of range at the
   *       moment of calling `finalize()`.
   */
  void mark_atom_erase(int atom_idx) { erased_atoms_.push_back(atom_idx); }

  /**
   * @brief Mark an atom to be erased.
   * @param atom An atom to erase.
   * @note The behavior is undefined if the atom does not belong to the
   *       molecule.
   */
  void mark_atom_erase(Molecule::Atom atom) { mark_atom_erase(atom.id()); }

  /**
   * @brief Clear all atoms and bonds of the molecule.
   */
  void clear_atoms() noexcept;

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @param bond The data of the bond to add.
   * @return A pair of (bond index, was added). If not added, the bond index is
   *         the index of the existing bond (the data is not modified).
   * @note The behavior is undefined if any of the atom indices is out of range,
   *       or if src == dst.
   */
  std::pair<int, bool> add_bond(int src, int dst, const BondData &bond);

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @param bond The data of the bond to add.
   * @return A pair of (bond index, was added). If not added, the bond index is
   *         the index of the existing bond (the data is not modified).
   * @note The behavior is undefined if any of the atom indices is out of range,
   *       or if src == dst.
   */
  std::pair<int, bool> add_bond(int src, int dst, BondData &&bond) noexcept;

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @param bond The bond to copy the data from.
   * @return A pair of (bond index, was added). If not added, the bond index is
   *         the index of the existing bond (the data is not modified).
   * @note The behavior is undefined if any of the atom indices is out of range,
   *       or if src == dst.
   */
  std::pair<int, bool> add_bond(int src, int dst, Molecule::Bond bond) {
    return add_bond(src, dst, bond.data());
  }

  /**
   * @brief Mark a bond to be erased.
   * @param bid The index of the bond to erase.
   * @note The behavior is undefined if the index is out of range.
   */
  void mark_bond_erase(int bid) { erased_bonds_.push_back(bid); }

  /**
   * @brief Mark a bond to be erased.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @note The behavior is undefined if any of the atom indices is out of range.
   *       This is a no-op if the bond does not exist.
   */
  void mark_bond_erase(int src, int dst);

  /**
   * @brief Mark a bond to be erased.
   * @param src The source atom of the bond.
   * @param dst The destination atom of the bond.
   * @note The behavior is undefined if any of the atom does not belong to the
   *       molecule. This is a no-op if the bond does not exist.
   */
  void mark_bond_erase(Molecule::Atom src, Molecule::Atom dst) {
    mark_bond_erase(src.id(), dst.id());
  }

  /**
   * @brief Clear all bonds of the molecule.
   */
  void clear_bonds() noexcept;

  /**
   * @brief Clear the molecule.
   */
  void clear() noexcept;

  /**
   * @brief Cancel all pending atom and bond removals.
   */
  void discard_erasure() noexcept;

  /**
   * @brief Finalize the mutation.
   * @note The mutator internally calls discard_erasure() after applying
   *       changes. Thus, successive calls to finalize() have no effect.
   * @sa Molecule::sanitize()
   *
   * This will effectively call Molecule::update_topology(), if any atoms or
   * bonds are added or removed.
   */
  void finalize() noexcept;

  // GCOV_EXCL_START
  Molecule &mol() noexcept {
    ABSL_ASSUME(mol_ != nullptr);
    return *mol_;
  }

  const Molecule &mol() const noexcept {
    ABSL_ASSUME(mol_ != nullptr);
    return *mol_;
  }
  // GCOV_EXCL_STOP

private:
  Molecule *mol_;
  int prev_num_atoms_;
  int prev_num_bonds_;

  std::vector<int> erased_atoms_;
  std::vector<int> erased_bonds_;
};

class MoleculeSanitizer {
public:
  /**
   * @brief Construct a new MoleculeSanitizer object.
   *
   * @param molecule The molecule to sanitize.
   */
  explicit MoleculeSanitizer(Molecule &molecule);

  MoleculeSanitizer(MoleculeSanitizer &&) noexcept = default;
  ~MoleculeSanitizer() noexcept = default;

  MoleculeSanitizer() = delete;
  MoleculeSanitizer(const MoleculeSanitizer &) = delete;
  MoleculeSanitizer &operator=(const MoleculeSanitizer &) = delete;
  MoleculeSanitizer &operator=(MoleculeSanitizer &&) noexcept = delete;

  /**
   * @brief Sanitize conjugated bonds.
   * @return Whether the molecule was successfully sanitized.
   */
  ABSL_MUST_USE_RESULT bool sanitize_conjugated();

  /**
   * @brief Sanitize aromaticity.
   * @pre Requires conjugated bonds to be sanitized. Do it manually, or call
   *      sanitize_conjugated() first.
   * @return Whether the molecule was successfully sanitized.
   */
  ABSL_MUST_USE_RESULT bool sanitize_aromaticity();

  /**
   * @brief Sanitize hybridization.
   * @pre Requires conjugated bonds to be sanitized. Do it manually, or call
   *      sanitize_conjugated() first.
   * @return Whether the molecule was successfully sanitized.
   */
  ABSL_MUST_USE_RESULT bool sanitize_hybridization();

  /**
   * @brief Sanitize valences.
   * @pre Requires conjugated bonds to be sanitized. Do it manually, or call
   *      sanitize_conjugated() first.
   * @return Whether the molecule was successfully sanitized.
   */
  ABSL_MUST_USE_RESULT bool sanitize_valence();

  /**
   * @brief Sanitize all.
   *
   * This is a shortcut for calling all of the above methods in the following
   * order:
   *   - sanitize_conjugated()
   *   - sanitize_aromaticity()
   *   - sanitize_hybridization()
   *   - sanitize_valence()
   *
   * The method will return on the first failure.
   *
   * @return Whether the molecule was successfully sanitized.
   */
  ABSL_MUST_USE_RESULT bool sanitize_all() {
    return sanitize_conjugated() && sanitize_aromaticity()
           && sanitize_hybridization() && sanitize_valence();
  }

  Molecule &mol() noexcept {
    ABSL_ASSUME(mol_ != nullptr);
    return *mol_;
  }

  const Molecule &mol() const noexcept {
    ABSL_ASSUME(mol_ != nullptr);
    return *mol_;
  }

private:
  Molecule *mol_;
  absl::FixedArray<int> valences_;
};

/* Out-of-line definitions for molecule */

template <class Iterator, class>
Molecule::Molecule(Iterator begin, Iterator end): Molecule() {
  MoleculeMutator m = mutator();
  for (auto it = begin; it != end; ++it) {
    m.add_atom(*it);
  }
}

inline MoleculeMutator Molecule::mutator() {
  return MoleculeMutator(*this);
}

/* Utility functions */

namespace internal {
  inline int common_valence(const Element &effective) {
    const int val_electrons = effective.valence_electrons(),
              common_valence = val_electrons <= 4 ? val_electrons
                                                  : 8 - val_electrons;
    return common_valence;
  }

  extern const Element &
  effective_element_or_element(const AtomData &data) noexcept;

  inline const Element &
  effective_element_or_element(Molecule::Atom atom) noexcept {
    return effective_element_or_element(atom.data());
  }

  extern int sum_bond_order_raw(Molecule::Atom atom, int implicit_hydrogens,
                                bool aromatic_correct);

  inline int sum_bond_order(Molecule::Atom atom, bool aromatic_correct) {
    return sum_bond_order_raw(atom, atom.data().implicit_hydrogens(),
                              aromatic_correct);
  }

  extern int steric_number(int total_degree, int nb_electrons);

  extern constants::Hybridization from_degree(int total_degree,
                                              int nb_electrons);

  extern int nonbonding_electrons(const AtomData &data, int total_valence);

  extern int count_pi_e(Molecule::Atom atom, int total_valence);

  extern int aromatic_pi_e(Molecule::Atom atom, int total_valence);
}  // namespace internal

/**
 * @brief Get the number of all neighbors of an atom.
 * @param atom An atom.
 * @return Number of all neighbors of the atom, including implicit hydrogens.
 */
inline int all_neighbors(Molecule::Atom atom) {
  return atom.degree() + atom.data().implicit_hydrogens();
}

/**
 * @brief Get the number of heavy atoms bonded to an atom.
 * @param atom An atom.
 * @return Number of heavy atoms bonded to the atom.
 * @note Dummy atom counts as a heavy atom.
 */
extern int count_heavy(Molecule::Atom atom);

/**
 * @brief Count the number of hydrogens of the atom.
 * @param atom An atom.
 * @return Number of hydrogens of the atom (including implicit hydrogens)
 * @note If the atom index is out of range, the behavior is undefined.
 */
extern int count_hydrogens(Molecule::Atom atom);

/**
 * @brief Get the approximate total bond order of the atom.
 * @param atom An atom.
 * @return Total bond order of the atom.
 * @note "Other bond" count as single bond.
 */
inline int sum_bond_order(Molecule::Atom atom) {
  return internal::sum_bond_order(atom, true);
}

/**
 * @brief Get the predicted non-bonding electron count of the atom.
 * @param atom An atom.
 * @return Predicted non-bonding electron count of the atom.
 * @note This function might return a negative value if the atom is not
 *       chemically valid.
 */
inline int nonbonding_electrons(Molecule::Atom atom) {
  return internal::nonbonding_electrons(atom.data(), sum_bond_order(atom));
}

/**
 * @brief Get the predicted steric number of the atom.
 * @param atom An atom.
 * @return Predicted steric number of the atom.
 * @note This function might return a negative value if the atom is not
 *       chemically valid.
 * @note Radicals don't count as lone pairs.
 */
inline int steric_number(Molecule::Atom atom) {
  int nbe = nonbonding_electrons(atom);
  if (nbe < 0)
    return nbe;

  return internal::steric_number(all_neighbors(atom), nbe);
}

/**
 * @brief Get "effective" element of the atom.
 * @param data Data of the atom.
 * @return "Effective" element of the atom: the returned element has atomic
 *         number of (original atomic number) - (formal charge). If the
 *         resulting atomic number is out of range, returns nullptr.
 */
inline const Element *effective_element(const AtomData &data) {
  const int effective_z = data.atomic_number() - data.formal_charge();
  return kPt.find_element(effective_z);
}

/**
 * @brief Get "effective" element of the atom.
 * @param atom An atom.
 * @return "Effective" element of the atom: the returned element has atomic
 *         number of (original atomic number) - (formal charge). If the
 *         resulting atomic number is out of range, returns nullptr.
 */
inline const Element *effective_element(Molecule::Atom atom) {
  return effective_element(atom.data());
}

/**
 * @brief Get fragments of the molecule.
 *
 * @param mol The molecule.
 * @return A list of fragments. Each fragment is a list of atom indices.
 */
extern std::vector<std::vector<int>> fragments(const Molecule &mol);
}  // namespace nuri

#endif /* NURI_CORE_MOLECULE_H_ */
