//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
/** @file */
#ifndef NURI_CORE_MOLECULE_H_
#define NURI_CORE_MOLECULE_H_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/attributes.h>
#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/graph.h"
#include "nuri/utils.h"

namespace nuri {
namespace constants {
  /**
   * @brief The hybridization state of an atom object.
   */
  enum Hybridization {
    kUnbound = 0,   // Unbound
    kTerminal = 1,  // Terminal
    kSP = 2,
    kSP2 = 3,
    kSP3 = 4,
    kSP3D = 5,
    kSP3D2 = 6,
    kOtherHyb = 7,  // Unknown/other
  };

  /**
   * @brief The bond order of a bond object.
   */
  enum BondOrder {
    kSingleBond = 1,
    kDoubleBond = 2,
    kTripleBond = 3,
    kQuadrupleBond = 4,
    kAromaticBond = 5,
  };

  extern constexpr inline double kBondOrderToDouble[] = { 0.0, 1.0, 2.0,
                                                          3.0, 4.0, 1.5 };
}  // namespace constants

class AtomData {
public:
  /**
   * @brief Creates a dummy atom with unknown hybridization.
   */
  AtomData(): AtomData(PeriodicTable::get()[0]) { }

  AtomData(const Element &element, int implicit_hydrogens = 0,
           int formal_charge = 0,
           constants::Hybridization hyb = constants::kUnbound,
           double partial_charge = 0.0, int mass_number = -1,
           bool is_aromatic = false, bool is_in_ring = false,
           bool is_chiral = false, bool is_right_handed = false);

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

  void set_element(const Element &element) { element_ = &element; }

  void set_element(int atomic_number) {
    set_element(PeriodicTable::get()[atomic_number]);
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

  void set_isotope(const Isotope &isotope) { isotope_ = &isotope; }

  void set_isotope(int mass_number) {
    isotope_ = element().find_isotope(mass_number);
  }

  /**
   * @brief Get the explicitly set isotope of the atom.
   * @return A pointer to the explicitly set isotope object. If none was
   *         explicitly given, returns `nullptr`. Normally isotope() would
   *         be the preferred method to get the isotope, which returns the
   *         representative isotope of the element if none was explicitly given.
   * @sa isotope()
   */
  const Isotope *explicit_isotope() const { return isotope_; }

  void set_hybridization(constants::Hybridization hyb) { hyb_ = hyb; }

  constants::Hybridization hybridization() const { return hyb_; }

  void set_implicit_hydrogens(int implicit_hydrogens) {
    implicit_hydrogens_ = implicit_hydrogens;
  }

  int implicit_hydrogens() const { return implicit_hydrogens_; }

  void set_aromatic(bool is_aromatic) {
    internal::update_flag(flags_, is_aromatic, AtomFlags::kAromatic);
  }

  bool is_aromatic() const {
    return internal::check_flag(flags_, AtomFlags::kAromatic);
  }

  void set_conjugated(bool is_conjugated) {
    internal::update_flag(flags_, is_conjugated, AtomFlags::kConjugated);
  }

  bool is_conjugated() const {
    return internal::check_flag(flags_, AtomFlags::kConjugated);
  }

  void set_ring_atom(bool is_ring_atom) {
    internal::update_flag(flags_, is_ring_atom, AtomFlags::kRing);
  }

  bool is_ring_atom() const {
    return internal::check_flag(flags_, AtomFlags::kRing);
  }

  void set_chiral(bool is_chiral) {
    internal::update_flag(flags_, is_chiral, AtomFlags::kChiral);
  }

  bool is_chiral() const {
    return internal::check_flag(flags_, AtomFlags::kChiral);
  }

  void set_right_handed(bool is_right_handed) {
    internal::update_flag(flags_, is_right_handed, AtomFlags::kRightHanded);
  }

  /**
   * @brief Get handedness of a chiral atom.
   *
   * @pre is_chiral() == `true`, otherwise return value would be meaningless.
   * @return Whether the chiral atom is "right-handed," i.e., `true` for (R)
   *         and `false` for (S).
   */
  bool is_right_handed() const {
    return internal::check_flag(flags_, AtomFlags::kRightHanded);
  }

  void reset_flags() { flags_ = static_cast<AtomFlags>(0); }

  void set_partial_charge(double charge) { partial_charge_ = charge; }

  double partial_charge() const { return partial_charge_; }

  void set_formal_charge(int charge) { formal_charge_ = charge; }

  int formal_charge() const { return formal_charge_; }

  const std::string *find_name() const { return internal::get_name(props_); }

  void set_name(std::string_view name) { internal::set_name(props_, name); }

  template <class KT, class VT>
  void add_prop(KT &&key, VT &&val) {
    props_.emplace_back(std::forward<KT>(key), std::forward<VT>(val));
  }

  std::vector<std::pair<std::string, std::string>> &props() { return props_; }

  const std::vector<std::pair<std::string, std::string>> &props() const {
    return props_;
  }

private:
  enum class AtomFlags : std::uint32_t {
    kAromatic = 0x1,
    kConjugated = 0x2,
    kRing = 0x4,
    kChiral = 0x8,
    kRightHanded = 0x10,
  };

  friend bool operator==(const AtomData &lhs, const AtomData &rhs) noexcept;

  const Element *element_;
  int implicit_hydrogens_;
  int formal_charge_;
  constants::Hybridization hyb_;
  AtomFlags flags_;
  double partial_charge_;
  const Isotope *isotope_;
  std::vector<std::pair<std::string, std::string>> props_;
};

inline bool operator==(const AtomData &lhs, const AtomData &rhs) noexcept {
  return lhs.element() == rhs.element()
         && lhs.hybridization() == rhs.hybridization()
         && lhs.flags_ == rhs.flags_
         && lhs.formal_charge() == rhs.formal_charge();
}

class BondData {
public:
  BondData(): BondData(constants::kSingleBond) { }

  explicit BondData(constants::BondOrder order)
      : order_(order), flags_(static_cast<BondFlags>(0)), length_(0) { }

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

  bool is_rotable() const {
    return !internal::check_flag(flags_,
                                 BondFlags::kConjugated | BondFlags::kRing);
  }

  bool is_ring_bond() const {
    return internal::check_flag(flags_, BondFlags::kRing);
  }

  void set_ring_bond(bool ring) {
    internal::update_flag(flags_, ring, BondFlags::kRing);
  }

  bool is_aromatic() const {
    return internal::check_flag(flags_, BondFlags::kAromatic);
  }

  void set_aromatic(bool aromatic) {
    internal::update_flag(flags_, aromatic, BondFlags::kAromatic);
  }

  bool is_conjugated() const {
    return internal::check_flag(flags_, BondFlags::kConjugated);
  }

  void set_conjugated(bool conj) {
    internal::update_flag(flags_, conj, BondFlags::kConjugated);
  }

  bool is_trans() const {
    return internal::check_flag(flags_, BondFlags::kEConfig);
  }

  void set_trans(bool trans) {
    internal::update_flag(flags_, trans, BondFlags::kEConfig);
  }

  void reset_flags() { flags_ = static_cast<BondFlags>(0); }

  /**
   * @brief Get the bond length.
   * @return The bond length of in angstroms \f$(\mathrm{Å})\f$. If the
   *         molecule has no 3D conformations, this will return 0.
   */
  double length() const { return length_; }

  /**
   * @brief Get the read-write reference to bond length.
   * @return The bond length of in angstroms \f$(\mathrm{Å})\f$. If the
   *         molecule has no 3D conformations, this will return 0.
   */
  double &length() { return length_; }

  const std::string *find_name() const { return internal::get_name(props_); }

  void set_name(std::string_view name) { internal::set_name(props_, name); }

  template <class KT, class VT>
  void add_prop(KT &&key, VT &&val) {
    props_.emplace_back(std::forward<KT>(key), std::forward<VT>(val));
  }

  std::vector<std::pair<std::string, std::string>> &props() { return props_; }

  const std::vector<std::pair<std::string, std::string>> &props() const {
    return props_;
  }

private:
  enum class BondFlags : std::uint32_t {
    kRing = 0x1,
    kAromatic = 0x2,
    kConjugated = 0x4,
    kEConfig = 0x8,
  };

  constants::BondOrder order_;
  BondFlags flags_;
  double length_;
  std::vector<std::pair<std::string, std::string>> props_;
};

class Molecule;

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

    using Neighbor = typename SubgraphType::ConstAdjRef;
    using neighbor_iterator = typename SubgraphType::adjacency_iterator;
    using const_neighbor_iterator =
        typename SubgraphType::const_adjacency_iterator;

    Substructure(const SubgraphType &sub): graph_(sub) { }

    Substructure(SubgraphType &&sub) noexcept: graph_(std::move(sub)) { }

    Substructure(const SubgraphType &sub, const std::string &name)
        : graph_(sub), name_(name) { }

    Substructure(SubgraphType &&sub, std::string &&name) noexcept
        : graph_(std::move(sub)), name_(std::move(name)) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    Substructure(const Substructure<other_const> &other)
        : graph_(other.graph_), name_(other.name_) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    Substructure(Substructure<other_const> &&other) noexcept
        : graph_(std::move(other.graph_)), name_(std::move(other.name_)) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    Substructure &operator=(const Substructure<other_const> &other) {
      graph_ = other.graph_;
      name_ = other.name_;
      return *this;
    }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    Substructure &operator=(Substructure<other_const> &&other) noexcept {
      graph_ = std::move(other.graph_);
      name_ = std::move(other.name_);
      return *this;
    }

    bool empty() const { return graph_.empty(); }
    int size() const { return graph_.size(); }
    int num_atoms() const { return graph_.num_nodes(); }

    void clear() {
      graph_.clear();
      name_.clear();
      id_ = 0;
      props_.clear();
    }

    void update(const std::vector<int> &atoms) { graph_.update(atoms); }
    void update(std::vector<int> &&atoms) noexcept {
      graph_.update(std::move(atoms));
    }

    void reserve(int n) { graph_.reserve(n); }

    void add_atom(int id) { graph_.add_node(id); }

    bool contains(int id) const { return graph_.contains(id); }
    bool contains(typename GraphType::ConstNodeRef atom) const {
      return graph_.contains(atom);
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
    void erase_atoms_of(UnaryPred &&pred) {
      graph_.erase_nodes_of(std::forward<UnaryPred>(pred));
    }

    iterator begin() { return graph_.begin(); }
    iterator end() { return graph_.end(); }

    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    const_iterator cbegin() const { return graph_.cbegin(); }
    const_iterator cend() const { return graph_.cend(); }

    const std::vector<int> &atom_ids() const { return graph_.node_ids(); }

    auto bonds() { return graph_.edges(); }
    auto bonds() const { return graph_.edges(); }

    int degree(int id) const { return graph_.degree(id); }

    neighbor_iterator find_neighbor(int src, int dst) {
      return graph_.find_adjacent(src, dst);
    }

    const_neighbor_iterator find_neighbor(int src, int dst) const {
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

    template <class KT, class VT>
    void add_prop(KT &&key, VT &&val) {
      props_.emplace_back(std::forward<KT>(key), std::forward<VT>(val));
    }

    std::vector<std::pair<std::string, std::string>> &props() { return props_; }

    const std::vector<std::pair<std::string, std::string>> &props() const {
      return props_;
    }

  private:
    friend Molecule;

    SubgraphType graph_;

    std::string name_;
    int id_;
    std::vector<std::pair<std::string, std::string>> props_;
  };

  template <class FT, bool is_const>
  class FindSubstructIter {
  public:
    using parent_type = const_if_t<is_const, FT>;

    using iterator_category = std::forward_iterator_tag;
    using value_type = const_if_t<is_const, Substructure<false>>;
    using reference = value_type &;
    using pointer = value_type *;
    using difference_type = int;

    using SubstructContainer = std::vector<Substructure<false>>;
    using ParentIterator =
        std::conditional_t<is_const, typename SubstructContainer::const_iterator,
                           typename SubstructContainer::iterator>;

    FindSubstructIter() = default;

    FindSubstructIter(parent_type &finder, ParentIterator it)
        : finder_(&finder), it_(finder.next(it)) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    FindSubstructIter(const FindSubstructIter<FT, other_const> &other)
        : finder_(&other.finder_), it_(other.it_) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    FindSubstructIter &
    operator=(const FindSubstructIter<FT, other_const> &other) {
      finder_ = other.finder_;
      it_ = other.it_;
      return *this;
    }

    FindSubstructIter &operator++() {
      it_ = finder_->next(++it_);
      return *this;
    }

    FindSubstructIter operator++(int) {
      auto tmp = *this;
      ++(*this);
      return tmp;
    }

    template <class OtherPred, bool other_const>
    bool
    operator==(const FindSubstructIter<OtherPred, other_const> &other) const {
      return it_ == other.it_;
    }

    template <class OtherPred, bool other_const>
    bool
    operator!=(const FindSubstructIter<OtherPred, other_const> &other) const {
      return !(*this == other);
    }

    reference operator*() const { return *it_; }

    pointer operator->() const { return &*it_; }

  private:
    template <class, bool>
    friend class FindSubstructIter;

    parent_type *finder_;
    ParentIterator it_;
  };

  template <class UnaryPred, bool is_const>
  class SubstructureFinder {
  public:
    using SubstructContainer = std::vector<Substructure<false>>;
    using parent_type = const_if_t<is_const, SubstructContainer>;

    using iterator = FindSubstructIter<SubstructureFinder, is_const>;
    using const_iterator = FindSubstructIter<SubstructureFinder, true>;

    SubstructureFinder(parent_type &substructs, UnaryPred &&pred)
        : substructs_(&substructs), pred_(std::forward<UnaryPred>(pred)) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    SubstructureFinder(const SubstructureFinder<UnaryPred, other_const> &other)
        : substructs_(other.substructs_), pred_(other.pred_) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    SubstructureFinder(
        SubstructureFinder<UnaryPred, other_const>
            &&other) noexcept(std::is_nothrow_move_constructible_v<UnaryPred>)
        : substructs_(other.substructs_), pred_(std::move(other.pred_)) { }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    SubstructureFinder &
    operator=(const SubstructureFinder<UnaryPred, other_const> &other) {
      substructs_ = other.substructs_;
      pred_ = other.pred_;
      return *this;
    }

    template <bool other_const,
              std::enable_if_t<is_const && !other_const, int> = 0>
    SubstructureFinder &
    operator=(SubstructureFinder<UnaryPred, other_const> &&other) noexcept(
        std::is_nothrow_move_constructible_v<UnaryPred>) {
      substructs_ = other.substructs_;
      pred_ = std::move(other.pred_);
      return *this;
    }

    iterator begin() { return iterator(*this, substructs_->begin()); }
    iterator end() { return iterator(*this, substructs_->end()); }

    const_iterator begin() const {
      return const_iterator(*this, substructs_->begin());
    }
    const_iterator end() const {
      return const_iterator(*this, substructs_->end());
    }

    const_iterator cbegin() const {
      return const_iterator(*this, substructs_->begin());
    }
    const_iterator cend() const {
      return const_iterator(*this, substructs_->end());
    }

  private:
    template <class, bool>
    friend class FindSubstructIter;

    typename SubstructContainer::iterator
    next(typename SubstructContainer::iterator begin) {
      return std::find_if(begin, substructs_->end(), pred_);
    }

    typename SubstructContainer::const_iterator
    next(typename SubstructContainer::const_iterator begin) const {
      return std::find_if(begin, substructs_->cend(), pred_);
    }

    parent_type *substructs_;
    UnaryPred pred_;
  };

  template <class UnaryPred>
  SubstructureFinder<UnaryPred, false>
  make_substructure_finder(std::vector<Substructure<false>> &substructs,
                           UnaryPred &&pred) {
    return { substructs, std::forward<UnaryPred>(pred) };
  }

  template <class UnaryPred>
  SubstructureFinder<UnaryPred, true>
  make_substructure_finder(const std::vector<Substructure<false>> &substructs,
                           UnaryPred &&pred) {
    return { substructs, std::forward<UnaryPred>(pred) };
  }
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
 *
 * If a new conformer is added, it must have same bond lengths with the existing
 * conformers (if any). If the added conformer is the first one, the bond
 * length will be calculated from its atomic coordinates.
 *
 * Even if the final conformer is erased, the bond lengths will **not** be
 * reset to the initial state.
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
  using bond_id_type = GraphType::edge_id_type;
  using bond_iterator = GraphType::edge_iterator;
  using const_bond_iterator = GraphType::const_edge_iterator;

  using Neighbor = GraphType::ConstAdjRef;
  using neighbor_iterator = GraphType::adjacency_iterator;
  using const_neighbor_iterator = GraphType::const_adjacency_iterator;

  friend class MoleculeMutator;

  /**
   * @brief Construct an empty Molecule object.
   */
  Molecule() noexcept = default;

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
   * @param bond_id Id of the bond to get.
   * @return A mutable view over bond \p bond_id of the molecule.
   * @note The returned reference is valid until the bond is erased from the
   *       molecule.
   */
  MutableBond bond(bond_id_type bond_id) { return graph_.edge(bond_id); }

  /**
   * @brief Get a bond of the molecule.
   * @param bond_id Id of the bond to get.
   * @return A read-only view over bond \p bond_id of the molecule.
   * @note The returned reference is valid until the bond is erased from the
   *       molecule.
   */
  Bond bond(bond_id_type bond_id) const { return graph_.edge(bond_id); }

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
  class MoleculeMutator mutator();

  void clear() noexcept;

  // TODO(jnooree): add_hydrogens
  // /**
  //  * @brief Add hydrogens to the molecule.
  //  */
  // void add_hydrogens();

  /**
   * @brief Erase all hydrogens from the molecule.
   */
  void erase_hydrogens();

  /**
   * @brief Check if the molecule has any 3D conformations.
   * @return `true` if the molecule has any 3D conformations, `false` otherwise.
   */
  bool is_3d() const { return !conformers_.empty(); }

  /**
   * @brief Get the atomic coordinates of ith conformer.
   *
   * @param i Index of the conformer.
   * @return Atomic coordinates of ith conformer.
   * @note If index is out of range, the behavior is undefined.
   */
  const MatrixX3d &conf(int i = 0) const { return conformers_[i]; }

  /**
   * @brief Get total number of conformers.
   * @return The number of conformers of the molecule.
   */
  int num_conf() const { return static_cast<int>(conformers_.size()); }

  /**
   * @brief Get all atomic coordinates of the conformers.
   * @return Atomic coordinates of the conformers of the molecule.
   */
  const std::vector<MatrixX3d> &all_conf() const { return conformers_; }

  /**
   * @brief Add a new conformer to the molecule.
   *
   * @param pos Atomic coordinates of the new conformer.
   * @return The index of the added conformer.
   * @note If number of rows of \p pos does not match the number of atoms in the
   *       molecule, the behavior is undefined.
   *
   * If no conformers exist, the bond lengths will be calculated from the
   * positions of the atoms in `pos`. Otherwise, the bond lengths will be left
   * unmodified, so it is the caller's responsibility to ensure that the bond
   * lengths are consistent with the new conformer.
   */
  int add_conf(const MatrixX3d &pos);

  /**
   * @brief Add a new conformer to the molecule.
   *
   * @param pos Atomic coordinates of the new conformer.
   * @return The index of the added conformer.
   * @note If number of rows of \p pos does not match the number of atoms in the
   *       molecule, the behavior is undefined.

   * If no conformers exist, the bond lengths will be calculated from the
   * positions of the atoms in `pos`. Otherwise, the bond lengths will be left
   * unmodified, so it is the caller's responsibility to ensure that the bond
   * lengths are consistent with the new conformer.
   */
  int add_conf(MatrixX3d &&pos) noexcept;

  /**
   * @brief Erase a conformer from the molecule.
   *
   * @param idx Index of the conformer to erase.
   * @note The behavior is undefined if the conformer index is out of range.
   */
  void erase_conf(int idx) { conformers_.erase(conformers_.begin() + idx); }

  /**
   * @brief Transform the molecule with the given affine transformation.
   * @param trans The affine transformation to apply.
   */
  void transform(const Affine3d &trans) {
    for (MatrixX3d &m: conformers_) {
      m.transpose() = trans * m.transpose();
    }
  }

  /**
   * @brief Transform a conformer of the molecule with the given affine
   *        transformation.
   * @param i The index of the conformer to transform.
   * @param trans The affine transformation to apply.
   * @note The behavior is undefined if the conformer index is out of range.
   */
  void transform(int i, const Affine3d &trans) {
    MatrixX3d &m = conformers_[i];
    m.transpose() = trans * m.transpose();
  }

  /**
   * @brief Rotate a bond.
   * @param ref_atom Index of the reference atom.
   * @param pivot_atom Index of the pivot atom.
   * @param angle Angle to rotate (in degrees).
   * @return `true` if the rotation was applied, `false` if the rotation was
   *         not applied (e.g. if the reference atom and the pivot atom are not
   *         connected by a bond, the bond is not rotatable, etc.).
   *
   * The rotation is applied to all conformers of the molecule.
   *
   * The part of the reference atom is fixed, and the part of the pivot atom
   * will be rotated about the reference atom -> pivot atom axis. Positive angle
   * means counter-clockwise rotation (as in the right-hand rule).
   */
  bool rotate_bond(int ref_atom, int pivot_atom, double angle);

  /**
   * @brief Rotate a bond.
   * @param bid The id of bond to rotate.
   * @param angle Angle to rotate (in degrees).
   * @return `true` if the rotation was applied, `false` if the rotation was
   *         not applied (e.g. the bond is not rotatable, etc.).
   *
   * The rotation is applied to all conformers of the molecule.
   *
   * The source atom of the bond is fixed, and the destination atom will be
   * rotated about the source atom -> destination atom axis. Positive angle
   * means counter-clockwise rotation (as in the right-hand rule).
   */
  bool rotate_bond(bond_id_type bid, double angle);

  /**
   * @brief Rotate a bond of a conformer.
   * @param i The index of the conformer to transform.
   * @param ref_atom Index of the pivot atom.
   * @param pivot_atom Index of the pivot atom.
   * @param angle Angle to rotate (in degrees).
   * @return `true` if the rotation was applied, `false` if the rotation was
   *         not applied (e.g. if the pivot atom and the pivot atom are not
   *         connected by a bond, the bond is not rotatable, etc.).
   *
   * The part of the pivot atom is fixed, and the part of the pivot atom will be
   * rotated about the pivot atom -> pivot atom axis. Positive angle means
   * counter-clockwise rotation (as in the right-hand rule).
   */
  bool rotate_bond(int i, int ref_atom, int pivot_atom, double angle);

  /**
   * @brief Rotate a bond of a conformer.
   * @param i The index of the conformer to transform.
   * @param bid The id of bond to rotate.
   * @param angle Angle to rotate (in degrees).
   * @return `true` if the rotation was applied, `false` if the rotation was
   *         not applied (e.g. the bond is not rotatable, etc.).
   *
   * The source atom of the bond is fixed, and the destination atom will be
   * rotated about the source atom -> destination atom axis. Positive angle
   * means counter-clockwise rotation (as in the right-hand rule).
   */
  bool rotate_bond(int i, bond_id_type bid, double angle);

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @return The new substructure.
   */
  Substructure substructure(const std::vector<int> &nodes) {
    return Subgraph(graph_, nodes);
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @return The new substructure.
   */
  Substructure substructure(std::vector<int> &&nodes) noexcept {
    return Subgraph(graph_, std::move(nodes));
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @return The new substructure.
   */
  ConstSubstructure substructure(const std::vector<int> &nodes) const {
    return Subgraph(graph_, nodes);
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @return The new substructure.
   */
  ConstSubstructure substructure(std::vector<int> &&nodes) const noexcept {
    return Subgraph(graph_, std::move(nodes));
  }

  /**
   * @brief Get a substructure of the molecule.
   *
   * @param i Index of the substructure to get.
   * @return A reference to the substructure.
   */
  Substructure &get_substructure(int i) { return substructs_[i]; }

  /**
   * @brief Get a substructure of the molecule.
   *
   * @param i Index of the substructure to get.
   * @return A const reference to the substructure.
   */
  const Substructure &get_substructure(int i) const { return substructs_[i]; }

  /**
   * @brief Store and return a new substurcture of the molecule.
   *
   * @return The new substructure.
   */
  Substructure &add_substructure() {
    return substructs_.emplace_back(Subgraph(graph_));
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @param idxs Indices of atoms in the substructure.
   * @return The new substructure.
   */
  Substructure &add_substructure(const std::vector<int> &idxs) {
    return substructs_.emplace_back(Subgraph(graph_, idxs));
  }

  /**
   * @brief Create and return a substurcture of the molecule.
   *
   * @param idxs Indices of atoms in the substructure.
   * @return The new substructure.
   */
  Substructure &add_substructure(std::vector<int> &&idxs) noexcept {
    return substructs_.emplace_back(Subgraph(graph_, std::move(idxs)));
  }

  /**
   * @brief Erase a substructure of the molecule.
   *
   * @param i Index of the substructure to erase.
   * @note If the index is out of range, the behavior is undefined.
   * @note Time complexity: \f$O(N)\f$, where \f$N\f$ is number of
   *       substructures.
   */
  void erase_substructure(int i) { substructs_.erase(substructs_.begin() + i); }

  /**
   * @brief Erase substructures of the molecule.
   *
   * @tparam UnaryPred Unary predicate type.
   * @param pred Unary predicate that accepts a substructure and returns true if
   *        the substructure should be erased.
   * @note Time complexity: \f$O(N)\f$, where \f$N\f$ is number of
   *       substructures.
   */
  template <class UnaryPred>
  void erase_substructures(UnaryPred &&pred) {
    erase_if(substructs_, std::forward<UnaryPred>(pred));
  }

  /**
   * @brief Erase all substructures of the molecule.
   */
  void clear_substructures() noexcept { substructs_.clear(); }

  /**
   * @brief Check if the molecule has substructures.
   *
   * @return `true` if the molecule has substructures, `false` otherwise.
   */
  bool has_substructures() const { return !substructs_.empty(); }

  /**
   * @brief Get the number of substructures.
   *
   * @return The number of substructures managed by the molecule.
   */
  int num_substructures() const { return static_cast<int>(substructs_.size()); }

  /**
   * @brief Get the substructures.
   *
   * @return A const reference to all substructures.
   */
  const std::vector<Substructure> &substructures() const { return substructs_; }

  /**
   * @brief Find substructures with given id.
   *
   * @return A ranged view of substructures with given id.
   */
  auto find_substructures(int id) {
    return internal::make_substructure_finder(
        substructs_, [id](const Substructure &sub) { return sub.id() == id; });
  }

  /**
   * @brief Find substructures with given name.
   *
   * @return A ranged view of substructures with given name.
   * @warning The owner of name must ensure that the name is valid during the
   *          lifetime of the returned view.
   */
  auto find_substructures(std::string_view name) {
    return internal::make_substructure_finder(substructs_,
                                              [name](const Substructure &sub) {
                                                return sub.name() == name;
                                              });
  }

  /**
   * @brief Find substructures with given id.
   *
   * @return A ranged constant view of substructures with given id.
   */
  auto find_substructures(int id) const {
    return internal::make_substructure_finder(
        substructs_, [id](const Substructure &sub) { return sub.id() == id; });
  }

  /**
   * @brief Find substructures with given name.
   *
   * @return A ranged constant view of substructures with given name.
   * @warning The owner of name must ensure that the name is valid during the
   *          lifetime of the returned view.
   */
  auto find_substructures(std::string_view name) const {
    return internal::make_substructure_finder(substructs_,
                                              [name](const Substructure &sub) {
                                                return sub.name() == name;
                                              });
  }

  /**
   * @brief Find substructures with given predicate.
   *
   * @return A ranged view of substructures which satisfy the predicate.
   */
  template <class UnaryPred>
  auto find_substructures_if(UnaryPred &&pred) {
    return internal::make_substructure_finder(substructs_,
                                              std::forward<UnaryPred>(pred));
  }

  /**
   * @brief Find substructures with given predicate.
   *
   * @return A ranged view of substructures which satisfy the predicate.
   */
  template <class UnaryPred>
  auto find_substructures_if(UnaryPred &&pred) const {
    return internal::make_substructure_finder(substructs_,
                                              std::forward<UnaryPred>(pred));
  }

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

    for (MatrixX3d &conf: conformers_) {
      conf.conservativeResize(size(), Eigen::NoChange);
    }
  }

  template <class KT, class VT>
  void add_prop(KT &&key, VT &&val) {
    props_.emplace_back(std::forward<KT>(key), std::forward<VT>(val));
  }

  std::vector<std::pair<std::string, std::string>> &props() { return props_; }

  const std::vector<std::pair<std::string, std::string>> &props() const {
    return props_;
  }

private:
  Molecule(GraphType &&graph, std::vector<MatrixX3d> &&conformers) noexcept
      : graph_(std::move(graph)), conformers_(std::move(conformers)) { }

  bool rotate_bond_common(int i, Bond b, int ref_atom, int pivot_atom,
                          double angle);

  GraphType graph_;
  std::vector<MatrixX3d> conformers_;
  std::string name_;
  std::vector<std::pair<std::string, std::string>> props_;

  std::vector<Substructure> substructs_;

  std::vector<std::vector<int>> ring_groups_;
  int num_fragments_;
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
  MoleculeMutator(Molecule &mol): mol_(&mol), init_num_atoms_(mol.size()) { }

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
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @param bond The data of the bond to add.
   * @return If added, pair of iterator to the added bond, and `true`. If the
   *         bond already exists, pair of iterator to the existing bond, and
   *         `false`.
   * @note The behavior is undefined if any of the atom indices is out of range,
   *       or if src == dst.
   *
   * If the bond is added and the molecule has at least one conformer, the bond
   * length will be calculated from the positions of the atoms in the first
   * conformer.
   */
  std::pair<Molecule::bond_iterator, bool> add_bond(int src, int dst,
                                                    const BondData &bond);

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @param bond The data of the bond to add.
   * @return If added, pair of iterator to the added bond, and `true`. If the
   *         bond already exists, pair of iterator to the existing bond, and
   *         `false`.
   * @note The behavior is undefined if any of the atom indices is out of range,
   *       or if src == dst.
   *
   * If the bond is added and the molecule has at least one conformer, the bond
   * length will be calculated from the positions of the atoms in the first
   * conformer.
   */
  std::pair<Molecule::bond_iterator, bool> add_bond(int src, int dst,
                                                    BondData &&bond) noexcept;

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond.
   * @param dst Index of the destination atom of the bond.
   * @param bond The bond to copy the data from.
   * @return If added, pair of iterator to the added bond, and `true`. If the
   *         bond already exists, pair of iterator to the existing bond, and
   *         `false`.
   * @note The behavior is undefined if any of the atom indices is out of range,
   *       or if src == dst.
   *
   * If the bond is added and the molecule has at least one conformer, the bond
   * length will be calculated from the positions of the atoms in the first
   * conformer.
   */
  std::pair<Molecule::bond_iterator, bool> add_bond(int src, int dst,
                                                    Molecule::Bond bond) {
    return add_bond(src, dst, bond.data());
  }

  /**
   * @brief Mark a bond to be erased.
   * @param src Index of the source atom of the bond, after all additions.
   * @param dst Index of the destination atom of the bond, after all additions.
   * @note The behavior is undefined if any of the atom indices is out of range,
   *       at the momenet of `finalize()` call. This is a no-op if the bond does
   *       not exist, also at the moment of `finalize()` call.
   */
  void mark_bond_erase(int src, int dst);

  /**
   * @brief Cancel all pending atom and bond removals.
   *
   * This effectively resets the mutator to the state after construction.
   */
  void discard_erasure() noexcept;

  /**
   * @brief Finalize the mutation.
   * @note The mutator internally calls discard_erasure() after applying
   *       changes. Thus, it's a no-op to call this method multiple times.
   * @sa Molecule::sanitize()
   *
   * This will effectively call Molecule::update_topology().
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
  int init_num_atoms_;

  std::vector<int> erased_atoms_;
  std::vector<std::pair<int, int>> erased_bonds_;
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
  effective_element_or_element(Molecule::Atom atom) noexcept;

  extern int sum_bond_order(Molecule::Atom atom, bool aromatic_correct);

  extern constants::Hybridization from_degree(int total_degree,
                                              int nb_electrons);

  extern int count_pi_e(Molecule::Atom atom, int total_valence);
}  // namespace internal

/**
 * @brief Get the number of all neighbors of an atom.
 * @param atom An atom.
 * @return Number of all neighbors of the atom, including implicit hydrogens.
 */
extern inline int all_neighbors(Molecule::Atom atom) {
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
 */
inline int sum_bond_order(Molecule::Atom atom) {
  return internal::sum_bond_order(atom, true);
}

/**
 * @brief Get "effective" element of the atom.
 * @param atom An atom.
 * @return "Effective" element of the atom: the returned element has atomic
 *         number of (original atomic number) - (formal charge). If the
 *         resulting atomic number is out of range, returns nullptr.
 */
extern const Element *effective_element(Molecule::Atom atom);

/* Important algorithms */

using Rings = std::vector<std::vector<int>>;

/**
 * @brief Find all elementary cycles in the molecular graph.
 * @param mol A molecule.
 * @return A pair of (all elementary cycles, success). If success is `false`,
 *         the vector is in an unspecified state. This will fail if and only if
 *         any atom is a member of more than 100 elementary cycles.
 *
 * This is based on the algorithm described in the following paper:
 *    Hanser, Th. *et al.* *J. Chem. Inf. Comput. Sci.* **1996**, *36* (6),
 *    1146-1152. DOI: [10.1021/ci960322f](https://doi.org/10.1021/ci960322f)
 *
 * The time complexity of this function is inherently exponential, but it is
 * expected to run in a reasonable time (\f$\sim\mathcal{O}(V^2)\f$) for most
 * molecules in practice.
 */
extern std::pair<Rings, bool> find_all_rings(const Molecule &mol);

namespace internal {
  struct FindRingsCommonData;
}  // namespace internal

/**
 * @brief Wrapper class of the common routines of find_sssr() and
 *        find_relevant_rings().
 * @sa nuri::find_relevant_rings(), nuri::find_sssr()
 *
 * Formally, SSSR (smallest set of smallest rings) is a *minimum cycle basis*
 * of the molecular graph. As discussed in many literatures, there is no unique
 * SSSR for a given molecular graph (even for simple molecules such as
 * 2-oxabicyclo[2.2.2]octane), and the SSSR is often counter-intuitive. For
 * example, the SSSR of cubane (although unique, due to symmetry reasons)
 * contains only five rings, which is not most chemists would expect.
 *
 * On the other hand, union of all SSSRs, sometimes called the *relevant
 * rings* in the literatures, is unique for a given molecule, and is the "all
 * smallest rings" of the molecule, chemically speaking. It is more appropriate
 * for most applications than SSSR.
 *
 * We provide two functions along with this class to find the relevant rings and
 * SSSR, respectively. If both are needed, it is recommended to construct this
 * class first, and call find_relevant_rings() and find_sssr() member functions
 * instead of calling the free functions directly.
 *
 * This is based on the algorithm described in the following paper:
 *    Vismara, P. *Electron. J. Comb.* **1997**, *4* (1), R9.
 *    DOI: [10.37236/1294](https://doi.org/10.37236/1294)
 *
 * Time complexity: theoretically \f$\mathcal{O}(\nu E^3)\f$, where \f$\nu =
 * \mathcal{O}(E)\f$ is size of SSSR. For most molecules, however, this is
 * \f$\mathcal{O}(V^3)\f$.
 */
class RingSetsFinder {
public:
  /**
   * @brief Construct a new Rings Finder object.
   * @param mol A molecule.
   */
  explicit RingSetsFinder(const Molecule &mol);

  RingSetsFinder(const RingSetsFinder &) = delete;
  RingSetsFinder &operator=(const RingSetsFinder &) = delete;
  RingSetsFinder(RingSetsFinder &&) noexcept;
  RingSetsFinder &operator=(RingSetsFinder &&) noexcept;

  ~RingSetsFinder() noexcept;

  /**
   * @brief Find the relevant rings of the molecule.
   * @return The relevant rings of the molecule.
   * @sa nuri::find_relevant_rings()
   */
  Rings find_relevant_rings() const;

  /**
   * @brief Find the SSSR of the molecule.
   * @return The smallest set of smallest rings (SSSR) of the molecule.
   * @sa nuri::find_sssr()
   * @note This function does not guarantee that the returned set is unique, nor
   * that the result is reproducible even for the same molecule.
   */
  Rings find_sssr() const;

private:
  const Molecule *mol_;
  std::unique_ptr<internal::FindRingsCommonData> data_;
};

/**
 * @brief Find union of the all SSSRs in the molecular graph.
 * @param mol A molecule.
 * @return Union of the all SSSRs in the molecular graph.
 * @sa find_sssr(), nuri::RingSetsFinder::find_relevant_rings()
 *
 * This is a convenience wrapper of the
 * nuri::RingSetsFinder::find_relevant_rings() member function.
 *
 * @note If both relevant rings and SSSR are needed, it is recommended to use
 * the nuri::RingSetsFinder class instead of the free functions.
 */
inline Rings find_relevant_rings(const Molecule &mol) {
  return RingSetsFinder(mol).find_relevant_rings();
}

/**
 * @brief Find a smallest set of smallest rings (SSSR) of the molecular graph.
 * @param mol A molecule.
 * @return *A* smallest set of smallest rings (SSSR) of the molecular graph.
 * @sa find_relevant_rings(), nuri::RingSetsFinder::find_sssr()
 * @note This function does not guarantee that the returned set is unique, nor
 *       that the result is reproducible even for the same molecule.
 *
 * This is a convenience wrapper of the nuri::RingSetsFinder::find_sssr() member
 * function.
 *
 * @note If both relevant rings and SSSR are needed, it is recommended to use
 * the nuri::RingSetsFinder class instead of the free functions.
 */
inline Rings find_sssr(const Molecule &mol) {
  return RingSetsFinder(mol).find_sssr();
}
}  // namespace nuri

#endif /* NURI_CORE_MOLECULE_H_ */
