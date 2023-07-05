//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
/** @file */
#ifndef NURI_CORE_MOLECULE_H_
#define NURI_CORE_MOLECULE_H_

#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>

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
    internal::set_flag_if(flags_, is_aromatic,
                          AtomFlags::kConjugated | AtomFlags::kRing);
  }

  bool is_aromatic() const {
    return internal::check_flag(flags_, AtomFlags::kAromatic);
  }

  void set_conjugated(bool is_conjugated) {
    internal::update_flag(flags_, is_conjugated, AtomFlags::kConjugated);
    internal::unset_flag_if(flags_, !is_conjugated, AtomFlags::kAromatic);
  }

  bool is_conjugated() const {
    return internal::check_flag(flags_, AtomFlags::kConjugated);
  }

  void set_ring_atom(bool is_ring_atom) {
    internal::update_flag(flags_, is_ring_atom, AtomFlags::kRing);
    internal::unset_flag_if(flags_, !is_ring_atom, AtomFlags::kAromatic);
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

private:
  enum class AtomFlags : uint32_t {
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
    return internal::check_flag(flags_, BondFlags::kRotable);
  }

  void set_rotable(bool rotable) {
    internal::update_flag(flags_, rotable, BondFlags::kRotable);
  }

  bool is_ring_bond() const {
    return internal::check_flag(flags_, BondFlags::kRing);
  }

  void set_ring_bond(bool ring) {
    internal::update_flag(flags_, ring, BondFlags::kRing);
    internal::unset_flag_if(flags_, !ring, BondFlags::kAromatic);
  }

  bool is_aromatic() const {
    return internal::check_flag(flags_, BondFlags::kAromatic);
  }

  void set_aromatic(bool aromatic) {
    internal::update_flag(flags_, aromatic, BondFlags::kAromatic);
    internal::set_flag_if(flags_, aromatic,
                          BondFlags::kRing | BondFlags::kConjugated);
  }

  bool is_conjugated() const {
    return internal::check_flag(flags_, BondFlags::kConjugated);
  }

  void set_conjugated(bool conj) {
    internal::update_flag(flags_, conj, BondFlags::kConjugated);
    internal::unset_flag_if(flags_, !conj, BondFlags::kAromatic);
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

private:
  enum class BondFlags : uint32_t {
    kRotable = 0x1,
    kRing = 0x2,
    kAromatic = 0x4,
    kConjugated = 0x8,
    kEConfig = 0x10,
  };

  constants::BondOrder order_;
  BondFlags flags_;
  double length_;
};

/**
 * @brief Read-only molecule class.
 *
 * The two only allowed mutable operation on the molecule is:
 *    - modifing conformers, and
 *    - adding/removing hydrogens.
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

  using Atom = GraphType::ConstNodeRef;
  using const_iterator = GraphType::const_iterator;
  using const_atom_iterator = const_iterator;

  using Bond = GraphType::ConstEdgeRef;
  using bond_id_type = GraphType::edge_id_type;
  using const_bond_iterator = GraphType::const_edge_iterator;

  using Neighbor = GraphType::ConstAdjRef;
  using const_neighbor_iterator = GraphType::const_adjacency_iterator;

  friend class MoleculeMutator;

  /**
   * @brief Construct an empty Molecule object.
   */
  Molecule() noexcept: circuit_rank_(0) { }

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
  const_iterator begin() const { return graph_.begin(); }
  /**
   * @brief The begin iterator of the molecule over atoms.
   * @sa begin()
   */
  const_iterator atom_begin() const { return begin(); }

  /**
   * @brief The past-the-end iterator of the molecule over atoms.
   */
  const_iterator end() const { return graph_.end(); }
  /**
   * @brief The past-the-end iterator of the molecule over atoms.
   */
  const_iterator atom_end() const { return end(); }

  /**
   * @brief Check if the molecule has any bonds.
   */
  bool bond_empty() const { return graph_.empty(); }

  /**
   * @brief Get the number of bonds in the molecule.
   */
  int num_bonds() const { return graph_.num_edges(); }

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
  const_bond_iterator find_bond(int src, int dst) const {
    return graph_.find_edge(src, dst);
  }

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
  const_bond_iterator bond_begin() const { return graph_.edge_begin(); }
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
  const_neighbor_iterator find_neighbor(int src, int dst) const {
    return graph_.find_adjacent(src, dst);
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
  const_neighbor_iterator neighbor_end(int atom_idx) const {
    return graph_.adj_end(atom_idx);
  }

  /**
   * @brief Get a MoleculeMutator object associated with the molecule.
   * @param sanitize Passed to the constructor of MoleculeMutator.
   * @return The MoleculeMutator object to update this molecule.
   * @sa MoleculeMutator
   */
  class MoleculeMutator mutator(bool sanitize = true);

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
   * @brief Fix all chemical errors in the molecule.
   * @param use_conformer If non-negative, use bond angles and lengths of the
   *        specified conformer to fix chemical errors. Otherwise, only the
   *        graph is used. *Unimplemented, currently a no-op.*
   * @return `true` if the molecule is valid after sanitization, `false` if the
   *         molecule is still invalid after sanitization.
   * @note This method has high overhead, and is not intended to be used in
   *       downstream code. It is mainly used from MoleculeMutator class.
   * @warning All molecule-related methods/functions assume that the molecule
   *          is valid. If the molecule is not sanitized after the last
   *          modification, all library code might return incorrect results. It
   *          is the caller's responsibility to ensure that the molecule is
   *          chemically valid before passing it to any library function, if the
   *          molecule was not sanitized after any modification.
   */
  bool sanitize(int use_conformer = -1);

  /**
   * @brief Get size of SSSR.
   * @return The number of rings in the smallest set of smallest rings.
   */
  int num_sssr() const { return circuit_rank_; }

  /**
   * @brief Set size of SSSR.
   * @warning If the size is incorrect, all library code might return incorrect
   *          results. It is the caller's responsibility to ensure that the size
   *          is correct.
   */
  void set_num_sssr(int n) { circuit_rank_ = n; }

  /**
   * @brief Observe the previous result of sanitize() call.
   * @return `true` if the last call to sanitize() resulted in a valid molecule,
   *         `false` otherwise.
   * @note Calling this method before calling sanitize() is meaningless.
   * @sa sanitize()
   */
  bool was_valid() const { return was_valid_; }

private:
  Molecule(GraphType &&graph, std::vector<MatrixX3d> &&conformers) noexcept
    : graph_(std::move(graph)), conformers_(std::move(conformers)) { }

  GraphType::NodeRef mutable_atom(int atom_idx) {
    return graph_.node(atom_idx);
  }

  GraphType::EdgeRef mutable_bond(bond_id_type bond_id) {
    return graph_.edge(bond_id);
  }

  GraphType::edge_iterator find_mutable_bond(int src, int dst) {
    return graph_.find_edge(src, dst);
  }

  bool rotate_bond_common(int i, Bond b, int ref_atom, int pivot_atom,
                          double angle);

  GraphType graph_;
  std::vector<MatrixX3d> conformers_;
  std::string name_;
  int circuit_rank_;
  bool was_valid_;
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
   * @brief Construct a new MoleculeMutator object
   *
   * @param mol The molecule to mutate.
   * @param sanitize If `true`, sanitize the molecule at finalize() call.
   * @param sanitize_with If non-negative, use bond angles and lengths of the
   *        specified conformer to fix chemical errors. Otherwise, only the
   *        graph is used. *Unimplemented, currently a no-op.*
   * @sa sanitize(), finalize()
   */
  MoleculeMutator(Molecule &mol, bool sanitize = true, int sanitize_with = -1)
    : mol_(&mol), conformer_idx_(sanitize_with), sanitize_(sanitize) { }

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
   * @param atom An atom to add.
   * @return The index of the added atom.
   */
  int add_atom(const Molecule::Atom &atom) { return add_atom(atom.data()); }

  /**
   * @brief Erase an atom from the molecule.
   * @param atom_idx Index of the atom to erase, after all atom additions.
   * @note The behavior is undefined if the atom index is out of range at the
   *       moment of calling `finalize()`.
   */
  void erase_atom(int atom_idx) { erased_atoms_.push_back(atom_idx); }

  /**
   * @brief Get data of an atom.
   * @param atom_idx Index of the atom after all atom additions, but before any
   *                 erasures.
   * @note The behavior is undefined if the atom index is out of range.
   */
  AtomData &atom_data(int atom_idx) {
    return mol().graph_.node(atom_idx).data();
  }

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond, after all atom additions.
   * @param dst Index of the destination atom of the bond, after all atom
   *            additions.
   * @param bond The data of the bond to add.
   * @return `true` if the bond was added, `false` if the bond already exists.
   * @note The behavior is undefined if any of the atom indices is out of range.
   */
  bool add_bond(int src, int dst, const BondData &bond);

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond, after all atom additions.
   * @param dst Index of the destination atom of the bond, after all atom
   *            additions.
   * @param bond The bond to add.
   * @return `true` if the bond was added, `false` if the bond already exists.
   * @note The behavior is undefined if any of the atom indices is out of range.
   */
  bool add_bond(int src, int dst, const Molecule::Bond &bond) {
    return add_bond(src, dst, bond.data());
  }

  /**
   * @brief Erase a bond from the molecule.
   * @param src Index of the source atom of the bond, after all atom additions.
   * @param dst Index of the destination atom of the bond, after all atom
   *            additions.
   * @note The behavior is undefined if any of the atom indices is out of range,
   *       **at the momenet of `finalize()` call**. This is a no-op if the bond
   *       does not exist, also **at the moment of `finalize()` call**.
   */
  void erase_bond(int src, int dst);

  /**
   * @brief Get data of a bond.
   * @param src Index of the source atom of the bond
   * @param dst Index of the destination atom of the bond
   * @return Pointer to the bond data, or `nullptr` if the bond does not exist.
   * @note The behavior is undefined if the atom index is out of range at the
   *       moment of calling this method.
   */
  BondData *bond_data(int src, int dst);

  bool &sanitize() { return sanitize_; }

  bool sanitize() const { return sanitize_; }

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
   * This will effectively call Molecule::sanitize() unless sanitize() == false.
   */
  void finalize() noexcept;

private:
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

  Molecule *mol_;

  std::vector<int> erased_atoms_;
  std::vector<std::pair<int, int>> erased_bonds_;

  int conformer_idx_;
  bool sanitize_;
};

/* Out-of-line definitions for molecule */

template <class Iterator, class>
Molecule::Molecule(Iterator begin, Iterator end): Molecule() {
  MoleculeMutator m = mutator(false);
  for (auto it = begin; it != end; ++it) {
    m.add_atom(*it);
  }
}

inline MoleculeMutator Molecule::mutator(bool sanitize) {
  return MoleculeMutator(*this, sanitize);
}

/* Utility functions */

/**
 * @brief Get the number of all neighbors of an atom.
 * @param atom An atom.
 * @return Number of all neighbors of the atom, including implicit hydrogens.
 */
extern inline int all_neighbors(Molecule::Atom atom) {
  return atom.degree() + atom.data().implicit_hydrogens();
}

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
extern int sum_bond_order(Molecule::Atom atom);

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
