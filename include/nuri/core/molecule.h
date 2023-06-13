//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
/** @file */
#ifndef NURI_CORE_MOLECULE_H_
#define NURI_CORE_MOLECULE_H_

#include <iterator>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/base/optimization.h>
#include <absl/container/flat_hash_set.h>

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
    kUnkHyb = 0,
    kTerminal = 1,
    kSP = 2,
    kSP2 = 3,
    kSP3 = 4,
    kSP3D = 5,
    kSP3D2 = 6,
    kOther = 7,
  };

  /**
   * @brief The bond order of a bond object.
   */
  enum BondOrder {
    kNoBond = 0,
    kSingleBond = 1,
    kDoubleBond = 2,
    kTripleBond = 3,
    kQuadrupleBond = 4,
    kAromaticBond = 5,
    kConjugatedBond = 6,
  };
}  // namespace constants

class AtomData {
public:
  /**
   * @brief Creates a dummy atom with unknown hybridization.
   */
  AtomData(): AtomData(PeriodicTable::get()[0]) { }

  AtomData(const Element &element,
           constants::Hybridization hyb = constants::kUnkHyb,
           int formal_charge = 0, double partial_charge = 0.0,
           int mass_number = -1, bool is_aromatic = false,
           bool is_in_ring = false, bool is_chiral = false,
           bool is_right_handed = false);

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

  void set_aromatic(bool is_aromatic) {
    internal::update_flag(flags_, is_aromatic, kAromaticAtom);
  }

  bool is_aromatic() const {
    return internal::check_flag(flags_, kAromaticAtom);
  }

  void set_ring_atom(bool is_ring_atom) {
    internal::update_flag(flags_, is_ring_atom, kRingAtom);
  }

  bool is_ring_atom() const { return internal::check_flag(flags_, kRingAtom); }

  void set_chiral(bool is_chiral) {
    internal::update_flag(flags_, is_chiral, kChiralAtom);
  }

  bool is_chiral() const { return internal::check_flag(flags_, kChiralAtom); }

  void set_right_handed(bool is_right_handed) {
    internal::update_flag(flags_, is_right_handed, kRightHandedAtom);
  }

  /**
   * @brief Get handedness of a chiral atom.
   *
   * @pre is_chiral() == `true`, otherwise return value would be meaningless.
   * @return Whether the chiral atom is "right-handed," i.e., `true` for (R)
   *         and `false` for (S).
   */
  bool is_right_handed() const {
    return internal::check_flag(flags_, kRightHandedAtom);
  }

  void set_partial_charge(double charge) { partial_charge_ = charge; }

  double partial_charge() const { return partial_charge_; }

  void set_formal_charge(int charge) { formal_charge_ = charge; }

  int formal_charge() const { return formal_charge_; }

private:
  enum Flags {
    kAromaticAtom = 0x1,
    kRingAtom = 0x2,
    kChiralAtom = 0x4,
    kRightHandedAtom = 0x8,
  };

  friend bool operator==(const AtomData &lhs, const AtomData &rhs) noexcept;

  const Element *element_;
  const Isotope *isotope_;
  constants::Hybridization hyb_;
  uint32_t flags_;
  int formal_charge_;
  double partial_charge_;
};

inline bool operator==(const AtomData &lhs, const AtomData &rhs) noexcept {
  return lhs.element() == rhs.element()
         && lhs.hybridization() == rhs.hybridization()
         && lhs.flags_ == rhs.flags_
         && lhs.formal_charge() == rhs.formal_charge();
}

class BondData {
public:
  explicit BondData(constants::BondOrder order)
    : order_(order), flags_(0), length_(0) { }

  /**
   * @brief Get the bond order of the bond.
   */
  constants::BondOrder order() const { return order_; }

  /**
   * @brief Get the read-write reference to bond order.
   */
  constants::BondOrder &order() { return order_; }

  bool is_rotable() const { return internal::check_flag(flags_, kRotableBond); }

  void set_rotable(bool rotable) {
    internal::update_flag(flags_, rotable, kRotableBond);
  }

  bool is_ring_bond() const { return internal::check_flag(flags_, kRingBond); }

  void set_ring_bond(bool ring) {
    internal::update_flag(flags_, ring, kRingBond);
  }

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
  enum Flags {
    kRotableBond = 0x1,
    kRingBond = 0x2,
  };

  constants::BondOrder order_;
  uint32_t flags_;
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
   * @brief The begin iterator of the molecule over bonds.
   */
  const_bond_iterator bond_begin() const { return graph_.edge_begin(); }
  /**
   * @brief The past-the-end iterator of the molecule over bonds.
   */
  const_bond_iterator bond_end() const { return graph_.edge_end(); }

  /**
   * @brief Get the number of bonds of the atom.
   * @param atom_idx Index of the atom.
   * @return Valence of the atom.
   * @note If the atom index is out of range, the behavior is undefined.
   */
  int valence(int atom_idx) const { return graph_.degree(atom_idx); }

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
   *
   * @return The MoleculeMutator object to update this molecule.
   */
  class MoleculeMutator mutator();

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

private:
  bool rotate_bond_common(int i, Bond b, int ref_atom, int pivot_atom,
                          double angle);

  Molecule(GraphType &&graph, std::vector<MatrixX3d> &&conformers) noexcept
    : graph_(std::move(graph)), conformers_(std::move(conformers)) { }

  GraphType graph_;
  std::vector<MatrixX3d> conformers_;
};

/**
 * @brief A class to mutate a molecule.
 *
 * The order of the mutations does not effect the result. The mutations are
 * applied in the following order **regardless of the order of the method
 * calls**:
 *
 * 1. Add atoms,
 * 2. Add bonds,
 * 3. Erase bonds, and
 * 4. Erase atoms.
 *
 * This is because the removal of atoms might change the atom indices of the
 * remaining atoms, thus giving unexpected results if the removal is done before
 * any other mutations.
 *
 * The mutations are applied when the `accept()` method is called. If the
 * method is not explicitly called, the destructor will automatically call it.
 *
 * Conformers of molecules will resize to the new number of atoms, but currently
 * there is no way to specify the new positions of the added atoms. Assign the
 * new positions manually after the mutation if necessary.
 *
 * @note Having multiple instances of this class for the same molecule might
 *       result in an inconsistent bond addition/removal.
 */
class MoleculeMutator {
public:
  MoleculeMutator(Molecule &mol): mol_(&mol) { }

  MoleculeMutator() = delete;
  MoleculeMutator(const MoleculeMutator &) = delete;
  MoleculeMutator &operator=(const MoleculeMutator &) = delete;
  MoleculeMutator(MoleculeMutator &&) noexcept = default;
  MoleculeMutator &operator=(MoleculeMutator &&) noexcept = default;

  ~MoleculeMutator() noexcept { accept(); }

  /**
   * @brief Add an atom to the molecule.
   * @param atom The data of the atom to add.
   * @return The index of the added atom.
   */
  int add_atom(const AtomData &atom) {
    const int ret = next_atom_idx();
    new_atoms_.push_back(atom);
    return ret;
  }

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
   *       moment of calling `accept()`.
   */
  void erase_atom(int atom_idx) { erased_atoms_.insert(atom_idx); }

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond, after all atom additions.
   * @param dst Index of the destination atom of the bond, after all atom
   *            additions.
   * @param bond The data of the bond to add.
   * @return `true` if the bond was added, `false` if the bond already exists.
   * @note Implementation detail: src, dst are swapped if src > dst.
   */
  bool add_bond(int src, int dst, const BondData &bond);

  /**
   * @brief Add a bond to the molecule.
   * @param src Index of the source atom of the bond, after all atom additions.
   * @param dst Index of the destination atom of the bond, after all atom
   *            additions.
   * @param bond The bond to add.
   * @return `true` if the bond was added, `false` if the bond already exists.
   * @note Implementation detail: src, dst are swapped if src > dst.
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
   *       **at the momenet of `accept()` call**. This is a no-op if the bond
   *       does not exist, also **at the moment of `accept()` call**.
   * @note Implementation detail: src, dst are swapped if src > dst.
   */
  void erase_bond(int src, int dst);

  /**
   * @brief Get the number of atoms in the molecule after the mutation.
   *
   * @return The number of atoms in the molecule after the mutation.
   */
  int num_atoms() const;

  /**
   * @brief Cancel all pending changes.
   * @note This effectively resets the mutator to the state after construction.
   */
  void discard() noexcept;

  /**
   * @brief Apply all changes made to the molecule.
   * @note The mutator internally discard() all changes after the call. Thus,
   *       it's a no-op to call this method multiple times.
   */
  void accept() noexcept;

private:
  struct AddedBond {
    std::pair<int, int> ends;
    BondData data;
  };

  int next_atom_idx() const;

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

  std::vector<AtomData> new_atoms_;
  absl::flat_hash_set<int> erased_atoms_;

  std::vector<AddedBond> new_bonds_;
  absl::flat_hash_set<std::pair<int, int>> new_bonds_set_;
  std::vector<std::pair<int, int>> erased_bonds_;
};

/* Out-of-line definitions */

template <class Iterator, class>
Molecule::Molecule(Iterator begin, Iterator end): Molecule() {
  MoleculeMutator mutator(*this);
  for (auto it = begin; it != end; ++it) {
    mutator.add_atom(*it);
  }
}

inline MoleculeMutator Molecule::mutator() {
  return MoleculeMutator(*this);
}
}  // namespace nuri

#endif /* NURI_CORE_MOLECULE_H_ */
