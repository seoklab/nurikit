//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//
#ifndef NURI_PYTHON_CORE_CORE_MODULE_H_
#define NURI_PYTHON_CORE_CORE_MODULE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/core/property_map.h"
#include "nuri/python/core/containers.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
enum class Chirality : int {
  kNone,
  kCW,
  kCCW,
};

inline Chirality chirality_of(const AtomData &data) {
  if (!data.is_chiral())
    return Chirality::kNone;

  return data.is_clockwise() ? Chirality::kCW : Chirality::kCCW;
}

inline void update_chirality(AtomData &data, Chirality kind) {
  data.set_chiral(kind != Chirality::kNone)
      .set_clockwise(kind == Chirality::kCW);
}

enum class BondConfig : int {
  kNone,
  kCis,
  kTrans,
};

inline BondConfig config_of(const BondData &data) {
  if (!data.has_config())
    return BondConfig::kNone;

  return data.is_trans() ? BondConfig::kTrans : BondConfig::kCis;
}

inline void update_config(BondData &data, BondConfig kind) {
  data.set_config(kind != BondConfig::kNone)
      .set_trans(kind == BondConfig::kTrans);
}

inline int check_conf(const Molecule &mol, int idx) {
  return py_check_index(static_cast<int>(mol.confs().size()), idx,
                        "conformer index out of range");
}

inline int get_or_throw_bond(Molecule &mol, int src, int dst) {
  auto it = mol.find_bond(src, dst);
  if (it == mol.bond_end())
    throw py::value_error("no such bond");
  return it->id();
}

inline int get_or_throw_nei(Molecule &mol, int src, int dst) {
  auto it = mol.find_neighbor(src, dst);
  if (it.end())
    throw py::value_error("not a neighbor");
  return it - mol.neighbor_begin(src);
}

class PyAtom;
class PyBond;
class PyNeigh;

class PyMol: public ParentWrapper<PyMol, Molecule> {
  using Base = ParentWrapper<PyMol, Molecule>;

public:
  using Base::Base;

  PyAtom pyatom(int idx);

  PyBond pybond(int idx);
  PyBond pybond(int src, int dst);

  PyNeigh pyneighbor(int src, int dst);

  void tick() {
    PyMol::Parent::tick();
    tock();
  }

  void tock() { substruct_version_++; }

  std::uint64_t sub_version() const { return substruct_version_; }

  void clear_atoms() {
    if (has_mutator_)
      throw std::runtime_error(
          "cannot clear atoms/bonds of molecule with active mutator");

    (**this).clear_atoms();
    tick();
  }

  void clear_bonds() {
    if (has_mutator_)
      throw std::runtime_error(
          "cannot clear bonds of molecule with active mutator");

    (**this).clear_bonds();
    tick();
  }

  void clear() {
    if (has_mutator_)
      throw std::runtime_error("cannot clear molecule with active mutator");

    (**this).clear();
    tick();
  }

  auto mutator() {
    if (has_mutator_)
      throw std::runtime_error("molecule already has a mutator");

    has_mutator_ = true;
    return std::make_unique<MoleculeMutator>(self()->mutator());
  }

private:
  friend class PyMutator;

  void mutator_finalized() {
    has_mutator_ = false;
    tick();
  }

  std::uint64_t substruct_version_ = 0;
  bool has_mutator_ = false;
};

class PyMutator {
public:
  explicit PyMutator(PyMol &pm): mol_(&pm) { }

  PyMutator &initialize() {
    if (mut_)
      throw std::runtime_error("mutator is already active");

    mut_ = mol_->mutator();
    return *this;
  }

  void finalize() {
    if (!mut_)
      throw std::runtime_error("mutator is not active");

    mut_.reset();
    mol_->mutator_finalized();
  }

  PyAtom add_atom(AtomData &&data);

  PyBond add_bond(int src, int dst, BondData &&data);

  void clear_atoms() {
    mut().clear_atoms();
    mol_->tick();
  }

  void clear_bonds() {
    mut().clear_bonds();
    mol_->tick();
  }

  void clear() {
    mut().clear();
    mol_->tick();
  }

  MoleculeMutator &mut() {
    if (!mut_)
      throw std::runtime_error("mutator is already finalized or not active");

    return *mut_;
  }

  Molecule &mol() { return **mol_; }

private:
  PyMol *mol_;
  std::unique_ptr<MoleculeMutator> mut_;
};

class PyNeigh: public ProxyWrapper<PyNeigh, std::pair<int, int>,
                                   Molecule::MutableNeighbor, PyMol> {
  using Base = ProxyWrapper<PyNeigh, std::pair<int, int>,
                            Molecule::MutableNeighbor, PyMol>;

public:
  using Base::Base;

  PyAtom src();

  PyAtom dst();

  PyBond bond();

  using Base::ok;

private:
  friend Base;

  static Molecule::MutableNeighbor deref(PyMol &mol,
                                         const std::pair<int, int> &idxs) {
    return mol->atom(idxs.first).neighbor(idxs.second);
  }
};

class PyAtom: public ProxyWrapper<PyAtom, int, Molecule::MutableAtom, PyMol> {
  using Base = ProxyWrapper<PyAtom, int, Molecule::MutableAtom, PyMol>;

public:
  using Base::Base;

  PyNeigh neighbor(int idx) {
    return PyNeigh(parent(), { raw(), idx }, version());
  }

  auto pos(int conf) {
    conf = check_conf(*parent(), conf);
    auto atom = **this;

    Matrix3Xd &mat = parent()->confs()[conf];
    if (mat.cols() <= atom.id()) {
      throw py::value_error(
          "atom position not available in the conformer. Are you in a "
          "mutation context?");
    }

    return mat.col(atom.id());
  }

  using Base::ok;

private:
  friend Base;

  static Molecule::MutableAtom deref(PyMol &mol, int idx) {
    return mol->atom(idx);
  }
};

class PyBond: public ProxyWrapper<PyBond, int, Molecule::MutableBond, PyMol> {
  using Base = ProxyWrapper<PyBond, int, Molecule::MutableBond, PyMol>;

public:
  using Base::Base;

  auto src() { return this->parent().pyatom((**this).src().id()); }

  auto dst() { return this->parent().pyatom((**this).dst().id()); }

  void rotate(double angle, bool reverse, bool strict, std::optional<int> conf);

  double length(int conf) {
    conf = check_conf(*parent(), conf);
    return parent()->distance(**this, conf);
  }

  double sqlen(int conf) {
    conf = check_conf(*parent(), conf);
    return parent()->distsq(**this, conf);
  }

  using Base::ok;

private:
  friend Base;

  static Molecule::MutableBond deref(PyMol &mol, int idx) {
    return mol->bond(idx);
  }
};

inline PyAtom PyMol::pyatom(int idx) {
  return PyAtom(*this, idx, version());
}

inline PyBond PyMol::pybond(int idx) {
  return PyBond(*this, idx, version());
}

inline PyBond PyMol::pybond(int src, int dst) {
  int idx = get_or_throw_bond(**this, src, dst);
  return PyBond(*this, idx, version());
}

inline PyNeigh PyMol::pyneighbor(int src, int dst) {
  auto idx = get_or_throw_nei(**this, src, dst);
  return PyNeigh(*this, { src, idx }, version());
}

inline PyAtom PyNeigh::src() {
  return PyAtom(parent(), (**this).src().id(), version());
}

inline PyAtom PyNeigh::dst() {
  return PyAtom(parent(), (**this).dst().id(), version());
}

inline PyBond PyNeigh::bond() {
  return PyBond(parent(), (**this).eid(), version());
}

template <class P>
class PySubAtom
    : public ProxyWrapper<PySubAtom<P>, int, Substructure::MutableAtom, P> {
  using Base = typename PySubAtom::Parent;

public:
  using Base::Base;

  PyAtom as_parent() const {
    this->check();
    PyMol &mol = this->parent().parent();
    return mol.pyatom(this->parent()->atom_ids()[this->raw()]);
  }

  auto pos(int conf) const { return as_parent().pos(conf); }

private:
  friend Base;

  bool ok() const { return this->parent().ok(this->version()); }

  static Substructure::MutableAtom deref(P &sub, int idx) {
    return sub->atom(idx);
  }
};

template <class P>
class PySubBond
    : public ProxyWrapper<PySubBond<P>, int, Substructure::MutableBond, P> {
  using Base = typename PySubBond::Parent;

public:
  using Base::Base;

  PyBond as_parent() const {
    this->check();
    PyMol &mol = this->parent().parent();
    return mol.pybond(this->parent()->bond_ids()[this->raw()]);
  }

  auto src() { return this->parent().pysubatom((**this).src().id()); }

  auto dst() { return this->parent().pysubatom((**this).dst().id()); }

  double length(int conf) { return as_parent().length(conf); }

  double sqlen(int conf) { return as_parent().sqlen(conf); }

private:
  friend Base;

  bool ok() const { return this->parent().ok(this->version()); }

  static Substructure::MutableBond deref(P &sub, int idx) {
    return sub->bond(idx);
  }
};

template <class P>
class PySubNeigh
    : public ProxyWrapper<PySubNeigh<P>, Substructure::MutableNeighbor,
                          Substructure::MutableNeighbor, P> {
  using Base = typename PySubNeigh::Parent;

public:
  using Base::Base;

  PyNeigh as_parent() const {
    this->check();
    PyMol &mol = this->parent().parent();
    int psrc = this->raw().src().as_parent().id(),
        pdst = this->raw().dst().as_parent().id();
    return mol.pyneighbor(psrc, pdst);
  }

  auto src() { return this->parent().pysubatom((**this).src().id()); }

  auto dst() { return this->parent().pysubatom((**this).dst().id()); }

  auto bond() { return this->parent().pysubbond((**this).eid()); }

private:
  friend Base;

  bool ok() const { return this->parent().ok(this->version()); }

  static Substructure::MutableNeighbor
  deref(P & /* sub */, Substructure::MutableNeighbor nei) {
    return nei;
  }
};

template <class Derived, class T>
class SubstructWrapper: public ProxyWrapper<Derived, T, Substructure &, PyMol> {
  using Base = typename SubstructWrapper::Parent;

public:
  using Base::Base;

  auto pysubatom(int idx) {
    return PySubAtom<Derived>(derived(), idx, derived().version_for_child());
  }

  auto pysubbond(int idx) {
    return PySubBond<Derived>(derived(), idx, derived().version_for_child());
  }

  auto pysubneighbor(Substructure::MutableNeighbor nei) {
    return PySubNeigh<Derived>(derived(), nei, derived().version_for_child());
  }

  auto pysubbond(Molecule::Atom src, Molecule::Atom dst) {
    Substructure &sub = *derived();
    auto bit = sub.find_bond(src, dst);
    if (bit == sub.bond_end())
      throw py::value_error(
          "no such bond, or one of the atoms is not in the substructure");
    return pysubbond(bit->id());
  }

  auto pysubbond(Substructure::Atom src, Substructure::Atom dst) {
    Substructure &sub = *derived();
    auto bit = sub.find_bond(src, dst);
    if (bit == sub.bond_end())
      throw py::value_error("no such bond");
    return pysubbond(bit->id());
  }

  auto pysubneighbor(Molecule::Atom src, Molecule::Atom dst) {
    Substructure &sub = *derived();
    auto sit = sub.find_atom(src), dit = sub.find_atom(dst);
    if (sit == sub.atom_end() || dit == sub.atom_end())
      throw py::value_error(
          "source or destination atom does not belong to the substructure");

    auto it = sub.find_neighbor(*sit, *dit);
    if (it.end())
      throw py::value_error("not a neighbor");

    return pysubneighbor(*it);
  }

  auto pysubneighbor(Substructure::Atom src, Substructure::Atom dst) {
    auto it = derived()->find_neighbor(src, dst);
    if (it.end())
      throw py::value_error("not a neighbor");

    return pysubneighbor(*it);
  }

private:
  Derived &derived() { return static_cast<Derived &>(*this); }
};

class ProxySubstruct: public SubstructWrapper<ProxySubstruct, int> {
  using Base = SubstructWrapper<ProxySubstruct, int>;

public:
  using Base::Base;

  void erase_hydrogens();

  void tick() {
    parent().tock();
    self_tick();
  }

  std::uint64_t version_for_child() const { return parent().sub_version(); }

  bool ok(std::uint64_t version) const {
    return parent().sub_version() == version;
  }

private:
  friend Base;
  friend ProxySubstruct::Parent;
  friend PySubAtom<ProxySubstruct>;
  friend PySubBond<ProxySubstruct>;
  friend PySubNeigh<ProxySubstruct>;

  bool ok() const { return ok(version()); }

  static Substructure &deref(PyMol &mol, int idx) {
    return mol->substructures()[idx];
  }
};

class PySubstruct: public SubstructWrapper<PySubstruct, Substructure> {
  using Base = SubstructWrapper<PySubstruct, Substructure>;

public:
  using Base::Base;

  void erase_hydrogens();

  static PySubstruct from_mol(PyMol &mol, Substructure &&sub) {
    return PySubstruct(mol, std::move(sub), mol.version_for_child());
  }

  static PySubstruct from_proxy(ProxySubstruct &proxy) {
    return PySubstruct(proxy.parent(), *proxy,
                       proxy.parent().version_for_child());
  }

  void tick() { sub_version_++; }

  std::uint64_t version_for_child() const { return sub_version_; }

  bool ok(std::uint64_t version) const {
    return ok() && version == sub_version_;
  }

private:
  friend Base;
  friend PySubstruct::Parent;
  friend PySubAtom<PySubstruct>;
  friend PySubBond<PySubstruct>;
  friend PySubNeigh<PySubstruct>;

  using Base::ok;

  void self_tick() {
    Base::self_tick();
    tick();
  }

  static Substructure &deref(PyMol & /* mol */, Substructure &sub) {
    return sub;
  }

  std::uint64_t sub_version_ = 0;
};

class ProxySubstructIterator;

class ProxySubstructContainer {
public:
  explicit ProxySubstructContainer(PyMol &mol): mol_(&mol) { }

  int size() const { return static_cast<int>(mol()->substructures().size()); }

  auto get(int idx) { return ProxySubstruct(mol(), idx, mol().sub_version()); }

  void set(int idx, Substructure &&sub) {
    mol().tock();
    mol()->substructures()[idx] = std::move(sub);
  }

  void del(int idx) {
    mol().tock();
    auto &subs = mol()->substructures();
    subs.erase(subs.begin() + idx);
  }

  void clear() {
    mol().tock();
    mol()->substructures().clear();
  }

  ProxySubstructIterator iter();

  PyMol &mol() { return *mol_; }

  const PyMol &mol() const { return *mol_; }

private:
  PyMol *mol_;
};

class ProxySubstructIterator
    : public PyIterator<ProxySubstructIterator, ProxySubstructContainer> {
  using Base = ProxySubstructIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module &m) {
    return Base::bind(m, "_ProxySubstructureIterator", kReturnsSubobject);
  }

private:
  friend Base;

  static int size_of(ProxySubstructContainer &cont) { return cont.size(); }

  static ProxySubstruct deref(ProxySubstructContainer &cont, int idx) {
    return cont.get(idx);
  }
};

inline ProxySubstructIterator ProxySubstructContainer::iter() {
  return ProxySubstructIterator(*this);
}

extern const Element &
element_from_symbol_or_name(std::string_view symbol_or_name);

extern const Isotope &isotope_from_element_and_mass(const Element &elem,
                                                    int mass_number);

extern const Element &get_or_throw_z(int z);
extern constants::Hybridization get_or_throw_hyb(constants::Hybridization hyb);
extern constants::BondOrder get_or_throw_ord(constants::BondOrder ord);
extern SubstructCategory get_or_throw_cat(SubstructCategory cat);

extern void log_aromatic_warning(const AtomData &atom);
extern void log_aromatic_warning(const BondData &bond);

template <class MatrixLike>
void assign_conf(MatrixLike &conf, const py::handle &obj) {
  static_assert(MatrixLike::RowsAtCompileTime == 3);

  auto arr = py_array_cast<3>(obj);
  auto mat = arr.eigen();
  if (mat.cols() != conf.cols()) {
    throw py::value_error(
        absl::StrCat("conformer has different number of atoms: expected ",
                     conf.cols(), ", got ", mat.cols()));
  }

  conf = mat;
}

inline int check_implicit_hydrogens(int n) {
  if (n < 0)
    throw py::value_error("negative number of implicit hydrogens");
  return n;
}

inline void update_hyb(AtomData &self, constants::Hybridization hyb) {
  self.set_hybridization(get_or_throw_hyb(hyb));
}

inline void update_ord(BondData &self, constants::BondOrder ord) {
  self.set_order(get_or_throw_ord(ord));
  log_aromatic_warning(self);
}

template <class T>
bool same_parent(const T &lhs, const T &rhs) {
  return &lhs == &rhs;
}

template <class T>
void check_parent(const T &lhs, const T &rhs, const char *what) {
  if (!same_parent(lhs, rhs))
    throw py::value_error(what);
}

inline int check_atom(const Molecule &mol, int idx) {
  return py_check_index(mol.num_atoms(), idx, "atom index out of range");
}

inline Molecule::MutableAtom check_atom(const Molecule &mol, PyAtom &atom) {
  check_parent(mol, *atom.parent(), "atom does not belong to the molecule");
  return *atom;
}

inline int check_bond(const Molecule &mol, int idx) {
  return py_check_index(mol.num_bonds(), idx, "bond index out of range");
}

inline Molecule::MutableBond check_bond(const Molecule &mol, PyBond &bond) {
  check_parent(mol, *bond.parent(), "bond does not belong to the molecule");
  return *bond;
}

extern std::pair<int, int> check_bond_ends(const Molecule &mol, int src,
                                           int dst);

extern std::pair<Molecule::MutableAtom, Molecule::MutableAtom>
check_bond_ends(const Molecule &mol, PyAtom &src, PyAtom &dst);

template <class T>
inline AtomData &atom_prolog(T &self) {
  return self->data();
}

template <>
inline AtomData &atom_prolog(AtomData &self) {
  return self;
}

template <class T>
py::class_<T> &add_atom_data_interface(py::class_<T> &cls) {
  cls.def_property(
      "hyb", [](T &self) { return atom_prolog(self).hybridization(); },
      [](T &self, constants::Hybridization hyb) {
        update_hyb(atom_prolog(self), hyb);
      },
      rvp::automatic,
      R"doc(
:type: Hyb

The hybridization of the atom.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "implicit_hydrogens",
      [](T &self) { return atom_prolog(self).implicit_hydrogens(); },
      [](T &self, int n) {
        n = check_implicit_hydrogens(n);
        atom_prolog(self).set_implicit_hydrogens(n);
      },
      rvp::automatic,
      R"doc(
:type: int

The number of implicit hydrogens of the atom. Guaranteed to be non-negative.

.. note::
  It is illegal to set the number of implicit hydrogens to a negative number.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "formal_charge",
      [](T &self) { return atom_prolog(self).formal_charge(); },
      [](T &self, int charge) { atom_prolog(self).set_formal_charge(charge); },
      rvp::automatic,
      R"doc(
:type: int

The formal charge of the atom.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "partial_charge",
      [](T &self) { return atom_prolog(self).partial_charge(); },
      [](T &self, double charge) {
        atom_prolog(self).set_partial_charge(charge);
      },
      rvp::automatic,
      R"doc(
:type: float

The partial charge of the atom.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "aromatic", [](T &self) { return atom_prolog(self).is_aromatic(); },
      [](T &self, bool is_aromatic) {
        AtomData &data = atom_prolog(self);
        data.set_aromatic(is_aromatic);
        log_aromatic_warning(data);
      },
      rvp::automatic,
      R"doc(
:type: bool

Whether the atom is aromatic.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "conjugated", [](T &self) { return atom_prolog(self).is_conjugated(); },
      [](T &self, bool is_conjugated) {
        AtomData &data = atom_prolog(self);
        data.set_conjugated(is_conjugated);
        log_aromatic_warning(data);
      },
      rvp::automatic,
      R"doc(
:type: bool

Whether the atom is conjugated.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "ring", [](T &self) { return atom_prolog(self).is_ring_atom(); },
      [](T &self, bool is_ring) {
        AtomData &data = atom_prolog(self);
        data.set_ring_atom(is_ring);
        log_aromatic_warning(data);
      },
      rvp::automatic,
      R"doc(
:type: bool

Whether the atom is a ring atom.

.. note::
  Beware updating this property when the atom is owned by a molecule. The
  molecule may not be aware of the change.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "chirality", [](T &self) { return chirality_of(atom_prolog(self)); },
      [](T &self, std::optional<Chirality> kind) {
        update_chirality(atom_prolog(self), kind.value_or(Chirality::kNone));
      },
      rvp::automatic,
      R"doc(
:type: Chirality

Explicit chirality of the atom. Note that this does *not* imply the atom is a
stereocenter chemically and might not correspond to the geometry of the
molecule. See :class:`Chirality` for formal definition.

.. tip::
  Assigning :obj:`None` clears the explicit chirality.

.. seealso::
  :class:`Chirality`, :meth:`update`
)doc");
  cls.def_property_readonly(
      "atomic_number",
      [](T &self) { return atom_prolog(self).atomic_number(); }, rvp::automatic,
      R"doc(
:type: int

The atomic number of the atom.

.. seealso::
  :meth:`set_element`, :meth:`update`
)doc");
  cls.def_property_readonly(
      "element_symbol",
      [](T &self) { return atom_prolog(self).element_symbol(); },
      rvp::automatic,
      R"doc(
:type: str

The IUPAC element symbol of the atom.

.. seealso::
  :meth:`set_element`
)doc");
  cls.def_property_readonly(
      "element_name", [](T &self) { return atom_prolog(self).element_name(); },
      rvp::automatic,
      R"doc(
:type: str

The IUPAC element name of the atom.

.. seealso::
  :meth:`set_element`
)doc");
  cls.def_property_readonly(
      "atomic_weight",
      [](T &self) { return atom_prolog(self).atomic_weight(); }, rvp::automatic,
      R"doc(
:type: float

The atomic weight of the atom. Equivalent to ``data.element.atomic_weight``.
)doc");
  cls.def_property(
      "element",
      [](T &self) -> const Element & { return atom_prolog(self).element(); },
      [](T &self, const Element &elem) { atom_prolog(self).set_element(elem); },
      rvp::reference, R"doc(
:type: Element

The element of the atom.

.. seealso::
  :meth:`set_element`, :meth:`update`
)doc");
  cls.def(
      "set_element",
      [](T &self, int atomic_number) -> T & {
        AtomData &data = atom_prolog(self);
        data.set_element(get_or_throw_z(atomic_number));
        return self;
      },
      py::arg("atomic_number"), R"doc(
Set the element of the atom.

:param atomic_number: The atomic number of the element to set.

.. seealso::
  :meth:`update`
)doc");
  cls.def(
      "set_element",
      [](T &self, std::string_view arg) -> T & {
        AtomData &data = atom_prolog(self);
        data.set_element(element_from_symbol_or_name(arg));
        return self;
      },
      py::arg("symbol_or_name"), R"doc(
Set the element of the atom.

:param symbol_or_name: The atomic symbol or name of the element to set.

.. note::
  The symbol or name is case-insensitive. Symbol is tried first, and if it
  fails, name is tried.
)doc");
  cls.def(
      "set_element",
      [](T &self, const Element &elem) -> T & {
        AtomData &data = atom_prolog(self);
        data.set_element(elem);
        return self;
      },
      py::arg("element"), R"doc(
Set the element of the atom.

:param element: The element to set.

.. seealso::
  :meth:`update`
)doc");
  cls.def(
      "get_isotope",
      [](T &self, bool expl) {
        if (expl)
          return atom_prolog(self).explicit_isotope();
        return &atom_prolog(self).isotope();
      },
      rvp::reference, py::arg("explicit") = false,
      R"doc(
Get the isotope of the atom.

:param explicit: If True, returns the explicit isotope of the atom. Otherwise,
  returns the isotope of the atom. Defaults to False.

:returns: The isotope of the atom. If the atom does not have an explicit
  isotope,

  * If ``explicit`` is False, the representative isotope of the element is
    returned.
  * If ``explicit`` is True, None is returned.
)doc");
  cls.def(
      "set_isotope",
      [](T &self, int mass_number) -> T & {
        AtomData &data = atom_prolog(self);
        data.set_isotope(
            isotope_from_element_and_mass(data.element(), mass_number));
        return self;
      },
      py::arg("mass_number"),
      R"doc(
Set the isotope of the atom.

:param mass_number: The mass number of the isotope to set.
)doc");
  cls.def(
      "set_isotope",
      [](T &self, const Isotope &iso) -> T & {
        atom_prolog(self).set_isotope(iso);
        return self;
      },
      py::arg("isotope"),
      R"doc(
Set the isotope of the atom.

:param isotope: The isotope to set.
)doc");
  cls.def_property(
      "name", [](T &self) { return atom_prolog(self).get_name(); },
      [](T &self, std::string_view name) { atom_prolog(self).set_name(name); },
      rvp::automatic,
      R"doc(
:type: str

The name of the atom. Returns an empty string if the name is not set.

.. seealso::
  :meth:`update`
)doc");
  cls.def(
      "update",
      [](T &self, std::optional<constants::Hybridization> hyb,
         std::optional<int> implicit_hydrogens,
         std::optional<int> formal_charge, std::optional<double> partial_charge,
         std::optional<int> atomic_number, const Element *element,
         std::optional<bool> ar, std::optional<bool> conj,
         std::optional<bool> ring, std::optional<Chirality> chirality,
         std::optional<std::string> name) -> T & {
        // Possibly throwing functions

        AtomData &data = atom_prolog(self);

        if (hyb)
          *hyb = get_or_throw_hyb(*hyb);

        if (implicit_hydrogens)
          *implicit_hydrogens = check_implicit_hydrogens(*implicit_hydrogens);

        if (atomic_number && element != nullptr)
          throw py::value_error(
              "atomic_number and element are mutually exclusive");
        if (atomic_number)
          element = &get_or_throw_z(*atomic_number);

        // Now we can update the atom

        if (hyb)
          data.set_hybridization(*hyb);

        if (implicit_hydrogens)
          data.set_implicit_hydrogens(*implicit_hydrogens);

        if (formal_charge)
          data.set_formal_charge(*formal_charge);

        if (partial_charge)
          data.set_partial_charge(*partial_charge);

        if (element != nullptr)
          data.set_element(*element);

        if (ar)
          data.set_aromatic(*ar);
        if (conj)
          data.set_conjugated(*conj);
        if (ring)
          data.set_ring_atom(*ring);
        if (chirality)
          update_chirality(data, *chirality);

        log_aromatic_warning(data);

        if (name)
          data.set_name(std::move(*name));

        return self;
      },
      py::kw_only(),
      py::arg("hyb") = py::none(),                 //
      py::arg("implicit_hydrogens") = py::none(),  //
      py::arg("formal_charge") = py::none(),       //
      py::arg("partial_charge") = py::none(),      //
      py::arg("atomic_number") = py::none(),       //
      py::arg("element") = py::none(),             //
      py::arg("aromatic") = py::none(),            //
      py::arg("conjugated") = py::none(),          //
      py::arg("ring") = py::none(),                //
      py::arg("chirality") = py::none(),           //
      py::arg("name") = py::none(),                //
      R"doc(
Update the atom data. If any of the arguments are not given, the corresponding
property is not updated.

.. note::
  ``atomic_number`` and ``element`` are mutually exclusive. If both are given,
  an exception is raised.
.. note::
  It is illegal to set the number of implicit hydrogens to a negative number.
)doc");
  cls.def(
      "update_from",
      [](T &self, PyAtom &other) -> T & {
        atom_prolog(self) = other->data();
        return self;
      },
      py::arg("atom"),
      R"doc(
Update the atom data.

:param atom: The atom to copy the data from.
)doc");
  cls.def(
      "update_from",
      [](T &self, PySubAtom<ProxySubstruct> &other) -> T & {
        atom_prolog(self) = other->data();
        return self;
      },
      py::arg("atom"),
      R"doc(
Update the atom data.

:param atom: The atom to copy the data from.
)doc");
  cls.def(
      "update_from",
      [](T &self, PySubAtom<PySubstruct> &other) -> T & {
        atom_prolog(self) = other->data();
        return self;
      },
      py::arg("atom"),
      R"doc(
Update the atom data.

:param atom: The atom to copy the data from.
)doc");
  cls.def(
      "update_from",
      [](T &self, const AtomData &other) -> T & {
        atom_prolog(self) = other;
        return self;
      },
      py::arg("data"),
      R"doc(
Update the atom data.

:param data: The atom data to update from.
)doc");

  return cls;
}

template <class T>
py::class_<T> &add_common_atom_interface(py::class_<T> &cls) {
  add_atom_data_interface(cls);
  cls.def(
      "get_pos",
      [](T &self, int conf) {
        auto blk = self.pos(conf);
        return eigen_as_numpy(blk);
      },
      py::arg("conf") = 0, R"doc(
Get the position of the atom.

:param conf: The index of the conformation to get the position from. Defaults to
  0.
:returns: The position of the atom.

.. note::
  The position could not be directly set from Python. Use the :meth:`set_pos`
  method to set the position.
)doc");
  cls.def(
      "set_pos",
      [](T &self, const py::handle &obj, int conf) {
        auto arr = py_array_cast<3, 1>(obj);
        auto blk = self.pos(conf);
        blk = arr.eigen();
      },
      py::arg("pos"), py::arg("conf") = 0, R"doc(
Set the position of the atom.

:param pos: The 3D vector to set the position to. Must be convertible to a numpy
  array of shape (3,).
:param conf: The index of the conformation to set the position to. Defaults to
  0.
)doc");
  def_property_subobject(
      cls, "props",
      [](T &self) {
        return ProxyPropertyMap(&self->data().props(), self.parent());
      },
      [](T &self, const internal::PropertyMap &props) {
        self->data().props() = props;
      },
      rvp::automatic,
      R"doc(
:type: collections.abc.MutableMapping[str, str]

A dictionary-like object to store additional properties of the atom. The keys
and values are both strings.

.. note::
  The properties are shared with the underlying :class:`AtomData` object. If the
  properties are modified, the underlying object is also modified.

  As a result, the property map is also invalidated when any changes are made
  to the molecule. If the properties must be kept alive, copy the properties
  first with ``copy()`` method.
)doc");
  cls.def(
      "copy_data", [](T &self) { return self->data(); },
      R"doc(
Copy the underlying :class:`AtomData` object.

:returns: A copy of the underlying :class:`AtomData` object.
)doc");
  return cls;
}

template <class T>
inline BondData &bond_prolog(T &self) {
  return self->data();
}

template <>
inline BondData &bond_prolog(BondData &self) {
  return self;
}

template <class T>
py::class_<T> &add_bond_data_interface(py::class_<T> &cls) {
  cls.def_property(
      "order", [](T &self) { return bond_prolog(self).order(); },
      [](T &self, constants::BondOrder ord) {
        return update_ord(bond_prolog(self), ord);
      },
      rvp::automatic,
      R"doc(
:type: BondOrder

The bond order of the bond.

.. seealso::
  :meth:`update`
)doc");
  cls.def(
      "approx_order", [](T &self) { return bond_prolog(self).approx_order(); },
      R"doc(
The approximate bond order of the bond.
)doc");
  cls.def(
      "rotatable", [](T &self) { return bond_prolog(self).is_rotatable(); },
      R"doc(
Whether the bond is rotatable.

.. note::
  The result is calculated as the bond order is :data:`BondOrder.Single` or
  :data:`BondOrder.Other`, and the bond is not a conjugated or a ring bond.
)doc");
  cls.def_property(
      "ring", [](T &self) { return bond_prolog(self).is_ring_bond(); },
      [](T &self, bool ring) {
        BondData &data = bond_prolog(self);
        data.set_ring_bond(ring);
        log_aromatic_warning(data);
      },
      rvp::automatic,
      R"doc(
:type: bool

Whether the bond is a ring bond.

.. note::
  Beware updating this property when the bond is owned by a molecule. The
  molecule may not be aware of the change.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "aromatic", [](T &self) { return bond_prolog(self).is_aromatic(); },
      [](T &self, bool is_aromatic) {
        BondData &data = bond_prolog(self);
        data.set_aromatic(is_aromatic);
        log_aromatic_warning(data);
      },
      rvp::automatic,
      R"doc(
:type: bool

Whether the bond is aromatic.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "conjugated", [](T &self) { return bond_prolog(self).is_conjugated(); },
      [](T &self, bool is_conjugated) {
        BondData &data = bond_prolog(self);
        data.set_conjugated(is_conjugated);
        log_aromatic_warning(data);
      },
      rvp::automatic,
      R"doc(
:type: bool

Whether the atom is conjugated.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "config", [](T &self) { return config_of(bond_prolog(self)); },
      [](T &self, std::optional<BondConfig> cfg) {
        update_config(bond_prolog(self), cfg.value_or(BondConfig::kNone));
      },
      rvp::automatic,
      R"doc(
:type: BondConfig

The explicit configuration of the bond. Note that this does *not* imply the bond
is a torsionally restricted bond chemically.

.. note::
  For bonds with more than 3 neighboring atoms, :attr:`BondConfig.Cis` or
  :attr:`BondConfig.Trans` configurations are not well defined terms. In such
  cases, this will return whether **the first neighbors are on the same side of
  the bond**. For example, in the following structure (assuming the neighbors
  are ordered in the same way as the atoms), the bond between atoms 0 and 1 is
  considered to be in a cis configuration (first neighbors are marked with angle
  brackets)::

    <2>     <4>
      \     /
       0 = 1
      /     \
     3       5

  On the other hand, when the neighbors are ordered in the opposite way, the
  bond between atoms 0 and 1 is considered to be in a trans configuration::

    <2>      5
      \     /
       0 = 1
      /     \
     3      <4>

.. tip::
  Assigning :obj:`None` clears the explicit bond configuration.

.. seealso::
  :meth:`update`
)doc");
  cls.def_property(
      "name", [](T &self) { return bond_prolog(self).get_name(); },
      [](T &self, std::string_view name) { bond_prolog(self).set_name(name); },
      rvp::automatic,
      R"doc(
:type: str

The name of the bond. Returns an empty string if the name is not set.

.. seealso::
  :meth:`update`
)doc");
  cls.def(
      "update",
      [](T &self, std::optional<constants::BondOrder> ord,
         std::optional<bool> ar, std::optional<bool> conj,
         std::optional<bool> ring, std::optional<BondConfig> cfg,
         std::optional<std::string> name) -> T & {
        BondData &data = bond_prolog(self);

        // Possibly throwing, must be done first
        if (ord)
          update_ord(data, *ord);

        if (ar)
          data.set_aromatic(*ar);
        if (conj)
          data.set_conjugated(*conj);
        if (ring)
          data.set_ring_bond(*ring);
        if (cfg)
          update_config(data, *cfg);

        log_aromatic_warning(data);

        if (name)
          data.set_name(std::move(*name));

        return self;
      },
      py::kw_only(),
      py::arg("order") = py::none(),       //
      py::arg("aromatic") = py::none(),    //
      py::arg("conjugated") = py::none(),  //
      py::arg("ring") = py::none(),        //
      py::arg("config") = py::none(),      //
      py::arg("name") = py::none(),        //
      R"doc(
Update the bond data. If any of the arguments are not given, the corresponding
property is not updated.
)doc");
  cls.def(
      "update_from",
      [](T &self, PyBond &other) -> T & {
        bond_prolog(self) = other->data();
        return self;
      },
      py::arg("bond"),
      R"doc(
Update the bond data.

:param bond: The bond to copy the data from.
)doc");
  cls.def(
      "update_from",
      [](T &self, PySubBond<ProxySubstruct> &other) -> T & {
        bond_prolog(self) = other->data();
        return self;
      },
      py::arg("bond"),
      R"doc(
Update the bond data.

:param bond: The bond to copy the data from.
)doc");
  cls.def(
      "update_from",
      [](T &self, PySubBond<PySubstruct> &other) -> T & {
        bond_prolog(self) = other->data();
        return self;
      },
      py::arg("bond"),
      R"doc(
Update the bond data.

:param bond: The bond to copy the data from.
)doc");
  cls.def(
      "update_from",
      [](T &self, const BondData &other) -> T & {
        bond_prolog(self) = other;
        return self;
      },
      py::arg("data"),
      R"doc(
Update the bond data.

:param data: The bond data to update from.
)doc");

  return cls;
}

template <class T>
py::class_<T> &add_common_bond_interface(py::class_<T> &cls) {
  add_bond_data_interface(cls);
  def_property_readonly_subobject(cls, "src", &T::src, rvp::automatic,
                                  R"doc(
:type: Atom

The source atom of the bond.
)doc");
  def_property_readonly_subobject(cls, "dst", &T::dst, rvp::automatic,
                                  R"doc(
:type: Atom

The destination atom of the bond.
)doc");
  cls.def("sqlen", &T::sqlen, py::arg("conf") = 0, R"doc(
Calculate the square of the length of the bond.

:param conf: The index of the conformation to calculate the length from.
  Defaults to 0.
:returns: The square of the length of the bond.
)doc");
  cls.def("length", &T::length, py::arg("conf") = 0, R"doc(
Calculate the length of the bond.

:param conf: The index of the conformation to calculate the length from.
  Defaults to 0.
:returns: The length of the bond.
)doc");
  def_property_subobject(
      cls, "props",
      [](T &self) {
        return ProxyPropertyMap(&self->data().props(), self.parent());
      },
      [](T &self, const internal::PropertyMap &props) {
        self->data().props() = props;
      },
      rvp::automatic,
      R"doc(
:type: collections.abc.MutableMapping[str, str]

A dictionary-like object to store additional properties of the bond. The keys
and values are both strings.

.. note::
  The properties are shared with the underlying :class:`BondData` object. If the
  properties are modified, the underlying object is also modified.

  As a result, the property map is invalidated when any changes are made to the
  molecule. If the properties must be kept alive, copy the properties first with
  ``copy()`` method.
)doc");
  cls.def(
      "copy_data", [](T &self) -> BondData { return self->data(); },
      R"doc(
Copy the underlying :class:`BondData` object.

:returns: A copy of the underlying :class:`BondData` object.
)doc");
  return cls;
}

using AtomsArg = pyt::Iterable<std::variant<PyAtom, int>>;
using BondsArg = pyt::Iterable<std::variant<PyBond, int>>;

extern Substructure create_substruct(Molecule &mol,
                                     const std::optional<AtomsArg> &atoms,
                                     const std::optional<BondsArg> &bonds,
                                     SubstructCategory cat);

/* Called from bind_molecule, don't call directly */
extern void bind_substructure(pybind11::module &m);

extern void bind_element(pybind11::module &m);
extern void bind_molecule(pybind11::module &m);
}  // namespace python_internal
}  // namespace nuri

#endif /* NURI_PYTHON_CORE_CORE_MODULE_H_ */
