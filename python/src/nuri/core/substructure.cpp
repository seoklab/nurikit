//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/cleanup/cleanup.h>
#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/options.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "nuri/eigen_config.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/core/property_map.h"
#include "nuri/python/core/containers.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
template <class P>
class PySubAtomIterator: public PyIterator<PySubAtomIterator<P>, P> {
  using Base = typename PySubAtomIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module &m, const char *name) {
    return Base::bind(m, name, kReturnsSubobject);
  }

private:
  friend Base;

  static int size_of(P &sub) { return sub->num_atoms(); }

  static PySubAtom<P> deref(P &sub, int idx) { return sub.pysubatom(idx); }
};

template <class P>
class PySubNeighIterator {
public:
  PySubNeighIterator(PySubAtom<P> &atom): atom_(&atom), iter_(atom->begin()) { }

  PySubNeigh<P> next() {
    atom_->check();

    if (iter_.end())
      throw py::stop_iteration();

    return atom_->parent().pysubneighbor(*iter_++);
  }

private:
  PySubAtom<P> *atom_;
  Substructure::neighbor_iterator iter_;
};

template <class P>
class PySubBondIterator: public PyIterator<PySubBondIterator<P>, P> {
  using Base = typename PySubBondIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module &m, const char *name) {
    return Base::bind(m, name, kReturnsSubobject);
  }

private:
  friend Base;

  static int size_of(P &sub) { return sub->num_bonds(); }

  static PySubBond<P> deref(P &sub, int idx) { return sub.pysubbond(idx); }
};

template <class P>
struct PySubBondsWrapper {
  P *sub;
};

template <class P>
class SubConformersIterator
    : public PyIterator<SubConformersIterator<P>, const std::vector<Matrix3Xd>> {
  using Base = typename SubConformersIterator::Parent;

public:
  SubConformersIterator(const std::vector<Matrix3Xd> &confs, P &sub)
      : Base(confs), sub_(&sub) { }

  static auto bind(py::module &m, const char *name) {
    return Base::bind(m, name);
  }

private:
  friend Base;

  static auto size_of(const std::vector<Matrix3Xd> &confs) {
    return confs.size();
  }

  auto deref(const std::vector<Matrix3Xd> &confs, int idx) const {
    auto sub = confs[idx](Eigen::all, (**sub_).atom_ids());
    return eigen_as_numpy(sub);
  }

  P *sub_;
};

int check_subatom(const Substructure &sub, int idx) {
  return py_check_index(sub.num_atoms(), idx, "atom index out of range");
}

template <class P>
Substructure::MutableAtom check_subatom(const Substructure &sub,
                                        PySubAtom<P> &atom) {
  check_parent(sub, *atom.parent(),
               "sub-atom does not belong to the same substructure");
  return *atom;
}

template <class P>
Substructure::MutableBond check_subbond(const Substructure &sub,
                                        PySubBond<P> &bond) {
  check_parent(sub, *bond.parent(),
               "sub-bond does not belong to the same substructure");
  return *bond;
}

template <class P>
std::pair<Substructure::MutableAtom, Substructure::MutableAtom>
check_subbond_ends(const Substructure &sub, PySubAtom<P> &src,
                   PySubAtom<P> &dst) {
  check_parent(sub, *src.parent(),
               "source sub-atom does not belong to the same substructure");
  check_parent(sub, *dst.parent(),
               "destination sub-atom does not belong to the same substructure");
  return { *src, *dst };
}

template <class P>
PySubAtom<P> subatom_from_parent(P &self, PyAtom &atom) {
  Substructure &sub = *self;
  auto cpp_atom = check_atom(*self.parent(), atom);
  auto it = sub.find_atom(cpp_atom);
  if (it == sub.end())
    throw py::key_error(py::repr(py::cast(atom)));

  return self.pysubatom(it->id());
}

template <class P>
PySubBond<P> subbond_from_parent(P &self, PyBond &bond) {
  Substructure &sub = *self;
  auto cpp_bond = check_bond(*self.parent(), bond);
  auto it = sub.find_bond(cpp_bond);
  if (it == sub.bond_end())
    throw py::key_error(py::repr(py::cast(bond)));

  return self.pysubbond(it->id());
}

void add_atom_single(std::vector<int> &idxs, Molecule &parent,
                     const py::handle &obj) {
  if (py::isinstance<py::int_>(obj)) {
    int idx = obj.cast<int>();
    check_atom(parent, idx);
    idxs.push_back(idx);
    return;
  }

  if (!py::isinstance<PyAtom>(obj))
    throw py::type_error("value must be Atom or int");

  PyAtom &atom = py::cast<PyAtom &>(obj);
  auto cpp_atom = check_atom(parent, atom);
  idxs.push_back(cpp_atom.id());
}

void add_bond_single(std::vector<int> &idxs, Molecule &parent,
                     const py::handle &obj) {
  if (py::isinstance<py::int_>(obj)) {
    int idx = obj.cast<int>();
    check_bond(parent, idx);
    idxs.push_back(idx);
    return;
  }

  if (!py::isinstance<PyBond>(obj))
    throw py::type_error("value must be Bond or int");

  PyBond &bond = py::cast<PyBond &>(obj);
  auto cpp_bond = check_bond(parent, bond);
  idxs.push_back(cpp_bond.id());
}

template <class P>
void bind_substructure_kind(py::class_<P> &sub,
                            py::class_<PySubBondsWrapper<P>> &sub_bonds,
                            py::class_<PySubAtom<P>> &sub_atom,
                            py::class_<PySubBond<P>> &sub_bond,
                            py::class_<PySubNeigh<P>> &sub_nei,
                            py::class_<PySubNeighIterator<P>> &nei_iter) {
  using Atom = PySubAtom<P>;
  using Bond = PySubBond<P>;
  using Nei = PySubNeigh<P>;

  add_common_atom_interface(sub_atom);
  sub_atom.def_property_readonly("id", &Atom::raw, rvp::automatic, R"doc(
:type: int

A unique identifier of the atom in the substructure. The identifier is
guaranteed to be unique within the atoms of the substructure.

This is a read-only property.

.. warning::
  This is not the same as the parent atom's identifier. Convert this to the
  parent atom using :meth:`as_parent` if you need the parent atom's identifier.
.. note::
  Implementation detail: the identifier is the index of the atom in the
  substructure's atom list.
)doc");
  sub_atom.def("as_parent", &Atom::as_parent, kReturnsSubobject,
               "Returns the parent atom of this atom.");
  sub_atom.def(
      "__iter__", [](Atom &atom) { return PySubNeighIterator<P>(atom); },
      kReturnsSubobject);

  add_common_bond_interface(sub_bond);
  sub_bond.def_property_readonly("id", &Bond::raw, rvp::automatic, R"doc(
:type: int

A unique identifier of the bond in the substructure. The identifier is
guaranteed to be unique within the atoms of the substructure.

This is a read-only property.

.. warning::
  This is not the same as the parent bond's identifier. Convert this to the
  parent bond using :meth:`as_parent` if you need the parent bond's identifier.
.. note::
  Implementation detail: the identifier is the index of the bond in the
  substructure's bond list.
)doc");
  sub_bond.def("as_parent", &Bond::as_parent, kReturnsSubobject,
               "Returns the parent bond of this bond.");

  def_property_readonly_subobject(sub_nei, "src", &Nei::src, rvp::automatic,
                                  "Source atom of the neighbor.");
  def_property_readonly_subobject(sub_nei, "dst", &Nei::dst, rvp::automatic,
                                  "Destination atom of the neighbor.");
  def_property_readonly_subobject(sub_nei, "bond", &Nei::bond, rvp::automatic,
                                  "Bond between the source and destination "
                                  "atoms.");
  sub_nei.def("as_parent", &Nei::as_parent, kReturnsSubobject,
              "Returns the parent version of this neighbor.");

  nei_iter.def("__iter__", pass_through<PySubNeighIterator<P>>);
  nei_iter.def("__next__", &PySubNeighIterator<P>::next, kReturnsSubobject);

  add_sequence_interface(
      sub_bonds,
      [](PySubBondsWrapper<P> &self) { return (**self.sub).num_bonds(); },
      [](PySubBondsWrapper<P> &self, int idx) {
        P &substruct = *self.sub;
        idx = py_check_index(substruct->num_bonds(), idx,
                             "bond index out of range");
        return substruct.pysubbond(idx);
      },
      [](PySubBondsWrapper<P> &self) {
        return PySubBondIterator<P>(*self.sub);
      });
  sub_bonds.def(
      "__getitem__",
      [](PySubBondsWrapper<P> &self, PyBond &bond) {
        return subbond_from_parent<P>(*self.sub, bond);
      },
      kReturnsSubobject, py::arg("bond"));
  sub_bonds.def("__contains__",
                [](PySubBondsWrapper<P> &self, PySubBond<P> &bond) {
                  bond.check();
                  return same_parent(*self.sub, bond.parent());
                });
  sub_bonds.def("__contains__", [](PySubBondsWrapper<P> &self, PyBond &bond) {
    P &psub = *self.sub;
    bond.check();
    const Substructure &substruct = *psub;
    if (!same_parent(psub.parent(), bond.parent()))
      return false;
    return substruct.contains_bond(*bond);
  });

  sub.def_property_readonly("molecule", &P::parent, rvp::reference, R"doc(
:type: Molecule

The parent molecule of the substructure.
)doc");
  sub.def(
      "atom",
      [](P &self, int idx) {
        idx = check_subatom(*self, idx);
        return self.pysubatom(idx);
      },
      kReturnsSubobject, py::arg("idx"), R"doc(
Get a substructure atom by index.

:param idx: The index of the atom.
:return: The atom at the given index.
:rtype: SubAtom

.. note::
  The returned atom is invalidated when the parent molecule is modified, or if
  the substructure is modified. If the atom must be kept alive, copy the atom
  data first.
)doc");
  sub.def("atom", subatom_from_parent<P>, kReturnsSubobject, py::arg("atom"),
          R"doc(
Get a substructure atom from a parent atom.

:param atom: The parent atom to get the sub-atom of.
:return: The sub-atom of the parent atom.
:rtype: SubAtom

.. note::
  The returned atom is invalidated when the parent molecule is modified, or if
  the substructure is modified. If the atom must be kept alive, copy the atom
  data first.
)doc");
  sub.def(
      "num_atoms", [](P &self) { return self->num_atoms(); },
      R"doc(
The number of atoms in the substructure. Equivalent to ``len(sub)``.
)doc");
  sub.def(
      "bonds", [](P &self) { return PySubBondsWrapper<P> { &self }; },
      kReturnsSubobject,
      R"doc(
:rtype: collections.abc.Sequence[SubBond]

Get a collection of bonds in the substructure. Invalidated when the parent
molecule is modified, or if the substructure is modified.
)doc");
  sub.def(
      "bond",
      [](P &self, int idx) {
        idx = py_check_index(self->num_bonds(), idx, "bond index out of range");
        return self.pysubbond(idx);
      },
      kReturnsSubobject, py::arg("idx"), R"doc(
Get a bond by index.

:param idx: The index of the bond.
:return: The bond at the given index.
:rtype: SubBond
)doc");
  sub.def("bond", subbond_from_parent<P>, kReturnsSubobject, py::arg("bond"),
          R"doc(
Get a substructure bond from a parent bond.

:param bond: The parent bond to get the sub-bond of.
:return: The sub-bond of the parent bond.
:rtype: SubBond

.. note::
  The returned bond is invalidated when the parent molecule is modified, or if
  the substructure is modified. If the bond must be kept alive, copy the bond
  data first.
)doc");
  sub.def(
      "bond",
      [](P &self, PySubAtom<P> &src, PySubAtom<P> &dst) {
        auto [sa, da] = check_subbond_ends(*self, src, dst);
        return self.pysubbond(sa, da);
      },
      kReturnsSubobject, py::arg("src"), py::arg("dst"), R"doc(
Get a bond of the substructure. ``src`` and ``dst`` are interchangeable.

:param src: The source sub-atom of the bond.
:param dst: The destination sub-atom of the bond.
:returns: The bond from the source to the destination sub-atom.
:rtype: SubBond
:raises ValueError: If the bond does not exist, or any of the sub-atoms does not
  belong to the substructure.

.. seealso::
  :meth:`neighbor`
.. note::
  The returned bond may not have ``bond.src.id == src.id`` and
  ``bond.dst.id == dst.id``, as the source and destination atoms of the bond may
  be swapped.
.. note::
  The returned bond is invalidated when the parent molecule is modified, or if
  the substructure is modified. If the bond must be kept alive, copy the bond
  data first.
)doc");
  sub.def(
      "bond",
      [](P &self, PyAtom &src, PyAtom &dst) {
        auto [sa, da] = check_bond_ends(*self.parent(), src, dst);
        return self.pysubbond(sa, da);
      },
      kReturnsSubobject, py::arg("src"), py::arg("dst"), R"doc(
Get a bond of the substructure. ``src`` and ``dst`` are interchangeable.

:param src: The source atom of the bond.
:param dst: The destination atom of the bond.
:returns: The bond from the source to the destination atom.
:rtype: SubBond
:raises ValueError: If the bond does not exist, the source or destination
  atom does not belong to the substructure, or any of the atoms does not belong
  to the same molecule.

.. seealso::
  :meth:`neighbor`
.. note::
  The source and destination atoms of the bond may be swapped.
.. note::
  The returned bond is invalidated when the parent molecule is modified, or if
  the substructure is modified. If the bond must be kept alive, copy the bond
  data first.
)doc");
  sub.def(
      "has_bond",
      [](P &self, PySubAtom<P> &src, PySubAtom<P> &dst) {
        auto [sa, da] = check_subbond_ends(*self, src, dst);
        return self->find_bond(sa, da) != self->bond_end();
      },
      py::arg("src"), py::arg("dst"), R"doc(
Check if two atoms are connected by a bond.

:param src: The source sub-atom of the bond.
:param dst: The destination sub-atom of the bond.
:returns: Whether the source and destination atoms are connected by a bond.
:raises ValueError: If any of the sub-atoms does not belong to the substructure.
)doc");
  sub.def(
      "has_bond",
      [](P &self, PyAtom &src, PyAtom &dst) {
        auto [sa, da] = check_bond_ends(*self.parent(), src, dst);
        return self->find_bond(sa, da) != self->bond_end();
      },
      py::arg("src"), py::arg("dst"), R"doc(
Check if two atoms are connected by a bond.

:param src: The source atom of the bond.
:param dst: The destination atom of the bond.
:returns: Whether the source and destination atoms are connected by a bond.
:raises ValueError: If any of the atoms does not belong to the molecule.
)doc");
  sub.def(
      "num_bonds", [](P &self) { return self->num_bonds(); },
      R"doc(
The number of bonds in the substructure. Equivalent to ``len(sub.bonds)``.
)doc");
  sub.def(
      "neighbor",
      [](P &self, PySubAtom<P> &src, PySubAtom<P> &dst) {
        auto [sa, da] = check_subbond_ends(*self, src, dst);
        return self.pysubneighbor(sa, da);
      },
      kReturnsSubobject, py::arg("src"), py::arg("dst"), R"doc(
Get a neighbor of the substructure.

:param src: The source sub-atom of the neighbor.
:param dst: The destination sub-atom of the neighbor.
:returns: The neighbor from the source to the destination sub-atom.
:rtype: SubNeighbor
:raises ValueError: If the underlying bond does not exist, or any of the
  sub-atoms does not belong to the substructure.

.. seealso::
  :meth:`bond`
.. note::
  Unlike :meth:`bond`, the returned neighbor is always guaranteed to have
  ``nei.src.id == src.id`` and ``nei.dst.id == dst.id``.
.. note::
  The returned neighbor is invalidated when the parent molecule is modified, or
  if the substructure is modified.
)doc");
  sub.def(
      "neighbor",
      [](P &self, PyAtom &src, PyAtom &dst) {
        auto [sa, da] = check_bond_ends(*self.parent(), src, dst);
        return self.pysubneighbor(sa, da);
      },
      kReturnsSubobject, py::arg("src"), py::arg("dst"), R"doc(
Get a neighbor of the substructure.

:param src: The source atom of the neighbor.
:param dst: The destination atom of the neighbor.
:returns: The neighbor from the source to the destination atom.
:rtype: SubNeighbor
:raises ValueError: If the underlying bond does not exist, the source or
  destination atom does not belong to the substructure, or any of the atoms does
  not belong to the same molecule.

.. seealso::
  :meth:`bond`
.. note::
  Unlike :meth:`bond`, the returned neighbor is always guaranteed to have
  the source and destination atoms in the same order as the arguments.
.. note::
  The returned neighbor is invalidated when the parent molecule is modified, or
  if the substructure is modified.
)doc");
  sub.def(
      "get_conf",
      [](P &self, int conf) {
        Molecule &mol = *self.parent();
        conf = check_conf(mol, conf);
        return eigen_as_numpy(mol.confs()[conf](Eigen::all, self->atom_ids()));
      },
      py::arg("conf") = 0, R"doc(
Get the coordinates of the atoms in a conformation of the substructure.

:param conf: The index of the conformation to get the coordinates from.
:returns: The coordinates of the atoms in the substructure, as a 2D array of
  shape ``(num_atoms, 3)``.

.. note::
  The returned array is a copy of the coordinates. To update the coordinates,
  use the :meth:`set_conf` method.
)doc");
  sub.def(
      "set_conf",
      [](P &self, const py::handle &obj, int conf) {
        Molecule &mol = *self.parent();
        conf = check_conf(mol, conf);
        auto block = mol.confs()[conf](Eigen::all, self->atom_ids());
        assign_conf(block, obj);
      },
      py::arg("coords"), py::arg("conf") = 0, R"doc(
Set the coordinates of the atoms in a conformation of the substructure.

:param coords: The coordinates of the atoms in the conformation. Must be
  convertible to a numpy array of shape ``(num_atoms, 3)``.
:param conf: The index of the conformation to set the coordinates to.

.. note::
  The coordinates of the atoms that are *not* in the substructure are not
  affected.
)doc");
  sub.def(
      "num_confs", [](P &self) { return self.parent()->confs().size(); },
      R"doc(
Get the number of conformations of the substructure.
)doc");
  sub.def(
      "conformers",
      [](P &self) {
        return SubConformersIterator<P>(self.parent()->confs(), self);
      },
      kReturnsSubobject, R"doc(
Get an iterable object of all conformations of the substructure. Each
conformation is a 2D array of shape ``(num_atoms, 3)``. It is not available to
update the coordinates from the returned conformers; you should manually assign
to the conformers to update the coordinates.

:rtype: collections.abc.Iterable[numpy.ndarray]

.. seealso::
  :meth:`get_conf`, :meth:`set_conf`
)doc");
  sub.def(
      "add_atoms",
      [](P &self, const AtomsArg &atoms, bool add_bonds) {
        Substructure &substruct = *self;
        Molecule &parent = *self.parent();

        std::vector<int> idxs;
        for (py::handle obj: atoms)
          add_atom_single(idxs, parent, obj);

        substruct.add_atoms(internal::IndexSet(std::move(idxs)), add_bonds);
        self.tick();
      },
      py::arg("atoms"), py::arg("add_bonds") = true, R"doc(
Add atoms to the substructure.

:param collections.abc.Iterable[Atom | int] atoms: The atoms to add to the
  substructure. The atoms must belong to the same molecule as the substructure.
  All duplicate atoms are ignored.
:param bool add_bonds: If True, the bonds between the added atoms are also added
  to the substructure. If False, the bonds are not added.
:raises TypeError: If any atom is not an :class:`Atom` or :class:`int`.
:raises ValueError: If any atom does not belong to the same molecule.
:raises IndexError: If any atom index is out of range.

.. note::
  Due to the implementation, it is much faster to add atoms in bulk than adding
  them one by one. Thus, we explicitly provide only the bulk addition method.
)doc");
  sub.def(
      "add_bonds",
      [](P &self, const BondsArg &bonds) {
        Substructure &substruct = *self;
        Molecule &parent = *self.parent();

        std::vector<int> idxs;
        for (py::handle obj: bonds)
          add_bond_single(idxs, parent, obj);

        substruct.add_bonds(internal::IndexSet(std::move(idxs)));
        self.tick();
      },
      py::arg("bonds"), R"doc(
Add bonds to the substructure. If any atom of the bond does not belong to the
substructure, the atom is also added to the substructure.

:param collections.abc.Iterable[Bond | int] bonds: The bonds to add to the
  substructure. The bonds must belong to the same molecule as the substructure.
  All duplicate bonds are ignored.
:raises TypeError: If any bond is not a :class:`Bond` or :class:`int`.
:raises ValueError: If any bond does not belong to the same molecule.
:raises IndexError: If any bond index is out of range.

.. note::
  Due to the implementation, it is much faster to add bonds in bulk than adding
  them one by one. Thus, we explicitly provide only the bulk addition method.
)doc");
  sub.def(
      "refresh_bonds",
      [](P &self) {
        self->refresh_bonds();
        self.tick();
      },
      R"doc(
Refresh the bonds of the substructure. All bonds between the atoms of the
substructure are removed, and new bonds are added based on the parent molecule.
)doc");
  sub.def(
      "erase_atom",
      [](P &self, PySubAtom<P> &atom) {
        Substructure &substruct = *self;
        auto cpp_atom = check_subatom(substruct, atom);
        substruct.erase_atom(cpp_atom);
        self.tick();
      },
      py::arg("sub_atom"), R"doc(
Remove an atom from the substructure. Any bonds connected to the atom are also
removed.

:param sub_atom: The sub-atom to remove.

.. note::
  The parent molecule is not modified by this operation.
)doc");
  sub.def(
      "erase_atom",
      [](P &self, PyAtom &atom) {
        Substructure &substruct = *self;
        auto cpp_atom = check_atom(*self.parent(), atom);
        auto it = substruct.find_atom(cpp_atom);
        if (it == substruct.end())
          throw py::value_error("atom not in substructure");

        substruct.erase_atom(*it);
        self.tick();
      },
      py::arg("atom"), R"doc(
Remove an atom from the substructure. Any bonds connected to the atom are also
removed.

:param atom: The parent atom to remove.
:raises ValueError: If the atom is not in the substructure.

.. note::
  The parent molecule is not modified by this operation.
)doc");
  sub.def(
      "erase_bond",
      [](P &self, PySubBond<P> &bond) {
        Substructure &substruct = *self;
        auto cpp_bond = check_subbond(substruct, bond);
        substruct.erase_bond(cpp_bond);
        self.tick();
      },
      py::arg("sub_bond"), R"doc(
Remove a bond from the substructure.

:param sub_bond: The sub-bond to remove.

.. note::
  The parent molecule is not modified by this operation.
)doc");
  sub.def(
      "erase_bond",
      [](P &self, PyBond &bond) {
        Substructure &substruct = *self;
        auto cpp_bond = check_bond(*self.parent(), bond);
        auto it = substruct.find_bond(cpp_bond);
        if (it == substruct.bond_end())
          throw py::value_error("bond not in substructure");

        substruct.erase_bond(*it);
        self.tick();
      },
      py::arg("bond"), R"doc(
Remove a bond from the substructure.

:param bond: The parent bond to remove.
:raises ValueError: If the bond is not in the substructure.

.. note::
  The parent molecule is not modified by this operation.
)doc");
  sub.def(
      "erase_bond",
      [](P &self, PySubAtom<P> &src, PySubAtom<P> &dst) {
        Substructure &substruct = *self;
        auto [sa, da] = check_subbond_ends(substruct, src, dst);

        auto it = substruct.find_bond(sa, da);
        if (it == substruct.bond_end())
          throw py::value_error("bond not in substructure");

        substruct.erase_bond(*it);
        self.tick();
      },
      py::arg("src"), py::arg("dst"), R"doc(
Remove a bond from the substructure. The source and destination atoms are
interchangeable.

:param src: The source atom of the bond.
:param dst: The destination atom of the bond.
:raises ValueError: If the bond is not in the substructure.

.. note::
  The parent molecule is not modified by this operation.
)doc");
  sub.def(
      "erase_bond",
      [](P &self, PyAtom &src, PyAtom &dst) {
        Substructure &substruct = *self;
        auto [sa, da] = check_bond_ends(*self.parent(), src, dst);

        auto it = substruct.find_bond(sa, da);
        if (it == substruct.bond_end())
          throw py::value_error("bond not in substructure or does not exist");

        substruct.erase_bond(*it);
        self.tick();
      },
      py::arg("src"), py::arg("dst"), R"doc(
Remove a bond from the substructure. The source and destination atoms are
interchangeable.

:param src: The source atom of the bond.
:param dst: The destination atom of the bond.
:raises ValueError: If the bond is not in the substructure or does not exist.

.. note::
  The parent molecule is not modified by this operation.
)doc");
  sub.def(
      "parent_atoms",
      [](P &self) {
        py::list ret = py::cast(self->atom_ids());
        return ret;
      },
      R"doc(
The parent atom indices of the substructure atoms. The indices are guaranteed to
be unique and in ascending order.

:rtype: list[int]

.. note::
  The returned list is a copy of the internal list, so modifying the list does
  not affect the substructure.
)doc");
  sub.def(
      "parent_bonds",
      [](P &self) {
        py::list ret = py::cast(self->bond_ids());
        return ret;
      },
      R"doc(
The parent bond indices of the substructure bonds. The indices are guaranteed to
be unique and in ascending order.

:rtype: list[int]

.. note::
  The returned list is a copy of the internal list, so modifying the list does
  not affect the substructure.
)doc");
  sub.def("conceal_hydrogens", &P::erase_hydrogens,
          R"doc(
Convert trivial explicit hydrogen atoms of the substructure to implicit
hydrogens.

Trivial explicit hydrogen atoms are the hydrogen atoms that are connected to
only one heavy atom with a single bond and have no other neighbors (including
implicit hydrogens).

.. note::
  Invalidates all atom and bond objects.
)doc");
  sub.def(
      "clear_atoms",
      [](P &self) {
        self->clear_atoms();
        self.tick();
      },
      R"doc(
Remove all atoms from the substructure. The bonds are also removed. Other
metadata of the substructure is not affected.
)doc");
  sub.def(
      "clear_bonds",
      [](P &self) {
        self->clear_bonds();
        self.tick();
      },
      R"doc(
Remove all bonds from the substructure. The atoms and other metadata of the
substructure is not affected.
)doc");
  sub.def(
      "clear",
      [](P &self) {
        self->clear();
        self.tick();
      },
      "Effectively reset the substructure to an empty state.");
  sub.def_property(
      "id", [](P &self) { return self->id(); },
      [](P &self, int id) { self->set_id(id); }, rvp::automatic,
      R"doc(
:type: int

An integral identifier of the substructure. The identifier is mostly for use in
the protein residue numbering system.

.. warning::
  This is *not* guaranteed to be unique within the molecule.
)doc");
  sub.def_property(
      "name", [](P &self) { return self->name(); },
      [](P &self, std::string_view name) { self->name() = name; },
      rvp::automatic,
      R"doc(
:type: str

A name of the substructure. This is for user convenience and has no effect on
the substructure's behavior.
)doc");
  sub.def_property(
      "category", [](P &self) { return self->category(); },
      [](P &self, SubstructCategory cat) {
        self->category() = get_or_throw_cat(cat);
      },
      rvp::automatic,
      R"doc(
:type: SubstructureCategory

The category of the substructure. This is used to categorize the substructure.
)doc");

  add_sequence_interface(
      sub, [](P &self) { return self->size(); },
      [](P &self, int idx) {
        idx = check_subatom(*self, idx);
        return self.pysubatom(idx);
      },
      [](P &self) { return PySubAtomIterator<P>(self); });
  sub.def("__contains__", [](P &self, PySubAtom<P> &atom) {
    atom.check();
    return same_parent(self, atom.parent());
  });
  sub.def("__contains__", [](P &self, PyAtom &atom) {
    atom.check();
    const Substructure &substruct = *self;
    if (!same_parent(self.parent(), atom.parent()))
      return false;
    return substruct.contains_atom(*atom);
  });
  sub.def("__getitem__", subatom_from_parent<P>, kReturnsSubobject,
          py::arg("atom"));
}

int check_sub(const Molecule &mol, int idx) {
  return py_check_index(static_cast<int>(mol.substructures().size()), idx,
                        "substructure index out of range");
}

void insert_substruct(ProxySubstructContainer &cont, int idx,
                      const std::optional<AtomsArg> &atoms,
                      const std::optional<BondsArg> &bonds,
                      SubstructCategory cat) {
  Molecule &mol = *cont.mol();
  auto &substructs = mol.substructures();
  substructs.insert(substructs.begin() + idx,
                    create_substruct(mol, atoms, bonds, cat));
}

void substruct_erase_hydrogens(PyMol &mol, Substructure &sub) {
  auto mut = mol.mutator();

  for (auto atom: sub) {
    if (atom.data().atomic_number() != 1)
      continue;

    auto patom = atom.as_parent();
    mut->mark_atom_erase(patom);
    for (auto nei: patom) {
      AtomData &data = nei.dst().data();
      data.set_implicit_hydrogens(data.implicit_hydrogens() + 1);
    }
  }

  mol.tick();
}
}  // namespace

void ProxySubstruct::erase_hydrogens() {
  Substructure &sub = **this;
  substruct_erase_hydrogens(parent(), sub);
  self_tick();
}

void PySubstruct::erase_hydrogens() {
  Molecule &mol = *parent();
  Substructure &substruct = mol.substructures().emplace_back(std::move(**this));
  absl::Cleanup c = [&]() {
    **this = std::move(substruct);
    mol.substructures().erase(--mol.substructures().end());
  };

  substruct_erase_hydrogens(parent(), substruct);
  self_tick();
}

SubstructCategory get_or_throw_cat(SubstructCategory cat) {
  if (cat < SubstructCategory::kUnknown || cat > SubstructCategory::kChain)
    throw py::value_error(
        absl::StrCat("invalid substructure category: ", static_cast<int>(cat)));

  return cat;
}

Substructure create_substruct(Molecule &mol,
                              const std::optional<AtomsArg> &atoms,
                              const std::optional<BondsArg> &bonds,
                              SubstructCategory cat) {
  cat = get_or_throw_cat(cat);

  std::vector<int> atom_idxs, bond_idxs;
  if (atoms)
    for (py::handle obj: *atoms)
      add_atom_single(atom_idxs, mol, obj);
  if (bonds)
    for (py::handle obj: *bonds)
      add_bond_single(bond_idxs, mol, obj);

  internal::IndexSet atoms_set(std::move(atom_idxs)),
      bonds_set(std::move(bond_idxs));

  if (!atoms && !bonds)
    return mol.substructure(cat);

  if (atoms && bonds)
    return mol.substructure(std::move(atoms_set), std::move(bonds_set), cat);

  if (atoms)
    return mol.atom_substructure(std::move(atoms_set), cat);

  /* if (bonds) */
  return mol.bond_substructure(std::move(bonds_set), cat);
}

void bind_substructure(py::module &m) {
  py::enum_<SubstructCategory>(m, "SubstructureCategory", R"doc(
The category of a substructure.

This is used to categorize the substructure. Mainly used for the proteins.
)doc")
      .value("Unknown", SubstructCategory::kUnknown)
      .value("Residue", SubstructCategory::kResidue)
      .value("Chain", SubstructCategory::kChain);

  py::class_<PySubstruct> sub(m, "Substructure", R"doc(
A substructure of a molecule.

This will invalidate when the parent molecule is modified.
)doc");
  py::class_<PySubBondsWrapper<PySubstruct>> sub_bonds(m, "_SubBonds", R"doc(
A collection of bonds in a substructure.

This is a read-only collection of bonds in a substructure. The collection is
invalidated when the parent molecule is modified, or if the substructure is
modified.
)doc");
  py::class_<PySubAtom<PySubstruct>> sub_atom(m, "SubAtom", R"doc(
Atom of a substructure.
)doc");
  py::class_<PySubBond<PySubstruct>> sub_bond(m, "SubBond", R"doc(
Bond of a substructure.
)doc");
  py::class_<PySubNeigh<PySubstruct>> sub_nei(m, "SubNeighbor", R"doc(
Neighbor of a substructure.
)doc");
  py::class_<PySubNeighIterator<PySubstruct>> nei_iter(m,
                                                       "_SubNeighborIterator");

  py::class_<ProxySubstruct> psub(m, "ProxySubstructure", R"doc(
This represents a substructure managed by a molecule. If a user wishes to
create a short-lived substructure not managed by a molecule, use
:meth:`Molecule.substructure` method instead.

This will invalidate when the parent molecule is modified, or any substructures
are removed from the parent molecule. If the substructure must be kept alive,
convert the substructure first with :meth:`copy` method.

.. seealso::
  :class:`Substructure`

Here, we only provide the methods that are additional to the
:class:`Substructure` class.
)doc");
  py::class_<PySubBondsWrapper<ProxySubstruct>> psub_bonds(m, "_ProxySubBonds");
  py::class_<PySubAtom<ProxySubstruct>> psub_atom(m, "ProxySubAtom");
  py::class_<PySubBond<ProxySubstruct>> psub_bond(m, "ProxySubBond");
  py::class_<PySubNeigh<ProxySubstruct>> psub_nei(m, "ProxySubNeighbor");
  py::class_<PySubNeighIterator<ProxySubstruct>> pnei_iter(
      m, "_ProxySubNeighborIterator");

  PySubAtomIterator<PySubstruct>::bind(m, "_SubAtomIterator");
  PySubBondIterator<PySubstruct>::bind(m, "_SubBondIterator");
  SubConformersIterator<PySubstruct>::bind(m, "_SubConformersIterator");

  PySubAtomIterator<ProxySubstruct>::bind(m, "_ProxySubAtomIterator");
  PySubBondIterator<ProxySubstruct>::bind(m, "_ProxySubBondIterator");
  SubConformersIterator<ProxySubstruct>::bind(m, "_ProxySubConformersIterator");
  ProxySubstructIterator::bind(m);

  bind_substructure_kind(sub, sub_bonds, sub_atom, sub_bond, sub_nei, nei_iter);
  bind_substructure_kind(psub, psub_bonds, psub_atom, psub_bond, psub_nei,
                         pnei_iter);

  def_property_subobject(
      sub, "props",
      [](PySubstruct &self) {
        return ProxyPropertyMap(
            &self->props(), 0, [](std::uint64_t /* unused */) { return true; });
      },
      [](PySubstruct &self, const internal::PropertyMap &props) {
        self->props() = props;
      },
      rvp::automatic,
      R"doc(
:type: collections.abc.MutableMapping[str, str]

A dictionary-like object to store additional properties of the substructure. The
keys and values are both strings.
)doc");

  def_property_subobject(
      psub, "props",
      [](ProxySubstruct &self) {
        return ProxyPropertyMap(&self->props(), self);
      },
      [](ProxySubstruct &self, const internal::PropertyMap &props) {
        self->props() = props;
      },
      rvp::automatic,
      R"doc(
:type: collections.abc.MutableMapping[str, str]

A dictionary-like object to store additional properties of the substructure. The
keys and values are both strings.
)doc")
      .def(
          "copy",
          [](ProxySubstruct &self) { return PySubstruct::from_proxy(self); },
          kReturnsSubobject, R"doc(
Create a copy of the substructure. The returned substructure is not managed by
the parent molecule.
)doc");

  py::class_<ProxySubstructContainer> subs(m, "SubstructureContainer", R"doc(
A collection of substructures of a molecule.
)doc");
  add_sequence_interface(
      subs, [](ProxySubstructContainer &self) { return self.size(); },
      [](ProxySubstructContainer &self, int idx) {
        idx = check_sub(*self.mol(), idx);
        return self.get(idx);
      },
      [](ProxySubstructContainer &self) { return self.iter(); })
      .def("__setitem__",
           [](ProxySubstructContainer &self, int idx, PySubstruct &other) {
             idx = check_sub(*self.mol(), idx);
             check_parent(self.mol(), other.parent(),
                          "substructure does not belong to the same molecule");
             self.set(idx, Substructure(*other));
           })
      .def("__setitem__",
           [](ProxySubstructContainer &self, int idx, ProxySubstruct &other) {
             idx = check_sub(*self.mol(), idx);
             check_parent(self.mol(), other.parent(),
                          "substructure does not belong to the same molecule");
             self.set(idx, Substructure(*other));
           })
      .def("__delitem__",
           [](ProxySubstructContainer &self, int idx) {
             idx = check_sub(*self.mol(), idx);
             self.del(idx);
           })
      .def(
          "pop",
          [](ProxySubstructContainer &self, const std::optional<int> &oi) {
            Molecule &mol = *self.mol();

            int idx;
            if (oi) {
              idx = check_sub(mol, *oi);
            } else {
              idx = static_cast<int>(mol.substructures().size() - 1);
            }

            PySubstruct ret = PySubstruct::from_mol(
                self.mol(), std::move(mol.substructures()[idx]));
            self.del(idx);
            return ret;
          },
          py::arg("idx") = py::none(), kReturnsSubobject, R"doc(
Remove a substructure from the collection and return it.

:param idx: The index of the substructure to remove. If not given, removes the
  last substructure.
)doc")
      .def(
          "add",
          [](ProxySubstructContainer &self,
             const std::optional<AtomsArg> &atoms,
             const std::optional<BondsArg> &bonds, SubstructCategory cat) {
            int idx = self.size();
            insert_substruct(self, idx, atoms, bonds, cat);
            return self.get(idx);
          },
          py::arg("atoms") = py::none(),  //
          py::arg("bonds") = py::none(),
          py::arg("cat") = SubstructCategory::kUnknown, kReturnsSubobject,
          R"doc(
Add a substructure to the collection and return it.

:param collections.abc.Iterable[Atom] atoms: The atoms to include in the
  substructure.
:param collections.abc.Iterable[Bond] bonds: The bonds to include in the
  substructure.
:param cat: The category of the substructure.
:returns: The newly added substructure.

This has three mode of operations:

#. If both ``atoms`` and ``bonds`` are given, a substructure is created with
   the given atoms and bonds. The atoms connected by the bonds will also be
   added to the substructure, even if they are not in the ``atoms`` list.
#. If only ``atoms`` are given, a substructure is created with the given atoms.
   All bonds between the atoms will also be added to the substructure.
#. If only ``bonds`` are given, a substructure is created with the given bonds.
   The atoms connected by the bonds will also be added to the substructure.
#. If neither ``atoms`` nor ``bonds`` are given, an empty substructure is
   created.

.. tip::
  Pass empty list to ``bonds`` to create an atoms-only substructure.
)doc")
      .def(
          "add",
          [](ProxySubstructContainer &self, int idx,
             const std::optional<AtomsArg> &atoms,
             const std::optional<BondsArg> &bonds, SubstructCategory cat) {
            idx = wrap_insert_index(self.size(), idx);
            insert_substruct(self, idx, atoms, bonds, cat);
            self.mol().tock();
            return self.get(idx);
          },
          py::arg("idx"),                 //
          py::arg("atoms") = py::none(),  //
          py::arg("bonds") = py::none(),
          py::arg("cat") = SubstructCategory::kUnknown, kReturnsSubobject,
          R"doc(
Add a substructure to the collection at the given index and return it.
Effectively invalidates all currently existing substructures.

:param idx: The index of the new substructure. If negative, counts from back to
  front (i.e., the new substructure will be created at
  ``max(0, len(subs) + idx)``). Otherwise, the substructure is added at
  ``min(idx, len(subs))``. This resembles the behavior of Python's
  :meth:`list.insert` method.
:param collections.abc.Iterable[Atom] atoms: The atoms to include in the
  substructure.
:param collections.abc.Iterable[Bond] bonds: The bonds to include in the
  substructure.
:param cat: The category of the substructure.
:returns: The newly added substructure.

This has three mode of operations:

#. If both ``atoms`` and ``bonds`` are given, a substructure is created with
   the given atoms and bonds. The atoms connected by the bonds will also be
   added to the substructure, even if they are not in the ``atoms`` list.
#. If only ``atoms`` are given, a substructure is created with the given atoms.
   All bonds between the atoms will also be added to the substructure.
#. If only ``bonds`` are given, a substructure is created with the given bonds.
   The atoms connected by the bonds will also be added to the substructure.
#. If neither ``atoms`` nor ``bonds`` are given, an empty substructure is
   created.

.. tip::
  Pass empty list to ``bonds`` to create an atoms-only substructure.
)doc")
      .def(
          "append",
          [](ProxySubstructContainer &self, PySubstruct &other) {
            check_parent(self.mol(), other.parent(),
                         "substructure does not belong to the same molecule");
            self.mol()->substructures().push_back(*other);
          },
          py::arg("other"), R"doc(
Add a substructure to the collection.

:param Substructure other: The substructure to add.

.. note::
  The given substructure is copied to the collection.
)doc")
      .def(
          "append",
          [](ProxySubstructContainer &self, ProxySubstruct &other) {
            check_parent(self.mol(), other.parent(),
                         "substructure does not belong to the same molecule");
            self.mol()->substructures().push_back(*other);
          },
          py::arg("other"), R"doc(
Add a substructure to the collection.

:param ProxySubstructure other: The substructure to add.

.. note::
  The given substructure is copied to the collection.
)doc")
      .def(
          "insert",
          [](ProxySubstructContainer &self, int idx, PySubstruct &other) {
            check_parent(self.mol(), other.parent(),
                         "substructure does not belong to the same molecule");
            idx = wrap_insert_index(self.size(), idx);
            auto &subsructs = self.mol()->substructures();
            subsructs.insert(subsructs.begin() + idx, *other);
            self.mol().tock();
          },
          py::arg("idx"), py::arg("other"), R"doc(
Add a substructure to the collection at the given index. Effectively invalidates
all currently existing substructures.

:param idx: The index of the new substructure. If negative, counts from back to
  front (i.e., the new substructure will be created at
  ``max(0, len(subs) + idx)``). Otherwise, the substructure is added at
  ``min(idx, len(subs))``. This resembles the behavior of Python's
  :meth:`list.insert` method.
:param Substructure other: The substructure to add.

.. note::
  The given substructure is copied to the collection, so modifying the given
  substructure does not affect the collection.
)doc")
      .def(
          "insert",
          [](ProxySubstructContainer &self, int idx, ProxySubstruct &other) {
            check_parent(self.mol(), other.parent(),
                         "substructure does not belong to the same molecule");
            idx = wrap_insert_index(self.size(), idx);
            auto &subsructs = self.mol()->substructures();
            subsructs.insert(subsructs.begin() + idx, *other);
            self.mol().tock();
          },
          py::arg("idx"), py::arg("other"), R"doc(
Add a substructure to the collection at the given index. Effectively invalidates
all currently existing substructures.

:param idx: The index of the new substructure. If negative, counts from back to
  front (i.e., the new substructure will be created at
  ``max(0, len(subs) + idx)``). Otherwise, the substructure is added at
  ``min(idx, len(subs))``. This resembles the behavior of Python's
  :meth:`list.insert` method.
:param ProxySubstructure other: The substructure to add.

.. note::
  The given substructure is copied to the collection, so modifying the given
  substructure does not affect the collection.
)doc")
      .def("clear", &ProxySubstructContainer::clear, R"doc(
Remove all substructures from the collection. Effectively invalidates all
currently existing substructures.
)doc");
}
}  // namespace python_internal
}  // namespace nuri
