//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/molecule.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/cleanup/cleanup.h>
#include <absl/log/absl_log.h>
#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/graph.h"
#include "nuri/core/property_map.h"
#include "nuri/python/core/containers.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
int check_nei(Molecule::Atom atom, int idx) {
  return py_check_index(atom.degree(), idx, "neighbor index out of range");
}

// NOLINTBEGIN(clang-diagnostic-unused-member-function)
class PyAtomIterator: public PyIterator<PyAtomIterator, PyMol> {
  using Base = PyAtomIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module &m) {
    return Base::bind(m, "_AtomIterator", kReturnsSubobject);
  }

private:
  friend Base;

  static int size_of(PyMol &mol) { return mol->num_atoms(); }

  static PyAtom deref(PyMol &mol, int idx) { return mol.pyatom(idx); }
};

class PyBondIterator: public PyIterator<PyBondIterator, PyMol> {
  using Base = PyBondIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module &m) {
    return Base::bind(m, "_BondIterator", kReturnsSubobject);
  }

private:
  friend Base;

  static int size_of(PyMol &mol) { return mol->num_bonds(); }

  static PyBond deref(PyMol &mol, int idx) { return mol.pybond(idx); }
};

class PyNeighborIterator: public PyIterator<PyNeighborIterator, PyAtom> {
  using Base = PyNeighborIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module &m) {
    return Base::bind(m, "_NeighborIterator", kReturnsSubobject);
  }

private:
  friend Base;

  static int size_of(PyAtom &atom) { return atom->degree(); }

  static PyNeigh deref(PyAtom &atom, int idx) { return atom.neighbor(idx); }
};

class ConformersIterator
    : public PyIterator<ConformersIterator, const std::vector<Matrix3Xd>> {
  using Base = ConformersIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module &m) {
    return Base::bind(m, "_ConformerIterator");
  }

private:
  friend Base;

  static auto size_of(const std::vector<Matrix3Xd> &confs) {
    return confs.size();
  }

  static py::array_t<double> deref(const std::vector<Matrix3Xd> &confs,
                                   int idx) {
    return eigen_as_numpy(confs[idx]);
  }
};

// NOLINTEND(clang-diagnostic-unused-member-function)

struct PyBondsWrapper {
  PyMol *mol;
};

void bind_enums(py::module &m) {
  py::enum_<constants::Hybridization>(m, "Hyb")
      .value("Unbound", constants::Hybridization::kUnbound)
      .value("Terminal", constants::Hybridization::kTerminal)
      .value("SP", constants::Hybridization::kSP)
      .value("SP2", constants::Hybridization::kSP2)
      .value("SP3", constants::Hybridization::kSP3)
      .value("SP3D", constants::Hybridization::kSP3D)
      .value("SP3D2", constants::Hybridization::kSP3D2)
      .value("Other", constants::Hybridization::kOtherHyb);

  py::implicitly_convertible<int, constants::Hybridization>();

  py::enum_<constants::BondOrder>(m, "BondOrder")
      .value("Other", constants::BondOrder::kOtherBond)
      .value("Single", constants::BondOrder::kSingleBond)
      .value("Double", constants::BondOrder::kDoubleBond)
      .value("Triple", constants::BondOrder::kTripleBond)
      .value("Quadruple", constants::BondOrder::kQuadrupleBond)
      .value("Aromatic", constants::BondOrder::kAromaticBond);

  py::implicitly_convertible<int, constants::BondOrder>();

  py::enum_<Chirality>(m, "Chirality", R"doc(
Chirality of an atom.

When viewed from the first neighboring atom of a "chiral" atom, the chirality
is determined by the spatial arrangement of the remaining neighbors. That is,
when the remaining neighbors are arranged in a clockwise direction, the
chirality is "clockwise" (:attr:`CW`), and when they are arranged in a
counter-clockwise direction, the chirality is "counter-clockwise" (:attr:`CCW`).
If the atom is not a stereocenter or the chirality is unspecified, the chirality
is "unknown" (:attr:`Unknown`).

If the atom has an implicit hydrogen, it will be always placed at the end of the
neighbor list. This is to ensure that the chirality of the atom is not affected
by adding back the implicit hydrogen (which will be placed at the end).

.. note::
  It is worth noting that this chirality definition ("NuriKit Chirality") is not
  strictly equivalent to the chirality definition in SMILES ("SMILES
  Chirality"), although it appears to be similar and often resolves to the same
  chirality.

  One notable difference is that most SMILES parser implementations place the
  implicit hydrogen where it appears in the SMILES string. [#fn-non-conforming]_
  For example, consider the stereocenter in the following SMILES string::

    [C@@H](F)(Cl)Br

  The SMILES Chirality of the atom is "clockwise" because the implicit hydrogen
  is interpreted as the first neighbor. On the other hand, the NuriKit Chirality
  of the atom is "counter-clockwise" because the implicit hydrogen is
  interpreted as the last neighbor.

  This is not a problem in most cases, because when the stereocenter is not the
  first atom of a fragment, the SMILES Chirality and the NuriKit Chirality are
  consistent. For example, a slightly modified SMILES string of the above
  example will result in a "counter-clockwise" configuration in both
  definitions::

    F[C@H](Cl)Br

  Another neighbor ordering inconsistency might occur when ring closure is
  involved. This is because a ring-closing bond **addition** could only be done
  after the partner atom is added, but the SMILES Chirality is resolved in the
  order of the **appearance** of the bonds in the SMILES string. For example,
  consider the following SMILES string, in which the two stereocenters are both
  "clockwise" in terms of the SMILES Chirality (atoms are numbered for
  reference)::

    1 2  3  4 5     6 7
    C[C@@H]1C[C@@]1(F)C

  The NuriKit Chirality of atom 2 is "counter-clockwise" because the order of
  the neighbors is 1, 3, 5, 4 in the SMILES Chirality (atom 5 precedes atom 4
  because the ring-closing bond between atoms 2 and 5 *appears before* the bond
  between atoms 2 and 4), but 1, 3, 4, 5 in the NuriKit Chirality (atom 4
  precedes atom 5 because the ring-closing bond is *added after* the bond
  between atoms 2 and 4).

  On the other hand, the NuriKit Chirality of atom 5 is "clockwise" because the
  order of the neighbors is 4, 2, 6, 7 in both definitions. Unlike the other
  stereocenter, the partner of the ring-closing bond (atom 2) is already added,
  and the ring-closing bond can now be added where it appears in the SMILES
  string.

  .. rubric:: Footnotes

  .. [#fn-non-conforming] Note that this behavior of the implementations is not
     strictly conforming to the OpenSMILES specification, which states that the
     implicit hydrogen should be considered to be the **first atom in the
     clockwise or anticlockwise accounting**.
)doc")
      .value("Unknown", Chirality::kNone)
      .value("CW", Chirality::kCW)
      .value("CCW", Chirality::kCCW);

  py::implicitly_convertible<int, Chirality>();

  py::enum_<BondConfig>(m, "BondConfig")
      .value("Unknown", BondConfig::kNone)
      .value("Trans", BondConfig::kTrans)
      .value("Cis", BondConfig::kCis);

  py::implicitly_convertible<int, BondConfig>();
}

void bind_atom(py::class_<AtomData> &atom_data, py::class_<PyAtom> &atom) {
  enable_copy(atom_data);
  add_atom_data_interface(atom_data)
      .def(py::init<>())
      .def(py::init([](int atomic_number) {
             return AtomData(get_or_throw_z(atomic_number));
           }),
           py::arg("atomic_number"))
      .def(py::init<const Element &>(), py::arg("element"))
      .def_property(
          "props", py::overload_cast<>(&AtomData::props),
          [](AtomData &self, const internal::PropertyMap &props) {
            self.props() = props;
          },
          R"doc(
:type: collections.abc.MutableMapping[str, str]

A dictionary-like object to store additional properties of the atom. The keys
and values are both strings.
)doc");

  add_common_atom_interface(atom)
      .def_property_readonly("id", &PyAtom::raw, rvp::automatic, R"doc(
:type: int

A unique identifier of the atom in the molecule. The identifier is guaranteed
to be unique within the atoms of the molecule.

This is a read-only property.

.. note::
  Implementation detail: the identifier is the index of the atom.
)doc")
      .def(
          "count_neighbors", [](PyAtom &self) { return all_neighbors(*self); },
          R"doc(
Count connected atoms to the atom. Includes both explicit and implicit
neighbors.

.. note::
  This is *not* same with ``len(atom)``. The length of the atom is the number of
  explicit neighbors, or, the iterable neighbors of the atom. Implicit hydrogens
  could not be iterated, thus not counted in the length.
)doc")
      .def(
          "count_heavy_neighbors",
          [](PyAtom &self) { return count_heavy(*self); },
          R"doc(
Count heavy neighbors connected to the atom. A heavy atom is an atom that is not
hydrogen.
)doc")
      .def(
          "count_hydrogens",
          [](PyAtom &self) { return count_hydrogens(*self); },
          R"doc(
Count hydrogen atoms connected to the atom. Includes both explicit and implicit
hydrogens.
)doc");

  add_sequence_interface(
      atom, [](PyAtom &self) { return self->degree(); },
      [](PyAtom &self, int idx) {
        Molecule::MutableAtom ma = *self;
        idx = check_nei(ma, idx);
        return self.neighbor(idx);
      },
      [](PyAtom &self) { return PyNeighborIterator(self); });
}

void bind_bond(py::class_<PyBondsWrapper> &bonds,
               py::class_<BondData> &bond_data, py::class_<PyBond> &bond) {
  enable_copy(bond_data);
  add_bond_data_interface(bond_data)
      .def(py::init<>())
      .def(py::init([](constants::BondOrder order) {
             return BondData(get_or_throw_ord(order));
           }),
           py::arg("order"), R"doc(
Create a bond data with the given bond order.

:param BondOrder|int order: The bond order of the bond.
)doc")
      .def_property(
          "props", py::overload_cast<>(&BondData::props),
          [](BondData &self, const internal::PropertyMap &props) {
            self.props() = props;
          },
          R"doc(
:type: collections.abc.MutableMapping[str, str]

A dictionary-like object to store additional properties of the bond. The keys
and values are both strings.
)doc");

  add_common_bond_interface(bond)
      .def_property_readonly("id", &PyBond::raw, rvp::automatic,
                             R"doc(
:type: int

A unique identifier of the bond in the molecule. The identifier is guaranteed
to be unique within the bonds of the molecule.

This is a read-only property.

.. note::
  Implementation detail: the identifier is the index of the bond.
)doc")
      .def("rotate", &PyBond::rotate, py::arg("angle"),
           py::arg("rotate_src") = false, py::arg("strict") = true,
           py::arg("conf") = py::none(), R"doc(
Rotate the bond by the given angle. The components connected only to the
destination atom (excluding this bond) are rotated around the bond axis.
Rotation is done in the direction of the right-hand rule, i.e., the rotation is
counter-clockwise with respect to the src -> dst vector.

:param angle: The angle to rotate the bond by, in *degrees*.
:param rotate_src: If True, the source atom side is rotated instead.
:param strict: If True, rotation will fail for multiple bonds and conjugated
  bonds. If False, the rotation will be attempted regardless.
:param conf: The index of the conformation to rotate the bond in. If not given,
  all conformations are rotated.
:raises ValueError: If the bond is not rotatable. If ``strict`` is False, it
  will be raised only if the bond is a member of a ring, as it will be
  impossible to rotate the bond without breaking the ring.
)doc");

  add_sequence_interface(
      bonds,
      [](const PyBondsWrapper &self) { return (*self.mol)->num_bonds(); },
      [](PyBondsWrapper &self, int idx) {
        PyMol &pm = *self.mol;
        idx = check_bond(*pm, idx);
        return pm.pybond(idx);
      },
      [](PyBondsWrapper &self) { return PyBondIterator(*self.mol); });
  bonds.def("__contains__", [](const PyBondsWrapper &self, PyBond &pb) {
    pb.check();
    return same_parent(*self.mol, pb.parent());
  });
}
}  // namespace

PyAtom PyMutator::add_atom(AtomData &&data) {
  int idx = mut().add_atom(std::move(data));
  return mol_->pyatom(idx);
}

PyBond PyMutator::add_bond(int src, int dst, BondData &&data) {
  if (src == dst)
    throw py::value_error("source and destination atoms are the same");

  auto [eid, ok] = mut().add_bond(src, dst, std::move(data));
  if (!ok)
    throw py::value_error("duplicate bond");
  return mol_->pybond(eid);
}

void PyBond::rotate(double angle, bool reverse, bool strict,
                    std::optional<int> conf) {
  Molecule::Bond bond = **this;
  if (strict && !bond.data().is_rotatable())
    throw py::value_error("bond is not rotatable");

  int src = bond.src().id(), dst = bond.dst().id();
  if (reverse)
    std::swap(src, dst);

  bool ok;
  if (conf) {
    check_conf(*parent(), *conf);
    ok = parent()->rotate_bond_conf(*conf, src, dst, angle);
  } else {
    ok = parent()->rotate_bond(src, dst, angle);
  }

  if (!ok)
    throw py::value_error("bond is not rotatable");
}

const Element &get_or_throw_z(int z) {
  const Element *elem = kPt.find_element(z);
  if (elem == nullptr)
    throw py::value_error(absl::StrCat("invalid atomic number: ", z));
  return *elem;
}

constants::Hybridization get_or_throw_hyb(constants::Hybridization hyb) {
  if (hyb < constants::Hybridization::kUnbound
      || hyb > constants::Hybridization::kOtherHyb)
    throw py::value_error(
        absl::StrCat("invalid hybridization: ", static_cast<int>(hyb)));

  return hyb;
}

constants::BondOrder get_or_throw_ord(constants::BondOrder ord) {
  if (ord < constants::BondOrder::kOtherBond
      || ord > constants::BondOrder::kAromaticBond)
    throw py::value_error(
        absl::StrCat("invalid bond order: ", static_cast<int>(ord)));

  return ord;
}

void log_aromatic_warning(const AtomData &atom) {
  if (!atom.is_aromatic())
    return;

  ABSL_LOG_IF(WARNING, !atom.is_conjugated())
      << "Aromatic atom is not conjugated";
  ABSL_LOG_IF(WARNING, !atom.is_ring_atom())
      << "Aromatic atom is not a ring atom";
}

void log_aromatic_warning(const BondData &bond) {
  if (!bond.is_aromatic()) {
    ABSL_LOG_IF(WARNING, bond.order() == constants::kAromaticBond)
        << "Non-aromatic bond has aromatic bond order";
    return;
  }

  ABSL_LOG_IF(WARNING, !bond.is_conjugated())
      << "Aromatic bond is not conjugated";
  ABSL_LOG_IF(WARNING, !bond.is_ring_bond())
      << "Aromatic bond is not a ring bond";
}

std::pair<int, int> check_bond_ends(const Molecule &mol, int src, int dst) {
  src = check_atom(mol, src);
  dst = check_atom(mol, dst);
  return { src, dst };
}

std::pair<Molecule::MutableAtom, Molecule::MutableAtom>
check_bond_ends(const Molecule &mol, PyAtom &src, PyAtom &dst) {
  check_parent(mol, *src.parent(),
               "source atom does not belong to the molecule");
  check_parent(mol, *dst.parent(),
               "destination atom does not belong to the molecule");
  return { *src, *dst };
}

void bind_molecule(py::module &m) {
  bind_enums(m);

  py::class_<PyMol> mol(m, "Molecule", R"doc(
A molecule.
Refer to the ``nuri::Molecule`` class in the |cppdocs| for more details.
)doc");
  py::class_<PyMutator> mutator(m, "Mutator", R"doc(
A mutator for a molecule. Use this as a context manager to make changes to a
molecule:

>>> from nuri.core import Molecule, AtomData
>>> mol = Molecule()
>>> print(mol.num_atoms())
0
>>> with mol.mutator() as mut:  # doctest: +IGNORE_RESULT
...     src = mut.add_atom(6)
...     dst = mut.add_atom(6)
...     mut.add_bond(src, dst)
>>> print(mol.num_atoms())
2
>>> print(mol.num_bonds())
1
>>> print(mol.atom(0).atomic_number)
6
>>> print(mol.bond(0).order)
BondOrder.Single

.. note::
  The mutator is invalidated when the context is exited. It is an error to use
  the mutator after the context is exited.
)doc");
  py::class_<PyNeigh> neigh(m, "Neighbor", R"doc(
A neighbor of an atom in a molecule.
)doc");
  py::class_<AtomData> atom_data(m, "AtomData", R"doc(
Data of an atom in a molecule.
Refer to the ``nuri::AtomData`` class in the |cppdocs| for more details.
)doc");
  py::class_<PyAtom> atom(m, "Atom", R"doc(
An atom of a molecule.

This is a proxy object to the :class:`AtomData` of the atom in a molecule. The
proxy object is invalidated when any changes are made to the molecule. If
underlying data must be kept alive, copy the data first with :meth:`copy_data`
method.

We only document the differences from the original class. Refer to the
:class:`AtomData` class for common properties and methods.

.. note:: Unlike the underlying data object, the atom cannot be created
  directly. Use the :meth:`Mutator.add_atom` method to add an atom to a
  molecule.
)doc");
  py::class_<BondData> bond_data(m, "BondData", R"doc(
Data of a bond in a molecule.
Refer to the ``nuri::BondData`` class in the |cppdocs| for more details.
)doc");
  py::class_<PyBond> bond(m, "Bond", R"doc(
A bond of a molecule.

This is a proxy object to the :class:`BondData` of the bond in a molecule. The
proxy object is invalidated when any changes are made to the molecule. If
underlying data must be kept alive, copy the data first with :meth:`copy_data`
method.

We only document the differences from the original class. Refer to the
:class:`BondData` class for common properties and methods.

.. note:: Unlike the underlying data object, the bond cannot be created
  directly. Use the :meth:`Mutator.add_bond` method to add a bond to a molecule.
)doc");
  py::class_<PyBondsWrapper> bonds(m, "_BondsWrapper");

  PyAtomIterator::bind(m);
  PyBondIterator::bind(m);
  PyNeighborIterator::bind(m);
  ConformersIterator::bind(m);

  bind_substructure(m);

  bind_atom(atom_data, atom);
  bind_bond(bonds, bond_data, bond);

  def_property_readonly_subobject(neigh, "src", &PyNeigh::src, rvp::automatic,
                                  R"doc(
:type: Atom

The source atom of the bond.
)doc");
  def_property_readonly_subobject(neigh, "dst", &PyNeigh::dst, rvp::automatic,
                                  R"doc(
:type: Atom

The destination atom of the bond.
)doc");
  def_property_readonly_subobject(neigh, "bond", &PyNeigh::bond, rvp::automatic,
                                  R"doc(
:type: Bond

The bond between the source and destination atoms.

.. note::
  There is no guarantee that ``nei.src.id == bond.src.id`` and
  ``nei.dst.id == bond.dst.id``; the source and destination atoms of the bond
  may be swapped.
)doc");

  enable_copy(mol);
  add_sequence_interface(
      mol, [](const PyMol &self) { return self->num_atoms(); },
      [](PyMol &self, int idx) {
        idx = check_atom(*self, idx);
        return self.pyatom(idx);
      },
      [](PyMol &self) { return PyAtomIterator(self); });
  mol.def("__contains__",
          [](PyMol &self, PyAtom &pa) {
            pa.check();
            return same_parent(self, pa.parent());
          })
      .def("__contains__", [](PyMol &self, PyBond &pb) {
        pb.check();
        return same_parent(self, pb.parent());
      });

  mol.def(py::init<>(), "Create an empty molecule.")
      .def(
          "atom",
          [](PyMol &self, int idx) {
            idx = check_atom(*self, idx);
            return self.pyatom(idx);
          },
          kReturnsSubobject, py::arg("idx"), R"doc(
Get an atom of the molecule.

:param idx: The index of the atom to get.
:returns: The atom at the index.
:rtype: Atom

.. note::
  The returned atom is invalidated when a mutator context is exited. If the atom
  must be kept alive, copy the atom data first with :meth:`Atom.copy_data`
  method.
)doc")
      .def(
          "num_atoms", [](PyMol &self) { return self->num_atoms(); }, R"doc(
Get the number of atoms in the molecule. Equivalent to ``len(mol)``.
)doc")
      .def(
          "bonds", [](PyMol &self) { return PyBondsWrapper { &self }; },
          kReturnsSubobject, R"doc(
:rtype: collections.abc.Sequence[Bond]

A wrapper object to access the bonds of the molecule. You can iterate the bonds
of the molecule with this object.
)doc")
      .def(
          "bond",
          [](PyMol &self, int idx) {
            idx = check_bond(*self, idx);
            return self.pybond(idx);
          },
          kReturnsSubobject, py::arg("idx"), R"doc(
Get a bond of the molecule.

:param idx: The index of the bond to get.
:returns: The bond at the index.
:rtype: Bond

.. note::
  The returned bond is invalidated when a mutator context is exited. If the bond
  must be kept alive, copy the bond data first with :meth:`Bond.copy_data`
  method.
)doc")
      .def(
          "bond",
          [](PyMol &self, int src, int dst) {
            std::tie(src, dst) = check_bond_ends(*self, src, dst);
            return self.pybond(src, dst);
          },
          kReturnsSubobject, py::arg("src"), py::arg("dst"), R"doc(
Get a bond of the molecule. ``src`` and ``dst`` are interchangeable.

:param src: The index of the source atom of the bond.
:param dst: The index of the destination atom of the bond.
:returns: The bond from the source to the destination atom.
:rtype: Bond
:raises ValueError: If the bond does not exist.
:raises IndexError: If the source or destination atom does not exist.

.. seealso::
  :meth:`neighbor`
.. note::
  The returned bond may not have ``bond.src.id == src`` and
  ``bond.dst.id == dst``, as the source and destination atoms of the bond may be
  swapped.
.. note::
  The returned bond is invalidated when a mutator context is exited. If the bond
  must be kept alive, copy the bond data first with :meth:`Bond.copy_data`
  method.
)doc")
      .def(
          "bond",
          [](PyMol &self, PyAtom &src, PyAtom &dst) {
            auto [sa, da] = check_bond_ends(*self, src, dst);
            return self.pybond(sa.id(), da.id());
          },
          kReturnsSubobject, py::arg("src"), py::arg("dst"), R"doc(
Get a bond of the molecule. ``src`` and ``dst`` are interchangeable.

:param src: The source atom of the bond.
:param dst: The destination atom of the bond.
:returns: The bond from the source to the destination atom.
:rtype: Bond
:raises ValueError: If the bond does not exist, or any of the atoms does not
  belong to the molecule.

.. seealso::
  :meth:`neighbor`
.. note::
  The returned bond may not have ``bond.src.id == src.id`` and
  ``bond.dst.id == dst.id``, as the source and destination atoms of the bond
  may be swapped.
.. note::
  The returned bond is invalidated when a mutator context is exited. If the bond
  must be kept alive, copy the bond data first with :meth:`Bond.copy_data`
  method.
)doc")
      .def(
          "has_bond",
          [](PyMol &self, int src, int dst) {
            std::tie(src, dst) = check_bond_ends(*self, src, dst);
            return self->find_bond(src, dst) != self->bond_end();
          },
          py::arg("src"), py::arg("dst"), R"doc(
Check if two atoms are connected by a bond.

:param src: The source atom of the bond.
:param dst: The destination atom of the bond.
:returns: Whether the source and destination atoms are connected by a bond.
:raises IndexError: If the source or destination atom does not exist.
)doc")
      .def(
          "has_bond",
          [](PyMol &self, PyAtom &src, PyAtom &dst) {
            auto [sa, da] = check_bond_ends(*self, src, dst);
            return self->find_bond(sa, da) != self->bond_end();
          },
          py::arg("src"), py::arg("dst"), R"doc(
Check if two atoms are connected by a bond.

:param src: The source atom of the bond.
:param dst: The destination atom of the bond.
:returns: Whether the source and destination atoms are connected by a bond.
:raises ValueError: If any of the atoms does not belong to the molecule.
)doc")
      .def(
          "num_bonds", [](PyMol &self) { return self->num_bonds(); }, R"doc(
Get the number of bonds in the molecule. Equivalent to ``len(mol.bonds)``.
)doc")
      .def(
          "neighbor",
          [](PyMol &self, int src, int dst) {
            std::tie(src, dst) = check_bond_ends(*self, src, dst);
            return self.pyneighbor(src, dst);
          },
          kReturnsSubobject, py::arg("src"), py::arg("dst"), R"doc(
Get a neighbor of the molecule.

:param src: The index of the source atom of the neighbor.
:param dst: The index of the destination atom of the neighbor.
:returns: The neighbor from the source to the destination atom.
:rtype: Neighbor
:raises ValueError: If the underlying bond does not exist.
:raises IndexError: If the source or destination atom does not exist.

.. seealso::
  :meth:`bond`
.. note::
  Unlike :meth:`bond`, the returned neighbor is always guaranteed to have
  ``nei.src.id == src`` and ``nei.dst.id == dst``.
.. note::
  The returned neighbor is invalidated when a mutator context is exited.
)doc")
      .def(
          "neighbor",
          [](PyMol &self, PyAtom &src, PyAtom &dst) {
            auto [sa, da] = check_bond_ends(*self, src, dst);
            return self.pyneighbor(sa.id(), da.id());
          },
          kReturnsSubobject, py::arg("src"), py::arg("dst"), R"doc(
Get a neighbor of the molecule.

:param src: The source atom of the neighbor.
:param dst: The destination atom of the neighbor.
:returns: The neighbor from the source to the destination atom.
:rtype: Neighbor
:raises ValueError: If the underlying bond does not exist, or any of the atoms
  does not belong to the molecule.

.. seealso::
  :meth:`bond`
.. note::
  Unlike :meth:`bond`, the returned neighbor is always guaranteed to have
  ``nei.src.id == src.id`` and ``nei.dst.id == dst.id``.
.. note::
  The returned neighbor is invalidated when a mutator context is exited.
)doc")
      .def(
          "sanitize",
          [](PyMol &self, bool conjugation, bool aromaticity, bool hyb,
             bool valence) {
            if (!conjugation && !aromaticity && !hyb && !valence) {
              ABSL_LOG(WARNING) << "no sanitization requested";
              return;
            }

            if (!conjugation && (aromaticity || hyb || valence)) {
              ABSL_LOG(WARNING)
                  << "turning conjugation on to satisfy other constraints";
              conjugation = true;
            }

            MoleculeSanitizer san(*self);
            if (conjugation && !san.sanitize_conjugated())
              throw py::value_error("failed to satisfy conjugation");
            if (aromaticity && !san.sanitize_aromaticity())
              throw py::value_error("failed to satisfy aromaticity");
            if (hyb && !san.sanitize_hybridization())
              throw py::value_error("failed to satisfy hybridization");
            if (valence && !san.sanitize_valence())
              throw py::value_error("failed to satisfy valence");
          },
          py::kw_only(),  //
          py::arg("conjugation") = true, py::arg("aromaticity") = true,
          py::arg("hyb") = true, py::arg("valence") = true, R"doc(
Sanitize the molecule.

:param conjugation: If True, sanitize conjugation.
:param aromaticity: If True, sanitize aromaticity.
:param hyb: If True, sanitize hybridization.
:param valence: If True, sanitize valence.
:raises ValueError: If the sanitization fails.

.. note::
  The sanitization is done in the order of conjugation, aromaticity,
  hybridization, and valence. If any of the sanitization fails, the subsequent
  sanitization will not be attempted.

.. note::
  The sanitization is done in place. The state of molecule will be mutated even
  if the sanitization fails.

.. note::
  If any of the other three sanitization is requested, the conjugation will be
  automatically turned on.

.. warning::
  This interface is experimental and may change in the future.
)doc")
      .def(
          "get_conf",
          [](PyMol &self, int conf) {
            conf = check_conf(*self, conf);
            return eigen_as_numpy(self->confs()[conf]);
          },
          py::arg("conf") = 0, R"doc(
Get the coordinates of the atoms in a conformation.

:param conf: The index of the conformation to get the coordinates from.
:returns: The coordinates of the atoms in the conformation, as a 2D array of
  shape ``(num_atoms, 3)``.

.. note::
  The returned array is a copy of the coordinates. To update the coordinates,
  use the :meth:`set_conf` method.
)doc")
      .def(
          "set_conf",
          [](PyMol &self, const py::handle &obj, int conf) {
            conf = check_conf(*self, conf);
            assign_conf(self->confs()[conf], obj);
          },
          py::arg("coords"), py::arg("conf") = 0, R"doc(
Set the coordinates of the atoms in a conformation.

:param coords: The coordinates of the atoms in the conformation. Must be
  convertible to a numpy array of shape ``(num_atoms, 3)``.
:param conf: The index of the conformation to set the coordinates to.
)doc")
      .def(
          "add_conf",
          [](PyMol &self, const py::handle &obj) {
            int ret = static_cast<int>(self->confs().size());
            Matrix3Xd new_conf(3, self->num_atoms());
            assign_conf(new_conf, obj);
            self->confs().emplace_back(std::move(new_conf));
            return ret;
          },
          py::arg("coords"), R"doc(
Add a conformation to the molecule at the end.

:param coords: The coordinates of the atoms in the conformation. Must be
  convertible to a numpy array of shape ``(num_atoms, 3)``.
:returns: The index of the added conformation.
)doc")
      .def(
          "add_conf",
          [](PyMol &self, const py::handle &obj, int conf) {
            conf =
                wrap_insert_index(static_cast<int>(self->confs().size()), conf);

            auto it = self->confs().begin() + conf;
            Matrix3Xd new_conf(3, self->num_atoms());
            assign_conf(new_conf, obj);
            self->confs().emplace(it, std::move(new_conf));
            return conf;
          },
          py::arg("coords"), py::arg("conf"), R"doc(
Add a conformation to the molecule.

:param coords: The coordinates of the atoms in the conformation. Must be
  convertible to a numpy array of shape ``(num_atoms, 3)``.
:param conf: The index of the conformation to add the coordinates to. If
  negative, counts from back to front (i.e., the new conformer
  will be created at ``max(0, num_confs() + conf)``). Otherwise, the
  coordinates are added at ``min(conf, num_confs())``. This resembles
  the behavior of Python's :meth:`list.insert` method.
:returns: The index of the added conformation.
)doc")
      .def(
          "del_conf",
          [](PyMol &self, int conf) {
            conf = check_conf(*self, conf);
            self->confs().erase(self->confs().begin() + conf);
          },
          py::arg("conf"), R"doc(
Remove a conformation from the molecule.

:param conf: The index of the conformation to remove.
)doc")
      .def(
          "num_confs", [](PyMol &self) { return self->confs().size(); },
          R"doc(
Get the number of conformations of the molecule.
)doc")
      .def(
          "clear_confs", [](PyMol &self) { self->confs().clear(); },
          R"doc(
Remove all conformations from the molecule.
)doc")
      .def(
          "conformers",
          [](PyMol &self) { return ConformersIterator(self->confs()); },
          kReturnsSubobject, R"doc(
Get an iterable object of all conformations of the molecule. Each conformation
is a 2D array of shape ``(num_atoms, 3)``. It is not available to update the
coordinates from the returned conformers; you should manually assign to the
conformers to update the coordinates.

:rtype: collections.abc.Iterable[numpy.ndarray]

.. seealso::
  :meth:`get_conf`, :meth:`set_conf`
)doc")
      .def(
          "mutator", [](PyMol &self) { return PyMutator(self); },
          kReturnsSubobject, R"doc(
Get a mutator for the molecule. Use this as a context manager to make changes to
the molecule.

.. note::
  The mutator will invalidate all atom and bond objects when the context is
  exited, whether or not the changes are made. If the objects must be kept
  alive, copy the data first with :meth:`Atom.copy_data` and
  :meth:`Bond.copy_data` methods.
.. note::
  Successive calls to this method will raise an exception if the previous
  mutator is not finalized.
)doc")
      .def(
          "reveal_hydrogens",
          [](PyMol &self, bool update_confs, bool optimize) {
            absl::Cleanup c = [&] { self.tick(); };
            if (!self->add_hydrogens(update_confs, optimize))
              throw py::value_error("failed to add hydrogens");
          },
          py::arg("update_confs") = true, py::arg("optimize") = true,
          R"doc(
Convert implicit hydrogen atoms of the molecule to explicit hydrogens.

:param update_confs: If True, the conformations of the molecule will be
  updated to include the newly added hydrogens. When set to False, the
  coordinates of the added hydrogens will have garbage values. Default to True.
:param optimize: If True, the conformations will be optimized after adding
  hydrogens. Default to True. This parameter is ignored if ``update_confs`` is
  False.
:raises ValueError: If the hydrogens cannot be added. This can only happen if
  ``update_confs`` is True and the molecule has at least one conformation.

.. note::
  Invalidates all atom and bond objects.
)doc")
      .def(
          "conceal_hydrogens",
          [](PyMol &self) {
            self->erase_hydrogens();
            self.tick();
          },
          R"doc(
Convert trivial explicit hydrogen atoms of the molecule to implicit hydrogens.

Trivial explicit hydrogen atoms are the hydrogen atoms that are connected to
only one heavy atom with a single bond and have no other neighbors (including
implicit hydrogens).

.. note::
  Invalidates all atom and bond objects.
)doc")
      .def("clear_atoms", &PyMol::clear_atoms,
           R"doc(
Clear all atoms and bonds of the molecule. Other metadata are left unmodified.

.. note::
  Invalidates all atom and bond objects.
.. warning::
  Molecules with active mutator context cannot clear atoms.
.. seealso::
  :meth:`Mutator.clear_atoms`
)doc")
      .def("clear_bonds", &PyMol::clear_bonds,
           R"doc(
Clear all bonds of the molecule. Atoms and other metadata are left unmodified.

.. note::
  Invalidates all atom and bond objects.
.. warning::
  Molecules with active mutator context cannot clear bonds.
.. seealso::
  :meth:`Mutator.clear_bonds`
)doc")
      .def("clear", &PyMol::clear,
           R"doc(
Effectively resets the molecule to an empty state.

.. note::
  Invalidates all atom and bond objects.

.. warning::
  Molecules with active mutator context cannot be cleared.
.. seealso::
  :meth:`Mutator.clear`
)doc")
      .def(
          "add_from",
          [](PyMol &self, const PyMol &other) {
            self->merge(*other);
            self.tick();
          },
          py::arg("other"),
          R"doc(
Add all atoms and bonds from another molecule to the molecule.

:param other: The molecule to add from.
)doc")
      .def(
          "add_from",
          [](PyMol &self, PySubstruct &other) {
            self->merge(*other);
            self.tick();
          },
          py::arg("other"),
          R"doc(
Add all atoms and bonds from a substructure to the molecule.

:param other: The substructure to add from.
)doc")
      .def(
          "add_from",
          [](PyMol &self, ProxySubstruct &other) {
            self->merge(*other);
            self.tick();
          },
          py::arg("other"),
          R"doc(
Add all atoms and bonds from a substructure to the molecule.

:param other: The substructure to add from.
)doc")
      .def(
          "sub",
          [](PyMol &self, const std::optional<AtomsArg> &as,
             const std::optional<BondsArg> &bs, SubstructCategory cat) {
            return PySubstruct::from_mol(self,
                                         create_substruct(*self, as, bs, cat));
          },
          py::arg("atoms") = py::none(),  //
          py::arg("bonds") = py::none(),
          py::arg("cat") = SubstructCategory::kUnknown, kReturnsSubobject,
          R"doc(
Create a substructure of the molecule.

:param collections.abc.Iterable[Atom | int] atoms: The atoms to include in the
  substructure.
:param collections.abc.Iterable[Bond | int] bonds: The bonds to include in the
  substructure.
:param cat: The category of the substructure.

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
)doc");
  def_property_readonly_subobject(
      mol, "subs", [](PyMol &self) { return ProxySubstructContainer(self); },
      rvp::automatic, R"doc(
:type: SubstructureContainer

A container of substructures of the molecule. You can iterate the substructures
of the molecule with this object.
)doc");
  def_property_subobject(
      mol, "props",
      [](PyMol &self) {
        return ProxyPropertyMap(
            &self->props(), 0, [](std::uint64_t /* unused */) { return true; });
      },
      [](PyMol &self, const internal::PropertyMap &props) {
        self->props() = props;
      },
      rvp::automatic,
      R"doc(
:type: collections.abc.MutableMapping[str, str]

A dictionary-like object to store additional properties of the molecule. The
keys and values are both strings.
)doc")
      .def_property(
          "name", [](PyMol &self) { return self->name(); },
          [](PyMol &self, std::string_view name) { self->name() = name; },
          rvp::automatic, ":type: str")
      .def(
          "num_sssr", [](PyMol &self) { return self->num_sssr(); }, R"doc(
The number of smallest set of smallest rings (SSSR) in the molecule.

.. warning::
   This might return incorrect value if the molecule is in a mutator context.
)doc")
      .def(
          "num_fragments", [](PyMol &self) { return self->num_fragments(); },
          R"doc(
The number of connected components (fragments) in the molecule.

.. warning::
   This might return incorrect value if the molecule is in a mutator context.
)doc");

  mutator
      .def(
          "add_atom",
          [](PyMutator &self, const AtomData *data) {
            return self.add_atom(default_if_null(data));
          },
          py::arg("data") = py::none(), kReturnsSubobject, R"doc(
Add an atom to the molecule.

:param data: The data of the atom to add. If not given, the atom is added with
  default properties.
:returns: The created atom.
)doc")
      .def(
          "add_atom",
          [](PyMutator &self, int atomic_number) {
            return self.add_atom(AtomData(get_or_throw_z(atomic_number)));
          },
          py::arg("atomic_number"), kReturnsSubobject, R"doc(
Add an atom to the molecule.

:param atomic_number: The atomic number of the atom to add. Other properties of
  the atom are set to default.
:returns: The created atom.
)doc")
      .def(
          "mark_atom_erase",
          [](PyMutator &self, int idx) {
            idx = check_atom(self.mol(), idx);
            self.mut().mark_atom_erase(idx);
          },
          R"doc(
Mark an atom to be erased from the molecule. The atom is not erased until the
context manager is exited.

:param idx: The index of the atom to erase.
)doc")
      .def(
          "mark_atom_erase",
          [](PyMutator &self, PyAtom &pa) {
            auto cpp_atom = check_atom(self.mol(), pa);
            self.mut().mark_atom_erase(cpp_atom);
          },
          R"doc(
Mark an atom to be erased from the molecule. The atom is not erased until the
context manager is exited.

:param atom: The atom to erase.
)doc")
      .def(
          "add_bond",
          [](PyMutator &self, int src, int dst, constants::BondOrder order) {
            std::tie(src, dst) = check_bond_ends(self.mol(), src, dst);
            return self.add_bond(src, dst, BondData(get_or_throw_ord(order)));
          },
          py::arg("src"), py::arg("dst"),
          py::arg("order") = constants::kSingleBond, kReturnsSubobject,
          R"doc(
Add a bond to the molecule.

:param src: The index of the source atom.
:param dst: The index of the destination atom.
:param order: The order of the bond to add. Other properties of the bond are set
  to default. If not given, the bond is added with single bond order.
:returns: The created bond.
)doc")
      .def(
          "add_bond",
          [](PyMutator &self, PyAtom &src, PyAtom &dst,
             constants::BondOrder order) {
            auto [sa, da] = check_bond_ends(self.mol(), src, dst);
            return self.add_bond(sa.id(), da.id(),
                                 BondData(get_or_throw_ord(order)));
          },
          py::arg("src"), py::arg("dst"),
          py::arg("order") = constants::kSingleBond, kReturnsSubobject,
          R"doc(
Add a bond to the molecule.

:param src: The source atom.
:param dst: The destination atom.
:param order: The order of the bond to add. Other properties of the bond are set
  to default. If not given, the bond is added with single bond order.
:returns: The created bond.
)doc")
      .def(
          "add_bond",
          [](PyMutator &self, int src, int dst, const BondData &data) {
            std::tie(src, dst) = check_bond_ends(self.mol(), src, dst);
            return self.add_bond(src, dst, BondData { data });
          },
          py::arg("src"), py::arg("dst"), py::arg("data"), kReturnsSubobject,
          R"doc(
Add a bond to the molecule.

:param src: The index of the source atom.
:param dst: The index of the destination atom.
:param data: The data of the bond to add.
:returns: The created bond.
)doc")
      .def(
          "add_bond",
          [](PyMutator &self, PyAtom &src, PyAtom &dst, const BondData &data) {
            auto [sa, da] = check_bond_ends(self.mol(), src, dst);
            return self.add_bond(sa.id(), da.id(), BondData { data });
          },
          py::arg("src"), py::arg("dst"), py::arg("data"), kReturnsSubobject,
          R"doc(
Add a bond to the molecule.

:param src: The source atom.
:param dst: The destination atom.
:param data: The data of the bond to add.
:returns: The created bond.
)doc")
      .def(
          "mark_bond_erase",
          [](PyMutator &self, int src, int dst) {
            std::tie(src, dst) = check_bond_ends(self.mol(), src, dst);
            int idx = get_or_throw_bond(self.mol(), src, dst);
            self.mut().mark_bond_erase(idx);
          },
          py::arg("src"), py::arg("dst"),
          R"doc(
Mark a bond to be erased from the molecule. The bond is not erased until the
context manager is exited.

:param src: The index of the source atom of the bond.
:param dst: The index of the destination atom of the bond.
:raises ValueError: If the bond does not exist.
)doc")
      .def(
          "mark_bond_erase",
          [](PyMutator &self, PyAtom &src, PyAtom &dst) {
            auto [sa, da] = check_bond_ends(self.mol(), src, dst);
            int idx = get_or_throw_bond(self.mol(), sa.id(), da.id());
            self.mut().mark_bond_erase(idx);
          },
          py::arg("src"), py::arg("dst"),
          R"doc(
Mark a bond to be erased from the molecule. The bond is not erased until the
context manager is exited.

:param src: The source atom of the bond.
:param dst: The destination atom of the bond.
:raises ValueError: If the bond does not exist.
)doc")
      .def(
          "mark_bond_erase",
          [](PyMutator &self, int idx) {
            idx = check_bond(self.mol(), idx);
            self.mut().mark_bond_erase(idx);
          },
          py::arg("idx"),
          R"doc(
Mark a bond to be erased from the molecule. The bond is not erased until the
context manager is exited.

:param idx: The index of the bond to erase.
)doc")
      .def(
          "mark_bond_erase",
          [](PyMutator &self, PyBond &pb) {
            auto cpp_bond = check_bond(self.mol(), pb);
            self.mut().mark_bond_erase(cpp_bond.id());
          },
          py::arg("bond"),
          R"doc(
Mark a bond to be erased from the molecule. The bond is not erased until the
context manager is exited.

:param bond: The bond to erase.
)doc")
      .def("clear_atoms", &PyMutator::clear_atoms, R"doc(
Clear all atoms and bonds of the molecule. Other metadata are left unmodified.

.. note::
  Invalidates all atom and bond objects.
)doc")
      .def("clear_bonds", &PyMutator::clear_bonds, R"doc(
Clear all bonds of the molecule. Atoms and other metadata are left unmodified.

.. note::
  Invalidates all atom and bond objects.
)doc")
      .def("clear", &PyMutator::clear, R"doc(
Effectively resets the molecule to an empty state.

.. note::
  Invalidates all atom and bond objects.
)doc")
      .def("__enter__", &PyMutator::initialize)
      .def("__exit__",
           [](PyMutator &self, const py::args & /* args */,
              const py::kwargs & /* kwargs */) { self.finalize(); });
}
}  // namespace python_internal
}  // namespace nuri
