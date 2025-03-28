//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/element.h"

#include <string>
#include <string_view>
#include <vector>

#include <absl/strings/ascii.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
std::string isotope_repr(const Isotope &iso) {
  return absl::StrCat("<Isotope ", iso.mass_number, " ",
                      kPt[iso.atomic_number].symbol(), ">");
}
}  // namespace

const Element &element_from_symbol_or_name(std::string_view symbol_or_name) {
  std::string arg(symbol_or_name);
  absl::AsciiStrToUpper(&arg);

  const Element *elem = kPt.find_element(arg);
  if (elem == nullptr) {
    elem = kPt.find_element_of_name(arg);
    if (elem == nullptr)
      throw py::key_error(std::string(symbol_or_name));
  }

  return *elem;
}

const Isotope &isotope_from_element_and_mass(const Element &elem,
                                             int mass_number) {
  const Isotope *iso = elem.find_isotope(mass_number);
  if (iso == nullptr) {
    throw py::value_error(  //
        absl::StrCat("invalid mass number ", mass_number, " for element ",
                     elem.symbol()));
  }
  return *iso;
}

void bind_element(py::module &m) {
  PyProxyCls<Element> py_elem(m, "Element", R"doc(
    An element.

    All instances of this class are immutable and singleton. If you want to
    compare two instances, just use the ``is`` operator. You can also compare
    two elements using the comparison operators, which in turn compares their
    :attr:`atomic_number` (added for convenience).

    >>> from nuri import periodic_table
    >>> periodic_table["H"] < periodic_table["He"]
    True

    Refer to the ``nuri::Element`` class in the |cppdocs| for more details.
  )doc");

  PyProxyCls<Isotope>(m, "Isotope", R"doc(
    An isotope of an element.

    All instances of this class are immutable and singleton. If you want to
    compare two instances, just use the ``is`` operator. You can also compare
    two elements using the comparison operators, which in turn compares their
    :attr:`mass_number` (added for convenience).

    Refer to the ``nuri::Element`` class in the |cppdocs| for more details.
  )doc")
      .def_property_readonly(
          "element",
          [](const Isotope &self) -> const Element & {
            return kPt[self.atomic_number];
          },
          rvp::reference,
          R"doc(
        :type: Element

        The element of this isotope.
)doc")
      .def_readonly("mass_number", &Isotope::mass_number, ":type: int")
      .def_readonly("atomic_weight", &Isotope::atomic_weight, ":type: float")
      .def_readonly("abundance", &Isotope::abundance, ":type: float")
      .def("__gt__",
           [](const Isotope &lhs, const Isotope &rhs) {
             return lhs.mass_number > rhs.mass_number;
           })
      .def("__lt__",
           [](const Isotope &lhs, const Isotope &rhs) {
             return lhs.mass_number < rhs.mass_number;
           })
      .def("__ge__",
           [](const Isotope &lhs, const Isotope &rhs) {
             return lhs.mass_number >= rhs.mass_number;
           })
      .def("__le__",
           [](const Isotope &lhs, const Isotope &rhs) {
             return lhs.mass_number <= rhs.mass_number;
           })
      .def("__repr__", isotope_repr);

  using IsotopeList = std::vector<Isotope>;
  PyProxyCls<IsotopeList>(m, "_IsotopeList")
      .def("__len__", &IsotopeList::size)
      .def(
          "__getitem__",
          [](const IsotopeList &self, int i) -> const Isotope & {
            i = py_check_index(static_cast<int>(self.size()), i,
                               "_IsotopeList index out of range");
            return self[i];
          },
          py::arg("index"), rvp::reference)
      .def("__repr__",
           [](const IsotopeList &self) {
             return absl::StrCat("<_IsotopeList of ",
                                 kPt[self[0].atomic_number].name(), ">");
           })
      .def("__str__",
           [](const IsotopeList &self) {
             return "<_IsotopeList ["
                    + absl::StrJoin(self, ", ",
                                    [](std::string *s, const Isotope &iso) {
                                      absl::StrAppend(s, isotope_repr(iso));
                                    })
                    + "]>";
           })
      .def("__iter__", [](const IsotopeList &self) {
        return py::make_iterator(self.begin(), self.end(), rvp::reference);
      });

  py_elem  //
      .def_property_readonly("atomic_number", &Element::atomic_number,
                             rvp::automatic, ":type: int")
      .def_property_readonly("symbol", &Element::symbol, rvp::automatic,
                             ":type: str")
      .def_property_readonly("name", &Element::name, rvp::automatic,
                             ":type: str")
      .def_property_readonly("period", &Element::period, rvp::automatic,
                             ":type: int")
      .def_property_readonly("group", &Element::group, rvp::automatic,
                             ":type: int")
      .def_property_readonly("atomic_weight", &Element::atomic_weight,
                             rvp::automatic, ":type: float")
      .def_property_readonly("covalent_radius", &Element::covalent_radius,
                             rvp::automatic, ":type: float")
      .def_property_readonly("vdw_radius", &Element::vdw_radius, rvp::automatic,
                             ":type: float")
      .def_property_readonly("eneg", &Element::eneg, rvp::automatic,
                             ":type: float")
      .def_property_readonly("major_isotope", &Element::major_isotope,
                             rvp::reference, ":type: Isotope")
      .def_property_readonly("isotopes", &Element::isotopes, rvp::reference,
                             ":type: collections.abc.Sequence[Isotope]")
      .def("get_isotope", isotope_from_element_and_mass, rvp::reference,
           py::arg("mass_number"), R"doc(
Get an isotope of this element by mass number.

:param mass_number: The mass number of the isotope.
:raises ValueError: If no such isotope exists.
)doc")
      .def("__gt__",
           [](const Element &lhs, const Element &rhs) {
             return lhs.atomic_number() > rhs.atomic_number();
           })
      .def("__lt__",
           [](const Element &lhs, const Element &rhs) {
             return lhs.atomic_number() < rhs.atomic_number();
           })
      .def("__ge__",
           [](const Element &lhs, const Element &rhs) {
             return lhs.atomic_number() >= rhs.atomic_number();
           })
      .def("__le__",
           [](const Element &lhs, const Element &rhs) {
             return lhs.atomic_number() <= rhs.atomic_number();
           })
      .def("__repr__", [](const Element &elem) {
        return absl::StrCat("<Element ", elem.symbol(), ">");
      });

  const py::arg an("atomic_number"), asn("atomic_symbol_or_name");

  PyProxyCls<PeriodicTable>(m, "PeriodicTable", R"doc(
The periodic table of elements.

The periodic table is a singleton object. You can access the periodic table via
the :data:`nuri.periodic_table` attribute, or the factory static method
:meth:`PeriodicTable.get()`. Both of them refer to the same object. Note that
:class:`PeriodicTable` object is *not* constructible from the Python side.

You can access the periodic table as a dictionary-like object. The keys are
atomic numbers, atomic symbols, and atomic names, tried in this order. The
returned values are :class:`Element` objects. For example:

>>> from nuri import periodic_table
>>> periodic_table[1]
<Element H>
>>> periodic_table["H"]
<Element H>
>>> periodic_table["Hydrogen"]
<Element H>

The symbols and names are case insensitive. If no such element exists, a
:exc:`KeyError` is raised.

>>> periodic_table[1000]
Traceback (most recent call last):
  ...
KeyError: '1000'

You can also test for the existence of an element using the ``in`` operator.

>>> 1 in periodic_table
True
>>> "H" in periodic_table
True
>>> "Hydrogen" in periodic_table
True
>>> 1000 in periodic_table
False

The periodic table itself is an iterable object. You can iterate over the
elements in the periodic table.

>>> for elem in periodic_table:
...     print(elem)
...
<Element Xx>
<Element H>
...
<Element Og>

Refer to the ``nuri::PeriodicTable`` class in the |cppdocs| for details.
)doc")
      .def_static("get", &PeriodicTable::get, rvp::reference,
                  R"doc(
    Get the singleton :class:`PeriodicTable` object (same as
    :data:`nuri.periodic_table`).
)doc")
      .def_static("__contains__",
                  py::overload_cast<int>(PeriodicTable::has_element), an)
      .def_static(
          "__contains__",
          [](std::string arg) {
            absl::AsciiStrToUpper(&arg);
            return kPt.has_element(arg) || kPt.has_element_of_name(arg);
          },
          asn)
      .def_static(
          "__getitem__",
          [](int z) {
            const Element *elem = kPt.find_element(z);
            if (elem == nullptr)
              throw py::key_error(absl::StrCat(z));
            return elem;
          },
          an, rvp::reference)
      .def_static("__getitem__", element_from_symbol_or_name, asn,
                  rvp::reference)
      .def_static("__len__", []() { return PeriodicTable::kElementCount_; })
      .def_static("__iter__", []() {
        return py::make_iterator(kPt.begin(), kPt.end(), rvp::reference);
      });

  m.attr("periodic_table") = py::cast(kPt, rvp::reference);
}
}  // namespace python_internal
}  // namespace nuri
