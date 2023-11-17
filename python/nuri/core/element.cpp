//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/element.h"

#include <string>
#include <string_view>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>

#include "utils.h"

namespace nuri {
namespace {
namespace py = pybind11;

using nuri_py::PyProxyCls;

std::string isotope_repr(const Isotope &iso) {
  return absl::StrCat("<Isotope ", iso.mass_number, " ",
                      PeriodicTable::get()[iso.atomic_number].symbol(), ">");
}

PYBIND11_MODULE(element, m) {
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
          [](const Isotope &self) {
            return &PeriodicTable::get()[self.atomic_number];
          },
          pybind11::return_value_policy::reference,
          R"doc(
        :type: :class:`Element`

        The element of this isotope.
)doc")
      .def_readonly("mass_number", &Isotope::mass_number, ":type: :class:`int`")
      .def_readonly("atomic_weight", &Isotope::atomic_weight,
                    ":type: :class:`float`")
      .def_readonly("abundance", &Isotope::abundance, ":type: :class:`float`")
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
          "__iter__",
          [](const IsotopeList &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::return_value_policy::reference_internal)
      .def(
          "__getitem__",
          [](const IsotopeList &self, int i) {
            if (i < 0) {
              i += static_cast<int>(self.size());
            }
            if (i < 0 || i >= self.size()) {
              throw py::index_error("_IsotopeList index out of range");
            }
            return self[i];
          },
          py::arg("index"), py::return_value_policy::reference)
      .def("__repr__",
           [](const IsotopeList &self) {
             return absl::StrCat(
                 "<_IsotopeList of ",
                 PeriodicTable::get()[self[0].atomic_number].name(), ">");
           })
      .def("__str__", [](const IsotopeList &self) {
        return "<_IsotopeList ["
               + absl::StrJoin(self, ", ",
                               [](std::string *s, const Isotope &iso) {
                                 absl::StrAppend(s, isotope_repr(iso));
                               })
               + "]>";
      });

  py_elem  //
      .def_property_readonly("atomic_number", &Element::atomic_number,
                             ":type: :class:`int`")
      .def_property_readonly("symbol", &Element::symbol, ":type: :class:`str`")
      .def_property_readonly("name", &Element::name, ":type: :class:`str`")
      .def_property_readonly("period", &Element::period, ":type: :class:`int`")
      .def_property_readonly("group", &Element::group, ":type: :class:`int`")
      .def_property_readonly("atomic_weight", &Element::atomic_weight,
                             ":type: :class:`float`")
      .def_property_readonly("covalent_radius", &Element::covalent_radius,
                             ":type: :class:`float`")
      .def_property_readonly("vdw_radius", &Element::vdw_radius,
                             ":type: :class:`float`")
      .def_property_readonly("eneg", &Element::eneg, ":type: :class:`float`")
      .def_property_readonly("major_isotope", &Element::major_isotope,
                             py::return_value_policy::reference,
                             ":type: :class:`Isotope`")
      .def_property_readonly(
          "isotopes", &Element::isotopes, py::return_value_policy::reference,
          ":type: :class:`collections.abc.Sequence` of :class:`Isotope`")
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

  PyProxyCls<PeriodicTable> pt(m, "PeriodicTable", R"doc(
    The periodic table of elements.

    The periodic table is a singleton object. You can access the periodic table
    via the :data:`nuri.periodic_table` attribute, or the factory static method
    :meth:`PeriodicTable.get()`. Both of them refer to the same object. Note
    that :class:`PeriodicTable` object is *not* constructible from the Python
    side.

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

    Note the symbols and names are case sensitive and default to *Titlecase*.
    We additionally support two common cases for all symbols and names: (1) all
    upper case, and (2) all lower case. For example, ``periodic_table["HE"]``
    and ``periodic_table["he"]`` both work, but ``periodic_table["hE"]`` would
    not. If no such element exists, a :exc:`KeyError` is raised.

    >>> periodic_table[1000]
    Traceback (most recent call last):
      ...
    KeyError: '1000'

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
  )doc");
  pt.def_static("get", &PeriodicTable::get, py::return_value_policy::reference,
                R"doc(
    Get the singleton :class:`PeriodicTable` object (same as
    :data:`nuri.periodic_table`).
)doc")
      .def(
          "__contains__",
          [](const PeriodicTable &, int z) {
            return PeriodicTable::has_element(z);
          },
          an)
      .def(
          "__contains__",
          [](const PeriodicTable &self, std::string_view arg) {
            return self.has_element(arg) || self.has_element_of_name(arg);
          },
          asn)
      .def(
          "__getitem__",
          [](const PeriodicTable &self, int z) {
            const Element *elem = self.find_element(z);
            if (elem == nullptr) {
              throw py::key_error(absl::StrCat(z));
            }
            return elem;
          },
          an, py::return_value_policy::reference)
      .def(
          "__getitem__",
          [](const PeriodicTable &self, const std::string &arg) {
            const Element *elem = self.find_element(arg);
            if (elem == nullptr) {
              elem = self.find_element_of_name(arg);
              if (elem == nullptr) {
                throw py::key_error(arg);
              }
            }
            return elem;
          },
          asn, py::return_value_policy::reference)
      .def(
          "__iter__",
          [](const PeriodicTable &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::return_value_policy::reference_internal)
      .def("__len__",
           [](const PeriodicTable &) { return PeriodicTable::kElementCount_; });

  m.attr("periodic_table") =
      py::cast(nuri::PeriodicTable::get(), py::return_value_policy::reference);
}
}  // namespace
}  // namespace nuri
