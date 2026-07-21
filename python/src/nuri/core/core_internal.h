//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_CORE_CORE_INTERNAL_H_
#define NURI_PYTHON_CORE_CORE_INTERNAL_H_

#include <optional>
#include <variant>

#include <pybind11/pybind11.h>
#include <pybind11/typing.h>

#include "nuri/core/molecule.h"
#include "nuri/python/core/core_module.h"

namespace nuri {
namespace python_internal {
using AtomsArg = pyt::Iterable<std::variant<PyAtom, int>>;
using BondsArg = pyt::Iterable<std::variant<PyBond, int>>;

extern Substructure create_substruct(Molecule &mol,
                                     const std::optional<AtomsArg> &atoms,
                                     const std::optional<BondsArg> &bonds,
                                     SubstructCategory cat);

/* Called from bind_molecule, don't call directly */
extern void bind_substructure_impl(py::module &m);

extern void bind_element_impl(py::module &m);
extern void bind_molecule_impl(py::module &m);

extern void bind_geometry(py::module &m);
}  // namespace python_internal
}  // namespace nuri

#endif /* NURI_PYTHON_CORE_CORE_INTERNAL_H_ */
