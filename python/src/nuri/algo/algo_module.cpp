//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <absl/strings/ascii.h>
#include <absl/strings/str_cat.h>

#include "nuri/algo/crdgen.h"
#include "nuri/algo/guess.h"
#include "nuri/algo/rings.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
int check_size(std::optional<int> size) {
  if (size && *size <= 0)
    throw py::value_error("max_size must be positive number");

  return size.value_or(-1);
}

template <class AtomIndexer>
std::vector<PySubstruct> rings_to_subs(PyMol &mol, const Rings &rings,
                                       AtomIndexer indexer) {
  std::vector<PySubstruct> subs;
  subs.reserve(rings.size());

  std::vector<int> bonds;
  for (const auto &ring: rings) {
    bonds.clear();

    for (int i = 0; i < ring.size(); ++i) {
      int j = (i + 1) % static_cast<int>(ring.size());
      auto bit = mol->find_bond(indexer(ring[i]), indexer(ring[j]));
      ABSL_DCHECK(bit != mol->bond_end());
      bonds.push_back(bit->id());
    }

    subs.push_back(PySubstruct::from_mol(
        mol, mol->bond_substructure(internal::IndexSet(std::move(bonds)))));
  }

  return subs;
}

std::vector<PySubstruct> rings_to_subs(PyMol &mol, const Rings &rings) {
  return rings_to_subs(mol, rings, pass_through<const int>);
}

std::vector<PySubstruct> rings_to_subs(PySubstruct &sub, const Rings &rings) {
  return rings_to_subs(sub.parent(), rings,
                       [&](int i) { return sub->atom_ids()[i]; });
}

std::vector<PySubstruct> rings_to_subs(ProxySubstruct &sub,
                                       const Rings &rings) {
  return rings_to_subs(sub.parent(), rings,
                       [&](int i) { return sub->atom_ids()[i]; });
}

template <class MoleculeLike>
std::vector<PySubstruct>
rings_to_subs(MoleculeLike &mol, const std::pair<Rings, bool> &rings_result) {
  if (!rings_result.second)
    throw py::value_error("Too many rings");

  return rings_to_subs(mol, rings_result.first);
}

void bind_rings(py::module_ &m) {
  m.def(
       "find_all_rings",
       [](PyMol &mol, std::optional<int> max_size) {
         return rings_to_subs(mol, find_all_rings(*mol, check_size(max_size)));
       },
       py::arg("mol"), py::arg("max_size") = py::none(), R"doc(
Find all rings (a.k.a. elementary circuits) in a molecule.

:param mol: The molecule to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]
:raises ValueError: If the molecule has too many rings to find. Currently, this
  will fail if and only if any atom is a member of more than 100 rings.
)doc")
      .def(
          "find_all_rings",
          [](PySubstruct &sub, std::optional<int> max_size) {
            return rings_to_subs(sub,
                                 find_all_rings(*sub, check_size(max_size)));
          },
          py::arg("sub"), py::arg("max_size") = py::none(), R"doc(
Find all rings (a.k.a. elementary circuits) in a substructure.

:param sub: The substructure to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]
:raises ValueError: If the substructure has too many rings to find. Currently,
  this will fail if and only if any atom is a member of more than 100 rings.
)doc")
      .def(
          "find_all_rings",
          [](ProxySubstruct &sub, std::optional<int> max_size) {
            return rings_to_subs(sub,
                                 find_all_rings(*sub, check_size(max_size)));
          },
          py::arg("sub"), py::arg("max_size") = py::none(), R"doc(
Find all rings (a.k.a. elementary circuits) in a substructure.

:param sub: The substructure to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]
:raises ValueError: If the substructure has too many rings to find. Currently,
  this will fail if and only if any atom is a member of more than 100 rings.

This is based on the algorithm by :cite:t:`algo:all-rings`.

.. note::
  The time complexity of this function is inherently exponential, but it is
  expected to run in a reasonable time for most molecules in practice. See the
  reference for more details.
)doc")
      .def(
          "find_relevant_rings",
          [](PyMol &mol, std::optional<int> max_size) {
            return rings_to_subs(
                mol, find_relevant_rings(*mol, check_size(max_size)));
          },
          py::arg("mol"), py::arg("max_size") = py::none(), R"doc(
Find union of all SSSR (smallest set of smallest rings) in a molecule.

:param mol: The molecule to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]
)doc")
      .def(
          "find_relevant_rings",
          [](PySubstruct &sub, std::optional<int> max_size) {
            return rings_to_subs(
                sub, find_relevant_rings(*sub, check_size(max_size)));
          },
          py::arg("sub"), py::arg("max_size") = py::none(), R"doc(
Find union of all SSSR (smallest set of smallest rings) in a substructure.

:param sub: The substructure to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]
)doc")
      .def(
          "find_relevant_rings",
          [](ProxySubstruct &sub, std::optional<int> max_size) {
            return rings_to_subs(
                sub, find_relevant_rings(*sub, check_size(max_size)));
          },
          py::arg("sub"), py::arg("max_size") = py::none(), R"doc(
Find union of all SSSR (smallest set of smallest rings) in a substructure.

:param sub: The substructure to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]
)doc")
      .def(
          "find_sssr",
          [](PyMol &mol, std::optional<int> max_size) {
            return rings_to_subs(mol, find_sssr(*mol, check_size(max_size)));
          },
          py::arg("mol"), py::arg("max_size") = py::none(), R"doc(
Find a SSSR (smallest set of smallest rings) in a molecule.

:param mol: The molecule to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]
)doc")
      .def(
          "find_sssr",
          [](PySubstruct &sub, std::optional<int> max_size) {
            return rings_to_subs(sub, find_sssr(*sub, check_size(max_size)));
          },
          py::arg("sub"), py::arg("max_size") = py::none(), R"doc(
Find a SSSR (smallest set of smallest rings) in a substructure.

:param sub: The substructure to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]
)doc")
      .def(
          "find_sssr",
          [](ProxySubstruct &sub, std::optional<int> max_size) {
            return rings_to_subs(sub, find_sssr(*sub, check_size(max_size)));
          },
          py::arg("sub"), py::arg("max_size") = py::none(), R"doc(
Find a SSSR (smallest set of smallest rings) in a substructure.

:param sub: The substructure to find rings in.
:param max_size: The maximum size of rings to find. If not specified, all rings
  are found.
:return: A list of substructures, each representing a ring.
:rtype: list[nuri.core.Substructure]

.. note::
  This function does not guarantee that the returned set is unique, nor that the
  result is reproducible even for the same molecule.
)doc");
}

void bind_guess(py::module_ &m) {
  m.def(
       "guess_everything",
       [](PyMutator &mut, int conf, double threshold) {
         conf = check_conf(mut.mol(), conf);
         bool success = guess_everything(mut.mut(), conf, threshold);
         if (!success)
           throw py::value_error("Failed to guess");
       },
       py::arg("mutator"), py::arg("conf") = 0,
       py::arg("threshold") = kDefaultThreshold, R"doc(
Guess connectivity information of a molecule, then guess types of atoms and
bonds, and number of hydrogens of a molecule.

:param mutator: The mutator of the molecule to be guessed.
:param conf: The index of the conformation used for guessing.
:param threshold: The threshold for guessing connectivity. Will be added to the
  sum of two covalent radii of the atoms to determine the maximum distance
  between two atoms to be considered as bonded.
:raises IndexError: If the conformer index is out of range.
:raises ValueError: If the guessing fails. The state of molecule is not
  guaranteed to be preserved in this case. If you want to preserve the state,
  copy the molecule before calling this function using
  :meth:`~nuri.core.Molecule.copy`.

This function is functionally equivalent to calling :func:`guess_connectivity()`
and :func:`guess_all_types()` in sequence, except that it is (slightly) more
efficient.

.. tip::
  If connectivity information is already present and is correct, consider using
  :func:`guess_all_types()`.

.. warning::
  The information present in the molecule is overwritten by this function.
)doc")
      .def(
          "guess_connectivity",
          [](PyMutator &mut, int conf, double threshold) {
            conf = check_conf(mut.mol(), conf);
            guess_connectivity(mut.mut(), conf, threshold);
          },
          py::arg("mutator"), py::arg("conf") = 0,
          py::arg("threshold") = kDefaultThreshold, R"doc(
Guess connectivity information of a molecule.

:param mutator: The mutator of the molecule to be guessed.
:param conf: The index of the conformation used for guessing.
:param threshold: The threshold for guessing connectivity. Will be added to the
  sum of two covalent radii of the atoms to determine the maximum distance
  between two atoms to be considered as bonded.
:raises IndexError: If the conformer index is out of range. This function never
  fails otherwise.

This function find extra bonds that are not in the input molecule. Unlike
:func:`guess_everything()`, this function does not touch other information
present in the molecule.

.. tip::
  If want to guess types of atoms and bonds as well, consider using
  :func:`guess_everything()`.
)doc")
      .def(
          "guess_all_types",
          [](PyMol &mol, int conf) {
            conf = check_conf(*mol, conf);
            bool success = guess_all_types(*mol, conf);
            if (!success)
              throw py::value_error("Failed to guess");
          },
          py::arg("mol"), py::arg("conf") = 0, R"doc(
Guess types of atoms and bonds, and number of hydrogens of a molecule.

:param mol: The molecule to be guessed.
:param conf: The index of the conformation used for guessing.
:raises IndexError: If the conformer index is out of range.
:raises ValueError: If the guessing fails. The state of molecule is not
  guaranteed to be preserved in this case. If you want to preserve the state,
  copy the molecule before calling this function using
  :meth:`~nuri.core.Molecule.copy`.

.. tip::
  If want to find extra bonds that are not in the input molecule, consider using
  :func:`guess_everything()`.
)doc");
}

void bind_crdgen(py::module_ &m) {
  using std::operator""s;

  m.def(
      "generate_coords",
      [](PyMol &mol, std::string_view method, int trial, int seed) {
        if (absl::AsciiStrToUpper(method) != "DG")
          throw py::value_error(absl::StrCat("Unsupported method: ", method));

        bool success = generate_coords(*mol, trial, seed);
        if (!success)
          throw py::value_error("Failed to generate coordinates");
      },
      py::arg("mol"), py::arg("method") = "DG", py::arg("max_trial") = 10,
      py::arg("seed") = 0,
      R"doc(
Generate 3D coordinates of a molecule. The generated coordinates are stored in
the last conformer of the molecule if the generation is successful.

:param mol: The molecule to generate coordinates.
:param method: The method to use for coordinate generation (case insensitive).
  Currently, only ``DG`` (distance geometry) is supported.
:param max_trial: The maximum number of trials to generate trial distances.
:param seed: The seed for the random number generator. Might not be used
  depending on the method used or if the algorithm succeeds before random
  initialization.
:raises ValueError: If the generation fails. On exception, the molecule is left
  unmodified.
)doc");
}

NURI_PYTHON_MODULE(m) {
  // For types
  py::module_::import("nuri.core");

  bind_guess(m);
  bind_rings(m);
  bind_crdgen(m);
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
