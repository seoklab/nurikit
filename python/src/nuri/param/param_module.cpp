//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/pdb.h"
#include "nuri/param/radii.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
enum class RadiiKind : int {
  kCovalent,
  kVdw,
  kConnolly,
};

std::optional<RadiiKind> parse_kind(std::string_view kind) {
  if (kind == "cov")
    return RadiiKind::kCovalent;
  if (kind == "vdw")
    return RadiiKind::kVdw;
  if (kind == "connolly")
    return RadiiKind::kConnolly;
  return std::nullopt;
}

RadiiKind try_parse_kind(std::string_view kind) {
  std::optional<RadiiKind> parsed = parse_kind(kind);
  if (!parsed) {
    throw py::value_error(absl::StrCat(
        "kind must be \"cov\", \"vdw\" or \"connolly\", got \"", kind, "\""));
  }
  return *parsed;
}

struct MissingPolicy {
  bool error;
  std::optional<RadiiKind> fallback;
};

MissingPolicy try_parse_missing(std::string_view missing) {
  if (missing == "ignore")
    return { false, std::nullopt };
  if (missing == "error")
    return { true, std::nullopt };
  if (missing == "cov" || missing == "vdw")
    return { false, parse_kind(missing) };

  throw py::value_error(absl::StrCat(
      "missing must be \"ignore\", \"error\", \"cov\" or \"vdw\", got \"",
      missing, "\""));
}

auto init_radii(py::handle templ, E::Index n) {
  if (templ.is_none()) {
    auto out = empty_numpy<E::Dynamic, 1>({ n });
    out.eigen().setConstant(kUnknownRadius);
    return out;
  }

  auto out = py_array_cast<E::Dynamic, 1>(templ, true);
  if (out.eigen().size() != n) {
    throw py::value_error(absl::StrCat("template size (", out.eigen().size(),
                                       ") does not match number of atoms (", n,
                                       ")"));
  }
  return out;
}

void assign_radii_kind(MutRef<ArrayXd> radii, const Molecule &mol, RadiiKind rk,
                       double scale) {
  auto run = [&](auto &&fill) {
    for (auto atom: mol) {
      if (radii[atom.id()] >= 0)
        continue;

      radii[atom.id()] = scale * fill(atom);
    }
  };

  switch (rk) {
  case RadiiKind::kCovalent:
    run([](Molecule::Atom atom) {
      return atom.data().element().covalent_radius();
    });
    break;
  case RadiiKind::kVdw:
    run([](Molecule::Atom atom) { return atom.data().element().vdw_radius(); });
    break;
  case RadiiKind::kConnolly:
    run(ConnollyRadii {});
    break;
  }
}

void assign_radii_kind(MutRef<ArrayXd> radii, const PDBModel &model,
                       RadiiKind rk, double scale) {
  auto run = [&](auto &&fill) {
    for (const PDBResidue &res: model.residues()) {
      for (int i: res.atom_idxs()) {
        if (radii[i] >= 0)
          continue;

        radii[i] = scale * fill(res, model.atoms()[i]);
      }
    }
  };

  switch (rk) {
  case RadiiKind::kCovalent:
    run([](const PDBResidue & /* res */, const PDBAtom &atom) {
      return atom.element().covalent_radius();
    });
    break;
  case RadiiKind::kVdw:
    run([](const PDBResidue & /* res */, const PDBAtom &atom) {
      return atom.element().vdw_radius();
    });
    break;
  case RadiiKind::kConnolly:
    run([cr = ConnollyRadii {}](const PDBResidue &res, const PDBAtom &atom) {
      return cr(res.name(), atom.name());
    });
    break;
  }
}

template <class T>
auto assign_radii(const T &obj, std::string_view kind, std::string_view missing,
                  double scale, py::handle templ, E::Index n) {
  const RadiiKind rk = try_parse_kind(kind);
  const MissingPolicy policy = try_parse_missing(missing);
  check_positive(scale, "scale");

  auto out = init_radii(templ, n);
  auto radii = out.eigen().array();

  assign_radii_kind(radii, obj, rk, scale);
  if (policy.fallback) {
    assign_radii_kind(radii, obj, *policy.fallback, scale);
  } else if (policy.error && (radii < 0).any()) {
    std::vector<int> uncovered;
    for (int i = 0; i < radii.size(); ++i) {
      if (radii[i] < 0)
        uncovered.push_back(i);
    }

    throw py::value_error(absl::StrCat("No ", kind, " radius for atoms: [",
                                       absl::StrJoin(uncovered, ", "), "]"));
  }

  return std::move(out).numpy();
}

NURI_PYTHON_MODULE(m) {
  m.def(
       "assign_radii",
       [](const PyMol &mol, std::string_view kind, std::string_view missing,
          double scale, py::handle templ) {
         return assign_radii(*mol, kind, missing, scale, templ,
                             mol->num_atoms());
       },
       py::arg("mol"), py::arg("kind"), py::kw_only(),
       py::arg("missing") = "error", py::arg("scale") = 1.0,
       py::arg("template") = py::none(), R"doc(
Look up per-atom radii of a molecule.

:param mol: The molecule.
:param kind: The radii set; see below.
:param missing: What to do with atoms the set does not cover: ``"error"``
  raises, ``"ignore"`` leaves them unassigned, and ``"cov"`` or ``"vdw"``
  fills them from that set.
:param scale: A factor applied to every radius this call assigns; `template`
  entries that are kept are not scaled.
:param template: Optional 1D array of shape ``(N,)``. Only its negative
  entries are assigned; the others are kept as given. The result is always a
  new array.
:returns: One radius per atom, in angstroms. Unassigned entries are negative.
:raises ValueError: If `kind` or `missing` is unknown, `scale` is not
  positive, the `template` shape mismatches, or an atom is not covered and
  `missing` is ``"error"``.

Supported radii sets:

* ``"cov"``: Covalent radii of the elements.
* ``"vdw"``: Van der Waals radii of the elements.
* ``"connolly"``: United-atom radii of Connolly's MS
  program\ :footcite:`param:radii:connolly-msp-1993`, classified by element,
  hydrogen count and aromaticity. Hydrogens and atoms outside the set are not
  assigned.

  The set relies on hydrogen counts, so reveal or set the implicit hydrogens
  of heavy-atom-only structures first.
)doc")
      .def(
          "assign_radii",
          [](const PDBModel &model, std::string_view kind,
             std::string_view missing, double scale, py::handle templ) {
            return assign_radii(
                model, kind, missing, scale, templ,
                static_cast<Eigen::Index>(model.atoms().size()));
          },
          py::arg("model"), py::arg("kind"), py::kw_only(),
          py::arg("missing") = "error", py::arg("scale") = 1.0,
          py::arg("template") = py::none(), R"doc(
Look up per-atom radii of a PDB model.

:param model: The model.
:param kind: The radii set; see below.
:param missing: What to do with atoms the set does not cover: ``"error"``
  raises, ``"ignore"`` leaves them unassigned, and ``"cov"`` or ``"vdw"``
  fills them from that set.
:param scale: A factor applied to every radius this call assigns; `template`
  entries that are kept are not scaled.
:param template: Optional 1D array of shape ``(N,)``. Only its negative
  entries are assigned; the others are kept as given. The result is always a
  new array.
:returns: One radius per atom, in angstroms, parallel to
  :attr:`~nuri.fmt.pdb.Model.atoms`. Unassigned entries are negative.
:raises ValueError: If `kind` or `missing` is unknown, `scale` is not
  positive, the `template` shape mismatches, or an atom is not covered and
  `missing` is ``"error"``.

Supported radii sets:

* ``"cov"``: Covalent radii of the elements.
* ``"vdw"``: Van der Waals radii of the elements.
* ``"connolly"``: United-atom radii of Connolly's MS
  program\ :footcite:`param:radii:connolly-msp-1993`, classified by residue and
  atom name. Covers the heavy atoms of the standard amino acids and MSE; other
  atoms are not assigned.
)doc");
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
