//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/pdb.h"

#include <filesystem>
#include <fstream>
#include <istream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>
#include <pybind11/typing.h>

#include "nuri/eigen_config.h"
#include "fmt_internal.h"
#include "nuri/python/exception.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
namespace fs = std::filesystem;

pyt::List<PDBModel> read_pdb_models(std::istream &is, bool skip_on_error) {
  PDBReader reader(is);

  pyt::List<PDBModel> models;
  std::vector<std::string> block;
  while (reader.getnext(block)) {
    PDBModel model = read_pdb_model(block);

    if (model.atoms().empty()) {
      if (skip_on_error)
        continue;

      throw py::value_error(
          absl::StrCat("Failed to read PDB model ", models.size()));
    }

    models.append(std::move(model));
  }

  return models;
}
}  // namespace

void bind_pdb(py::module &m) {
  py::class_<PDBResidueId>(m, "ResidueId")
      .def_readonly("chain_id", &PDBResidueId::chain_id)
      .def_readonly("res_seq", &PDBResidueId::res_seq)
      .def_property_readonly("ins_code",
                             [](const PDBResidueId &self) -> std::string_view {
                               return self.ins_code == ' '
                                          ? ""
                                          : std::string_view(&self.ins_code, 1);
                             });

  py::class_<PDBAtomSite>(m, "AtomSite")
      .def_property_readonly("altloc", &PDBAtomSite::altloc)
      .def_property_readonly("pos",
                             [](const PDBAtomSite &self) {
                               return eigen_as_numpy(self.pos());
                             })
      .def_property_readonly("occupancy", &PDBAtomSite::occupancy)
      .def_property_readonly("tempfactor", &PDBAtomSite::tempfactor);
  py::bind_vector<std::vector<PDBAtomSite>>(m, "_AtomSiteList")
      .def("__repr__", [](const std::vector<PDBAtomSite> &self) {
        return absl::StrCat("<_AtomSiteList of ", self.size(), " sites>");
      });

  py::class_<PDBAtom>(m, "Atom")
      .def_property_readonly("name", &PDBAtom::name)
      .def_property_readonly("element", &PDBAtom::element, rvp::reference)
      .def_property_readonly("hetero", &PDBAtom::hetero)
      .def_property_readonly("sites", &PDBAtom::sites);
  py::bind_vector<std::vector<PDBAtom>>(m, "_AtomList")
      .def("__repr__", [](const std::vector<PDBAtom> &self) {
        return absl::StrCat("<_AtomList of ", self.size(), " atoms>");
      });

  py::class_<PDBResidue>(m, "Residue")
      .def_property_readonly("id", &PDBResidue::id)
      .def_property_readonly("name", &PDBResidue::name)
      .def_property_readonly("atom_idxs", [](const PDBResidue &self) {
        Eigen::Map<const ArrayXi> idxs_view(
            self.atom_idxs().data(), static_cast<int>(self.atom_idxs().size()));
        return eigen_as_numpy(idxs_view);
      });
  py::bind_vector<std::vector<PDBResidue>>(m, "_ResidueList")
      .def("__repr__", [](const std::vector<PDBResidue> &self) {
        return absl::StrCat("<_ResidueList of ", self.size(), " residues>");
      });

  py::class_<PDBChain>(m, "Chain")
      .def("__repr__",
           [](const PDBChain &self) {
             return absl::StrFormat(
                 "<nuri.fmt.pdb.Chain id='%c' with %d residues>", self.id(),
                 self.res_idxs().size());
           })
      .def_property_readonly("id", &PDBChain::id)
      .def_property_readonly("res_idxs", [](const PDBChain &self) {
        Eigen::Map<const ArrayXi> idxs_view(
            self.res_idxs().data(), static_cast<int>(self.res_idxs().size()));
        return eigen_as_numpy(idxs_view);
      });
  py::bind_vector<std::vector<PDBChain>>(m, "_ChainList")
      .def("__repr__", [](const std::vector<PDBChain> &self) {
        return absl::StrCat("<_ChainList of ", self.size(), " chains>");
      });

  py::class_<PDBModel> model(m, "Model");
  model
      .def("__repr__",
           [](const PDBModel &self) {
             return absl::StrCat("<nuri.fmt.pdb.Model with ",
                                 self.chains().size(), " chains, ",
                                 self.residues().size(), " residues, ",
                                 self.atoms().size(), " atoms>");
           })
      .def_property_readonly("chains", &PDBModel::chains)
      .def_property_readonly("residues", &PDBModel::residues)
      .def_property_readonly("atoms", &PDBModel::atoms)
      .def_property_readonly("major_conf",
                             [](const PDBModel &self) {
                               return eigen_as_numpy(self.major_conf());
                             })
      .def_property_readonly("props", py::overload_cast<>(&PDBModel::props));

  m.def(
      "read_models",
      [](const fs::path &path, bool skip_on_error) {
        std::ifstream ifs(path);
        if (!ifs)
          throw file_error(path.c_str());

        return read_pdb_models(ifs, skip_on_error);
      },
      py::arg("path"), py::arg("skip_on_error") = false, R"doc(
Read PDB models from a file.

:param path: The path to the PDB file.
:param skip_on_error: Whether to skip a model if an error occurs, instead of
  raising an exception.
:raises OSError: If any file-related error occurs.
:raises ValueError: If reading a model fails, unless `skip_on_error` is set.
)doc");
}
}  // namespace python_internal
}  // namespace nuri
