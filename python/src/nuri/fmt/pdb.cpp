//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/pdb.h"

#include <filesystem>
#include <fstream>
#include <istream>
#include <string>
#include <utility>
#include <vector>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen/matrix.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/typing.h>

#include "fmt_internal.h"
#include "nuri/python/exception.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
namespace fs = std::filesystem;

template <class T>
class VectorIterator
    : public PyIterator<VectorIterator<T>, const std::vector<T>> {
  using Base = typename VectorIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m, const char *name) {
    return Base::bind(m, name, kReturnsSubobject);
  }

private:
  friend Base;

  static auto size_of(const std::vector<T> &v) { return v.size(); }

  static const auto &deref(const std::vector<T> &v, int idx) { return v[idx]; }
};

template <class T>
py::class_<std::vector<T>> bind_readonly_vector(py::module &m, const char *name,
                                                const char *iter_name) {
  using V = std::vector<T>;
  using I = VectorIterator<T>;

  I::bind(m, iter_name);

  py::class_<V> cls(m, name);
  add_sequence_interface(
      cls, &V::size,
      [](const V &v, int idx) {
        idx = py_check_index(static_cast<int>(v.size()), idx,
                             "Index out of range");
        return v[idx];
      },
      [](const V &v) { return I(v); });
  cls.def("__repr__", [name](const V &v) {
    return absl::StrCat("<", name, " of ", v.size(), " items>");
  });

  return cls;
}

py::str icode_as_py(char icode) {
  if (icode == ' ')
    return py::str("");

  return py::str(&icode, 1);
}

template <class T>
py::array_t<T> stdvec_as_numpy(const std::vector<T> &vec) {
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> map(
      vec.data(), static_cast<int>(vec.size()));
  return eigen_as_numpy(map);
}

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

using py::literals::operator""_a;

py::dict resid_as_dict(PDBResidueId rid) {
  py::dict res("chain_id"_a = rid.chain_id, "res_seq"_a = rid.res_seq,
               "ins_code"_a = icode_as_py(rid.ins_code));
  return res;
}

py::list sites_as_list(const std::vector<PDBAtomSite> &sites) {
  py::list res(sites.size());
  for (int i = 0; i < sites.size(); ++i) {
    const PDBAtomSite &site = sites[i];
    res[i] = py::dict("altloc"_a = site.altloc(),
                      "pos"_a = eigen_as_numpy(site.pos()),
                      "occupancy"_a = site.occupancy(),
                      "tempfactor"_a = site.tempfactor());
  }
  return res;
}

py::dict model_as_dict(const PDBModel &self) {
  py::list atoms(self.atoms().size());
  for (int i = 0; i < self.atoms().size(); ++i) {
    const PDBAtom &atom = self.atoms()[i];

    atoms[i] =
        py::dict("res_id"_a = resid_as_dict(atom.rid()), "name"_a = atom.name(),
                 "element"_a = py::cast(atom.element(), rvp::reference),
                 "formal_charge"_a = atom.fcharge(), "hetero"_a = atom.hetero(),
                 "sites"_a = sites_as_list(atom.sites()));
  }

  py::list residues(self.residues().size());
  for (int i = 0; i < self.residues().size(); ++i) {
    const PDBResidue &residue = self.residues()[i];

    residues[i] = py::dict(
        "id"_a = resid_as_dict(residue.id()), "name"_a = residue.name(),
        "atom_idxs"_a = stdvec_as_numpy(residue.atom_idxs()));
  }

  py::list chains(self.chains().size());
  for (int i = 0; i < self.chains().size(); ++i) {
    const PDBChain &chain = self.chains()[i];

    chains[i] = py::dict("id"_a = chain.id(),
                         "res_idxs"_a = stdvec_as_numpy(chain.res_idxs()));
  }

  py::dict res("chains"_a = chains, "residues"_a = residues, "atoms"_a = atoms,
               "major_conf"_a = eigen_as_numpy(self.major_conf()),
               "props"_a = py::dict(py::cast(self.props())));
  return res;
}
}  // namespace

void bind_pdb(py::module &m) {
  py::class_<PDBResidueId>(m, "ResidueId")
      .def_readonly("chain_id", &PDBResidueId::chain_id)
      .def_readonly("res_seq", &PDBResidueId::res_seq)
      .def_property_readonly("ins_code", [](const PDBResidueId &self) {
        return icode_as_py(self.ins_code);
      });

  py::class_<PDBAtomSite>(m, "AtomSite")
      .def_property_readonly("altloc", &PDBAtomSite::altloc)
      .def_property_readonly("pos",
                             [](const PDBAtomSite &self) {
                               return eigen_as_numpy(self.pos());
                             })
      .def_property_readonly("occupancy", &PDBAtomSite::occupancy)
      .def_property_readonly("tempfactor", &PDBAtomSite::tempfactor);
  bind_readonly_vector<PDBAtomSite>(m, "_AtomSiteList",
                                    "_AtomSiteListIterator");

  py::class_<PDBAtom>(m, "Atom")
      .def_property_readonly("res_id", &PDBAtom::rid)
      .def_property_readonly("name", &PDBAtom::name)
      .def_property_readonly("element", &PDBAtom::element, rvp::reference)
      .def_property_readonly("formal_charge", &PDBAtom::fcharge)
      .def_property_readonly("hetero", &PDBAtom::hetero)
      .def_property_readonly("sites", &PDBAtom::sites);
  bind_readonly_vector<PDBAtom>(m, "_AtomList", "_AtomListIterator");

  py::class_<PDBResidue>(m, "Residue")
      .def_property_readonly("id", &PDBResidue::id)
      .def_property_readonly("name", &PDBResidue::name)
      .def_property_readonly("atom_idxs", [](const PDBResidue &self) {
        return stdvec_as_numpy(self.atom_idxs());
      });
  bind_readonly_vector<PDBResidue>(m, "_ResidueList", "_ResidueListIterator");

  py::class_<PDBChain>(m, "Chain")
      .def("__repr__",
           [](const PDBChain &self) {
             return absl::StrFormat(
                 "<nuri.fmt.pdb.Chain id='%c' with %d residues>", self.id(),
                 self.res_idxs().size());
           })
      .def_property_readonly("id", &PDBChain::id)
      .def_property_readonly("res_idxs", [](const PDBChain &self) {
        return stdvec_as_numpy(self.res_idxs());
      });
  bind_readonly_vector<PDBChain>(m, "_ChainList", "_ChainListIterator");

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
      .def_property_readonly("props", py::overload_cast<>(&PDBModel::props))
      .def("as_dict", &model_as_dict, "Convert the PDB model to a dictionary.");

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
