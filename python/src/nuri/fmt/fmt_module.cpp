//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl/filesystem.h>

#include <absl/log/absl_log.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/types/span.h>

#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/fmt/mol2.h"
#include "nuri/fmt/sdf.h"
#include "nuri/fmt/smiles.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/exception.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
class PyMoleculeReader {
public:
  PyMoleculeReader(std::unique_ptr<std::istream> is, std::string_view fmt,
                   bool sanitize, bool skip_on_error)
      : stream_(std::move(is)), skip_on_error_(skip_on_error) {
    if (!*stream_)
      throw py::value_error(absl::StrCat("Invalid stream object"));

    const MoleculeReaderFactory *factory =
        MoleculeReaderFactory::find_factory(fmt);
    if (factory == nullptr)
      throw py::value_error(absl::StrCat("Unknown format: ", fmt));

    reader_ = factory->from_stream(*stream_);
    if (!reader_)
      throw py::value_error(absl::StrCat("Failed to create reader for ", fmt));

    sanitize_ = sanitize && !reader_->sanitized();
  }

  auto next() {
    do {
      if (!reader_->getnext(block_))
        break;

      ABSL_LOG_IF(WARNING, block_.empty())
          << "Recieved an empty block for molecule";

      Molecule mol = reader_->parse(block_);
      if (mol.empty()) {
        std::string text;
        if (block_.empty()) {
          absl::StrAppend(&text, "Empty block for molecule.");
        } else {
          absl::StrAppend(&text,
                          "Failed to parse molecule or an empty molecule "
                          "supplied. The first lines of block are: \n\n  ",
                          absl::StrJoin(absl::MakeSpan(block_).subspan(0, 5),
                                        "\n  "));
        }
        log_or_throw(text.c_str());
        continue;
      }

      if (sanitize_ && !MoleculeSanitizer(mol).sanitize_all()) {
        log_or_throw("Failed to sanitize molecule");
        continue;
      }

      return PyMol(std::move(mol));
    } while (skip_on_error_);

    throw py::stop_iteration();
  }

private:
  void log_or_throw(const char *what) const {
    if (skip_on_error_)
      ABSL_LOG(ERROR) << what;
    else
      throw py::value_error(what);
  }

  std::unique_ptr<std::istream> stream_;
  std::unique_ptr<MoleculeReader> reader_;
  std::vector<std::string> block_;
  bool sanitize_;
  bool skip_on_error_;
};

template <class F, class... Args>
std::string try_write(const Molecule &mol, std::string_view fmt, F writer,
                      Args &&...args) {
  std::string buf;
  if (!writer(buf, mol, std::forward<Args>(args)...))
    throw py::value_error(absl::StrCat("Failed to convert molecule to ", fmt));
  return buf;
}

int writer_check_conf(const Molecule &mol, std::optional<int> oconf) {
  if (!oconf)
    return -1;
  return check_conf(mol, *oconf);
}

namespace fs = std::filesystem;

NURI_PYTHON_MODULE(m) {
  // For types
  py::module_::import("nuri.core");

  py::class_<PyMoleculeReader>(m, "MoleculeReader")
      .def("__iter__", pass_through<PyMoleculeReader>)
      .def("__next__", &PyMoleculeReader::next);

  m.def(
       "readfile",
       [](std::string_view fmt, const fs::path &path, bool sanitize,
          bool skip_on_error) {
         auto pifs = std::make_unique<std::ifstream>(path);
         if (!*pifs)
           throw file_error(path.c_str());

         return PyMoleculeReader(std::move(pifs), fmt, sanitize, skip_on_error);
       },
       py::arg("fmt"), py::arg("path"), py::arg("sanitize") = true,
       py::arg("skip_on_error") = false,
       R"doc(
Read a molecule from a file.

:param fmt: The format of the file.
:param path: The path to the file.
:param sanitize: Whether to sanitize the produced molecule. Note that if the
  underlying reader produces a sanitized molecule, this option is ignored and
  the molecule is always sanitized.
:param skip_on_error: Whether to skip a molecule if an error occurs, instead of
  raising an exception.
:raises OSError: If any file-related error occurs.
:raises ValueError: If the format is unknown or sanitization fails, unless
  `skip_on_error` is set.
:rtype: collections.abc.Iterable[Molecule]
)doc")
      .def(
          "readstring",
          [](std::string_view fmt, std::string_view data, bool sanitize,
             bool skip_on_error) {
            return PyMoleculeReader(
                std::make_unique<std::istringstream>(std::string(data)), fmt,
                sanitize, skip_on_error);
          },
          py::arg("fmt"), py::arg("data"), py::arg("sanitize") = true,
          py::arg("skip_on_error") = false,
          R"doc(
Read a molecule from string.

:param fmt: The format of the file.
:param data: The string to read.
:param sanitize: Whether to sanitize the produced molecule. Note that if the
  underlying reader produces a sanitized molecule, this option is ignored and
  the molecule is always sanitized.
:param skip_on_error: Whether to skip a molecule if an error occurs, instead of
  raising an exception.
:raises ValueError: If the format is unknown or sanitization fails, unless
  `skip_on_error` is set.
:rtype: collections.abc.Iterable[Molecule]

The returned object is an iterable of molecules.

>>> for mol in nuri.readstring("smi", "C"):
...     print(mol[0].atomic_number)
6
)doc");

  m.def(
       "to_smiles",
       [](const PyMol &mol) {
         return try_write(*mol, "smiles", write_smiles, false);
       },
       py::arg("mol"), kThreadSafe, R"doc(
Convert a molecule to SMILES string.

:param mol: The molecule to convert.
:raises ValueError: If the conversion fails.
)doc")
      .def(
          "to_mol2",
          [](const PyMol &mol, std::optional<int> oconf, bool write_sub) {
            int conf = writer_check_conf(*mol, oconf);
            return try_write(*mol, "Mol2", write_mol2, conf, write_sub);
          },
          py::arg("mol"), py::arg("conf") = py::none(),
          py::arg("write_sub") = true, kThreadSafe, R"doc(
Convert a molecule to Mol2 string.

:param mol: The molecule to convert.
:param conf: The conformation to convert. If not specified, writes all
  conformations. Ignored if the molecule has no conformations.
:param write_sub: Whether to write the substructures.
:raises IndexError: If the molecule has any conformations and `conf` is out of
  range.
:raises ValueError: If the conversion fails.
)doc")
      .def(
          "to_sdf",
          [](const PyMol &mol, std::optional<int> oconf,
             std::optional<int> oversion) {
            int conf = writer_check_conf(*mol, oconf);

            SDFVersion version = SDFVersion::kAutomatic;
            if (oversion) {
              int user_version = *oversion;
              if (user_version == 2000)
                version = SDFVersion::kV2000;
              else if (user_version == 3000)
                version = SDFVersion::kV3000;
              else
                throw py::value_error(
                    absl::StrCat("Invalid SDF version: ", user_version));
            }

            return try_write(*mol, "SDF", write_sdf, conf, version);
          },
          py::arg("mol"), py::arg("conf") = py::none(),
          py::arg("version") = py::none(), kThreadSafe, R"doc(
Convert a molecule to SDF string.

:param mol: The molecule to convert.
:param conf: The conformation to convert. If not specified, writes all
  conformations. Ignored if the molecule has no conformations.
:param version: The SDF version to write. If not specified, the version is
  automatically determined. Only 2000 and 3000 are supported.
:raises IndexError: If the molecule has any conformations and `conf` is out of
  range.
:raises ValueError: If the conversion fails, or if the version is invalid.
)doc");
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
