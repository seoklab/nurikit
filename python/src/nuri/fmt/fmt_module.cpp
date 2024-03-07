//
// Project nurikit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pyerrors.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl/filesystem.h>

#include <absl/strings/str_cat.h>

#include "nuri/fmt/base.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
class PyMoleculeReader {
public:
  PyMoleculeReader(std::unique_ptr<std::istream> is, std::string_view fmt)
      : stream_(std::move(is)) {
    if (!*stream_)
      throw py::value_error(absl::StrCat("Invalid stream object"));

    const MoleculeReaderFactory *factory =
        MoleculeReaderFactory::find_factory(fmt);
    if (factory == nullptr)
      throw py::value_error(absl::StrCat("Unknown format: ", fmt));

    reader_ = factory->from_stream(*stream_);
    if (!reader_)
      throw py::value_error(absl::StrCat("Failed to create reader for ", fmt));
  }

  auto next() {
    if (!reader_->getnext(block_))
      throw py::stop_iteration();

    return PyMol(reader_->parse(block_));
  }

private:
  std::unique_ptr<std::istream> stream_;
  std::unique_ptr<MoleculeReader> reader_;
  std::vector<std::string> block_;
};

namespace fs = std::filesystem;

NURI_PYTHON_MODULE(m) {
  // For types
  py::module_::import("nuri.core");

  py::class_<PyMoleculeReader>(m, "MoleculeReader")
      .def("__iter__", pass_through<PyMoleculeReader>)
      .def("__next__", &PyMoleculeReader::next);

  m.def(
       "readfile",
       [](std::string_view fmt, const fs::path &path) {
         auto pifs = std::make_unique<std::ifstream>(path);
         if (!*pifs) {
           PyErr_SetFromErrnoWithFilename(PyExc_OSError, path.c_str());
           throw py::error_already_set();
         }
         return PyMoleculeReader(std::move(pifs), fmt);
       },
       py::arg("fmt"), py::arg("path"),
       R"doc(
Read a molecule from a file.

:param fmt: The format of the file.
:param path: The path to the file.
:raises ValueError: If the format is unknown.
:rtype: collections.abc.Iterable[Molecule]
)doc")
      .def(
          "readstring",
          [](std::string_view fmt, std::string_view data) {
            return PyMoleculeReader(
                std::make_unique<std::istringstream>(std::string(data)), fmt);
          },
          py::arg("fmt"), py::arg("data"),
          R"doc(
Read a molecule from string.

:param fmt: The format of the file.
:param data: The string to read.
:raises ValueError: If the format is unknown.
:rtype: collections.abc.Iterable[Molecule]

The returned object is an iterable of molecules.

>>> for mol in nuri.readstring("smi", "C"):
...     print(mol[0].atomic_number)
6
)doc");
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
