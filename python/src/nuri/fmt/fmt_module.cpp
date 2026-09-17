//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <deque>
#include <filesystem>
#include <fstream>
#include <istream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <absl/algorithm/container.h>
#include <absl/cleanup/cleanup.h>
#include <absl/log/absl_log.h>
#include <absl/strings/str_cat.h>
#include <absl/synchronization/mutex.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl/filesystem.h>

#include "nuri/eigen_config.h"
#include "fmt_internal.h"
#include "nuri/algo/guess.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/fmt/mol2.h"
#include "nuri/fmt/parse_result.h"
#include "nuri/fmt/pdb.h"
#include "nuri/fmt/sdf.h"
#include "nuri/fmt/smiles.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/exception.h"
#include "nuri/python/stream.h"
#include "nuri/python/typing.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
bool all_confs_finite(const Molecule &mol) {
  return absl::c_all_of(mol.confs(),
                        [](const Matrix3Xd &conf) { return conf.allFinite(); });
}

class PyMoleculeReader {
public:
  PyMoleculeReader(std::unique_ptr<std::istream> is, std::string_view fmt,
                   bool sanitize, bool skip_on_error)
      : stream_(std::move(is)), sanitize_(sanitize),
        skip_on_error_(skip_on_error) {
    init(fmt);
  }

  PyMoleculeReader(std::unique_ptr<PyIStream> is, std::string_view fmt,
                   bool sanitize, bool skip_on_error)
      : pybuf_(&is->buf()), stream_(std::move(is)), sanitize_(sanitize),
        skip_on_error_(skip_on_error) {
    init(fmt);
  }

  PyMoleculeReader(py::object data, std::string_view fmt, bool sanitize,
                   bool skip_on_error)
      : owner_(std::move(data)), sanitize_(sanitize),
        skip_on_error_(skip_on_error) {
    stream_ = std::make_unique<internal::ViewIStream>(borrow_utf8(owner_));
    init(fmt);
  }

  auto next() {
    do {
      auto res = next_molecule();
      if (res.status() == ParseStatus::kEOF)
        break;

      if (!res) {
        std::string buf =
            absl::StrCat("Failed to parse molecule: ", res.error_msg());
        log_or_throw(buf.c_str());
        continue;
      }

      if (!all_confs_finite(*res)) {
        log_or_throw("Molecule has non-finite (NaN or infinite) coordinates.");
        continue;
      }

      if (guess_ && res->is_3d()) {
        if (!internal::guess_update_subs(*res)) {
          log_or_throw("Failed to guess molecule atom/bond types");
          continue;
        }
      } else if (sanitize_ && !MoleculeSanitizer(*res).sanitize_all()) {
        ABSL_LOG_IF(WARNING, guess_ && !res->is_3d())
            << "Reader might produce molecules with invalid bonds, but the "
               "molecule is missing 3D coordinates; guessing is disabled.";

        log_or_throw("Failed to sanitize molecule");
        continue;
      }

      return PyMol(std::move(*res));
    } while (skip_on_error_);

    throw py::stop_iteration();
  }

private:
  void init(std::string_view fmt) {
    if (!*stream_)
      throw py::value_error(absl::StrCat("Invalid stream object"));

    const MoleculeReaderFactory *factory =
        MoleculeReaderFactory::find_factory(fmt);
    if (factory == nullptr)
      throw py::value_error(absl::StrCat("Unknown format: ", fmt));

    reader_ = factory->from_stream(*stream_);
    if (!reader_)
      throw py::value_error(absl::StrCat("Failed to create reader for ", fmt));
    check_stream();

    guess_ = sanitize_ && !reader_->bond_valid();
  }

  void check_stream() {
    if (pybuf_ != nullptr)
      pybuf_->rethrow_pending();
  }

  ParseResult<Molecule> next_molecule() {
    std::unique_lock<absl::Mutex> lock(mutex_);

    while (pending_.empty()) {
      auto record = reader_->make_record();
      const bool advanced = reader_->getnext(*record);
      check_stream();
      if (!advanced) {
        mutex_.Await(absl::Condition(
            +[](PyMoleculeReader *self) {
              return !self->pending_.empty() || self->in_flight_ == 0;
            },
            this));
        if (!pending_.empty())
          break;

        return ParseResult<Molecule>::eof();
      }

      ++in_flight_;
      lock.unlock();
      absl::Cleanup finish = [&] {
        if (!lock.owns_lock())
          lock.lock();
        --in_flight_;
      };

      auto result = record->parse();
      lock.lock();

      if (!result)
        return ParseResult<Molecule>::error(std::move(result).error_msg());

      pending_.insert(pending_.end(),
                      std::make_move_iterator(result->data().begin()),
                      std::make_move_iterator(result->data().end()));
    }

    Molecule mol = std::move(pending_.front());
    pending_.pop_front();
    return mol;
  }

  void log_or_throw(const char *what) const {
    if (skip_on_error_)
      ABSL_LOG(ERROR) << what;
    else
      throw py::value_error(what);
  }

  py::object owner_;
  PyStreamBuf *pybuf_ = nullptr;
  std::unique_ptr<std::istream> stream_;
  std::unique_ptr<MoleculeReader> reader_;

  absl::Mutex mutex_;
  std::deque<Molecule> pending_;
  int in_flight_ = 0;

  bool sanitize_;
  bool skip_on_error_;
  bool guess_;
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

  py::class_<PyMoleculeReader>(m, "_MoleculeReader")
      .def("__iter__", pass_through<PyMoleculeReader>, kThreadSafe)
      .def("__next__", &PyMoleculeReader::next, kThreadSafe);

  m.def(
       "readfile",
       [](std::string_view fmt, const fs::path &path, bool sanitize,
          bool skip_on_error) {
         auto pifs = std::make_unique<std::ifstream>(path);
         if (!*pifs)
           throw file_error(path.c_str());

         return masquerade_cast<pyt::Iterator<PyMol>>(
             std::make_unique<PyMoleculeReader>(std::move(pifs), fmt, sanitize,
                                                skip_on_error));
       },
       py::arg("fmt"), py::arg("path"), py::arg("sanitize") = true,
       py::arg("skip_on_error") = false,
       R"doc(
Read a molecule from a file.

:param fmt: The format of the file.
:param path: The path to the file.
:param sanitize: Whether to sanitize the produced molecule. For formats that is
  known to produce molecules with insufficient bond information (e.g. PDB), this
  option will trigger guessing based on the 3D coordinates
  (:func:`nuri.algo.guess_everything()`).
:param skip_on_error: Whether to skip a molecule if an error occurs, instead of
  raising an exception.
:raises OSError: If any file-related error occurs.
:raises ValueError: If the format is unknown, or if a molecule cannot be read
  or sanitized, unless `skip_on_error` is set.

.. note::
  The yielded molecules always have finite coordinates; NaN or infinite
  coordinates are considered an error.
)doc")
      .def(
          "readstring",
          [](std::string_view fmt, pyt::Union<py::str, py::bytes> data,
             bool sanitize, bool skip_on_error) {
            return masquerade_cast<pyt::Iterator<PyMol>>(
                std::make_unique<PyMoleculeReader>(std::move(data), fmt,
                                                   sanitize, skip_on_error));
          },
          py::arg("fmt"), py::arg("data"), py::arg("sanitize") = true,
          py::arg("skip_on_error") = false,
          R"doc(
Read a molecule from string.

:param fmt: The format of the file.
:param data: The text to read, as :class:`str` or :class:`bytes`. Other
  bytes-like objects are accepted and copied.
:param sanitize: Whether to sanitize the produced molecule. For formats that is
  known to produce molecules with insufficient bond information (e.g. PDB), this
  option will trigger guessing based on the 3D coordinates
  (:func:`nuri.algo.guess_everything()`).
:param skip_on_error: Whether to skip a molecule if an error occurs, instead of
  raising an exception.
:raises TypeError: If `data` is neither bytes-like nor :class:`str`.
:raises ValueError: If the format is unknown, or if a molecule cannot be read
  or sanitized, unless `skip_on_error` is set.

The returned object is an iterator of molecules.

>>> for mol in nuri.readstring("smi", "C"):
...     print(mol[0].atomic_number)
6

.. note::
  The yielded molecules always have finite coordinates; NaN or infinite
  coordinates are considered an error.
)doc")
      .def(
          "readstream",
          [](std::string_view fmt, IO stream, bool sanitize,
             bool skip_on_error) {
            return masquerade_cast<pyt::Iterator<PyMol>>(
                std::make_unique<PyMoleculeReader>(
                    std::make_unique<PyIStream>(std::move(stream)), fmt,
                    sanitize, skip_on_error));
          },
          py::arg("fmt"), py::arg("stream"), py::arg("sanitize") = true,
          py::arg("skip_on_error") = false,
          R"doc(
Read molecules from a file-like object.

:param fmt: The format of the stream.
:param stream: A readable file-like object. Its ``read(n)`` method must return
  :class:`bytes` (binary mode) or :class:`str` (text mode; encoded as UTF-8
  before parsing). Binary mode is recommended: some formats (e.g. PDB) must
  seek, and only seekable binary streams support this.
:param sanitize: Whether to sanitize the produced molecule. For formats that is
  known to produce molecules with insufficient bond information (e.g. PDB), this
  option will trigger guessing based on the 3D coordinates
  (:func:`nuri.algo.guess_everything()`).
:param skip_on_error: Whether to skip a molecule if an error occurs, instead of
  raising an exception.
:raises TypeError: If `stream` has no ``read()`` method, or ``read()`` returns
  an object that is neither bytes-like nor :class:`str`.
:raises OSError: If the format requires seeking but `stream` does not support
  it, e.g. a text-mode or non-seekable stream. Streams from the :mod:`io` module
  raise :exc:`io.UnsupportedOperation` in that case.
:raises ValueError: If the format is unknown, or if a molecule cannot be read
  or sanitized, unless `skip_on_error` is set.

The returned object is an iterator of molecules. Any exception raised by
`stream` while reading propagates from the iterator regardless of
`skip_on_error`; molecules read before the failure are still yielded.

>>> import io
>>> for mol in nuri.readstream("smi", io.StringIO("C")):
...     print(mol[0].atomic_number)
6

Compressed files can be read without decompressing them to disk:

>>> import gzip
>>> with gzip.open("molecules.sdf.gz", "rb") as f:  # doctest: +SKIP
...     mols = list(nuri.readstream("sdf", f))

.. note::
  The yielded molecules always have finite coordinates; NaN or infinite
  coordinates are considered an error.
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
)doc")
      .def(
          "to_pdb",
          [](const PyMol &pmol, std::optional<int> oconf) {
            int conf = writer_check_conf(*pmol, oconf);
            return try_write(*pmol, "PDB",
                             [&](std::string &buf, const Molecule &mol) {
                               return write_pdb(buf, mol, -1, conf) >= 0;
                             });
          },
          py::arg("mol"), py::arg("conf") = py::none(), kThreadSafe, R"doc(
Convert a molecule to PDB string.

:param mol: The molecule to convert.
:param conf: The conformation to convert. If not specified, writes all
  conformations. Ignored if the molecule has no conformations.
:raises IndexError: If the molecule has any conformations and `conf` is out of
  range.
:raises ValueError: If the conversion fails.

.. note::
  Unlike most other formats, PDB does not support writing multiple different
  molecules in a single file. Simply concatenating the results of this function
  will not produce a valid PDB file.
)doc");

  py::module_ cif =
      m.def_submodule("cif", "CIF format-specific handlers and utilities.");
  bind_cif(cif);

  // Backward-compatible aliases: CIF used to live directly in nuri.fmt.
  m.attr("CifValue") = cif.attr("Value");
  m.attr("CifTable") = cif.attr("Table");
  m.attr("CifFrame") = cif.attr("Frame");
  m.attr("CifBlock") = cif.attr("Block");
  m.attr("read_cif") = cif.attr("read_blocks");
  m.attr("write_cif") = cif.attr("write");
  m.attr("cif_ddl2_frame_as_dict") = cif.attr("_frame_as_ddl2_dict");
  m.attr("mmcif_load_frame") = cif.attr("_frame_as_mols");

  py::module_ pdb =
      m.def_submodule("pdb", "PDB format-specific handlers and utilities.");
  bind_pdb(pdb);
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
