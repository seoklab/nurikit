//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/cif.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/base/nullability.h>
#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/strip.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include "fmt_internal.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/mmcif.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/exception.h"
#include "nuri/python/utils.h"
#include "nuri/utils.h"

namespace nuri {
namespace python_internal {
namespace {
namespace fs = std::filesystem;

pyt::List<pyt::Optional<py::str>> cif_table_row(const internal::CifTable &table,
                                                int row) {
  pyt::List<pyt::Optional<py::str>> data;
  for (int i = 0; i < table.cols(); ++i) {
    const internal::CifValue &val = table.data()[row][i];
    data.append(val.is_null() ? py::none().cast<py::object>()
                              : py::str(*val).cast<py::object>());
  }
  return data;
}

// NOLINTBEGIN(*-unused-member-function)

class PyCifTableIterator
    : public PyIterator<PyCifTableIterator, const internal::CifTable> {
public:
  using Parent::Parent;

  static auto bind(py::module &m) {
    return Parent::bind(m, "_CifTableIterator");
  }

private:
  friend Parent;

  static pyt::List<pyt::Optional<py::str>>
  deref(const internal::CifTable &table, int row) {
    return cif_table_row(table, row);
  }

  static size_t size_of(const internal::CifTable &table) {
    return table.size();
  }
};

class PyCifTable {
public:
  explicit PyCifTable(const internal::CifTable &table): table_(&table) {
    for (const std::string &key: table_->keys())
      keys_.append(key);
  }

  PyCifTableIterator iter() const { return PyCifTableIterator(*table_); }

  pyt::List<pyt::Optional<py::str>> get(int row) const {
    row = py_check_index(static_cast<int>(table_->size()), row,
                         "CifTable row index out of range");
    return cif_table_row(*table_, row);
  }

  size_t size() const { return table_->size(); }

  pyt::List<py::str> keys() const { return keys_.attr("copy")(); }

private:
  absl::Nonnull<const internal::CifTable *> table_;
  pyt::List<py::str> keys_;
};

class PyCifFrameIterator
    : public PyIterator<PyCifFrameIterator, const internal::CifFrame> {
public:
  using Parent::PyIterator;

  static auto bind(py::module &m) {
    return Parent::bind(m, "_CifFrameIterator", kReturnsSubobject);
  }

private:
  friend Parent;

  static PyCifTable deref(const internal::CifFrame &frame, int idx) {
    return PyCifTable(frame[idx]);
  }

  static size_t size_of(const internal::CifFrame &frame) {
    return frame.size();
  }
};

class PyCifFrame {
public:
  explicit PyCifFrame(const internal::CifFrame &frame): frame_(&frame) { }

  std::string_view name() const { return frame_->name(); }

  PyCifFrameIterator iter() const { return PyCifFrameIterator(*frame_); }

  PyCifTable get(int idx) const {
    idx = py_check_index(static_cast<int>(frame_->size()), idx,
                         "CifFrame table index out of range");
    return PyCifTable((*frame_)[idx]);
  }

  size_t size() const { return frame_->size(); }

  pyt::Optional<PyCifTable> prefix_search_first(std::string_view prefix) const {
    auto it = frame_->prefix_search(prefix);
    if (it.empty() || !absl::StartsWith(it.begin()->first, prefix))
      return py::none();

    return py::cast(PyCifTable((*frame_)[it.begin()->second.first]));
  }

  const internal::CifFrame &operator*() const { return *frame_; }

private:
  absl::Nonnull<const internal::CifFrame *> frame_;
};

class PyCifBlock {
public:
  explicit PyCifBlock(internal::CifBlock &&block): block_(std::move(block)) {
    if (block_.type() == internal::CifBlock::Type::kEOF)
      throw py::stop_iteration();

    if (block_.type() == internal::CifBlock::Type::kError)
      throw py::value_error(std::string(block_.error_msg()));
  }

  PyCifFrame data() const { return PyCifFrame(block_.data()); }

  std::string_view name() const { return block_.name(); }

  const std::vector<internal::CifFrame> &save_frames() const {
    return block_.save_frames();
  }

  bool is_global() const {
    return block_.type() == internal::CifBlock::Type::kGlobal;
  }

private:
  internal::CifBlock block_;
};

class PyCifParser {
public:
  PyCifParser(const PyCifParser &) = delete;
  PyCifParser(PyCifParser &&) noexcept = delete;
  PyCifParser &operator=(const PyCifParser &) = delete;
  PyCifParser &operator=(PyCifParser &&) noexcept = delete;
  ~PyCifParser() = default;

  explicit PyCifParser(std::ifstream &&ifs)
      : ifs_(std::move(ifs)), parser_(ifs_) { }

  static std::unique_ptr<PyCifParser> from_file(const fs::path &path) {
    std::ifstream ifs(path);
    if (!ifs)
      throw file_error(path.c_str());

    return std::make_unique<PyCifParser>(std::move(ifs));
  }

  PyCifBlock next() { return PyCifBlock(parser_.next()); }

private:
  std::ifstream ifs_;
  CifParser parser_;
};

// NOLINTEND(*-unused-member-function)

PyCifFrame wrap_cif_frame(const internal::CifFrame &frame) {
  return PyCifFrame(frame);
}

template <class CppType, class PyType, auto wrapper>
void bind_opaque_vector(py::module &m, const char *name, const char *onerror) {
  using CifList = std::vector<CppType>;

  PyProxyCls<CifList> cl(m, name);
  cl.def("__iter__", [](const CifList &self) {
    return py::make_iterator(
        internal::make_transform_iterator<wrapper>(self.begin()),
        internal::make_transform_iterator<wrapper>(self.end()),
        kReturnsSubobject);
  });
  cl.def(
      "__getitem__",
      [onerror](const CifList &self, int i) {
        i = py_check_index(static_cast<int>(self.size()), i, onerror);
        return wrapper(self[i]);
      },
      kReturnsSubobject);
  cl.def("__len__", &CifList::size);
  cl.def("__repr__", [name](const CifList &self) {
    return absl::StrCat("<", name, " of ", self.size(), " tables>");
  });
}

pyt::Dict<py::str, pyt::List<pyt::Dict<py::str, pyt::Optional<py::str>>>>
cif_ddl2_frame_as_dict(const PyCifFrame &frame) {
  absl::flat_hash_map<
      std::string_view,
      std::pair<std::vector<py::str>, std::vector<std::vector<py::object>>>>
      grouped;

  std::vector<std::string_view> parent_keys;
  std::vector<decltype(grouped)::iterator> slots;
  for (const internal::CifTable &table: *frame) {
    parent_keys.clear();
    parent_keys.reserve(table.cols());
    for (std::string_view key: table.keys()) {
      std::string_view pk = key.substr(0, key.find('.'));
      std::string_view sk = key.substr(pk.size() + 1);

      pk = absl::StripPrefix(pk, "_");

      parent_keys.push_back(pk);
      grouped[pk].first.push_back(sk);
    }

    slots.clear();
    slots.reserve(table.cols());
    for (std::string_view pk: parent_keys) {
      auto it = grouped.find(pk);
      ABSL_DCHECK(it != grouped.end());
      slots.push_back(it);

      auto &data = it->second.second;
      data.resize(table.size());

      for (auto &row: data)
        row.reserve(row.size() + it->second.first.size());
    }

    for (int i = 0; i < table.size(); ++i) {
      for (int j = 0; j < table.cols(); ++j) {
        auto it = slots[j];

        auto &data = it->second.second;

        const internal::CifValue &val = table[i][j];
        data[i].push_back(val.is_null() ? py::none().cast<py::object>()
                                        : py::str(*val).cast<py::object>());
      }
    }
  }

  py::dict tagged;
  for (auto &group: grouped) {
    py::str pk(group.first);
    py::list rows;
    for (const auto &row: group.second.second) {
      py::dict entry;
      for (int i = 0; i < row.size(); ++i)
        entry[group.second.first[i]] = row[i];
      rows.append(entry);
    }
    tagged[pk] = rows;
  }

  return tagged;
}

pyt::List<PyMol> mmcif_load_cif_frame(const PyCifFrame &frame) {
  std::vector<Molecule> mols = mmcif_load_frame(*frame);

  pyt::List<PyMol> pymols(mols.size());
  for (int i = 0; i < mols.size(); ++i)
    pymols[i] = PyMol(std::move(mols[i]));
  return pymols;
}
}  // namespace

void bind_cif(py::module &m) {
  PyCifTableIterator::bind(m);

  py::class_<PyCifTable>(m, "CifTable")
      .def("__iter__", &PyCifTable::iter, kReturnsSubobject)
      .def("__getitem__", &PyCifTable::get, py::arg("idx"))
      .def("__len__", &PyCifTable::size)
      .def(
          "__contains__",
          [](const PyCifTable &self, int idx) {
            return 0 <= idx && idx < self.size();
          },
          py::arg("idx"))
      .def("keys", &PyCifTable::keys);

  PyCifFrameIterator::bind(m);

  py::class_<PyCifFrame> cf(m, "CifFrame");
  add_sequence_interface(cf, &PyCifFrame::size, &PyCifFrame::get,
                         &PyCifFrame::iter);
  cf.def("prefix_search_first", &PyCifFrame::prefix_search_first,
         py::arg("prefix"), kReturnsSubobject, R"doc(
Search for the first table containing a column starting with the given prefix.

:param prefix: The prefix to search for.
:return: The first table containing the given prefix, or None if not found.
)doc");
  cf.def_property_readonly("name", &PyCifFrame::name);

  bind_opaque_vector<internal::CifFrame, PyCifFrame, wrap_cif_frame>(
      m, "_CifFrameList", "_CifFrameList index out of range");

  py::class_<PyCifBlock> cb(m, "CifBlock");
  cb.def_property_readonly("name", &PyCifBlock::name)
      .def_property_readonly("is_global", &PyCifBlock::is_global)
      .def_property_readonly("save_frames", &PyCifBlock::save_frames);
  def_property_readonly_subobject(cb, "data", &PyCifBlock::data);

  py::class_<PyCifParser>(m, "CifParser")
      .def("__iter__", pass_through<PyCifParser>)
      .def("__next__", &PyCifParser::next);

  m.def("read_cif", &PyCifParser::from_file, py::arg("path"), R"doc(
Create a parser object from a CIF file path.

:param path: The path to the CIF file.
:return: A parser object that can be used to iterate over the blocks in the file.
)doc")
      .def("cif_ddl2_frame_as_dict", cif_ddl2_frame_as_dict, py::arg("frame"),
           R"doc(
Convert a CIF frame to a dictionary of lists of dictionaries.

:param frame: The CIF frame to convert.
:return: A dictionary of lists of dictionaries, where the keys are the parent
  keys and the values are the rows of the table.
)doc")
      .def("mmcif_load_frame", mmcif_load_cif_frame, py::arg("frame"),
           R"doc(
Load a CIF frame as a list of molecules.

:param frame: The CIF frame to load.
:return: A list of molecules loaded from the frame.
)doc");
}
}  // namespace python_internal
}  // namespace nuri
