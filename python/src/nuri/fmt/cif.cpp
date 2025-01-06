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
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include "fmt_internal.h"
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
}  // namespace

void bind_cif(py::module &m) {
  PyCifTableIterator::bind(m);

  py::class_<PyCifTable>(m, "CifTable")  //
      .def("__iter__", &PyCifTable::iter, kReturnsSubobject)
      .def("__getitem__", &PyCifTable::get)
      .def("__len__", &PyCifTable::size)
      .def("keys", &PyCifTable::keys);

  PyCifFrameIterator::bind(m);

  py::class_<PyCifFrame>(m, "CifFrame")  //
      .def("__iter__", &PyCifFrame::iter, kReturnsSubobject)
      .def("__getitem__", &PyCifFrame::get, kReturnsSubobject)
      .def("__len__", &PyCifFrame::size)
      .def("prefix_search_first", &PyCifFrame::prefix_search_first,
           py::arg("prefix"), kReturnsSubobject, R"doc(
Search for the first table containing a column starting with the given prefix.

:param prefix: The prefix to search for.
:return: The first table containing the given prefix, or None if not found.
)doc")
      .def_property_readonly("name", &PyCifFrame::name);

  bind_opaque_vector<internal::CifFrame, PyCifFrame, wrap_cif_frame>(
      m, "_CifFrameList", "_CifFrameList index out of range");

  py::class_<PyCifBlock>(m, "CifBlock")  //
      .def_property_readonly("data", &PyCifBlock::data)
      .def_property_readonly("name", &PyCifBlock::name)
      .def_property_readonly("save_frames", &PyCifBlock::save_frames)
      .def_property_readonly("is_global", &PyCifBlock::is_global);

  py::class_<PyCifParser>(m, "CifParser")
      .def("__iter__", pass_through<PyCifParser>)
      .def("__next__", &PyCifParser::next);

  m.def("read_cif", &PyCifParser::from_file, py::arg("path"), R"doc(
Create a parser object from a CIF file path.

:param path: The path to the CIF file.
:return: A parser object that can be used to iterate over the blocks in the file.
)doc");
}
}  // namespace python_internal
}  // namespace nuri
