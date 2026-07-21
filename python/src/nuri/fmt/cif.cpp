//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/cif.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/strip.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/typing.h>

#include "fmt_internal.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/mmcif.h"
#include "nuri/python/core/core_module.h"
#include "nuri/python/exception.h"
#include "nuri/python/typing.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
namespace fs = std::filesystem;

class PyCifTable;

// Per-column formatting options, mirroring the CifValue constructor keywords.
// Applied when a plain scalar (or None) cell is turned into a CifValue.
struct ColumnFormat {
  int width = 0;             // int:   zero-pad to at least this many digits
  int precision = -1;        // float: digits after '.', <0 = shortest
  bool raw = false;          // str:   generic literal vs. quoted string
  bool short_form = false;   // bool:  y/n vs. yes/no
  bool null_unknown = true;  // None:  '?' (unknown) vs. '.' (inapplicable)
  bool coerce_nonfinite = false;  // float: coerce NaN/Inf to a safe value
};

using CifCell = pyt::Union<py::str, py::int_, py::float_, py::bool_,
                           internal::CifValue, py::none>;
using CifKeys = Sequence<py::str>;
using CifRow = Sequence<CifCell>;
using CifRows = Sequence<CifRow>;
using CifColumnFormats = pyt::Dict<py::str, ColumnFormat>;
using CifTables = Sequence<PyCifTable>;

pyt::List<pyt::Optional<py::str>> cif_table_row(const internal::CifTable &table,
                                                int row) {
  pyt::List<pyt::Optional<py::str>> data(table.cols());
  for (int i = 0; i < table.cols(); ++i) {
    const internal::CifValue &val = table.data()[row][i];
    data[i] = val.is_null() ? py::none().cast<py::object>()
                            : py::str(*val).cast<py::object>();
  }
  return data;
}

// Convert a table cell. ``None`` becomes the null value whose kind is decided
// by the column (``?`` if @p fmt.null_unknown, else ``.``).
internal::CifValue py_to_cif_value(py::handle cell, const ColumnFormat &fmt) {
  if (cell.is_none())
    return internal::CifValue::null(fmt.null_unknown);
  if (py::isinstance<py::str>(cell)) {
    auto text = cell.cast<std::string_view>();
    return fmt.raw ? internal::CifValue::generic(text)
                   : internal::CifValue::string(text);
  }
  if (py::isinstance<py::bool_>(cell))
    return cif_value(cell.cast<bool>(), fmt.short_form);
  if (py::isinstance<py::int_>(cell))
    return cif_value(cell.cast<std::int64_t>(), fmt.width);
  if (py::isinstance<py::float_>(cell))
    return cif_value(cell.cast<double>(), fmt.precision, fmt.coerce_nonfinite,
                     fmt.null_unknown);
  if (py::isinstance<internal::CifValue>(cell))
    return cell.cast<internal::CifValue>();

  throw py::type_error(
      "CIF value must be str, int, bool, float, None, or CifValue");
}

// Form a full CIF data name: prepend the leading ``_`` (the inverse of
// cif_ddl2_frame_as_dict) and, when non-empty, the DDL2 @p category
// (``_<category>.<key>``). Validity is checked by CifFrame::validate when the
// table is placed into a frame.
std::string make_cif_key(std::string_view category, std::string_view key) {
  if (!category.empty())
    return absl::StrCat("_", category, ".", key);
  return absl::StrCat("_", key);
}

// Map a CIF null token string to CifValue::null's boolean (``?`` -> unknown,
// ``.`` -> inapplicable).
bool parse_null_token(std::string_view token) {
  if (token == "?")
    return true;
  if (token == ".")
    return false;
  throw py::value_error(
      absl::StrCat(R"(null_token must be "?" or ".", got: )", token));
}

// Resolve per-column formatting options from the ``column_formats`` mapping,
// keyed by (underscore-less) column key. Unknown column keys raise ValueError.
std::vector<ColumnFormat>
parse_column_formats(const CifKeys &keys,
                     const CifColumnFormats &column_formats) {
  std::vector<ColumnFormat> formats(py::len(keys));

  absl::flat_hash_map<std::string, int> index;
  int i = 0;
  for (py::handle key: keys)
    index.emplace(key.cast<std::string>(), i++);

  for (auto item: column_formats) {
    auto col = item.first.cast<std::string_view>();
    auto it = index.find(col);
    if (it == index.end()) {
      throw py::value_error(
          absl::StrCat("Unknown key in column_formats: ", col));
    }

    formats[it->second] = item.second.cast<ColumnFormat>();
  }

  return formats;
}

internal::CifTable build_cif_table(const CifKeys &keys, const CifRows &rows,
                                   std::string_view category,
                                   const CifColumnFormats &column_formats) {
  internal::CifTable table;

  size_t ncols = py::len(keys);
  for (py::handle key: keys)
    table.add_key(make_cif_key(category, key.cast<std::string_view>()));

  std::vector<ColumnFormat> formats =
      parse_column_formats(keys, column_formats);

  for (py::handle row_handle: rows) {
    auto row = row_handle.cast<CifRow>();
    if (py::len(row) != ncols) {
      throw py::value_error(absl::StrCat("CIF row has ", py::len(row),
                                         " values but the table has ", ncols,
                                         " keys"));
    }
    for (size_t j = 0; j < ncols; ++j)
      table.add_data(py_to_cif_value(row[j], formats[j]));
  }

  std::string err = table.validate();
  if (!err.empty())
    throw py::value_error(err);

  return table;
}

// NOLINTBEGIN(*-unused-member-function)

class PyCifTableIterator
    : public PyIterator<PyCifTableIterator, const internal::CifTable> {
public:
  using Parent::Parent;

  static auto bind(py::module &m) { return Parent::bind(m, "_TableIterator"); }

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
  // View over a parser-owned table.
  explicit PyCifTable(const internal::CifTable &table): ext_(&table) {
    init_keys();
  }

  // Owning table built from Python.
  PyCifTable(const CifKeys &keys, const CifRows &rows,
             std::string_view category, const CifColumnFormats &column_formats)
      : owned_(build_cif_table(keys, rows, category, column_formats)) {
    init_keys();
  }

  const internal::CifTable &table() const { return owned_ ? *owned_ : *ext_; }

  auto iter() const { return PyCifTableIterator::make(table()); }

  pyt::List<pyt::Optional<py::str>> get(int row) const {
    row = py_check_index(static_cast<int>(table().size()), row,
                         "CIF table row index out of range");
    return cif_table_row(table(), row);
  }

  size_t size() const { return table().size(); }

  pyt::List<py::str> keys() const { return keys_.attr("copy")(); }

private:
  void init_keys() {
    for (const std::string &key: table().keys())
      keys_.append(key);
  }

  std::optional<internal::CifTable> owned_;
  const internal::CifTable *ext_ = nullptr;
  pyt::List<py::str> keys_;
};

class PyCifFrameIterator
    : public PyIterator<PyCifFrameIterator, const internal::CifFrame> {
public:
  using Parent::PyIterator;

  static auto bind(py::module &m) {
    return Parent::bind(m, "_FrameIterator", kReturnsSubobject);
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
  // View over a parser-owned frame.
  explicit PyCifFrame(const internal::CifFrame &frame): ext_(&frame) { }

  // Owning frame built from Python tables.
  PyCifFrame(std::string name, const CifTables &tables) {
    std::vector<internal::CifTable> ts;
    ts.reserve(py::len(tables));
    for (py::handle table_handle: tables)
      ts.push_back(table_handle.cast<const PyCifTable &>().table());

    owned_.emplace(std::move(ts), std::move(name));

    // Each table was already validated by its PyCifTable; only the frame-level
    // invariant (keys unique across tables) remains.
    std::string err = owned_->validate(false);
    if (!err.empty())
      throw py::value_error(err);
  }

  const internal::CifFrame &frame() const { return owned_ ? *owned_ : *ext_; }

  std::string_view name() const { return frame().name(); }

  auto iter() const { return PyCifFrameIterator::make(frame()); }

  PyCifTable get(int idx) const {
    idx = py_check_index(static_cast<int>(frame().size()), idx,
                         "CIF frame table index out of range");
    return PyCifTable(frame()[idx]);
  }

  size_t size() const { return frame().size(); }

  pyt::Optional<PyCifTable> prefix_search_first(std::string_view prefix) const {
    auto it = frame().prefix_search(prefix);
    if (it.empty() || !absl::StartsWith(it.begin()->first, prefix))
      return py::none();

    return py::cast(PyCifTable(frame()[it.begin()->second.first]));
  }

  const internal::CifFrame &operator*() const { return frame(); }

private:
  std::optional<internal::CifFrame> owned_;
  const internal::CifFrame *ext_ = nullptr;
};

class PyCifBlock {
public:
  PyCifBlock(internal::CifBlock &&block): block_(std::move(block)) {
    if (block_.type() == internal::CifBlock::Type::kEOF)
      throw py::stop_iteration();

    if (block_.type() == internal::CifBlock::Type::kError)
      throw py::value_error(std::string(block_.error_msg()));
  }

  const internal::CifBlock &block() const { return block_; }

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

internal::CifFrame clone_cif_frame(const internal::CifFrame &frame) {
  std::vector<internal::CifTable> ts(frame.tables());
  return internal::CifFrame(std::move(ts), std::string(frame.name()));
}

PyCifBlock build_cif_block(const PyCifFrame &data,
                           const pyt::Iterable<PyCifFrame> &save) {
  std::vector<internal::CifFrame> saves;
  for (py::handle sf: save)
    saves.push_back(clone_cif_frame(*sf.cast<const PyCifFrame &>()));

  internal::CifBlock block(clone_cif_frame(*data), std::move(saves),
                           internal::CifBlock::Type::kData);

  // The frames were validated when their PyCifFrame wrappers were built; only
  // the block-level invariant (save-frame names) remains.
  std::string err = block.validate(false);
  if (!err.empty())
    throw py::value_error(err);

  return PyCifBlock(std::move(block));
}

class PyCifParser {
public:
  PyCifParser(const PyCifParser &) = delete;
  PyCifParser(PyCifParser &&) noexcept = delete;
  PyCifParser &operator=(const PyCifParser &) = delete;
  PyCifParser &operator=(PyCifParser &&) noexcept = delete;
  ~PyCifParser() = default;

  explicit PyCifParser(std::ifstream &&ifs)
      : ifs_(std::move(ifs)), parser_(ifs_) { }

  static pyt::Iterator<PyCifBlock> from_file(const fs::path &path) {
    std::ifstream ifs(path);
    if (!ifs)
      throw file_error(path.c_str());

    return py::cast(std::make_unique<PyCifParser>(std::move(ifs)));
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
  add_sequence_interface(
      cl, &CifList::size,
      [onerror](const CifList &self, int i) {
        i = py_check_index(static_cast<int>(self.size()), i, onerror);
        return wrapper(self[i]);
      },
      [](const CifList &self) {
        return py::make_iterator(
            internal::make_transform_iterator<wrapper>(self.begin()),
            internal::make_transform_iterator<wrapper>(self.end()),
            kReturnsSubobject);
      });
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

py::str write_cif_from_block(const PyCifBlock &block, bool align) {
  ABSL_LOG_IF(WARNING, block.is_global())
      << "global_ blocks are a STAR feature; output will be an invalid CIF 1.1 "
         "file";

  std::string out;
  if (!write_cif_block(out, block.block(), align))
    throw py::value_error(out);

  return py::str(out);
}

py::str write_cif_from_frame(const PyCifFrame &frame, bool align) {
  std::string out;
  if (!write_cif_frame(out, *frame, internal::CifFrame::Type::kData, align))
    throw py::value_error(out);

  return py::str(out);
}
}  // namespace

void bind_cif(py::module &m) {
  // Register Value first so it resolves to its Python name (not the C++ type
  // name) in the Table cell annotation below.
  py::class_<internal::CifValue> cv(m, "Value", R"doc(
An explicit CIF value, for use as a :class:`Table` cell.

Most cells can be given as plain Python objects (:class:`str`, :class:`int`,
:class:`bool`, :class:`float`, or ``None``); construct a :class:`Value`
directly only when you need control over the formatting. The constructor is
overloaded on the type of ``value``.
)doc");

  py::class_<ColumnFormat>(m, "ColumnFormat")
      .def(py::init([](int width, std::optional<int> precision, bool raw,
                       bool short_form, std::string_view null_token,
                       bool coerce_nonfinite) {
             if (precision.has_value() && *precision < 0)
               throw py::value_error("precision must be non-negative");
             return ColumnFormat {
               width,      precision.value_or(-1),       raw,
               short_form, parse_null_token(null_token), coerce_nonfinite
             };
           }),
           py::kw_only(), py::arg("width") = 0,
           py::arg("precision") = py::none(), py::arg("raw") = false,
           py::arg("short_form") = false, py::arg("null_token") = "?",
           py::arg("coerce_nonfinite") = false, R"doc(
Per-column value formatting for :class:`Table`.

Groups the keyword options of the :class:`Value` constructor so a whole column
of plain scalar (or ``None``) cells can be formatted at once. Each option
applies only to cells of its matching type; the others are ignored. Explicit
:class:`Value` cells are never affected.

:param width: For :class:`int` cells, zero-pad the number to at least this many
  digits.
:param precision: For :class:`float` cells, digits after the decimal point; if
  ``None`` (the default), yields at most 6 significant digits. Must be
  non-negative if provided.
:param raw: For :class:`str` cells, store the text as an unquoted generic
  literal instead of a quoted string.
:param short_form: For :class:`bool` cells, use ``y``/``n`` instead of
  ``yes``/``no``.
:param null_token: The CIF null token used for ``None`` cells (and for ``NaN``
  when ``coerce_nonfinite`` is set): ``"?"`` (the default) for unknown, or
  ``"."`` for inapplicable.
:param coerce_nonfinite: For :class:`float` cells, coerce a non-finite value to
  a safe representation (``NaN`` to the ``null_token``, ``+/-Inf`` to a
  sentinel) instead of raising when serialized.
:raises ValueError: If ``precision`` is negative or ``null_token`` is not
  ``"?"`` or ``"."``.
)doc")
      .def_readonly("width", &ColumnFormat::width)
      .def_property_readonly(
          "precision",
          [](const ColumnFormat &f) -> pyt::Optional<py::int_> {
            if (f.precision < 0)
              return py::none();
            return py::int_(f.precision);
          })
      .def_readonly("raw", &ColumnFormat::raw)
      .def_readonly("short_form", &ColumnFormat::short_form)
      .def_property_readonly("null_token",
                             [](const ColumnFormat &f) {
                               return f.null_unknown ? "?" : ".";
                             })
      .def_readonly("coerce_nonfinite", &ColumnFormat::coerce_nonfinite)
      .def("__repr__", [](const ColumnFormat &f) {
        auto pybool = [](bool v) { return v ? "True" : "False"; };
        return absl::StrCat(
            "ColumnFormat(width=", f.width, ", precision=",
            f.precision < 0 ? std::string("None") : absl::StrCat(f.precision),
            ", raw=", pybool(f.raw), ", short_form=", pybool(f.short_form),
            ", null_token='", f.null_unknown ? "?" : ".",
            "', coerce_nonfinite=", pybool(f.coerce_nonfinite), ")");
      });

  PyCifTableIterator::bind(m);

  py::class_<PyCifTable> table(m, "Table");
  table.def(py::init<const CifKeys &, const CifRows &, std::string_view,
                     const CifColumnFormats &>(),
            py::arg("keys"), py::arg("rows"), py::arg("category") = "",
            py::arg("column_formats") = py::dict(), R"doc(
Construct a CIF table from column keys and rows of values.

:param keys: The column keys, given **without** the leading underscore (e.g.
  ``["atom_site.id", "atom_site.type_symbol"]``); a ``_`` is prepended to each.
:param rows: The rows of the table. Each row must have exactly ``len(keys)``
  cells. A cell may be a :class:`str`, :class:`int`, :class:`bool`,
  :class:`float`, ``None``, or a :class:`Value`.
:param category: Optional DDL2 category. When non-empty, each key is formed as
  ``_<category>.<key>``, so ``keys`` may list just the attribute names (e.g.
  ``category="atom_site", keys=["id", "type_symbol"]``). Empty (the default)
  leaves keys as given.
:param column_formats: Optional per-column formatting options, mapping a column
  key (as given in ``keys``) to a :class:`ColumnFormat` applied when a plain
  scalar (or ``None``) cell in that column is converted. Each option applies
  only to cells of its matching type (see :class:`ColumnFormat`); columns not
  listed use the defaults. Explicit :class:`Value` cells are unaffected.
:raises ValueError: If a row length differs from the number of keys, a key is
  not a valid or unique CIF data name, or ``column_formats`` names an unknown
  key.
:raises TypeError: If a cell has an unsupported type.
)doc");
  add_sequence_interface(table, &PyCifTable::size, &PyCifTable::get,
                         &PyCifTable::iter, rvp::automatic);
  table.def("keys", &PyCifTable::keys);

  PyCifFrameIterator::bind(m);

  py::class_<PyCifFrame> cf(m, "Frame");
  cf.def(py::init<std::string, const CifTables &>(), py::arg("name"),
         py::arg("tables"), R"doc(
Construct a CIF frame (data block body or save frame) from tables.

:param name: The frame name.
:param tables: The tables that make up the frame.
:raises ValueError: If a column key is duplicated across the tables.
)doc");
  add_sequence_interface(cf, &PyCifFrame::size, &PyCifFrame::get,
                         &PyCifFrame::iter);
  cf.def("prefix_search_first", &PyCifFrame::prefix_search_first,
         py::arg("prefix"), kReturnsSubobject, R"doc(
Search for the first table containing a column starting with the given prefix.

:param prefix: The prefix to search for.
:return: The first table containing the given prefix, or None if not found.
)doc");
  cf.def_property_readonly("name", &PyCifFrame::name);
  cf.def("as_ddl2_dict", cif_ddl2_frame_as_dict, R"doc(
Convert this DDL2 (mmCIF) frame to a dictionary of lists of dictionaries.

:return: A dictionary of lists of dictionaries, where the keys are the parent
  keys and the values are the rows of the table.
)doc");
  cf.def("as_mols", mmcif_load_cif_frame, R"doc(
Load this frame as a list of molecules.

:return: A list of molecules loaded from the frame.
)doc");

  // Compat shims; might be removed in a future release.
  // Need to bind as free functions additionally, otherwise stubgen will not
  // generate the correct signatures.
  m.def("_frame_as_ddl2_dict", cif_ddl2_frame_as_dict, py::arg("frame"),
        R"doc(Deprecated alias for :meth:`Frame.as_ddl2_dict`.)doc");
  m.def("_frame_as_mols", mmcif_load_cif_frame, py::arg("frame"),
        R"doc(Deprecated alias for :meth:`Frame.as_mols`.)doc");

  bind_opaque_vector<internal::CifFrame, PyCifFrame, wrap_cif_frame>(
      m, "_FrameList", "_FrameList index out of range");

  py::class_<PyCifBlock> cb(m, "Block");
  cb.def(py::init(&build_cif_block), py::arg("data"),
         py::arg("save") = py::tuple(), R"doc(
Construct a CIF block.

:param data: The main frame of the block; its name is used as the block name.
:param save: The save frames contained in the block.
:raises ValueError: If a save frame has an empty name or two share a name.
)doc");
  cb.def_property_readonly("name", &PyCifBlock::name)
      .def_property_readonly("is_global", &PyCifBlock::is_global);
  def_property_readonly_subobject(
      cb, "save_frames",
      [](const PyCifBlock &self) {
        return masquerade_cast<Sequence<PyCifFrame>>(self.save_frames(),
                                                     rvp::reference);
      },
      ":type: collections.abc.Sequence[Frame]");
  def_property_readonly_subobject(cb, "data", &PyCifBlock::data);

  py::class_<PyCifParser>(m, "_Parser")
      .def("__iter__", pass_through<PyCifParser>)
      .def("__next__", &PyCifParser::next);

  m.def("read_blocks", PyCifParser::from_file, py::arg("path"), R"doc(
Create a parser object from a CIF file path.

:param path: The path to the CIF file.
:return: An iterator over the blocks in the file.
)doc");

  cv.def(py::init([](std::string_view value, bool raw) {
           return raw ? internal::CifValue::generic(value)
                      : internal::CifValue::string(value);
         }),
         py::arg("value"), py::arg("raw") = false, R"doc(
Store text as a CIF value.

:param value: The text to store.
:param raw: If ``True``, store the text as an unquoted generic literal (e.g. a
  standard-uncertainty token like ``1.234(5)``). Defaults to ``False``: a
  string, quoted on write only if its content requires it.
)doc");
  cv.def(py::init(&cif_value<bool>), py::arg("value"),
         py::arg("short_form") = false, R"doc(
Store a boolean CIF value.

:param value: The boolean to store.
:param short_form: Use ``y``/``n`` instead of ``yes``/``no``.
)doc");
  cv.def(py::init(&cif_value<std::int64_t>), py::arg("value"),
         py::arg("width") = 0, R"doc(
Store an integer CIF value.

:param value: The integer to store.
:param width: If positive, zero-pad the number to at least this many digits.
)doc");
  cv.def(py::init([](double value, std::optional<int> prec,
                     bool coerce_nonfinite, std::string_view null_token) {
           if (prec.has_value() && prec.value() < 0)
             throw py::value_error("precision must be non-negative");

           return cif_value(value, prec.value_or(-1), coerce_nonfinite,
                            parse_null_token(null_token));
         }),
         py::arg("value"), py::arg("precision") = py::none(),
         py::arg("coerce_nonfinite") = false, py::arg("null_token") = "?",
         R"doc(
Store a floating-point CIF value.

:param value: The number to store.
:param precision: Digits after the decimal point; if ``None`` (the default),
  yields at most 6 significant digits. Must be non-negative if provided.
:param coerce_nonfinite: How to handle a non-finite ``value``. If ``False`` (the
  default), a non-finite value has no valid CIF representation and raises
  :exc:`ValueError` when serialized. If ``True``, ``NaN`` becomes the null value
  chosen by ``null_token`` and ``+/-Inf`` becomes the sentinel
  ``+/-8e+88888888``, which any IEEE-conformant parser (up to ``binary256``)
  reads back as the original infinity.
:param null_token: The CIF null token that ``NaN`` is coerced to when
  ``coerce_nonfinite`` is ``True``: ``"?"`` (the default) for unknown, or
  ``"."`` for inapplicable. Ignored for finite values.
:raises ValueError: If ``null_token`` is not ``"?"`` or ``"."``.
)doc");
  cv.def(py::init([](const py::none &, std::string_view null_token) {
           return internal::CifValue::null(parse_null_token(null_token));
         }),
         py::arg("value"), py::arg("null_token") = "?", R"doc(
Store a null CIF value.

:param value: ``None``.
:param null_token: The CIF null token: ``"?"`` (the default) for the unknown
  value, or ``"."`` for the inapplicable value.
:raises ValueError: If ``null_token`` is not ``"?"`` or ``"."``.
)doc");

  m.def("write", write_cif_from_frame, py::arg("frame"),
        py::arg("align") = false, R"doc(
Serialize a CIF frame to a CIF 1.1 string.

:param frame: The :class:`Frame` to serialize, written as a ``data_`` block.
  Both freshly constructed and parsed objects are accepted.
:param align: Whether to pad the columns of ``loop_`` tables so that values
  line up. Defaults to ``False``.
:return: The serialized CIF string.
:raises ValueError: If a value cannot be represented in CIF 1.1.
)doc")
      .def("write", write_cif_from_block, py::arg("block"),
           py::arg("align") = false, R"doc(
Serialize a CIF block to a CIF 1.1 string.

:param block: The :class:`Block` to serialize.
:param align: Whether to pad the columns of ``loop_`` tables so that values
  line up. Defaults to ``False``.
:return: The serialized CIF string.
:raises ValueError: If a value cannot be represented in CIF 1.1.

.. note::
  A ``global_`` block (only ever produced by the parser, from a STAR file) is
  written back as-is, i.e. as a ``global_`` block. This is a STAR construct and
  is **not** valid CIF 1.1; a warning is logged in that case.

>>> import nuri
>>> table = nuri.fmt.cif.Table(["x.a", "x.b"], [[1, "two words"], [None, nuri.fmt.cif.Value(None, null_token=".")]])
>>> block = nuri.fmt.cif.Block(nuri.fmt.cif.Frame("demo", [table]))
>>> print(nuri.fmt.cif.write(block))
data_demo
loop_
_x.a
_x.b
1 'two words'
? .
<BLANKLINE>
)doc");
}
}  // namespace python_internal
}  // namespace nuri
