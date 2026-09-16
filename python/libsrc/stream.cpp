//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/python/stream.h"

#include <ios>
#include <optional>
#include <utility>

#include <abstract.h>
#include <bytesobject.h>
#include <pyerrors.h>
#include <pyport.h>
#include <unicodeobject.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "nuri/utils.h"

namespace nuri {
namespace python_internal {
PyStreamBuf::PyStreamBuf(py::object stream): stream_(std::move(stream)) {
  const py::gil_scoped_acquire gil;

  if (!py::hasattr(stream_, "read"))
    throw py::type_error("stream must be a readable file-like object");
  read_ = stream_.attr("read");
}

void PyStreamBuf::rethrow_pending() {
  if (pending_) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    throw *std::exchange(pending_, std::nullopt);
  }
}

PyStreamBuf::int_type PyStreamBuf::underflow() {
  if (gptr() < egptr())
    return traits_type::to_int_type(*gptr());

  try {
    return fill();
  } catch (py::error_already_set &e) {
    if (!pending_)
      pending_ = std::move(e);
    return traits_type::eof();
  }
}

PyStreamBuf::pos_type PyStreamBuf::seekoff(off_type off,
                                           std::ios_base::seekdir dir,
                                           std::ios_base::openmode which) {
  static const PyStreamBuf::pos_type bad_pos =
      static_cast<PyStreamBuf::off_type>(-1);

  if ((which & std::ios_base::in) == 0)
    return bad_pos;

  try {
    return seek(off, dir);
  } catch (py::error_already_set &e) {
    if (!pending_)
      pending_ = std::move(e);
    return bad_pos;
  }
}

namespace {
internal::Nonnull<const char *> read_chunk(py::object &chunk,
                                           Py_ssize_t &size) {
  if (py::isinstance<py::str>(chunk)) {
    const char *data = PyUnicode_AsUTF8AndSize(chunk.ptr(), &size);
    if (data == nullptr)
      throw py::error_already_set();
    return data;
  }

  if (!py::isinstance<py::bytes>(chunk)) {
    if (!py::isinstance<py::buffer>(chunk)) {
      py::str msg = py::str("read() must return bytes-like or str, got {!r}")
                        .format(py::type::of(chunk));
      py::set_error(PyExc_TypeError, msg);
      throw py::error_already_set();
    }
    chunk = py::reinterpret_steal<py::object>(PyBytes_FromObject(chunk.ptr()));
    if (!chunk)
      throw py::error_already_set();
  }

  char *raw;
  if (PyBytes_AsStringAndSize(chunk.ptr(), &raw, &size) < 0 || raw == nullptr)
    throw py::error_already_set();
  return raw;
}
}  // namespace

PyStreamBuf::int_type PyStreamBuf::fill() {
  constexpr py::ssize_t chunk_size = 1 << 16;

  const py::gil_scoped_acquire gil;

  py::object chunk = read_(chunk_size);
  if (chunk.is_none())
    return traits_type::eof();

  Py_ssize_t size;
  const char *data = read_chunk(chunk, size);
  if (size == 0)
    return traits_type::eof();

  chunk_ = std::move(chunk);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  char *begin = const_cast<char *>(data);
  setg(begin, begin, begin + size);
  return traits_type::to_int_type(*begin);
}

namespace {
PyStreamBuf::off_type as_offset(const py::object &pos) {
  const Py_ssize_t off = PyNumber_AsSsize_t(pos.ptr(), PyExc_OverflowError);
  if (off == -1 && PyErr_Occurred() != nullptr)
    throw py::error_already_set();
  return off;
}
}  // namespace

PyStreamBuf::pos_type PyStreamBuf::seek(off_type off,
                                        std::ios_base::seekdir dir) {
  const py::gil_scoped_acquire gil;

  const off_type buffered = egptr() - gptr();
  int whence = 0;
  if (dir == std::ios_base::cur) {
    if (off == 0)
      return pos_type(as_offset(stream_.attr("tell")()) - buffered);

    off -= buffered;
    whence = 1;
  } else if (dir == std::ios_base::end) {
    whence = 2;
  }

  py::object ret = stream_.attr("seek")(off, whence);
  setg(nullptr, nullptr, nullptr);
  chunk_ = py::none();

  if (ret.is_none())
    return pos_type(as_offset(stream_.attr("tell")()));
  return pos_type(as_offset(ret));
}
}  // namespace python_internal
}  // namespace nuri
