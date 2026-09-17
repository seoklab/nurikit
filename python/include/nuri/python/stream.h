//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_STREAM_H_
#define NURI_PYTHON_STREAM_H_

#include <ios>
#include <istream>
#include <optional>
#include <streambuf>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace nuri {
namespace python_internal {
/**
 * @brief Read-only std::streambuf over a Python file-like object.
 *
 * read() may return bytes-like (binary mode) or str (text mode; UTF-8
 * encoded). Seeking is delegated to the object's seek()/tell() and refused
 * for text-mode streams. Python exceptions raised inside stream operations
 * cannot cross std::istream (they would be swallowed into badbit), so they are
 * parked and must be re-raised with rethrow_pending() by the caller.
 *
 * The GIL is acquired inside each operation; callers may hold or not hold it.
 */
class PYBIND11_EXPORT PyStreamBuf final: public std::streambuf {
public:
  explicit PyStreamBuf(py::object stream);

  void rethrow_pending();

protected:
  int_type underflow() override;

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override;

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(static_cast<off_type>(pos), std::ios_base::beg, which);
  }

private:
  int_type fill();

  pos_type seek(off_type off, std::ios_base::seekdir dir);

  py::object stream_;
  py::object read_;
  py::object chunk_;
  std::optional<py::error_already_set> pending_;
};

class PYBIND11_EXPORT PyIStream final: public std::istream {
public:
  explicit PyIStream(py::object stream)
      : std::istream(nullptr), buf_(std::move(stream)) {
    rdbuf(&buf_);
  }

  PyStreamBuf &buf() { return buf_; }

private:
  PyStreamBuf buf_;
};
}  // namespace python_internal
}  // namespace nuri

#endif /* NURI_PYTHON_STREAM_H_ */
