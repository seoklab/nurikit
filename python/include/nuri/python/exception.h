//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cerrno>

#include <pyerrors.h>
#include <pybind11/pybind11.h>

namespace nuri {
namespace python_internal {
class os_error: public py::builtin_exception {
public:
  explicit os_error(const char *what = "")
      : py::builtin_exception(what), errno_(errno) { }

  void set_error() const final {
    errno = errno_;
    set_error_impl();
  }

private:
  virtual void set_error_impl() const { PyErr_SetFromErrno(PyExc_OSError); }

  int errno_;
};

class file_error: public os_error {
public:
  explicit file_error(const char *fname): os_error(fname) { }

private:
  void set_error_impl() const override {
    PyErr_SetFromErrnoWithFilename(PyExc_OSError, what());
  }
};
}  // namespace python_internal
}  // namespace nuri
