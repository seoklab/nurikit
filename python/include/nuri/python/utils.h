//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_UTILS_H_
#define NURI_PYTHON_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>
#include <pybind11/attr.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <absl/strings/str_cat.h>

#include "nuri/eigen_config.h"
#include "nuri/utils.h"

namespace nuri {
namespace python_internal {
template <class CppType, class... Args>
using PyProxyCls =
    py::class_<CppType, std::unique_ptr<CppType, py::nodelete>, Args...>;

template <class Derived, class T>
class ParentWrapper {
public:
  ParentWrapper() = default;

  explicit ParentWrapper(T &&data): data_(std::move(data)) { }

  T &operator*() { return data_; }

  const T &operator*() const { return data_; }

  T *operator->() { return &data_; }

  const T *operator->() const { return &data_; }

  std::uint64_t version_for_child() const { return version(); }

  void tick() { version_++; }

  bool ok(std::uint64_t version) const { return version == version_; }

protected:
  using Parent = ParentWrapper;

  std::uint64_t version() const { return version_; }

  Derived &self() { return *static_cast<Derived *>(this); }

  const Derived &self() const { return *static_cast<const Derived *>(this); }

private:
  T data_;
  std::uint64_t version_ = 0;
};

template <class T>
class ArrowHelper {
public:
  explicit ArrowHelper(T &&obj) noexcept: obj_(std::move(obj)) { }

  constexpr T *operator->() noexcept { return &obj_; }

private:
  T obj_;
};

template <class T>
class ArrowHelper<T &> {
public:
  explicit ArrowHelper(T &obj) noexcept: obj_(&obj) { }

  ArrowHelper(T &&obj) = delete;

  constexpr T *operator->() noexcept { return obj_; }

private:
  T *obj_;
};

template <class Derived, class T, class R, class U>
class ProxyWrapper {
public:
  ProxyWrapper(U &parent, const T &proxy, std::uint64_t version) noexcept
      : parent_(&parent), proxy_(proxy), version_(version) { }

  ProxyWrapper(U &parent, T &&proxy, std::uint64_t version) noexcept
      : parent_(&parent), proxy_(std::move(proxy)), version_(version) { }

  R operator*() {
    check();
    return deref();
  }

  ArrowHelper<R> operator->() {
    check();
    return ArrowHelper<R>(deref());
  }

  void check() const {
    bool safe = static_cast<const Derived *>(this)->ok();
    if (!safe)
      throw std::runtime_error("parent object modified after proxy creation");
  }

  const T &raw() const { return proxy_; }

  U &parent() const { return *parent_; }

protected:
  using Parent = ProxyWrapper;

  T &proxy() { return proxy_; }

  void self_tick() { version_++; }

  std::uint64_t version() const { return version_; }

  /* Default implementations */

  bool ok() const { return parent_->version_for_child() == version(); }

private:
  template <class, class, class, class>
  friend class ProxyWrapper;

  template <class>
  friend class TypeErasedProxyWrapper;

  R deref() { return static_cast<Derived *>(this)->deref(*parent_, proxy_); }

  U *parent_;
  T proxy_;
  std::uint64_t version_;
};

template <class T>
class TypeErasedProxyWrapper {
public:
  template <class U>
  TypeErasedProxyWrapper(T &&proxy, const U &parent)
      : proxy_(std::move(proxy)), version_(parent.version_for_child()),
        version_check_([&parent](std::uint64_t v) { return parent.ok(v); }) { }

  template <class... Args>
  TypeErasedProxyWrapper(T &&proxy, std::uint64_t version, Args &&...args)
      : proxy_(std::move(proxy)), version_(version),
        version_check_(std::forward<Args>(args)...) { }

  template <class U>
  auto sibling(U &&proxy) const {
    return TypeErasedProxyWrapper<std::remove_reference_t<U>>(
        std::forward<U>(proxy), version_, version_check_);
  }

  T &operator*() {
    check();
    return proxy_;
  }

  T *operator->() {
    check();
    return &proxy_;
  }

  const T &operator*() const {
    check();
    return proxy_;
  }

  const T *operator->() const {
    check();
    return &proxy_;
  }

protected:
  void check() const {
    bool safe = version_check_(version_);
    if (!safe)
      throw std::runtime_error("parent object modified after proxy creation");
  }

private:
  template <class>
  friend class TypeErasedProxyWrapper;

  T proxy_;
  std::uint64_t version_;
  std::function<bool(std::uint64_t)> version_check_;
};

template <class T>
TypeErasedProxyWrapper(T &&)
    -> TypeErasedProxyWrapper<std::remove_reference_t<T>>;

template <class T, class U>
TypeErasedProxyWrapper(T &&, const U &)
    -> TypeErasedProxyWrapper<std::remove_reference_t<T>>;

template <class T>
T &pass_through(T &x) {
  return x;
}

template <class T, std::enable_if_t<std::is_default_constructible_v<T>
                                        && std::is_copy_constructible_v<T>,
                                    int> = 0>
T default_if_null(const T *x) {
  return x == nullptr ? T() : T(*x);
}

template <class Derived, class C>
class PyIterator {
public:
  PyIterator(C &cont)
      : container_(&cont), index_(0), size_(size_of(*container_)) { }

  decltype(auto) next() {
    if (index_ >= size_)
      throw py::stop_iteration();

    if (index_ >= size_of(*container_))
      throw std::runtime_error("container changed size during iteration");

    return static_cast<const Derived *>(this)->deref(*container_, index_++);
  }

protected:
  using Parent = PyIterator;

  template <class... Extra>
  static py::class_<Derived> bind(py::module &m, const char *name,
                                  Extra &&...extra) {
    return py::class_<Derived>(m, name)
        .def("__iter__", pass_through<Derived>)
        .def("__next__", &Derived::next, std::forward<Extra>(extra)...);
  }

  static int size_of(C &cont) {
    return static_cast<int>(Derived::size_of(cont));
  }

private:
  C *container_;
  int index_;
  int size_;
};

using DynamicStrides = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

template <Eigen::Index Rows = Eigen::Dynamic>
using PyVectorMap = Eigen::Map<const Vector<double, Rows>, Eigen::Unaligned,
                               Eigen::InnerStride<>>;

template <Eigen::Index Rows = Eigen::Dynamic, Eigen::Index Cols = Eigen::Dynamic>
using PyMatrixMap = Eigen::Map<const Matrix<double, Rows, Cols>,
                               Eigen::Unaligned, DynamicStrides>;

template <Eigen::Index Rows = Eigen::Dynamic>
inline PyVectorMap<Rows> map_py_vector(const py::handle &buf) {
  auto arr = py::array_t<double>::ensure(buf);
  if (arr.ndim() != 1) {
    throw py::value_error(
        absl::StrCat("expected 1-dimensional array, got ", arr.ndim()));
  }

  if constexpr (Rows != Eigen::Dynamic) {
    if (arr.size() != Rows) {
      throw py::value_error(
          absl::StrCat("expected ", Rows, " elements, got ", arr.size()));
    }
  }

  Eigen::InnerStride<> stride(arr.strides()[0]
                              / static_cast<py::ssize_t>(sizeof(double)));
  PyVectorMap<Rows> map(arr.data(), arr.size(), stride);
  return map;
}

/**
 * @brief Converts a buffer object to an Eigen matrix.
 *
 * @param buf buffer object to convert.
 * @return "Transposed" Eigen matrix mapped from the buffer object. Effectively,
 *         the matrix is mapped from the buffer which is row-major, so the
 *         resulting matrix is column-major (Eigen-style).
 *         For example, if the buffer is a 4x3 array, the resulting matrix will
 *         be a 3x4 matrix.
 */
template <Eigen::Index Rows = Eigen::Dynamic, Eigen::Index Cols = Eigen::Dynamic>
inline Eigen::Map<const Matrix<double, Rows, Cols>, Eigen::Unaligned,
                  DynamicStrides>
map_py_matrix(const py::handle &buf) {
  auto arr = py::array_t<double>::ensure(buf);
  if (arr.ndim() != 2) {
    throw py::value_error(
        absl::StrCat("expected 2-dimensional array, got ", arr.ndim()));
  }

  const auto py_rows = arr.shape()[0], py_cols = arr.shape()[1];

  if constexpr (Cols != Eigen::Dynamic) {
    if (Cols != py_rows) {
      throw py::value_error(
          absl::StrCat("expected ", Cols, " rows, got ", py_rows));
    }
  }

  if constexpr (Rows != Eigen::Dynamic) {
    if (Rows != py_cols) {
      throw py::value_error(
          absl::StrCat("expected ", Rows, " columns, got ", py_cols));
    }
  }

  DynamicStrides strides(
      arr.strides()[0] / static_cast<py::ssize_t>(sizeof(double)),
      arr.strides()[1] / static_cast<py::ssize_t>(sizeof(double)));
  PyMatrixMap<Rows, Cols> map(arr.data(), py_cols, py_rows, strides);
  return map;
}

template <auto Rows, auto Cols, class Scalar = double>
using TransposedView =
    Eigen::Map<const Matrix<Scalar, Rows, Cols, Eigen::RowMajor>>;

template <class ML>
using Transposed =
    Eigen::Matrix<typename ML::Scalar, ML::ColsAtCompileTime,
                  ML::RowsAtCompileTime,
                  ML::IsRowMajor ? Eigen::ColMajor : Eigen::RowMajor>;

template <class MatrixLike>
auto transpose_view(const MatrixLike &mat) {
  // Swapped rows and cols for the transposed view
  constexpr auto rows = MatrixLike::RowsAtCompileTime;
  constexpr auto cols = MatrixLike::ColsAtCompileTime;
  return TransposedView<cols, rows, typename MatrixLike::Scalar>(
      mat.data(), mat.cols(), mat.rows());
}

inline int py_check_index(int size, int idx, const char *onerror) {
  if (idx < 0)
    idx += size;
  if (idx < 0 || idx >= size)
    throw py::index_error(onerror);
  return idx;
}

constexpr py::keep_alive<0, 1> kReturnsSubobject {};
constexpr py::call_guard<py::gil_scoped_release> kThreadSafe {};

template <class T, class Size, class Getter, class Iter>
py::class_<T> &add_sequence_interface(py::class_<T> &cls, Size size,
                                      Getter &&getter, Iter &&iter) {
  cls.def("__len__", size);
  cls.def(
      "__contains__",
      [size](T &self, int idx) { return 0 <= idx && idx < size(self); },
      py::arg("idx"));
  cls.def("__getitem__", std::forward<Getter>(getter), kReturnsSubobject,
          py::arg("idx"));
  cls.def("__iter__", std::forward<Iter>(iter), kReturnsSubobject);
  return cls;
}

template <class T, std::enable_if_t<std::is_copy_constructible_v<T>, int> = 0>
py::class_<T> &enable_copy(py::class_<T> &cls, bool add_copy_method = true) {
  cls.def("__copy__", [](const T &self) { return self; });
  cls.def(
      "__deepcopy__",
      [](const T &self, const py::dict & /* unused */) { return self; },
      py::arg("memo"));

  if (add_copy_method) {
    cls.def(
        "copy", [](const T &self) { return self; },
        "Return a deep copy of self.");
  }

  return cls;
}

inline int wrap_insert_index(int size, int idx) {
  if (idx < 0) {
    idx = nonnegative(size + idx);
  } else {
    idx = std::min(idx, size);
  }
  return idx;
}

template <class T, class Getter, class Setter, class... Extra>
py::class_<T> &def_property_subobject(py::class_<T> &self, const char *name,
                                      Getter &&getter, Setter &&setter,
                                      Extra &&...extra) {
  return self.def_property(
      name, py::cpp_function(std::forward<Getter>(getter), kReturnsSubobject),
      std::forward<Setter>(setter), std::forward<Extra>(extra)...);
}

template <class T, class Getter, class... Extra>
py::class_<T> &
def_property_readonly_subobject(py::class_<T> &self, const char *name,
                                Getter &&getter, Extra &&...extra) {
  return self.def_property_readonly(
      name, py::cpp_function(std::forward<Getter>(getter), kReturnsSubobject),
      std::forward<Extra>(extra)...);
}
}  // namespace python_internal
}  // namespace nuri

#endif /* NURI_PYTHON_UTILS_H_ */
