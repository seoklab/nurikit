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
#include <vector>

#include <object.h>
#include <pyerrors.h>
#include <absl/algorithm/container.h>
#include <absl/base/nullability.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/str_cat.h>
#include <Eigen/Dense>
#include <pybind11/attr.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "nuri/eigen_config.h"
#include "nuri/meta.h"
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
  absl::Nonnull<C *> container_;
  int index_;
  int size_;
};

template <class DT, int Flags>
py::ssize_t eigen_stride(const py::array_t<DT, Flags> &arr, int dim) {
  ABSL_DCHECK_LT(dim, arr.ndim());
  return arr.strides()[dim] / static_cast<py::ssize_t>(sizeof(DT));
}

template <Eigen::Index Rows = Eigen::Dynamic, Eigen::Index Cols = 1,
          class DT = double, bool Const = true>
using PyVectorMap = std::enable_if_t<
    Rows == 1 || Cols == 1,
    Eigen::Map<internal::const_if_t<Const, Matrix<DT, Rows, Cols>>>>;

template <Eigen::Index Rows = Eigen::Dynamic, Eigen::Index Cols = Eigen::Dynamic,
          class DT = double, bool Const = true>
using PyMatrixMap =
    Eigen::Map<internal::const_if_t<Const, Matrix<DT, Rows, Cols>>,
               Eigen::Unaligned, Eigen::OuterStride<>>;

template <Eigen::Index Rows = Eigen::Dynamic,
          Eigen::Index Cols = Eigen::Dynamic, class DT = double>
class NpArrayWrapper;

template <Eigen::Index Rows, Eigen::Index Cols, class DT>
void numpy_to_eigen_check_compat(const py::array_t<DT> &arr) {
  constexpr bool is_vector = Rows == 1 || Cols == 1;
  constexpr Eigen::Index size = Rows == Eigen::Dynamic || Cols == Eigen::Dynamic
                                    ? Eigen::Dynamic
                                    : Rows * Cols;

  if constexpr (is_vector) {
    if (arr.ndim() != 1)
      throw py::value_error(
          absl::StrCat("expected 1D array, got ", arr.ndim(), "D"));

    if constexpr (size != Eigen::Dynamic) {
      if (arr.size() != size)
        throw py::value_error(
            absl::StrCat("expected ", size, " elements, got ", arr.size()));
    }
  } else {
    if (arr.ndim() != 2)
      throw py::value_error(
          absl::StrCat("expected 2D array, got ", arr.ndim(), "D"));

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
  }
}

template <class ML>
using NpArrayLike = NpArrayWrapper<ML::RowsAtCompileTime, ML::ColsAtCompileTime,
                                   typename ML::Scalar>;

template <Eigen::Index Rows, Eigen::Index Cols, class DT>
class NpArrayWrapper: private py::array_t<DT> {
private:
  using Parent = py::array_t<DT>;

  constexpr static bool kIsVector = Rows == 1 || Cols == 1;

  template <class Ptr>
  decltype(auto) eigen_helper(Ptr *data) const {
    if constexpr (kIsVector) {
      PyVectorMap<Rows, Cols, DT, std::is_const_v<Ptr>> map(data, this->size());
      return map;
    } else {
      PyMatrixMap<Rows, Cols, DT, std::is_const_v<Ptr>> map(
          data, this->shape()[1], this->shape()[0],
          Eigen::OuterStride<> { eigen_stride(*this, 0) });
      return map;
    }
  }

public:
  decltype(auto) eigen() & { return eigen_helper(this->mutable_data()); }

  decltype(auto) eigen() const & { return eigen_helper(this->data()); }

  // Temporary lifetime extension does not work here
  decltype(auto) eigen() && = delete;

  Parent numpy() const & { return *this; }

  Parent numpy() && { return std::move(*this); }

private:
  explicit NpArrayWrapper(std::vector<py::ssize_t> &&shape)
      : Parent(std::move(shape)) {
    check_invariants();
  }

  explicit NpArrayWrapper(const Parent &arr): Parent(arr) {
    check_invariants();
  }

  explicit NpArrayWrapper(Parent &&arr): Parent(std::move(arr)) {
    check_invariants();
  }

  void check_invariants() const {
    constexpr int req_ndim = kIsVector ? 1 : 2;

    numpy_to_eigen_check_compat<Rows, Cols, DT>(*this);

    const auto inner_stride = eigen_stride(*this, req_ndim - 1);
    // numpy returns stride 0 for empty arrays
    if (inner_stride != 1 && this->size() > 0) {
      throw std::runtime_error(
          absl::StrCat("Unexpected inner stride (", inner_stride, " != 1)"));
    }
  }

  template <Eigen::Index R, Eigen::Index C, class DU>
  friend NpArrayWrapper<R, C, DU>
  empty_numpy(std::vector<py::ssize_t> &&eigen_shape);

  template <class ML>
  friend NpArrayLike<ML> empty_like(const ML &mat);

  template <Eigen::Index R, Eigen::Index C, class DU>
  friend NpArrayWrapper<R, C, DU> py_array_cast(py::handle h);
};

template <Eigen::Index Rows = Eigen::Dynamic,
          Eigen::Index Cols = Eigen::Dynamic, class DT = double>
NpArrayWrapper<Rows, Cols, DT>
empty_numpy(std::vector<py::ssize_t> &&eigen_shape) {
  absl::c_reverse(eigen_shape);
  return NpArrayWrapper<Rows, Cols, DT>(std::move(eigen_shape));
}

template <class ML>
NpArrayLike<ML> empty_like(const ML &mat) {
  std::vector<py::ssize_t> shape;
  if constexpr (ML::RowsAtCompileTime == 1 || ML::ColsAtCompileTime == 1) {
    shape.push_back(mat.size());
  } else {
    shape.assign({ mat.cols(), mat.rows() });
  }
  return NpArrayLike<ML>(std::move(shape));
}

template <Eigen::Index Rows = Eigen::Dynamic,
          Eigen::Index Cols = Eigen::Dynamic, class DT = double>
NpArrayWrapper<Rows, Cols, DT> py_array_cast(py::handle h) {
  if (h.is_none())
    throw py::type_error("expected array-like object, got None");

  PyObject *result = NpArrayWrapper<Rows, Cols, DT>::raw_array_t(h.ptr());
  if (result == nullptr) {
    py::error_already_set current_exc;
    py::raise_from(current_exc, PyExc_ValueError,
                   "cannot convert object to numpy array");
    throw py::error_already_set();
  }

  py::array_t<DT> arr = py::reinterpret_steal<py::array_t<DT>>(result);
  numpy_to_eigen_check_compat<Rows, Cols, DT>(arr);

  auto maybe_copy = [&arr](Eigen::Index rows, Eigen::Index cols, auto strides) {
    if (strides.inner() == 1)
      return NpArrayWrapper<Rows, Cols, DT> { std::move(arr) };

    ABSL_DLOG(INFO) << "copy triggered";

    Eigen::Map<const Matrix<DT, Rows, Cols>, Eigen::Unaligned, decltype(strides)>
        data(arr.data(), rows, cols, strides);

    auto wrapper = empty_like(data);
    wrapper.eigen() = data;
    return wrapper;
  };

  if constexpr (Rows == 1 || Cols == 1) {
    Eigen::Index rows, cols;
    if constexpr (Cols == 1) {
      rows = arr.size();
      cols = 1;
    } else {
      rows = 1;
      cols = arr.size();
    }
    return maybe_copy(rows, cols,
                      Eigen::InnerStride<> { eigen_stride(arr, 0) });
  } else {
    return maybe_copy(arr.shape()[1], arr.shape()[0],
                      py::EigenDStride { eigen_stride(arr, 0),
                                         eigen_stride(arr, 1) });
  }
}

template <class ML>
decltype(auto) eigen_as_numpy(const ML &mat) {
  auto arr = empty_like(mat);
  arr.eigen() = mat;
  return std::move(arr).numpy();
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
      [size](T &self, int idx) {
        return 0 <= idx && idx < std::invoke(size, self);
      },
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
