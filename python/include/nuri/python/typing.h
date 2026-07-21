//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_TYPING_H_
#define NURI_PYTHON_TYPING_H_

#include <string_view>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/typing.h>

namespace nuri {
namespace python_internal {
// Typed ``collections.abc.Sequence`` annotation, mirroring pybind11's own
// ``typing::List`` (which pybind11 2.13 provides but without a ``Sequence``
// counterpart). Inheriting ``py::sequence`` keeps identical runtime
// behavior -- the element type ``T`` is annotation-only, as with
// ``pyt::List``/``Iterable``.
template <class T>
class Sequence: public py::sequence {
  using sequence::sequence;
};

template <class K, class V>
class Mapping: public py::object {
public:
  PYBIND11_OBJECT_DEFAULT(Mapping, object, PyMapping_Check)
};

template <class K, class V>
class MutableMapping: public py::object {
public:
  PYBIND11_OBJECT_DEFAULT(MutableMapping, object, PyMapping_Check)
};

// Renders as T in signatures; the runtime object stays whatever was cast.
template <class T>
class As: public py::object {
public:
  using object::object;
  // NOLINTNEXTLINE(google-explicit-constructor)
  As(py::object obj) noexcept: py::object(std::move(obj)) { }
};

constexpr inline std::string_view kAbcSequence = "Sequence";
constexpr inline std::string_view kAbcMutableMapping = "MutableMapping";

// abc.Sequence/Mapping lack Iterator's structural __subclasshook__, so
// masqueraded wrappers need an explicit register() for isinstance to match.
inline void register_abc(py::handle cls, std::string_view name) {
  py::module_::import("collections.abc")
      .attr(py::str(name.data(), name.size()))
      .attr("register")(cls);
}
}  // namespace python_internal
}  // namespace nuri

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
template <class T>
struct handle_type_name<nuri::python_internal::Sequence<T>> {
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr static auto name = const_name("collections.abc.Sequence[")
                               + make_caster<T>::name + const_name("]");
};

template <class K, class V>
struct handle_type_name<nuri::python_internal::Mapping<K, V>> {
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr static auto name = const_name("collections.abc.Mapping[")
                               + make_caster<K>::name + const_name(", ")
                               + make_caster<V>::name + const_name("]");
};

template <class K, class V>
struct handle_type_name<nuri::python_internal::MutableMapping<K, V>> {
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr static auto name = const_name("collections.abc.MutableMapping[")
                               + make_caster<K>::name + const_name(", ")
                               + make_caster<V>::name + const_name("]");
};

template <class T>
struct handle_type_name<nuri::python_internal::As<T>> {
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr static auto name = make_caster<T>::name;
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#endif /* NURI_PYTHON_TYPING_H_ */
