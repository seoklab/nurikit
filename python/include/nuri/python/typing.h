//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_PYTHON_TYPING_H_
#define NURI_PYTHON_TYPING_H_

#include <string_view>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/typing.h>

namespace nuri {
namespace python_internal {
// Typed ``collections.abc.Sequence`` annotation, mirroring pybind11's own
// ``typing::List`` (which pybind11 2.13 provides but without a ``Sequence``
// counterpart). Inheriting ``pybind11::sequence`` keeps identical runtime
// behavior -- the element type ``T`` is annotation-only, as with
// ``pyt::List``/``Iterable``.
template <class T>
class Sequence: public pybind11::sequence {
  using sequence::sequence;
};

// As Sequence, but for mappings; the PyMapping_Check guard validates inputs
// when used as a parameter (Sequence is return-only).
template <class K, class V>
class Mapping: public pybind11::object {
public:
  PYBIND11_OBJECT_DEFAULT(Mapping, object, PyMapping_Check)
};

template <class K, class V>
class MutableMapping: public pybind11::object {
public:
  PYBIND11_OBJECT_DEFAULT(MutableMapping, object, PyMapping_Check)
};

constexpr inline std::string_view kAbcSequence = "Sequence";
constexpr inline std::string_view kAbcMutableMapping = "MutableMapping";

// Sequence/Mapping lack the structural __subclasshook__ that abc.Iterator has,
// so a masqueraded wrapper needs explicit registration for isinstance to match.
inline void register_abc(pybind11::handle cls, std::string_view name) {
  pybind11::module_::import("collections.abc")
      .attr(pybind11::str(name.data(), name.size()))
      .attr("register")(cls);
}
}  // namespace python_internal
}  // namespace nuri

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)
template <class T>
struct handle_type_name<nuri::python_internal::Sequence<T>> {
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr static auto name =
      const_name("Sequence[") + make_caster<T>::name + const_name("]");
};

template <class K, class V>
struct handle_type_name<nuri::python_internal::Mapping<K, V>> {
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr static auto name = const_name("Mapping[") + make_caster<K>::name
                               + const_name(", ") + make_caster<V>::name
                               + const_name("]");
};

template <class K, class V>
struct handle_type_name<nuri::python_internal::MutableMapping<K, V>> {
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr static auto name = const_name("MutableMapping[")
                               + make_caster<K>::name + const_name(", ")
                               + make_caster<V>::name + const_name("]");
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

#endif /* NURI_PYTHON_TYPING_H_ */
