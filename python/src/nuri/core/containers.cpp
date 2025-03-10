//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/python/core/containers.h"

#include <string>
#include <string_view>
#include <utility>

#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/typing.h>

#include "nuri/core/property_map.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
// NOLINTBEGIN(clang-diagnostic-unused-member-function)

class MapKeyIterator: public PyIterator<MapKeyIterator, internal::PropertyMap> {
  using Base = MapKeyIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) { return Base::bind(m, "_PropertyMapKeys"); }

private:
  friend Base;

  static auto size_of(const internal::PropertyMap &map) { return map.size(); }

  static const std::string &deref(const internal::PropertyMap &map, int idx) {
    return map.sequence()[idx].first;
  }
};

class MapValIterator: public PyIterator<MapValIterator, internal::PropertyMap> {
  using Base = MapValIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) {
    return Base::bind(m, "_PropertyMapValues");
  }

private:
  friend Base;

  static auto size_of(const internal::PropertyMap &map) { return map.size(); }

  static const std::string &deref(const internal::PropertyMap &map, int idx) {
    return map.sequence()[idx].second;
  }
};

class MapPairIterator
    : public PyIterator<MapPairIterator, internal::PropertyMap> {
  using Base = MapPairIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) {
    return Base::bind(m, "_PropertyMapItems");
  }

private:
  friend Base;

  static auto size_of(const internal::PropertyMap &map) { return map.size(); }

  static const auto &deref(const internal::PropertyMap &map, int idx) {
    return map.sequence()[idx];
  }
};

class ProxyMapKeyIterator
    : public PyIterator<ProxyMapKeyIterator, ProxyPropertyMap> {
  using Base = ProxyMapKeyIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) {
    return Base::bind(m, "_ProxyPropertyMapKeys");
  }

private:
  friend Base;

  static auto size_of(const ProxyPropertyMap &map) { return (**map).size(); }

  static const std::string &deref(const ProxyPropertyMap &map, int idx) {
    return (**map).sequence()[idx].first;
  }
};

class ProxyMapValIterator
    : public PyIterator<ProxyMapValIterator, ProxyPropertyMap> {
  using Base = ProxyMapValIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) {
    return Base::bind(m, "_ProxyPropertyMapValues");
  }

private:
  friend Base;

  static auto size_of(const ProxyPropertyMap &map) { return (**map).size(); }

  static const std::string &deref(const ProxyPropertyMap &map, int idx) {
    return (**map).sequence()[idx].second;
  }
};

class ProxyMapPairIterator
    : public PyIterator<ProxyMapPairIterator, ProxyPropertyMap> {
  using Base = ProxyMapPairIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) {
    return Base::bind(m, "_ProxyPropertyMapItems");
  }

private:
  friend Base;

  static auto size_of(const ProxyPropertyMap &map) { return (**map).size(); }

  static const auto &deref(const ProxyPropertyMap &map, int idx) {
    return (**map).sequence()[idx];
  }
};

// NOLINTEND(clang-diagnostic-unused-member-function)

template <class P>
struct MapTraits;

template <>
struct MapTraits<internal::PropertyMap> {
  using iterator = MapPairIterator;
  using key_iterator = MapKeyIterator;
  using value_iterator = MapValIterator;
};

template <>
struct MapTraits<ProxyPropertyMap> {
  using iterator = ProxyMapPairIterator;
  using key_iterator = ProxyMapKeyIterator;
  using value_iterator = ProxyMapValIterator;
};

void map_setitem_py(internal::PropertyMap &self, const py::handle &key,
                    const py::handle &val) {
  if (!py::isinstance<py::str>(key))
    throw py::type_error("keys must be strings");
  if (!py::isinstance<py::str>(val))
    throw py::type_error("values must be strings");

  internal::set_key(self, key.cast<std::string_view>(),
                    val.cast<std::string_view>());
}

void map_setitem_unpack(internal::PropertyMap &self, const py::handle &pair) {
  if (py::len(pair) != 2)
    throw py::value_error("update expected at most 2-item tuples");

  py::handle key = pair[py::int_(0)], val = pair[py::int_(1)];
  map_setitem_py(self, key, val);
}

template <class T>
internal::PropertyMap &prolog(T &self);

template <>
internal::PropertyMap &prolog(internal::PropertyMap &self) {
  return self;
}

template <>
internal::PropertyMap &prolog(ProxyPropertyMap &self) {
  return **self;
}

template <class T>
py::class_<T> &add_map_interface(py::class_<T> &cls) {
  cls.def("__getitem__", [](T &self, std::string_view key) {
    internal::PropertyMap &map = prolog(self);

    auto it = map.find(key);
    if (it == map.end())
      throw py::key_error(std::string(key));

    return it->second;
  });
  cls.def("__setitem__",
          [](T &self, std::string_view key, std::string_view value) {
            internal::set_key(prolog(self), key, value);
          });
  cls.def("__delitem__", [](T &self, std::string_view key) {
    internal::PropertyMap &map = prolog(self);

    auto it = map.find(key);
    if (it == map.end())
      throw py::key_error(std::string(key));

    map.erase(it);
  });
  cls.def("__contains__", [](T &self, std::string_view key) {
    const internal::PropertyMap &map = prolog(self);
    return map.contains(key);
  });
  cls.def("__len__", [](T &self) { return prolog(self).size(); });
  cls.def(
      "__iter__",
      [](T &self) { return typename MapTraits<T>::key_iterator { self }; },
      kReturnsSubobject);
  cls.def(
      "keys",
      [](T &self) { return typename MapTraits<T>::key_iterator { self }; },
      kReturnsSubobject);
  cls.def(
      "values",
      [](T &self) { return typename MapTraits<T>::value_iterator { self }; },
      kReturnsSubobject);
  cls.def(
      "items", [](T &self) { return typename MapTraits<T>::iterator { self }; },
      kReturnsSubobject);
  cls.def("get", [](T &self, std::string_view key) {
    internal::PropertyMap &map = prolog(self);

    auto it = map.find(key);
    if (it == map.end())
      throw py::key_error(std::string(key));

    return it->second;
  });
  cls.def(
      "get",
      [](T &self, std::string_view key, const py::str &def) {
        internal::PropertyMap &map = prolog(self);
        auto it = map.find(key);
        return it == map.end() ? def : py::str(it->second);
      },
      py::arg("key"), py::arg("default"));
  cls.def("pop", [](T &self, std::string_view key) {
    internal::PropertyMap &map = prolog(self);

    auto it = map.find(key);
    if (it == map.end())
      throw py::key_error(std::string(key));

    py::str value = it->second;
    map.erase(it);
    return value;
  });
  cls.def(
      "pop",
      [](T &self, std::string_view key, py::str def) {
        internal::PropertyMap &map = prolog(self);

        auto it = map.find(key);
        if (it == map.end())
          return def;

        py::str value = it->second;
        map.erase(it);
        return value;
      },
      py::arg("key"), py::arg("default"));
  cls.def("popitem", [](T &self) {
    internal::PropertyMap &map = prolog(self);

    if (map.empty())
      throw py::key_error("popitem from an empty mapping");

    auto bit = --map.end();
    std::pair ret = std::move(*bit);
    map.erase(bit);
    return ret;
  });
  cls.def("clear", [](T &self) { prolog(self).clear(); });
  cls.def("update", [](T &self, const py::dict &other) {
    internal::PropertyMap &map = prolog(self);
    for (auto item: other)
      map_setitem_py(map, item.first, item.second);
  });
  cls.def("update", [](T &self, const py::kwargs &kwargs) {
    internal::PropertyMap &map = prolog(self);
    for (auto item: kwargs)
      map_setitem_py(map, item.first, item.second);
  });
  cls.def("update",
          [](T &self,
             const pyt::Iterable<pyt::Tuple<py::str, py::str>> &other) {
            internal::PropertyMap &map = prolog(self);
            for (auto item: other)
              map_setitem_unpack(map, item);
          });
  cls.def("setdefault", [](T &self, std::string_view key, const py::str &def) {
    internal::PropertyMap &map = prolog(self);

    auto [it, _] = map.emplace(key, def.cast<std::string_view>());
    return py::str(it->second);
  });
  cls.def("copy", [](T &self) { return internal::PropertyMap(prolog(self)); });
  cls.def("__copy__",
          [](T &self) { return internal::PropertyMap(prolog(self)); });
  cls.def(
      "__deepcopy__",
      [](T &self, const py::dict & /* unused */) {
        return internal::PropertyMap(prolog(self));
      },
      py::arg("memo"));

  return cls;
}

void bind_property_map(py::module &m) {
  py::class_<internal::PropertyMap> pm(m, "_PropertyMap");
  py::class_<ProxyPropertyMap> ppm(m, "_ProxyPropertyMap");

  MapKeyIterator::bind(m);
  MapValIterator::bind(m);
  MapPairIterator::bind(m);

  ProxyMapKeyIterator::bind(m);
  ProxyMapValIterator::bind(m);
  ProxyMapPairIterator::bind(m);

  add_map_interface(pm);
  pm.def(py::init<>())
      .def(py::init([](const ProxyPropertyMap &other) {
        return internal::PropertyMap(**other);
      }))
      .def(py::init([](const py::dict &dict) {
        internal::PropertyMap map;
        map.reserve(dict.size());
        for (auto item: dict)
          map_setitem_py(map, item.first, item.second);
        return map;
      }));

  add_map_interface(ppm);

  py::implicitly_convertible<ProxyPropertyMap, internal::PropertyMap>();
  py::implicitly_convertible<py::dict, internal::PropertyMap>();
}
}  // namespace

void bind_containers(py::module &m) {
  bind_property_map(m);
}
}  // namespace python_internal
}  // namespace nuri
