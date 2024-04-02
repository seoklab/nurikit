//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/python/core/containers.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/typing.h>

#include <absl/algorithm/container.h>

#include "nuri/python/config.h"
#include "nuri/python/utils.h"

namespace nuri {
namespace python_internal {
namespace {
// NOLINTBEGIN(clang-diagnostic-unused-member-function)

class MapKeyIterator: public PyIterator<MapKeyIterator, PropertyMap> {
  using Base = MapKeyIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) { return Base::bind(m, "_PropertyMapKeys"); }

private:
  friend Base;

  static auto size_of(const PropertyMap &map) { return map.size(); }

  static const std::string &deref(const PropertyMap &map, int idx) {
    return map[idx].first;
  }
};

class MapValIterator: public PyIterator<MapValIterator, PropertyMap> {
  using Base = MapValIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) {
    return Base::bind(m, "_PropertyMapValues");
  }

private:
  friend Base;

  static auto size_of(const PropertyMap &map) { return map.size(); }

  static const std::string &deref(const PropertyMap &map, int idx) {
    return map[idx].second;
  }
};

class MapPairIterator: public PyIterator<MapPairIterator, PropertyMap> {
  using Base = MapPairIterator::Parent;

public:
  using Base::Base;

  static auto bind(py::module_ &m) {
    return Base::bind(m, "_PropertyMapItems");
  }

private:
  friend Base;

  static auto size_of(const PropertyMap &map) { return map.size(); }

  static const auto &deref(const PropertyMap &map, int idx) { return map[idx]; }
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
    return (**map)[idx].first;
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
    return (**map)[idx].second;
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
    return (**map)[idx];
  }
};

// NOLINTEND(clang-diagnostic-unused-member-function)

template <class P>
struct MapTraits;

template <>
struct MapTraits<PropertyMap> {
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

auto find_map(PropertyMap &self, std::string_view key) {
  return absl::c_find_if(self,
                         [&](const auto &pair) { return pair.first == key; });
}

void map_setitem(PropertyMap &self, std::string_view key,
                 std::string_view value) {
  auto it = find_map(self, key);
  if (it == self.end())
    self.emplace_back(key, value);
  else
    it->second = value;
}

void map_setitem_py(PropertyMap &self, const py::handle &key,
                    const py::handle &val) {
  if (!py::isinstance<py::str>(key))
    throw py::type_error("keys must be strings");
  if (!py::isinstance<py::str>(val))
    throw py::type_error("values must be strings");

  map_setitem(self, key.cast<std::string_view>(), val.cast<std::string_view>());
}

void map_setitem_unpack(PropertyMap &self, const py::handle &pair) {
  if (py::len(pair) != 2)
    throw py::value_error("update expected at most 2-item tuples");

  py::handle key = pair[py::int_(0)], val = pair[py::int_(1)];
  map_setitem_py(self, key, val);
}

template <class T>
PropertyMap &prolog(T &self);

template <>
PropertyMap &prolog(PropertyMap &self) {
  return self;
}

template <>
PropertyMap &prolog(ProxyPropertyMap &self) {
  return **self;
}

template <class T>
py::class_<T> &add_map_interface(py::class_<T> &cls) {
  cls.def("__getitem__", [](T &self, std::string_view key) {
    PropertyMap &map = prolog(self);

    auto it = find_map(map, key);
    if (it == map.end())
      throw py::key_error(std::string(key));

    return it->second;
  });
  cls.def("__setitem__",
          [](T &self, std::string_view key, std::string_view value) {
            map_setitem(prolog(self), key, value);
          });
  cls.def("__delitem__", [](T &self, std::string_view key) {
    PropertyMap &map = prolog(self);

    auto it = find_map(map, key);
    if (it == map.end())
      throw py::key_error(std::string(key));

    map.erase(it);
  });
  cls.def("__contains__", [](T &self, std::string_view key) {
    PropertyMap &map = prolog(self);
    return find_map(map, key) != map.end();
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
    PropertyMap &map = prolog(self);

    auto it = find_map(map, key);
    if (it == map.end())
      throw py::key_error(std::string(key));

    return it->second;
  });
  cls.def(
      "get",
      [](T &self, std::string_view key, const py::str &def) {
        PropertyMap &map = prolog(self);
        auto it = find_map(map, key);
        return it == map.end() ? def : py::str(it->second);
      },
      py::arg("key"), py::arg("default"));
  cls.def("pop", [](T &self, std::string_view key) {
    PropertyMap &map = prolog(self);

    auto it = find_map(map, key);
    if (it == map.end())
      throw py::key_error(std::string(key));

    py::str value = it->second;
    map.erase(it);
    return value;
  });
  cls.def(
      "pop",
      [](T &self, std::string_view key, py::str def) {
        PropertyMap &map = prolog(self);

        auto it = find_map(map, key);
        if (it == map.end())
          return def;

        py::str value = it->second;
        map.erase(it);
        return value;
      },
      py::arg("key"), py::arg("default"));
  cls.def("popitem", [](T &self) {
    PropertyMap &map = prolog(self);

    if (map.empty())
      throw py::key_error("popitem from an empty mapping");

    std::pair ret = std::move(map.back());
    map.pop_back();
    return ret;
  });
  cls.def("clear", [](T &self) { prolog(self).clear(); });
  cls.def("update", [](T &self, const py::dict &other) {
    PropertyMap &map = prolog(self);
    for (auto item: other)
      map_setitem_py(map, item.first, item.second);
  });
  cls.def("update", [](T &self, const py::kwargs &kwargs) {
    PropertyMap &map = prolog(self);
    for (auto item: kwargs)
      map_setitem_py(map, item.first, item.second);
  });
  cls.def("update",
          [](T &self,
             const pyt::Iterable<pyt::Tuple<py::str, py::str>> &other) {
            PropertyMap &map = prolog(self);
            for (auto item: other)
              map_setitem_unpack(map, item);
          });
  cls.def("setdefault", [](T &self, std::string_view key, py::str def) {
    PropertyMap &map = prolog(self);

    auto it = find_map(map, key);
    if (it != map.end())
      return py::str(it->second);

    map.emplace_back(key, def);
    return def;
  });
  cls.def("copy", [](T &self) { return PropertyMap(prolog(self)); });
  cls.def("__copy__", [](T &self) { return PropertyMap(prolog(self)); });
  cls.def(
      "__deepcopy__",
      [](T &self, const py::dict & /* unused */) {
        return PropertyMap(prolog(self));
      },
      py::arg("memo"));

  return cls;
}

void bind_property_map(py::module &m) {
  py::class_<PropertyMap> pm(m, "_PropertyMap");
  py::class_<ProxyPropertyMap> ppm(m, "_ProxyPropertyMap");

  MapKeyIterator::bind(m);
  MapValIterator::bind(m);
  MapPairIterator::bind(m);

  ProxyMapKeyIterator::bind(m);
  ProxyMapValIterator::bind(m);
  ProxyMapPairIterator::bind(m);

  add_map_interface(pm);
  pm.def(py::init<>())
      .def(py::init(
          [](const ProxyPropertyMap &other) { return PropertyMap(**other); }))
      .def(py::init([](const py::dict &dict) {
        PropertyMap map;
        map.reserve(dict.size());
        for (auto item: dict)
          map_setitem_py(map, item.first, item.second);
        return map;
      }));

  add_map_interface(ppm);

  py::implicitly_convertible<ProxyPropertyMap, PropertyMap>();
  py::implicitly_convertible<py::dict, PropertyMap>();
}
}  // namespace

void bind_containers(py::module &m) {
  bind_property_map(m);
}
}  // namespace python_internal
}  // namespace nuri
