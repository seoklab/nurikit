//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_CORE_PROPERTY_MAP_H_
#define NURI_CORE_PROPERTY_MAP_H_

//! @cond
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/container/flat_map.hpp>
//! @endcond

namespace nuri {
namespace internal {
  using PropertyMap = boost::container::flat_map<
      std::string, std::string, std::less<>,
      std::vector<std::pair<std::string, std::string>>>;

  // RDKit-compatible key for name
  constexpr std::string_view kNameKey = "_Name";

  template <
      class PT,
      std::enable_if_t<std::is_same_v<PropertyMap, std::decay_t<PT>>, int> = 0>
  auto find_key(PT &props, std::string_view key) {
    return props.find(key);
  }

  template <
      class PT,
      std::enable_if_t<std::is_same_v<PropertyMap, std::decay_t<PT>>, int> = 0>
  bool has_key(PT &props, std::string_view key) {
    return props.contains(key);
  }

  template <
      class PT,
      std::enable_if_t<std::is_same_v<PropertyMap, std::decay_t<PT>>, int> = 0>
  std::string_view get_key(PT &props, std::string_view key) {
    auto it = find_key(props, key);
    if (it == props.end())
      return "";
    return it->second;
  }

  template <
      class PT, class ST,
      std::enable_if_t<std::is_same_v<PropertyMap, std::decay_t<PT>>, int> = 0>
  void set_key(PT &props, std::string_view key, ST &&value) {
    auto it = props.lower_bound(key);

    if (it != props.end() && it->first == key) {
      it->second = std::forward<ST>(value);
    } else {
      props.emplace_hint(it, key, std::forward<ST>(value));
    }
  }

  template <
      class PT,
      std::enable_if_t<std::is_same_v<PropertyMap, std::decay_t<PT>>, int> = 0>
  auto find_name(PT &props) -> decltype(props.begin()) {
    return find_key(props, kNameKey);
  }

  template <
      class PT,
      std::enable_if_t<std::is_same_v<PropertyMap, std::decay_t<PT>>, int> = 0>
  bool has_name(PT &props) {
    return has_key(props, kNameKey);
  }

  template <
      class PT,
      std::enable_if_t<std::is_same_v<PropertyMap, std::decay_t<PT>>, int> = 0>
  std::string_view get_name(PT &props) {
    return get_key(props, kNameKey);
  }

  template <
      class PT, class ST,
      std::enable_if_t<std::is_same_v<PropertyMap, std::decay_t<PT>>, int> = 0>
  void set_name(PT &props, ST &&name) {
    set_key(props, kNameKey, std::forward<ST>(name));
  }
}  // namespace internal
}  // namespace nuri

#endif /* NURI_CORE_PROPERTY_MAP_H_ */
