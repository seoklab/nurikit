//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/base.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_log.h>

namespace nuri {
namespace {
absl::flat_hash_map<std::string, const MoleculeReaderFactory *> &
reader_factory_registry() {
  static absl::flat_hash_map<std::string, const MoleculeReaderFactory *> ret;
  return ret;
}
}  // namespace

const MoleculeReaderFactory *
MoleculeReaderFactory::find_factory(std::string_view name) {
  const absl::flat_hash_map<std::string, const MoleculeReaderFactory *> &reg =
      reader_factory_registry();

  auto it = reg.find(name);
  if (it == reg.end()) {
    return nullptr;
  }
  return it->second;
}

bool MoleculeReaderFactory::register_factory(
    std::unique_ptr<MoleculeReaderFactory> factory,
    const std::vector<std::string> &names) {
  static std::vector<std::unique_ptr<MoleculeReaderFactory>> factories;

  MoleculeReaderFactory *f = factories.emplace_back(std::move(factory)).get();
  // GCOV_EXCL_START
  ABSL_LOG_IF(WARNING, names.empty()) << "Empty name list for factory";
  // GCOV_EXCL_STOP

  for (const auto &name: names) {
    register_for_name(f, name);
  }

  return true;
}

void MoleculeReaderFactory::register_for_name(
    const MoleculeReaderFactory *factory, std::string_view name) {
  auto [_, inserted] =
      reader_factory_registry().insert_or_assign(name, factory);
  // GCOV_EXCL_START
  ABSL_LOG_IF(WARNING, !inserted)
      << "Duplicate factory name: " << name
      << ". Overwriting existing factory (is this intended?).";
  // GCOV_EXCL_STOP
}
}  // namespace nuri
