//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/base.h"

#include <absl/log/absl_log.h>

namespace nuri {
bool MoleculeStreamFactory::register_factory(
  std::unique_ptr<MoleculeStreamFactory> factory,
  const std::vector<std::string> &names) {
  static std::vector<std::unique_ptr<MoleculeStreamFactory>> factories;

  MoleculeStreamFactory *f = factories.emplace_back(std::move(factory)).get();
  // GCOV_EXCL_START
  ABSL_LOG_IF(WARNING, names.empty()) << "Empty name list for factory";
  // GCOV_EXCL_STOP

  for (const auto &name: names) {
    register_for_name(f, name);
  }

  return true;
}

void MoleculeStreamFactory::register_for_name(
  const MoleculeStreamFactory *factory, std::string_view name) {
  auto [_, inserted] = registry().insert_or_assign(name, factory);
  // GCOV_EXCL_START
  ABSL_LOG_IF(WARNING, !inserted)
    << "Duplicate factory name: " << name
    << ". Overwriting existing factory (is this intended?).";
  // GCOV_EXCL_STOP
}

absl::flat_hash_map<std::string, const MoleculeStreamFactory *> &
MoleculeStreamFactory::registry() {
  static absl::flat_hash_map<std::string, const MoleculeStreamFactory *> ret;
  return ret;
}
}  // namespace nuri
