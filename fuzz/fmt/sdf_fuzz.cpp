//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <new>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <absl/base/call_once.h>
#include <absl/base/log_severity.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>

#include "fuzz_utils.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/sdf.h"

NURI_FUZZ_MAIN(data, size) {
  static absl::once_flag flag;
  absl::call_once(flag, []() {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverity::kFatal);
  });

  std::istringstream iss(
      std::string { reinterpret_cast<const char *>(data), size });
  nuri::SDFReader reader(iss);

  std::vector<std::string> block;
  std::string sdf;
  while (reader.getnext(block)) {
    nuri::Molecule mol;

    try {
      mol = reader.parse(block);
    } catch (const std::length_error & /* e */) {
      return -1;
    } catch (const std::bad_alloc & /* e */) {
      return -1;
    }

    if (mol.empty())
      continue;

    nuri::write_sdf(sdf, mol, -1, nuri::SDFVersion::kAutomatic);
    nuri::write_sdf(sdf, mol, -1, nuri::SDFVersion::kV2000);
    nuri::write_sdf(sdf, mol, -1, nuri::SDFVersion::kV3000);

    mol.confs().clear();

    nuri::write_sdf(sdf, mol, -1, nuri::SDFVersion::kAutomatic);
    nuri::write_sdf(sdf, mol, -1, nuri::SDFVersion::kV2000);
    nuri::write_sdf(sdf, mol, -1, nuri::SDFVersion::kV3000);
  }

  return 0;
}
