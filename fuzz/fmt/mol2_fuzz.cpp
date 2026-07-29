//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <utility>
#include <vector>

#include <absl/base/call_once.h>
#include <absl/base/log_severity.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>

#include "fuzz_utils.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/mol2.h"
#include "nuri/fmt/parse_result.h"

NURI_FUZZ_MAIN(data, size) {
  static absl::once_flag flag;
  absl::call_once(flag, []() {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverity::kFatal);
  });

  std::istringstream iss(
      std::string { reinterpret_cast<const char *>(data), size });
  nuri::Mol2Reader reader(iss);

  std::vector<std::string> block;
  std::string mol2;
  while (reader.getnext(block)) {
    nuri::ParseResult<nuri::Molecule> res = reader.parse(block);
    if (!res)
      continue;

    nuri::Molecule mol = *std::move(res);

    nuri::write_mol2(mol2, mol);
    mol.confs().clear();
    nuri::write_mol2(mol2, mol);
  }

  return 0;
}
