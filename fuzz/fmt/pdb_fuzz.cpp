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
#include "nuri/fmt/parse_result.h"
#include "nuri/fmt/pdb.h"

NURI_FUZZ_MAIN(data, size) {
  static absl::once_flag flag;
  absl::call_once(flag, []() {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverity::kFatal);
  });

  std::istringstream iss(
      std::string { reinterpret_cast<const char *>(data), size });
  nuri::PDBReader reader(iss);

  auto record = reader.make_record();
  while (reader.getnext(*record)) {
    nuri::ParseResult<nuri::Molecule> res = record->next();
    if (!res)
      continue;

    nuri::Molecule mol = *std::move(res);

    std::string buf;
    nuri::write_pdb(buf, mol);

    mol.confs().clear();
    nuri::write_pdb(buf, mol);
  }

  return 0;
}
