//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/python/core/containers.h"
#include "nuri/python/core/core_module.h"

namespace nuri {
namespace python_internal {
namespace {
NURI_PYTHON_MODULE(m) {
  bind_containers(m);
  bind_element(m);
  bind_molecule(m);
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
