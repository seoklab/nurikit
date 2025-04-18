#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

add_custom_target(NuriPython ALL)

if(NURI_BUILD_LIB)
  add_dependencies(NuriPython NuriLib)
else()
  find_package("${PROJECT_NAME}" "${PROJECT_VERSION}" REQUIRED)
endif()

include(NuriKitPythonUtils)

nuri_python_generate_stubs(nuri)

nuri_python_add_module(_log_interface nuri/_log_interface_module.cpp)
target_link_libraries(NuriPythonMod-_log_interface PRIVATE absl::log_initialize)

nuri_python_add_module(_core OUTPUT_DIRECTORY "core"
  nuri/core/_core_module.cpp
  nuri/core/containers.cpp
  nuri/core/element.cpp
  nuri/core/molecule.cpp
  nuri/core/substructure.cpp
)

nuri_python_add_module(geometry OUTPUT_DIRECTORY "core"
  nuri/core/geometry_module.cpp
)

nuri_python_add_module(algo
  nuri/algo/algo_module.cpp
)

nuri_python_add_module(fmt
  nuri/fmt/fmt_module.cpp
  nuri/fmt/cif.cpp
)
target_link_libraries(NuriPythonMod-fmt PRIVATE absl::span)

nuri_python_add_module(tm OUTPUT_DIRECTORY "tools"
  nuri/tools/tm_module.cpp
)
