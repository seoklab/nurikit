#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

add_custom_target(nuri_python)
add_dependencies(nuri_python nuri_lib)

if(NURI_BUILD_PYTHON)
  set_target_properties(nuri_python PROPERTIES EXCLUDE_FROM_ALL OFF)
endif()

include(NuriKitPythonUtils)

nuri_python_generate_stubs(nuri)

nuri_python_add_module(_log_interface nuri/_log_interface_module.cpp)
target_link_libraries(nuri_python_log_interface PRIVATE absl::log_initialize)

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
target_link_libraries(nuri_python_fmt PRIVATE absl::span)

nuri_python_add_module(tm OUTPUT_DIRECTORY "tools"
  nuri/tools/tm_module.cpp
)
