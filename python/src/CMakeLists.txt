#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

if(NURI_BUILD_PYTHON_STUBS)
  find_program(PYBIND11_STUBGEN pybind11-stubgen)
  mark_as_advanced(PYBIND11_STUBGEN)

  if(PYBIND11_STUBGEN)
    message(NOTICE "Generating Python stubs for NuriKit")
  else()
    message(WARNING "pybind11-stubgen not found, cannot generate stubs")
    set(NURI_BUILD_PYTHON_STUBS OFF)
  endif()
endif()

add_custom_target(NuriPython ALL)
clear_coverage_data(NuriPython)

if(NURI_BUILD_LIB)
  add_dependencies(NuriPython NuriLib)
else()
  find_package("${PROJECT_NAME}" "${PROJECT_VERSION}" REQUIRED)
endif()

include(NuriKitPythonUtils)

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

if(NURI_BUILD_PYTHON_STUBS)
  file(
    GLOB_RECURSE NURI_PYTHON_FILES
    RELATIVE "${CMAKE_CURRENT_LIST_DIR}/"
    "nuri/*.py"
  )

  foreach(pyfile IN LISTS NURI_PYTHON_FILES)
    string(REGEX REPLACE "\\.py$" "" module "${pyfile}")
    string(REGEX REPLACE "/+" "." module "${module}")

    string(REGEX REPLACE "\\.py$" ".pyi" stubfile "${pyfile}")

    nuri_python_generate_stubs(
      "${module}"
      "${stubfile}"
      "${CMAKE_CURRENT_LIST_DIR}/${pyfile}"
    )
  endforeach()

  add_custom_target(NuriPythonStubs
    ALL
    DEPENDS NuriPython ${NURI_STUB_FILES}
  )
  clear_coverage_data(NuriPythonStubs)
endif()
