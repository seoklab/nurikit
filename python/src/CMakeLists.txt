#
# Project nurikit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

function(nuri_python_add_module name)
  cmake_parse_arguments(ARG "" "OUTPUT_DIRECTORY" "" ${ARGN})

  if(NOT ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "nuri_python_add_module: No sources provided")
  endif()

  set(sources ${ARG_UNPARSED_ARGUMENTS})

  if(ARG_OUTPUT_DIRECTORY)
    set(subdir "${ARG_OUTPUT_DIRECTORY}")
  else()
    set(subdir "")
  endif()

  set(target_name "nuri_python_${name}")
  string(REGEX REPLACE "_+" "_" target_name "${target_name}")

  file(RELATIVE_PATH dir_inv
    "${CMAKE_CURRENT_LIST_DIR}/nuri/${ARG_OUTPUT_DIRECTORY}"
    "${CMAKE_CURRENT_LIST_DIR}/nuri/")

  pybind11_add_module("${target_name}" OPT_SIZE "${sources}")
  target_link_libraries("${target_name}" PRIVATE nuri_lib)
  target_compile_definitions(
    "${target_name}"
    PRIVATE "NURI_PYTHON_MODULE_NAME=${name}")
  set_target_properties("${target_name}"
    PROPERTIES
    OUTPUT_NAME "${name}"
    INSTALL_RPATH "$ORIGIN/${dir_inv}${CMAKE_INSTALL_LIBDIR}")

  if(NOT SKBUILD)
    set_target_properties("${target_name}"
      PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/nuri/${subdir}")
  endif()

  install(TARGETS "${target_name}"
    LIBRARY
    DESTINATION "./${subdir}")

  add_dependencies(nuri_python "${target_name}")
endfunction()

add_custom_target(nuri_python)
add_dependencies(nuri_python nuri_lib)

nuri_python_add_module(_log_interface nuri/_log_interface.cpp)
target_link_libraries(nuri_python_log_interface PRIVATE absl::log_initialize)

nuri_python_add_module(_core OUTPUT_DIRECTORY "core"
  nuri/core/_core.cpp
  nuri/core/containers.cpp
  nuri/core/element.cpp
  nuri/core/molecule.cpp
  nuri/core/substructure.cpp
)

nuri_python_add_module(algo
  nuri/algo/algo_module.cpp
)

nuri_python_add_module(fmt
  nuri/fmt/fmt_module.cpp
)

find_program(PYBIND11_STUBGEN pybind11-stubgen)
mark_as_advanced(PYBIND11_STUBGEN)

if(PYBIND11_STUBGEN STREQUAL "PYBIND11_STUBGEN-NOTFOUND")
  message("pybind11-stubgen not found, skipping stub generation")
  return()
endif()

add_custom_command(
  TARGET nuri_python
  POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E remove_directory
  "${CMAKE_CURRENT_LIST_DIR}/stubs"
  COMMAND "${PYBIND11_STUBGEN}"
  --enum-class-locations .*:nuri.core._core
  --numpy-array-remove-parameters
  nuri
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  VERBATIM)
