#
# Project nurikit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

function(nuri_python_add_module name directory)
  set(target_name "nuri_python${name}")

  file(RELATIVE_PATH dir_inv
    "${CMAKE_CURRENT_LIST_DIR}/${directory}"
    "${CMAKE_CURRENT_LIST_DIR}/")

  pybind11_add_module("${target_name}" OPT_SIZE "${ARGN}")
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
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${directory}")
  endif()

  install(TARGETS "${target_name}"
    LIBRARY
    DESTINATION "./${directory}")

  add_dependencies(nuri_python "${target_name}")
endfunction()

add_custom_target(nuri_python)
add_dependencies(nuri_python nuri_lib)

nuri_python_add_module(_log_interface "nuri" "nuri/_log_interface.cpp")
target_link_libraries(nuri_python_log_interface PRIVATE absl::log_initialize)

nuri_python_add_module(_core "nuri/core"
  nuri/core/_core.cpp
  nuri/core/containers.cpp
  nuri/core/element.cpp
  nuri/core/molecule.cpp
  nuri/core/substructure.cpp
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
