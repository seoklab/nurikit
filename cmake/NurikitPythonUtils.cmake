#
# Project nurikit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

add_custom_command(
  TARGET nuri_python
  POST_BUILD
  COMMAND "${CMAKE_COMMAND}"
  -P "${PROJECT_SOURCE_DIR}/cmake/NurikitClearStubs.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  VERBATIM)

find_program(PYBIND11_STUBGEN pybind11-stubgen)
mark_as_advanced(PYBIND11_STUBGEN)

if(PYBIND11_STUBGEN STREQUAL "PYBIND11_STUBGEN-NOTFOUND")
  message(NOTICE "pybind11-stubgen not found, skipping stub generation")
else()
  set(PYBIND11_STUBGEN_FOUND ON)
endif()

function(nuri_python_generate_stubs module)
  if(NOT PYBIND11_STUBGEN_FOUND)
    message(WARNING "pybind11-stubgen not found, skipping stub generation")
    return()
  endif()

  add_custom_command(
    TARGET nuri_python
    POST_BUILD
    COMMAND ${EXECUTE_WITH_SAN}
    "${PYBIND11_STUBGEN}"
    -o "${CMAKE_CURRENT_LIST_DIR}"
    --enum-class-locations .*:nuri.core._core
    --numpy-array-remove-parameters
    ${ARGN}
    "${module}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    VERBATIM
  )
endfunction()

function(nuri_python_add_module name)
  cmake_parse_arguments(ARG "" "OUTPUT_DIRECTORY" "STUBGEN_ARGS" ${ARGN})

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
    "${CMAKE_CURRENT_LIST_DIR}/nuri/${subdir}"
    "${CMAKE_CURRENT_LIST_DIR}/nuri/")

  pybind11_add_module("${target_name}" OPT_SIZE "${sources}")
  target_link_libraries("${target_name}" PRIVATE nuri_lib)
  target_compile_definitions(
    "${target_name}"
    PRIVATE "NURI_PYTHON_MODULE_NAME=${name}")
  set_target_properties("${target_name}"
    PROPERTIES
    OUTPUT_NAME "${name}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/nuri/${subdir}"
    INSTALL_RPATH "$ORIGIN/${dir_inv}${CMAKE_INSTALL_LIBDIR}"
  )

  if(PYBIND11_STUBGEN_FOUND)
    file(RELATIVE_PATH submodule
      "${CMAKE_CURRENT_LIST_DIR}/"
      "${CMAKE_CURRENT_LIST_DIR}/nuri/${subdir}/${name}")
    string(REGEX REPLACE "/" "." submodule "${submodule}")
    message(STATUS "Generating stubs for ${submodule}")

    nuri_python_generate_stubs("${submodule}" ${ARG_STUBGEN_ARGS})
  endif()

  install(TARGETS "${target_name}" LIBRARY DESTINATION "./${subdir}")

  add_dependencies(nuri_python "${target_name}")
endfunction()
