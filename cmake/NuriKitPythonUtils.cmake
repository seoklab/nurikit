#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

function(nuri_python_generate_stubs module output dependency)
  if(NOT NURI_BUILD_PYTHON_STUBS)
    return()
  endif()

  set(pypath_orig "$ENV{PYTHONPATH}")

  if(pypath_orig)
    set(pypath_orig "${pypath_orig}:")
  endif()

  set(output_full "${CMAKE_CURRENT_LIST_DIR}/${output}")

  add_custom_command(
    OUTPUT "${output_full}"
    COMMAND ${CMAKE_COMMAND}
    -E env "PYTHONPATH=${pypath_orig}${CMAKE_CURRENT_LIST_DIR}" ${SANITIZER_ENVS}
    "${PYBIND11_STUBGEN}"
    -o "${CMAKE_CURRENT_LIST_DIR}"
    --enum-class-locations .*:nuri.core._core
    --numpy-array-remove-parameters
    ${ARGN}
    "${module}"
    DEPENDS "${dependency}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    COMMENT "Generating Python stubs for ${module}"
    VERBATIM
  )

  list(APPEND NURI_STUB_FILES "${output_full}")
  set(NURI_STUB_FILES "${NURI_STUB_FILES}" PARENT_SCOPE)
endfunction()

function(nuri_python_add_module name)
  cmake_parse_arguments(ARG "" "OUTPUT_DIRECTORY" "STUBGEN_ARGS" ${ARGN})

  if(NOT ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "nuri_python_add_module: No sources provided")
  endif()

  set(sources ${ARG_UNPARSED_ARGUMENTS})

  if(ARG_OUTPUT_DIRECTORY)
    set(subdir "${ARG_OUTPUT_DIRECTORY}/")
  else()
    set(subdir "")
  endif()

  set(target_name "NuriPythonMod-${subdir}-${name}")
  string(REGEX REPLACE "/+" "-" target_name "${target_name}")
  string(REGEX REPLACE "-+" "-" target_name "${target_name}")

  pybind11_add_module("${target_name}" OPT_SIZE "${sources}")
  target_link_libraries("${target_name}" PRIVATE "${PROJECT_NAME}::NuriLib")
  target_compile_definitions(
    "${target_name}"
    PRIVATE "NURI_PYTHON_MODULE_NAME=${name}"
  )
  set_target_properties(
    "${target_name}"
    PROPERTIES
    OUTPUT_NAME "${name}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/nuri/${subdir}"
  )

  if(NURI_BUILD_LIB AND NURI_INSTALL_RPATH)
    file(RELATIVE_PATH dir_inv
      "${CMAKE_CURRENT_LIST_DIR}/nuri/${subdir}"
      "${CMAKE_CURRENT_LIST_DIR}/nuri/"
    )
    set_target_properties(
      "${target_name}"
      PROPERTIES
      INSTALL_RPATH "${NURI_RPATH_PREFIX}/${dir_inv}${CMAKE_INSTALL_LIBDIR}"
    )
  endif()

  if(NURI_BUILD_PYTHON_STUBS)
    file(RELATIVE_PATH submodule
      "${CMAKE_CURRENT_LIST_DIR}/"
      "${CMAKE_CURRENT_LIST_DIR}/nuri/${subdir}${name}"
    )
    string(REGEX REPLACE "/" "." submodule "${submodule}")

    nuri_python_generate_stubs(
      "${submodule}"
      "nuri/${subdir}${name}.pyi"
      "${target_name}"
      ${ARG_STUBGEN_ARGS}
    )
    set(NURI_STUB_FILES "${NURI_STUB_FILES}" PARENT_SCOPE)
  endif()

  if(SKBUILD)
    install(TARGETS "${target_name}" LIBRARY DESTINATION "./${subdir}")
  endif()

  add_dependencies(NuriPython "${target_name}")
endfunction()
