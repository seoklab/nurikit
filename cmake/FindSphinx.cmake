#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

find_program(SPHINX_EXECUTABLE
  NAMES sphinx-build)
mark_as_advanced(SPHINX_EXECUTABLE)

option(NURI_PYDOC_DEPEND_PYTHON
  "Auto-rebuild python module before building docs" OFF)
mark_as_advanced(NURI_PYDOC_DEPEND_PYTHON)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)

function(add_sphinx_docs target)
  add_custom_target("${target}"
    COMMAND ${CMAKE_COMMAND} -E env ${SANITIZER_ENVS}
    ${SPHINX_EXECUTABLE}
    -E
    -b html
    -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMENT "Building Sphinx documentation for ${target}"
    VERBATIM)

  if(NURI_BUILD_PYTHON_DOCS)
    set_target_properties("${target}" PROPERTIES EXCLUDE_FROM_ALL OFF)
  endif()

  if(TARGET nuri_docs)
    add_dependencies("${target}" nuri_docs)
  endif()

  if(NURI_PYDOC_DEPEND_PYTHON)
    message("${target} will rebuild python modules")
    add_dependencies("${target}" nuri_python)
  endif()

  add_custom_target("${target}_doctest"
    COMMAND ${CMAKE_COMMAND} -E env ${SANITIZER_ENVS}
    ${SPHINX_EXECUTABLE}
    -E
    -b doctest
    -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/doctest
    COMMENT "Running doctest for ${target}"
    VERBATIM)
endfunction()
