#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

find_program(SPHINX_EXECUTABLE NAMES sphinx-build)
mark_as_advanced(SPHINX_EXECUTABLE)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)

if(Sphinx_FOUND)
  message(STATUS "Sphinx found")
elseif(NURI_BUILD_PYTHON_DOCS)
  message(FATAL_ERROR "Sphinx not found; cannot build docs")
else()
  message(STATUS "Sphinx not found; not building docs")
  return()
endif()

configure_file(conf.py.in "${CMAKE_CURRENT_LIST_DIR}/conf.py" @ONLY)

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/html/index.html"
  COMMAND ${CMAKE_COMMAND} -E env ${SANITIZER_ENVS}
  ${SPHINX_EXECUTABLE}
  -E
  -b html
  -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/html
  MAIN_DEPENDENCY "${CMAKE_CURRENT_LIST_DIR}/conf.py"
  DEPENDS NuriPython
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Building Sphinx documentation for NuriPython"
  VERBATIM
)
add_custom_target(
  NuriPythonDocs
  ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/html/index.html"
)

if(TARGET NuriDocs)
  add_dependencies(NuriPythonDocs NuriDocs)
endif()

add_custom_target(NuriPythonDoctest
  COMMAND ${CMAKE_COMMAND} -E env ${SANITIZER_ENVS}
  ${SPHINX_EXECUTABLE}
  -E
  -b doctest
  -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/doctest
  COMMENT "Running doctest on NuriPythonDocs"
  VERBATIM
)
