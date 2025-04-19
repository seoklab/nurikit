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
  message(NOTICE "Sphinx not found; cannot run doctest")
  return()
endif()

configure_file(conf.py.in "${CMAKE_CURRENT_LIST_DIR}/conf.py" @ONLY)

if(NURI_BUILD_PYTHON_DOCS)
  add_custom_target(NuriPythonDocs
    ALL
    COMMAND ${CMAKE_COMMAND} -E env ${SANITIZER_ENVS}
    ${SPHINX_EXECUTABLE}
    -E
    -b html
    -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMENT "Building Sphinx documentation for NuriPython"
    VERBATIM
  )

  if(TARGET NuriPython)
    add_dependencies(NuriPythonDocs NuriPython)
  endif()

  if(TARGET NuriDocs)
    add_dependencies(NuriPythonDocs NuriDocs)
  endif()
endif()

if(BUILD_TESTING)
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
endif()
