#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

find_package(Doxygen 1.8)

if(DOXYGEN_FOUND)
  message(STATUS "Doxygen found")
else()
  if(NURI_BUILD_DOCS)
    message(WARNING "Doxygen not found; cannot build docs")
  else()
    message(STATUS "Doxygen not found; not building docs")
  endif()
  return()
endif()

set(nuri_public_include_dirs "${NURI_INCLUDE_DIRECTORIES}")
get_target_property(nuri_lib_include_dirs nuri_lib INTERFACE_INCLUDE_DIRECTORIES)
list(APPEND nuri_public_include_dirs ${nuri_lib_include_dirs})
list(JOIN nuri_public_include_dirs " " NURI_INCLUDE_PATHS)

set(DOXYGEN_IN ${CMAKE_CURRENT_LIST_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

file(GLOB_RECURSE nuri_public_headers "${PROJECT_SOURCE_DIR}/include/nuri/*.h")

set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(DOXYGEN_OUTPUT_DIR "${DOXYGEN_OUTPUT_DIR}" PARENT_SCOPE)
add_custom_command(
  OUTPUT "${DOXYGEN_OUTPUT_DIR}/html/cpp/index.html"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIR}/html"
  COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYGEN_OUT}"
  MAIN_DEPENDENCY "${DOXYGEN_OUT}"
  DEPENDS ${nuri_public_headers} ${NURI_BIBTEX_FILE}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM
)
add_custom_target(nuri_docs DEPENDS "${DOXYGEN_OUTPUT_DIR}/html/cpp/index.html")

if(NURI_BUILD_DOCS)
  set_target_properties(nuri_docs PROPERTIES EXCLUDE_FROM_ALL OFF)
endif()
