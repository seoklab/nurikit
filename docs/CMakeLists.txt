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
    message(NOTICE "Doxygen not found; not building docs")
  endif()
  return()
endif()

set(
  NURI_INCLUDE_PATHS
  "$<JOIN:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:NuriLib,INCLUDE_DIRECTORIES>>, >"
)

set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(DOXYGEN_OUTPUT_DIR "${DOXYGEN_OUTPUT_DIR}" PARENT_SCOPE)
configure_file(Doxyfile.in Doxyfile.in2 @ONLY)
file(GENERATE OUTPUT Doxyfile INPUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.in2")

file(
  GLOB_RECURSE nuri_public_headers
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/nuri/*.h"
)
file(
  GLOB nuri_doc_pages
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/pages/*.md"
)
add_custom_command(
  OUTPUT "${DOXYGEN_OUTPUT_DIR}/html/cpp/index.html"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIR}/html"
  COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
  MAIN_DEPENDENCY "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
  DEPENDS
    ${nuri_public_headers}
    ${nuri_doc_pages}
    ${NURI_BIBTEX_FILE}
    "${CMAKE_CURRENT_LIST_DIR}/formulas.tex"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM
)
add_custom_target(
  NuriDocs
  ALL
  DEPENDS "${DOXYGEN_OUTPUT_DIR}/html/cpp/index.html"
)
