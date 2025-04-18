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

set(nuri_public_include_dirs "${NURI_INCLUDE_DIRECTORIES}")

get_target_property(nuri_lib_deps NuriLib INTERFACE_LINK_LIBRARIES)

foreach(nuri_dep IN LISTS nuri_lib_deps)
  get_target_property(
    nuri_dep_include_dirs
    "${nuri_dep}"
    INTERFACE_INCLUDE_DIRECTORIES
  )

  if(NOT nuri_dep_include_dirs)
    list(APPEND nuri_public_include_dirs ${nuri_dep_include_dirs})
  endif()
endforeach()

list(REMOVE_DUPLICATES nuri_public_include_dirs)
list(JOIN nuri_public_include_dirs " " NURI_INCLUDE_PATHS)

set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(DOXYGEN_OUTPUT_DIR "${DOXYGEN_OUTPUT_DIR}" PARENT_SCOPE)
configure_file(Doxyfile.in Doxyfile @ONLY)

file(GLOB_RECURSE nuri_public_headers "${PROJECT_SOURCE_DIR}/include/nuri/*.h")
add_custom_command(
  OUTPUT "${DOXYGEN_OUTPUT_DIR}/html/cpp/index.html"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIR}/html"
  COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
  MAIN_DEPENDENCY "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
  DEPENDS ${nuri_public_headers} ${NURI_BIBTEX_FILE}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM
)
add_custom_target(
  NuriDocs
  ALL
  DEPENDS "${DOXYGEN_OUTPUT_DIR}/html/cpp/index.html"
)
