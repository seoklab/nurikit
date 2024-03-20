#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
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
get_target_property(nuri_lib_include_dirs
  nuri_lib INTERFACE_INCLUDE_DIRECTORIES)
list(APPEND nuri_public_include_dirs
  ${nuri_lib_include_dirs}
  "${PROJECT_SOURCE_DIR}/third-party/abseil-cpp")
list(JOIN nuri_public_include_dirs " " NURI_INCLUDE_PATHS)

set(DOXYGEN_IN ${CMAKE_CURRENT_LIST_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
set(DOXYGEN_OUTPUT_DIR "${PROJECT_BINARY_DIR}/generated/docs")

configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
add_custom_target(_nuri_docs_setup
  COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIR}"
  VERBATIM)
add_custom_target(nuri_docs
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
  DEPENDS _nuri_docs_setup
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM)

if(NURI_BUILD_DOCS)
  set_target_properties(nuri_docs PROPERTIES EXCLUDE_FROM_ALL OFF)
endif()
