#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
)

add_library(NuriPyLib SHARED
  stream.cpp
)
add_library("${PROJECT_NAME}::NuriPyLib" ALIAS NuriPyLib)
clear_coverage_data(NuriPyLib)

target_include_directories(NuriPyLib
  PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../include"
)
target_link_libraries(NuriPyLib
  PUBLIC "${PROJECT_NAME}::NuriLib" pybind11::module
)
set_target_properties(
  NuriPyLib
  PROPERTIES
  OUTPUT_NAME nuripy
  SOVERSION 0
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

if(NURI_INSTALL_RPATH)
  set_property(
    TARGET NuriPyLib
    APPEND PROPERTY INSTALL_RPATH
    "${NURI_RPATH_PREFIX}" "${NURI_RPATH_PREFIX}/../lib64"
  )
endif()

install(
  TARGETS NuriPyLib
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
