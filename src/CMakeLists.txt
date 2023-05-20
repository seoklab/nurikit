#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

add_executable(nuri_exe "main.cpp")
set_target_properties(nuri_exe
  PROPERTIES
  OUTPUT_NAME nuri
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS nuri_exe RUNTIME)

file(GLOB_RECURSE NURIKIT_SRCS *.cpp)
list(FILTER NURIKIT_SRCS EXCLUDE REGEX "main\.cpp$")

if(NURIKIT_SRCS)
  add_library(nuri_lib SHARED "${NURIKIT_SRCS}")
  target_link_libraries(nuri_lib
    PUBLIC
    absl::strings absl::flat_hash_map absl::absl_check)
  set_target_properties(nuri_lib PROPERTIES OUTPUT_NAME nuri)
  install(TARGETS nuri_lib LIBRARY)

  target_link_libraries(nuri_exe PRIVATE nuri_lib)
endif()
