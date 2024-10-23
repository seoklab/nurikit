#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(NuriKitUtils)

find_or_fetch_eigen()
find_or_fetch_spectra()

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

file(GLOB_RECURSE NURI_SRCS *.cpp)
list(FILTER NURI_SRCS EXCLUDE REGEX "main\.cpp$")

if(NOT NURI_SRCS)
  message(FATAL_ERROR "No source files found for nuri")
endif()

add_library(nuri_lib SHARED "${NURI_SRCS}")
set_target_properties(nuri_lib PROPERTIES OUTPUT_NAME nuri)
target_link_libraries(nuri_lib
  PUBLIC
  absl::strings
  absl::flat_hash_map
  absl::absl_check
  absl::absl_log
)
target_system_include_directories(nuri_lib Eigen3::Eigen Spectra::Spectra)
handle_boost_dependency(nuri_lib)

if(NURI_LIBRARY_FLAGS)
  target_compile_options(nuri_lib PRIVATE ${NURI_LIBRARY_FLAGS})
  target_link_options(nuri_lib PRIVATE ${NURI_LIBRARY_FLAGS})
endif()

add_executable(nuri_exe "main.cpp")
set_target_properties(nuri_exe
  PROPERTIES
  OUTPUT_NAME nuri
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
target_link_libraries(nuri_exe PRIVATE nuri_lib)

install(TARGETS nuri_lib LIBRARY)
install(TARGETS nuri_exe RUNTIME)
