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
target_system_include_directories(nuri_lib
  Eigen3::Eigen
  Spectra::Spectra
  absl::strings
  absl::flat_hash_map
  absl::absl_log
  absl::absl_check
)
handle_boost_dependency(nuri_lib)
clear_coverage_data(nuri_lib)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND NURI_LIBRARY_FLAGS
    -fno-math-errno
    -fno-signed-zeros
    -fno-trapping-math
    -fassociative-math
    -freciprocal-math
    -fno-rounding-math
    -ffp-contract=fast
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    list(APPEND NURI_LIBRARY_FLAGS -fno-signaling-nans -fexcess-precision=fast)
  endif()
endif()

if(NURI_LIBRARY_FLAGS)
  target_compile_options(nuri_lib PRIVATE ${NURI_LIBRARY_FLAGS})
  target_link_options(nuri_lib PRIVATE ${NURI_LIBRARY_FLAGS})
endif()

add_executable(nuri_exe "main.cpp")
set_target_properties(nuri_exe
  PROPERTIES
  OUTPUT_NAME nuri
  INSTALL_RPATH "${NURI_RPATH_PREFIX}/../${CMAKE_INSTALL_LIBDIR}")
target_link_libraries(nuri_exe PRIVATE nuri_lib absl::log_initialize)

install(TARGETS nuri_lib LIBRARY)
install(TARGETS nuri_exe RUNTIME)
