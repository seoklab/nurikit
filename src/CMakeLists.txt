#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)
include(NuriKitUtils)

find_or_add_package(
  NAME Spectra
  CPM_VERSION 1.0
  OPTIONS "BUILD_TESTING OFF"
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG v1.0.1
)

if(NOT TARGET Spectra::Spectra)
  add_library(Spectra::Spectra ALIAS Spectra)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
)

file(GLOB_RECURSE NURI_SRCS *.cpp)
list(FILTER NURI_SRCS EXCLUDE REGEX "main\.cpp$")

add_library(NuriLib SHARED "${NURI_SRCS}")
add_library("${PROJECT_NAME}::NuriLib" ALIAS NuriLib)
clear_coverage_data(NuriLib)

target_compile_features(NuriLib PUBLIC cxx_std_17)
target_compile_definitions(NuriLib
  PUBLIC
  $<$<CONFIG:Debug>:NURI_DEBUG;EIGEN_RUNTIME_NO_MALLOC>
)

if(ABSL_USES_OLD_ABI)
  target_compile_definitions(NuriLib PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
endif()

target_link_libraries(NuriLib
  PUBLIC
  Eigen3::Eigen
  Boost::headers
  absl::strings
  absl::flat_hash_map
  absl::absl_check
  absl::absl_log
  PRIVATE
  Spectra::Spectra
)
set_target_properties(NuriLib PROPERTIES OUTPUT_NAME nuri)

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
  target_compile_options(NuriLib PRIVATE ${NURI_LIBRARY_FLAGS})
  target_link_options(NuriLib PRIVATE ${NURI_LIBRARY_FLAGS})
endif()

add_executable(NuriExe "main.cpp")
add_executable("${PROJECT_NAME}::NuriExe" ALIAS NuriExe)

target_link_libraries(NuriExe PRIVATE NuriLib absl::log_initialize)
set_target_properties(NuriExe PROPERTIES OUTPUT_NAME nuri)

if(NURI_INSTALL_RPATH)
  set_target_properties(
    NuriExe
    PROPERTIES
    INSTALL_RPATH "${NURI_RPATH_PREFIX}/../${CMAKE_INSTALL_LIBDIR}"
  )
endif()

install(
  TARGETS NuriLib NuriExe
  EXPORT "${PROJECT_NAME}Targets"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

if(NOT SKBUILD)
  install(
    EXPORT "${PROJECT_NAME}Targets"
    NAMESPACE "${PROJECT_NAME}::"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  )
endif()
