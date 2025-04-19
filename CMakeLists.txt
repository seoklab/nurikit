#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # Version for Ubuntu 20.04 LTS

# This snippet is copied from boostorg/boost/CMakeLists.txt
# Distributed under the Boost Software License, Version 1.0.
# See http://www.boost.org/LICENSE_1_0.txt
# The default build type must be set before project()
if(
  CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
  AND NOT CMAKE_BUILD_TYPE
  AND NOT CMAKE_CONFIGURATION_TYPES
)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo"
  )
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NuriKitUtils)

nuri_get_version()
project(NuriKit VERSION "${NURI_CORE_VERSION}" LANGUAGES CXX)

option(NURI_BUILD_LIB "Build NuriKit library" ON)
option(NURI_BUILD_PYTHON "Build Python bindings" ON)
option(NURI_BUILD_DOCS "Build documentation" OFF)
option(NURI_BUILD_PYTHON_DOCS "Build python documentation" OFF)

option(NURI_PREBUILT_ABSL "Download prebuilt abseil binary" ON)
option(NURI_INSTALL_RPATH "Install with RPATH when building full library" ON)

set(NURI_OPTIMIZATION_LEVEL "O3" CACHE STRING "Optimization level")
option(NURI_ENABLE_IPO "Do interprocedural optimization" ON)

option(NURI_TEST_COVERAGE "Enable coverage build" OFF)
option(NURI_ENABLE_SANITIZERS "Enable sanitizers for debug build" OFF)
option(NURI_BUILD_FUZZING "Enable fuzzing build" OFF)

option(NURI_BUILD_PYTHON_STUBS "Build python stub files" OFF)
set(NURI_FORCE_VERSION "" CACHE STRING "Set NuriKit version to this value")
mark_as_advanced(NURI_BUILD_PYTHON_STUBS NURI_FORCE_VERSION)

# cmake includes
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake"
)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/cmake")

include(CPM)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest) # For BUILD_TESTING
include(ProcessorCount)

# Global configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# System/compiler dependent flags
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    add_both_options(-stdlib=libstdc++)
  endif()

  add_compile_options(-Wno-gnu-zero-variadic-macro-arguments)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(CMAKE_SYSTEM_NAME MATCHES Darwin)
    add_both_options(-stdlib=libc++)
  endif()
else()
  message(FATAL_ERROR
    "Unsupported compiler '${CMAKE_CXX_COMPILER_ID}' detected! "
    "Please use clang or gcc as a compiler.")
endif()

# Optimization flags
set(nuri_is_release "$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>")

add_both_options(
  "$<$<CONFIG:Debug>:-O0>"
  "$<${nuri_is_release}:-${NURI_OPTIMIZATION_LEVEL}>"
)

if(NURI_ENABLE_IPO)
  # Next line is required for absl, etc.
  set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_link_options("$<${nuri_is_release}:-flto=auto>")
  endif()
endif()

# Option-dependent configurations
if(NURI_BUILD_LIB AND NURI_INSTALL_RPATH)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    set(NURI_RPATH_PREFIX "$ORIGIN")
  elseif(CMAKE_SYSTEM_NAME MATCHES Darwin)
    set(NURI_RPATH_PREFIX "@loader_path")
  else()
    message(FATAL_ERROR
      "Unsupported system '${CMAKE_SYSTEM_NAME}' detected! "
      "Only Linux and macOS are supported.")
  endif()
endif()

if(NURI_BUILD_FUZZING)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Fuzzing build is only supported with Clang")
  endif()

  message(NOTICE "Fuzzing build: enabling sanitizers and sanitizer coverage")
  set(NURI_ENABLE_SANITIZERS ON)
endif()

set_sanitizer_envs()

if(NURI_ENABLE_SANITIZERS)
  message(NOTICE "Enabling sanitizers as requested")

  add_both_options(
    -fsanitize=address
    -fsanitize=undefined
    -fno-sanitize-recover=all
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_both_options(
      -shared-libasan
      -fsanitize=float-divide-by-zero,implicit-conversion,local-bounds,nullability
      -fsanitize-recover=implicit-conversion # too many false positives
    )
  endif()
endif()

# Find/download dependencies
if(NURI_BUILD_LIB OR NURI_BUILD_PYTHON)
  find_or_fetch_abseil()

  find_or_add_package(
    NAME Eigen3
    MIN_VERSION 3.4
    CPM_VERSION 3.4.0
    OPTIONS
    "BUILD_TESTING OFF"
    "EIGEN_BUILD_DOC OFF"
    "EIGEN_BUILD_PKGCONFIG OFF"
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )

  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)
  find_or_add_package(
    NAME Boost
    MIN_VERSION 1.82
    CPM_VERSION 1.87.0
    OPTIONS
    "BUILD_TESTING OFF"
    "BOOST_ENABLE_CMAKE ON"
    "BOOST_SKIP_INSTALL_RULES OFF"
    "BOOST_ENABLE_COMPATIBILITY_TARGETS ON"
    "BOOST_INCLUDE_LIBRARIES \
spirit\\\;fusion\\\;optional\\\;container\\\;iterator\\\;range"
    URL "https://github.com/boostorg/boost/releases/download/boost-1.87.0/boost-1.87.0-cmake.tar.xz"
  )

  if(BUILD_TESTING AND NURI_BUILD_LIB)
    find_or_add_package(
      NAME GTest
      CPM_VERSION 1.16.0
      GIT_REPOSITORY https://github.com/google/googletest.git
    )
  endif()
endif()

# Coverage & fuzzing
# Put here so don't check coverage/fuzz external projects
if(NURI_TEST_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(NOTICE "GNU/Clang detected, enabling coverage")
  add_both_options(--coverage)
endif()

if(NURI_BUILD_FUZZING)
  add_both_options(
    -fsanitize=fuzzer
    -fprofile-instr-generate
    -fcoverage-mapping
  )
endif()

# Global project compile options / variables
set(NURI_BIBTEX_FILE "${CMAKE_CURRENT_LIST_DIR}/docs/refs.bib")

add_compile_options(
  -pedantic
  -Wall
  -Wextra
  -Wno-sign-compare
  -Wno-nullable-to-nonnull-conversion # abseil/abseil-cpp#1793
)

if(NURI_BUILD_LIB)
  message(STATUS "Building NuriKit Library")

  configure_file(
    "include/nuri/nurikit_config.h.in"
    "include/nuri/nurikit_config.h"
    @ONLY
  )
  set(
    NURI_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
  )
  include_directories(${NURI_INCLUDE_DIRECTORIES})

  if(NOT SKBUILD)
    install(
      FILES "${CMAKE_CURRENT_BINARY_DIR}/include/nuri/nurikit_config.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nuri"
    )

    install(
      DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include/nuri"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
      FILES_MATCHING PATTERN "*.h"
    )
  endif()

  add_subdirectory(src)

  if(NOT SKBUILD)
    configure_package_config_file(
      "cmake/${PROJECT_NAME}Config.cmake.in"
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
    write_basic_package_version_file(
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
      COMPATIBILITY ExactVersion # TODO: make this at least SameMinorVersion
    )
    install(
      FILES
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
      "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
    )
  endif()

  if(NURI_BUILD_DOCS)
    add_subdirectory(docs)
  endif()

  if(BUILD_TESTING)
    add_subdirectory(test)
  endif()

  if(NURI_BUILD_FUZZING)
    add_subdirectory(fuzz)
  endif()
endif()

if(NURI_BUILD_PYTHON OR NURI_BUILD_PYTHON_DOCS)
  if(NOT SKBUILD)
    configure_file(
      pyproject.toml.in
      "${CMAKE_CURRENT_LIST_DIR}/pyproject.toml"
      @ONLY
    )
  endif()

  add_subdirectory(python)
endif()
