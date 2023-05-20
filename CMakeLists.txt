#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # Version for Ubuntu 20.04 LTS

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(NurikitUtils)

nurikit_get_version()
project(nurikit VERSION "${NURIKIT_VERSION}" LANGUAGES CXX)

# Compile options
set(NURIKIT_OPTIMIZATION_LEVEL "O3" CACHE STRING "Optimization level")
option(NURIKIT_BUILD_PYTHON "Build Python bindings" ON)
option(NURIKIT_ENABLE_IPO "Do interprocedural optimization" ON)
option(NURIKIT_ENABLE_AVX2 "Use -mavx2 flag for optimization" ON)
option(NURIKIT_ENABLE_ARCH_NATIVE "Use -march=native flag for optimization" OFF)

option(NURIKIT_TEST_COVERAGE "Enable coverage build" OFF)
option(NURIKIT_BUILD_DOCS "Build documentation" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -stdlib=libstdc++")
elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  message(FATAL_ERROR
    "Unsupported compiler '${CMAKE_CXX_COMPILER_ID}' detected! "
    "Please use clang or gcc as a compiler.")
endif()

if(NURIKIT_BUILD_PYTHON)
  set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
  message(STATUS "Python bindings will be built, using Ninja generator")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NURIKIT_GLOBAL_OPTFLAGS " -${NURIKIT_OPTIMIZATION_LEVEL}")

if(NURIKIT_ENABLE_ARCH_NATIVE)
  string(APPEND NURIKIT_GLOBAL_OPTFLAGS " -march=native")
elseif(NURIKIT_ENABLE_AVX2)
  string(APPEND NURIKIT_GLOBAL_OPTFLAGS " -mavx2")
endif()

string(APPEND CMAKE_CXX_FLAGS_RELEASE "${NURIKIT_GLOBAL_OPTFLAGS}")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO "${NURIKIT_GLOBAL_OPTFLAGS}")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Bulid type")

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-O0)
  add_compile_definitions("DEBUG")
elseif(NURIKIT_ENABLE_IPO)
  # Next line is required for absl, etc.
  set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
  add_link_options(-s)
endif()

add_subdirectory(third-party EXCLUDE_FROM_ALL)

add_compile_options(-pedantic -Wall -Wextra -Wno-sign-compare)
set(NURIKIT_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(CMAKE_BUILD_TYPE STREQUAL "Debug"
  AND NURIKIT_TEST_COVERAGE
  AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(STATUS "GNU/Clang detected, enabling coverage")
  string(APPEND CMAKE_CXX_FLAGS " --coverage")
endif()

add_subdirectory(include)
include_directories(${NURIKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src)
add_subdirectory(docs EXCLUDE_FROM_ALL)

if(SKBUILD)
  add_subdirectory(python)
elseif(NURIKIT_BUILD_PYTHON)
  set(NURIKIT_VERBOSE_CMAKE "false")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(NURIKIT_VERBOSE_CMAKE "true")
  endif()

  configure_file(
    pyproject.toml.in "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" @ONLY)

  add_subdirectory(python EXCLUDE_FROM_ALL)
endif()

include(CTest)
add_subdirectory(test EXCLUDE_FROM_ALL)
