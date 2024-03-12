#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # Version for Ubuntu 20.04 LTS

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NurikitUtils)

nuri_get_version()
project(nurikit VERSION "${NURI_CORE_VERSION}" LANGUAGES CXX)

# Compile options
set(NURI_OPTIMIZATION_LEVEL "O3" CACHE STRING "Optimization level")
option(NURI_BUILD_PYTHON "Build Python bindings" ON)
option(NURI_ENABLE_IPO "Do interprocedural optimization" ON)
option(NURI_ENABLE_AVX2 "Use -mavx2 flag for optimization" ON)
option(NURI_ENABLE_ARCH_NATIVE "Use -march=native flag for optimization" OFF)

option(NURI_ENABLE_SANITIZERS "Enable sanitizers for debug build" OFF)

option(NURI_TEST_COVERAGE "Enable coverage build" OFF)
option(NURI_BUILD_DOCS "Build documentation" OFF)
option(NURI_BUILD_PYTHON_DOCS "Build python documentation" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS
    " -stdlib=libstdc++ -Wno-gnu-zero-variadic-macro-arguments"
  )
elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  message(FATAL_ERROR
    "Unsupported compiler '${CMAKE_CXX_COMPILER_ID}' detected! "
    "Please use clang or gcc as a compiler.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NURI_GLOBAL_OPTFLAGS " -${NURI_OPTIMIZATION_LEVEL}")

if(NURI_ENABLE_ARCH_NATIVE)
  string(APPEND NURI_GLOBAL_OPTFLAGS " -march=native")
elseif(NURI_ENABLE_AVX2)
  string(APPEND NURI_GLOBAL_OPTFLAGS " -mavx2")
endif()

string(APPEND CMAKE_CXX_FLAGS_RELEASE "${NURI_GLOBAL_OPTFLAGS}")
string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO "${NURI_GLOBAL_OPTFLAGS}")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Bulid type")

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-O0)
  add_compile_definitions("DEBUG")
elseif(NURI_ENABLE_IPO)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(NURI_LIBRARY_FLAGS "-flto=auto")
  endif()

  # Next line is required for absl, etc.
  set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

set_sanitizer_envs()

if(NURI_ENABLE_SANITIZERS)
  message("Enabling sanitizers as requested")
  add_compile_options(
    -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all)
  add_link_options(
    -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(sanitizer_extra_args
      -shared-libasan
      -fsanitize=float-divide-by-zero,implicit-conversion,local-bounds,nullability
      -fsanitize-recover=implicit-conversion # too many false positives
    )
    add_compile_options(${sanitizer_extra_args})
    add_link_options(${sanitizer_extra_args})
  endif()
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release|MinSizeRel")
  add_link_options(-s)
endif()

add_subdirectory(third-party EXCLUDE_FROM_ALL)

add_compile_options(-pedantic -Wall -Wextra -Wno-sign-compare)
set(NURI_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")

if(CMAKE_BUILD_TYPE STREQUAL "Debug"
  AND NURI_TEST_COVERAGE
  AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(STATUS "GNU/Clang detected, enabling coverage")
  string(APPEND CMAKE_CXX_FLAGS " --coverage")
endif()

add_subdirectory(include)
include_directories(${NURI_INCLUDE_DIRECTORIES})

add_subdirectory(src)
add_subdirectory(docs EXCLUDE_FROM_ALL)

if(SKBUILD)
  add_subdirectory(python)
elseif(NURI_BUILD_PYTHON OR NURI_BUILD_PYTHON_DOCS)
  configure_file(
    pyproject.toml.in "${CMAKE_CURRENT_LIST_DIR}/pyproject.toml" @ONLY)

  add_subdirectory(python EXCLUDE_FROM_ALL)
endif()

include(CTest)
add_subdirectory(test EXCLUDE_FROM_ALL)
