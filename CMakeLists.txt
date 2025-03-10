#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # Version for Ubuntu 20.04 LTS

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(NuriKitUtils)

nuri_get_version()
project(NuriKit VERSION "${NURI_CORE_VERSION}" LANGUAGES CXX)

# Compile options
set(NURI_OPTIMIZATION_LEVEL "O3" CACHE STRING "Optimization level")
option(NURI_BUILD_PYTHON "Build Python bindings" ON)
option(NURI_ENABLE_IPO "Do interprocedural optimization" ON)
option(NURI_ENABLE_AVX2 "Use -mavx2 flag for optimization" OFF)
option(NURI_ENABLE_ARCH_NATIVE "Use -march=native flag for optimization" OFF)

option(NURI_ENABLE_SANITIZERS "Enable sanitizers for debug build" OFF)
option(NURI_BUILD_FUZZING "Enable fuzzing build" OFF)
option(NURI_PREBUILT_ABSL "Download prebuilt abseil binary" ON)

option(NURI_TEST_COVERAGE "Enable coverage build" OFF)
option(NURI_BUILD_DOCS "Build documentation" OFF)
option(NURI_BUILD_PYTHON_DOCS "Build python documentation" OFF)

option(NURI_POSTINSTALL_TEST "Run tests after installation" OFF)
mark_as_advanced(NURI_POSTINSTALL_TEST)

if(CMAKE_SYSTEM_NAME MATCHES Linux)
  set(NURI_RPATH_PREFIX "$ORIGIN")
elseif(CMAKE_SYSTEM_NAME MATCHES Darwin)
  set(NURI_RPATH_PREFIX "@loader_path")
else()
  message(FATAL_ERROR
    "Unsupported system '${CMAKE_SYSTEM_NAME}' detected! "
    "Only Linux and macOS are supported.")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  if(CMAKE_SYSTEM_NAME MATCHES Linux)
    string(APPEND CMAKE_CXX_FLAGS " -stdlib=libstdc++")
  endif()

  string(APPEND CMAKE_CXX_FLAGS " -Wno-gnu-zero-variadic-macro-arguments")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(CMAKE_SYSTEM_NAME MATCHES Darwin)
    string(APPEND CMAKE_CXX_FLAGS " -stdlib=libc++")
  endif()
else()
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
  add_compile_definitions("DEBUG" "EIGEN_RUNTIME_NO_MALLOC")
elseif(NURI_ENABLE_IPO)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(NURI_LIBRARY_FLAGS "-flto=auto")
  endif()

  # Next line is required for absl, etc.
  set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if(NURI_BUILD_FUZZING)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Fuzzing build is only supported with Clang")
  endif()

  message("Fuzzing build: enabling sanitizers and sanitizer coverage")
  set(NURI_ENABLE_SANITIZERS ON)
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

find_or_fetch_abseil()
add_subdirectory(third-party EXCLUDE_FROM_ALL)

if(NURI_BUILD_FUZZING)
  add_compile_options(-fsanitize=fuzzer)
  string(APPEND CMAKE_CXX_FLAGS " -fprofile-instr-generate -fcoverage-mapping")
endif()

add_compile_options(
  -pedantic
  -Wall
  -Wextra
  -Wno-sign-compare
  -Wno-nullable-to-nonnull-conversion # abseil/abseil-cpp#1793
)
set(NURI_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")

if(CMAKE_BUILD_TYPE STREQUAL "Debug"
  AND NURI_TEST_COVERAGE
  AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(STATUS "GNU/Clang detected, enabling coverage")
  string(APPEND CMAKE_CXX_FLAGS " --coverage")
endif()

add_subdirectory(include)
include_directories(${NURI_INCLUDE_DIRECTORIES})
string(APPEND CMAKE_CXX_FLAGS
  " -include"
  " ${CMAKE_CURRENT_BINARY_DIR}/include/generated/include/nuri/nurikit_config.h"
)

set(NURI_BIBTEX_FILE "${CMAKE_CURRENT_LIST_DIR}/docs/refs.bib")

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
add_subdirectory(fuzz EXCLUDE_FROM_ALL)

if(BUILD_TESTING AND NURI_POSTINSTALL_TEST)
  message(NOTICE "Running tests after post-installation step")
  include(ProcessorCount)

  ProcessorCount(NPROC)
  install(CODE
    "execute_process(
      COMMAND \"\${CMAKE_CTEST_COMMAND}\" -T test -j ${NPROC} --output-on-failure
      WORKING_DIRECTORY \"$<1:${CMAKE_CURRENT_BINARY_DIR}>\"
      RESULT_VARIABLE test_result
    )
    if(test_result)
      message(FATAL_ERROR \"Tests failed with error code \${test_result}\")
    endif()"
  )
endif()
