#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # Version for Ubuntu 20.04 LTS
project(nurikit VERSION 0.1.0 LANGUAGES CXX)

# Compile options
set(NURIKIT_OPTIMIZATION_LEVEL "O3" CACHE STRING "Optimization level")
option(NURIKIT_ENABLE_IPO "Do interprocedural optimization" ON)
option(NURIKIT_ENABLE_AVX2 "Use -mavx2 flag for optimization" ON)
option(NURIKIT_ENABLE_ARCH_NATIVE "Use -march=native flag for optimization" OFF)

option(NURIKIT_TEST_COVERAGE "Enable coverage build" OFF)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Bulid type")
if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -stdlib=libstdc++")
elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  message(FATAL_ERROR
    "Unsupported compiler '${CMAKE_CXX_COMPILER_ID}' detected! "
    "Please use clang or gcc as a compiler.")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Only do these if this is the main project, and not if it is included through
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
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

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0)
    add_compile_definitions("DEBUG")
  elseif(NURIKIT_ENABLE_IPO)
    # Next line is required for absl, etc.
    set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endif()

add_subdirectory(third-party EXCLUDE_FROM_ALL)

configure_file(cmake/nurikit_config.h.in
  generated/include/nuri/nurikit_config.h @ONLY)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/generated/include/nuri/nurikit_config.h
  DESTINATION include/nuri
  COMPONENT Development)

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}/generated/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_compile_options(-pedantic -Wall -Wextra -Wno-sign-compare)

add_subdirectory(include)
add_subdirectory(src)

include(CTest)
add_subdirectory(test EXCLUDE_FROM_ALL)
