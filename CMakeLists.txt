cmake_minimum_required(VERSION 3.16 FATAL_ERROR)  # Version for Ubuntu 20.04 LTS
project(c-like-template LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Bulid type")

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compile options
set(OPTIMIZATION_LEVEL "O3" CACHE STRING "Optimization level")
option(ENABLE_IPO "Do interprocedural optimization" ON)
option(ENABLE_AVX2 "Use -mavx2 flag for optimization" ON)
option(ENABLE_ARCH_NATIVE "Use -march=native flag for optimization" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -stdlib=libstdc++")
elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  message(FATAL_ERROR
    "Unsupported compiler '${CMAKE_CXX_COMPILER_ID}' detected! "
    "Please use clang or gcc as a compiler.")
endif()

add_compile_options(-pedantic -Wall -Wextra -Wno-sign-compare)

# Only do these if this is the main project, and not if it is included through
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_EXTENSIONS ON)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

  set(GLOBAL_OPTFLAGS " -${OPTIMIZATION_LEVEL} -fno-rtti -fno-exceptions")
  if(ENABLE_ARCH_NATIVE)
    string(APPEND GLOBAL_OPTFLAGS " -march=native")
  elseif(ENABLE_AVX2)
    string(APPEND GLOBAL_OPTFLAGS " -mavx2")
  endif()
  string(APPEND CMAKE_CXX_FLAGS_RELEASE "${GLOBAL_OPTFLAGS}")
  string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO "${GLOBAL_OPTFLAGS}")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0)
    add_compile_definitions("DEBUG")
  elseif(ENABLE_IPO)
    # Next line is required for absl, etc.
    # set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endif()

# Uncomment next line for third-party libraries
# add_subdirectory(third-party EXCLUDE_FROM_ALL)
add_subdirectory(src)
