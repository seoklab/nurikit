#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

macro(nuri_get_version)
  if(SKBUILD)
    return()
  endif()

  include(FetchContent)
  Fetchcontent_Declare(
    CMakeExtraUtils
    GIT_REPOSITORY https://github.com/LecrisUT/CMakeExtraUtils.git
    GIT_TAG v0.4.1
  )
  FetchContent_MakeAvailable(CMakeExtraUtils)

  include(DynamicVersion)
  dynamic_version(PROJECT_PREFIX NuriKit_)

  string(TIMESTAMP NURI_YEAR "%Y" UTC)
endmacro()

function(nuri_make_available_deponly target)
  include(FetchContent)

  FetchContent_GetProperties(${target})

  if(NOT ${target}_POPULATED)
    FetchContent_Populate(${target})
    add_subdirectory(
      ${${target}_SOURCE_DIR} ${${target}_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endfunction()

function(_target_system_include target dep)
  get_target_property(
    interface_include_dirs "${dep}" INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories("${target}" SYSTEM PUBLIC ${interface_include_dirs})
endfunction()

function(target_system_include_directories target)
  foreach(dep IN LISTS ARGN)
    _target_system_include("${target}" "${dep}")
  endforeach()
endfunction()

function(find_or_fetch_eigen)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(BUILD_TESTING OFF)
  set(EIGEN_BUILD_DOC OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)

  find_package(Eigen3 3.4 QUIET)

  if(Eigen3_FOUND)
    message(STATUS "Found Eigen3 ${Eigen3_VERSION}")
  else()
    include(FetchContent)
    message(NOTICE "Could not find compatible Eigen3. Fetching from gitlab.")

    FetchContent_Declare(
      eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.4.0)
    nuri_make_available_deponly(eigen)
  endif()
endfunction()

function(find_or_fetch_spectra)
  set(BUILD_TESTING OFF)

  find_package(Spectra 1.0 QUIET)

  if(Spectra_FOUND)
    message(STATUS "Found Spectra ${Spectra_VERSION}")
  else()
    include(FetchContent)
    message(NOTICE "Could not find compatible Spectra. Fetching from github.")

    FetchContent_Declare(
      spectra
      GIT_REPOSITORY https://github.com/yixuan/spectra.git
      GIT_TAG v1.0.1)
    nuri_make_available_deponly(spectra)
    add_library(Spectra::Spectra ALIAS Spectra)
  endif()
endfunction()

function(find_or_fetch_pybind11)
  set(BUILD_TESTING OFF)
  set(PYBIND11_FINDPYTHON ON)

  # FindPythonInterp/FindPythonLibs deprecated since cmake 3.12
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
  endif()

  find_package(pybind11 2.13 QUIET)

  if(pybind11_FOUND)
    message(STATUS "Found pybind11 ${pybind11_VERSION}")
  else()
    include(FetchContent)
    message(NOTICE "Could not find compatible pybind11. Fetching from github.")

    Fetchcontent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
    )
    nuri_make_available_deponly(pybind11)
  endif()
endfunction()

function(find_or_fetch_abseil)
  set(BUILD_TESTING OFF)
  set(BUILD_SHARED_LIBS OFF)
  set(ABSL_BUILD_TESTING OFF)
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_USE_SYSTEM_INCLUDES ON)

  if(NURI_ENABLE_SANITIZERS)
    message(
      NOTICE
      "abseil must be built with sanitizers enabled; ignoring system abseil"
    )
  elseif(NURI_PREBUILT_ABSL AND NOT absl_ROOT)
    include(FetchContent)
    message(NOTICE "Fetching prebuilt abseil binary.")

    if(CMAKE_SYSTEM_NAME MATCHES Linux)
      set(os_arch "manylinux2014_x86_64")
    elseif(CMAKE_SYSTEM_NAME MATCHES Darwin)
      set(os_arch "macosx_universal2")
    endif()

    Fetchcontent_Declare(
      absl
      URL "https://github.com/jnooree/abseil-cpp/releases/latest/download/libabsl-static-${os_arch}.tar.gz"
    )
    Fetchcontent_MakeAvailable(absl)

    set(absl_ROOT "${absl_SOURCE_DIR}")
    find_package(absl)

    if(absl_FOUND AND absl_VERSION VERSION_GREATER_EQUAL 20240116)
      message(STATUS "Found abseil ${absl_VERSION}")

      if(CMAKE_SYSTEM_NAME MATCHES Linux)
        string(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
      endif()

      return()
    endif()
  else()
    find_package(absl QUIET)

    # 20240116 required for VLOG()
    if(absl_FOUND AND absl_VERSION VERSION_GREATER_EQUAL 20240116)
      message(STATUS "Found abseil ${absl_VERSION}")
      return()
    endif()
  endif()

  include(FetchContent)
  message(NOTICE "Could not find compatible abseil. Fetching from github.")

  Fetchcontent_Declare(
    absl
    URL https://github.com/abseil/abseil-cpp/releases/download/20240722.0/abseil-cpp-20240722.0.tar.gz
  )
  nuri_make_available_deponly(absl)
endfunction()

function(handle_boost_dependency target)
  set(BUILD_TESTING OFF)

  # FindBoost deprecated since cmake 3.30
  if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
  endif()

  find_package(Boost 1.82 QUIET)

  if(Boost_FOUND)
    message(STATUS "Found Boost ${Boost_VERSION}")
    target_include_directories("${target}" SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
    return()
  endif()

  message(NOTICE "Could not find compatible Boost. Fetching from boostorg.")
  include(FetchContent)
  FetchContent_Declare(
    boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.82.0/boost-1.82.0.tar.xz
  )

  set(Boost_ENABLE_CMAKE ON)
  set(Boost_USE_STATIC_LIBS ON)
  nuri_make_available_deponly(boost)

  target_system_include_directories(
    "${target}"
    Boost::spirit Boost::fusion Boost::mpl Boost::optional
    Boost::iterator Boost::config
  )
  target_link_libraries(
    "${target}"
    PUBLIC Boost::iterator Boost::config
    PRIVATE Boost::spirit Boost::fusion Boost::mpl Boost::optional
  )
endfunction()

function(set_sanitizer_envs)
  if(NOT NURI_ENABLE_SANITIZERS)
    set(SANITIZER_ENVS "LD_PRELOAD=$ENV{LD_PRELOAD}" PARENT_SCOPE)
    return()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(filename libasan.so)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(filename libclang_rt.asan-x86_64.so)
  endif()

  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${filename}"
    OUTPUT_VARIABLE asan_lib_path
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=libubsan.so"
    OUTPUT_VARIABLE ubsan_lib_path
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(SANITIZER_ENVS
    "LD_PRELOAD=${asan_lib_path} ${ubsan_lib_path} $ENV{LD_PRELOAD}"
    PARENT_SCOPE)
endfunction()

function(clear_coverage_data target)
  if(NOT NURI_TEST_COVERAGE)
    return()
  endif()

  add_custom_command(
    TARGET "${target}"
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
    "-DNURI_COVERAGE_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    -P "${PROJECT_SOURCE_DIR}/cmake/NuriKitClearCoverage.cmake"
    VERBATIM
  )
endfunction()
