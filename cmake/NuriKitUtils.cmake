#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

macro(_nuri_get_git_version_impl)
  find_package(Git)

  if(NOT Git_FOUND)
    return()
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE nuri_revision
    ERROR_QUIET)
  string(STRIP "${nuri_revision}" nuri_revision)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match --abbrev=0
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE NURI_REF
    ERROR_QUIET)

  if(git_result EQUAL 0)
    string(STRIP "${NURI_REF}" NURI_REF)
    string(REGEX REPLACE "^v" "" NURI_FULL_VERSION "${NURI_REF}")
    message(STATUS "NuriKit version from git: ${NURI_FULL_VERSION}")
  else()
    execute_process(
      COMMAND ${GIT_EXECUTABLE} symbolic-ref -q --short HEAD
      RESULT_VARIABLE git_result
      OUTPUT_VARIABLE NURI_REF
      ERROR_QUIET)

    if(NOT git_result EQUAL 0)
      set(NURI_REF "${nuri_revision}")
    endif()

    string(STRIP "${NURI_REF}" NURI_REF)
  endif()
endmacro()

function(nuri_get_version)
  if(SKBUILD)
    # Version correctly set via scikit-build-core; skip git versioning.
    set(NURI_FULL_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
    message(
      STATUS "NuriKit version from scikit-build-core: ${NURI_FULL_VERSION}"
    )
  else()
    _nuri_get_git_version_impl()
  endif()

  if(NURI_FORCE_VERSION)
    message(STATUS "Using explicit NuriKit version: ${NURI_FORCE_VERSION}")
    set(NURI_FULL_VERSION "${NURI_FORCE_VERSION}")
  endif()

  if(NURI_REF)
    message(STATUS "NuriKit ref from git: ${NURI_REF}")
  else()
    message(NOTICE "NuriKit ref not found! Using unknown.")
    set(NURI_REF "unknown")
  endif()

  if(nuri_revision)
    message(STATUS "NuriKit revision from git: ${nuri_revision}")
  else()
    message(NOTICE "NuriKit revision not found! Using unknown.")
    set(nuri_revision "unknown")
  endif()

  if(NOT NURI_FULL_VERSION)
    set(NURI_FULL_VERSION "0.1.0.dev0+${nuri_revision}")
    message(NOTICE "NuriKit version not found! Using ${NURI_FULL_VERSION}")
  endif()

  string(REGEX REPLACE "\\+.+$" "" NURI_VERSION "${NURI_FULL_VERSION}")
  string(
    REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+"
    NURI_CORE_VERSION "${NURI_VERSION}"
  )

  string(TIMESTAMP NURI_YEAR "%Y" UTC)
  set(NURI_YEAR "${NURI_YEAR}" PARENT_SCOPE)
  set(NURI_VERSION "${NURI_VERSION}" PARENT_SCOPE)
  set(NURI_CORE_VERSION "${NURI_CORE_VERSION}" PARENT_SCOPE)
  set(NURI_FULL_VERSION "${NURI_FULL_VERSION}" PARENT_SCOPE)
  set(NURI_REF "${NURI_REF}" PARENT_SCOPE)
endfunction()

function(find_or_add_package)
  cmake_parse_arguments(_pkg "" "NAME;MIN_VERSION;CPM_VERSION" "" ${ARGN})

  if(NOT _pkg_MIN_VERSION)
    set(_pkg_MIN_VERSION "${_pkg_CPM_VERSION}")
  endif()

  find_package("${_pkg_NAME}" "${_pkg_MIN_VERSION}" QUIET)

  if(${_pkg_NAME}_FOUND)
    message(STATUS "Found ${_pkg_NAME} ${${_pkg_NAME}_VERSION}")

    set("${_pkg_NAME}_FOUND" "${${_pkg_NAME}_FOUND}" PARENT_SCOPE)
    set("${_pkg_NAME}_VERSION" "${${_pkg_NAME}_VERSION}" PARENT_SCOPE)
    return()
  endif()

  message(STATUS "Could not find ${_pkg_NAME}. Adding with CPM.")

  include(CPM)
  set(CPM_USE_LOCAL_PACKAGES OFF)
  CPMAddPackage(
    NAME "${_pkg_NAME}"
    VERSION "${_pkg_CPM_VERSION}"
    EXCLUDE_FROM_ALL ON
    SYSTEM ON
    ${_pkg_UNPARSED_ARGUMENTS}
  )

  # emulate find_package behavior
  set("${_pkg_NAME}_FOUND" ON PARENT_SCOPE)
  set("${_pkg_NAME}_VERSION" "${_pkg_CPM_VERSION}" PARENT_SCOPE)
endfunction()

function(find_or_fetch_abseil)
  if(NURI_ENABLE_SANITIZERS)
    message(
      NOTICE
      "abseil must be built with sanitizers enabled; ignoring system abseil"
    )

    if(CPM_USE_LOCAL_PACKAGES)
      message(
        WARNING
        "CPM_USE_LOCAL_PACKAGES is not supported with sanitizers, ignoring"
      )
    endif()
  else()
    if(NURI_PREBUILT_ABSL)
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

      set(absl_findpackage_args PATHS "${absl_SOURCE_DIR}" NO_DEFAULT_PATH)
    else()
      set(absl_findpackage_args QUIET)
    endif()

    find_package(absl ${absl_findpackage_args})

    if(absl_FOUND AND absl_VERSION VERSION_GREATER_EQUAL 20240116)
      message(STATUS "Found absl ${absl_VERSION}")
      set(absl_FOUND "${absl_FOUND}" PARENT_SCOPE)
      set(absl_VERSION "${absl_VERSION}" PARENT_SCOPE)

      if(NURI_PREBUILT_ABSL AND CMAKE_SYSTEM_NAME MATCHES Linux)
        set(ABSL_USES_OLD_ABI ON PARENT_SCOPE)
        set(
          CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"
          PARENT_SCOPE
        )
      endif()
      return()
    endif()
  endif()

  message(NOTICE "Could not find compatible abseil. Fetching from github.")

  include(CPM)
  set(CPM_USE_LOCAL_PACKAGES OFF)
  CPMAddPackage(
    NAME absl
    OPTIONS
    "BUILD_TESTING OFF"
    "BUILD_SHARED_LIBS OFF"
    "ABSL_ENABLE_INSTALL ON"
    "ABSL_BUILD_TESTING OFF"
    "ABSL_PROPAGATE_CXX_STD ON"
    "ABSL_USE_SYSTEM_INCLUDES ON"
    URL https://github.com/jnooree/abseil-cpp/releases/latest/download/abseil-cpp-latest.tar.gz
    EXCLUDE_FROM_ALL ON
    SYSTEM ON
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
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=libubsan.so"
    OUTPUT_VARIABLE ubsan_lib_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(SANITIZER_ENVS
    "LD_PRELOAD=${asan_lib_path} ${ubsan_lib_path} $ENV{LD_PRELOAD}"
    PARENT_SCOPE
  )
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

macro(add_both_options)
  add_compile_options(${ARGV})
  add_link_options(${ARGV})
endmacro()
