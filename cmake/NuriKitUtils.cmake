#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

function(_nuri_get_git_version_impl)
  set(_nuri_version_next "0.1.0.dev0")

  find_package(Git QUIET)

  if(NOT Git_FOUND)
    message(NOTICE "Git not found! Using ${_nuri_version_next}+unknown")
    set(NURI_FULL_VERSION "${_nuri_version_next}+unknown" PARENT_SCOPE)
    set(NURI_REF "unknown" PARENT_SCOPE)
    return()
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE nuri_revision
    ERROR_QUIET
  )
  string(STRIP "${nuri_revision}" nuri_revision)

  if(NOT nuri_revision)
    set(nuri_revision "unknown")
  endif()

  execute_process(
    COMMAND ${GIT_EXECUTABLE} symbolic-ref -q --short HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE nuri_branch
    ERROR_QUIET
  )
  string(STRIP "${nuri_branch}" nuri_branch)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match --abbrev=0
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE nuri_tag
    ERROR_QUIET
  )
  string(STRIP "${nuri_tag}" nuri_tag)

  if(nuri_tag)
    set(NURI_REF "${nuri_tag}")
    string(REGEX REPLACE "^v" "" NURI_FULL_VERSION "${NURI_REF}")
    message(STATUS "NuriKit version from git: ${NURI_FULL_VERSION}")
  else()
    set(NURI_FULL_VERSION "${_nuri_version_next}+${nuri_revision}")
    message(NOTICE
      "NuriKit version not found! "
      "Assuming ${_nuri_version_next}+${nuri_revision}"
    )

    if(nuri_branch)
      set(NURI_REF "${nuri_branch}")
    else()
      set(NURI_REF "${nuri_revision}")
    endif()
  endif()

  set(NURI_FULL_VERSION "${NURI_FULL_VERSION}" PARENT_SCOPE)
  set(NURI_REF "${NURI_REF}" PARENT_SCOPE)
endfunction()

function(nuri_get_version)
  if(NURI_FORCE_VERSION)
    set(NURI_FULL_VERSION "${NURI_FORCE_VERSION}")
    set(NURI_REF "v${NURI_FULL_VERSION}")
    message(STATUS "Using explicit NuriKit version: ${NURI_FORCE_VERSION}")
  elseif(SKBUILD)
    # Version correctly set via scikit-build-core; skip git versioning.
    set(NURI_FULL_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
    set(NURI_REF "v${NURI_FULL_VERSION}")
    message(
      STATUS "NuriKit version from scikit-build-core: ${NURI_FULL_VERSION}"
    )
  else()
    _nuri_get_git_version_impl()
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
  cmake_parse_arguments(
    _pkg ""
    "NAME;MIN_VERSION;CPM_VERSION;COMPONENTS"
    ""
    "${ARGN}"
  )

  if(NOT _pkg_MIN_VERSION)
    set(_pkg_MIN_VERSION "${_pkg_CPM_VERSION}")
  endif()

  find_package(
    "${_pkg_NAME}"
    "${_pkg_MIN_VERSION}"
    QUIET
    COMPONENTS "${_pkg_COMPONENTS}"
    NO_MODULE
  )

  if(${_pkg_NAME}_FOUND)
    message(STATUS "Found ${_pkg_NAME} ${${_pkg_NAME}_VERSION}")

    set("${_pkg_NAME}_FOUND" "${${_pkg_NAME}_FOUND}" PARENT_SCOPE)
    set("${_pkg_NAME}_VERSION" "${${_pkg_NAME}_VERSION}" PARENT_SCOPE)
    return()
  endif()

  message(STATUS "Could not find ${_pkg_NAME}. Adding with CPM.")

  string(REPLACE "\\" "\\\\" extra_args "${_pkg_UNPARSED_ARGUMENTS}")

  include(CPM)
  set(CPM_USE_LOCAL_PACKAGES OFF)
  CPMAddPackage(
    NAME "${_pkg_NAME}"
    VERSION "${_pkg_CPM_VERSION}"
    EXCLUDE_FROM_ALL ON
    SYSTEM ON
    "${extra_args}"
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
    "ASAN_OPTIONS=detect_leaks=0"
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
    -P "${PROJECT_SOURCE_DIR}/cmake/NuriKitClearCoverage.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM
  )
endfunction()

macro(add_both_options)
  add_compile_options(${ARGV})
  add_link_options(${ARGV})
endmacro()
