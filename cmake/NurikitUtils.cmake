#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

macro(_nurikit_get_git_version_impl)
  find_package(Git)

  if(NOT Git_FOUND)
    message(WARNING "Git not found!")
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
    string(REGEX REPLACE "^v" "" NURI_VERSION "${NURI_REF}")
    message(STATUS "Nurikit version from git: ${NURI_VERSION}")
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

function(nurikit_get_version)
  if(SKBUILD)
    # Version correctly set via scikit-build-core; skip git versioning.
    set(NURI_VERSION "${SKBUILD_PROJECT_VERSION}")
    message(STATUS "Nurikit version from scikit-build-core: ${NURI_VERSION}")
  else()
    _nurikit_get_git_version_impl()
  endif()

  if(NURI_REF)
    message(STATUS "Nurikit ref from git: ${NURI_REF}")
  else()
    message(NOTICE "Nurikit ref not found! Using unknown.")
    set(nuri_revision "unknown")
    set(NURI_REF "unknown")
  endif()

  if(NURI_VERSION)
    set(NURI_FULL_VERSION "${NURI_VERSION}")
  else()
    set(NURI_VERSION "0.1.0.dev0")
    message(NOTICE "Nurikit version not found! Using ${NURI_VERSION}")
    set(NURI_FULL_VERSION "${NURI_VERSION}+${nuri_revision}")
  endif()

  string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+"
    NURI_CORE_VERSION "${NURI_VERSION}")

  string(TIMESTAMP NURI_YEAR "%Y" UTC)
  set(NURI_YEAR "${NURI_YEAR}" PARENT_SCOPE)
  set(NURI_VERSION "${NURI_VERSION}" PARENT_SCOPE)
  set(NURI_CORE_VERSION "${NURI_CORE_VERSION}" PARENT_SCOPE)
  set(NURI_FULL_VERSION "${NURI_FULL_VERSION}" PARENT_SCOPE)
  set(NURI_REF "${NURI_REF}" PARENT_SCOPE)
endfunction()

function(nurikit_make_available_deponly target)
  include(FetchContent)

  FetchContent_GetProperties(${target})

  if(NOT ${target}_POPULATED)
    FetchContent_Populate(${target})
    add_subdirectory(
      ${${target}_SOURCE_DIR} ${${target}_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endfunction()

function(find_or_fetch_eigen)
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

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_DOC OFF)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    nurikit_make_available_deponly(eigen)
  endif()

  get_target_property(
    EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  set(EIGEN_INCLUDE_DIRS "${EIGEN_INCLUDE_DIRS}" PARENT_SCOPE)
endfunction()
