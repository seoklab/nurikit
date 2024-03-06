#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

macro(_nuri_get_git_version_impl)
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

function(nuri_get_version)
  if(SKBUILD)
    # Version correctly set via scikit-build-core; skip git versioning.
    set(NURI_VERSION "${SKBUILD_PROJECT_VERSION}")
    message(STATUS "Nurikit version from scikit-build-core: ${NURI_VERSION}")
  else()
    _nuri_get_git_version_impl()
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

function(nuri_make_available_deponly target)
  include(FetchContent)

  FetchContent_GetProperties(${target})

  if(NOT ${target}_POPULATED)
    FetchContent_Populate(${target})
    add_subdirectory(
      ${${target}_SOURCE_DIR} ${${target}_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
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

function(find_or_fetch_boost)
  set(BUILD_TESTING OFF)

  find_package(Boost 1.82)

  if(Boost_FOUND)
    message(STATUS "Found Boost ${Boost_VERSION}")
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

  # emulate find_package(Boost) behavior
  add_library(Boost::boost INTERFACE IMPORTED)
  target_link_libraries(Boost::boost INTERFACE
    Boost::iterator Boost::config # for iterators
    Boost::spirit Boost::fusion Boost::mpl Boost::optional # For parsers
  )
  target_compile_definitions(Boost::boost INTERFACE
    BOOST_ALL_NO_LIB
    BOOST_ERROR_CODE_HEADER_ONLY
  )
endfunction()

function(find_or_fetch_pybind11)
  set(BUILD_TESTING OFF)

  find_package(pybind11 2.10.4)

  if(pybind11_FOUND)
    message(STATUS "Found pybind11 ${pybind11_VERSION}")
  else()
    include(FetchContent)
    message(NOTICE "Could not find compatible pybind11. Fetching from github.")

    Fetchcontent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.10.4
    )
    nuri_make_available_deponly(pybind11)
  endif()
endfunction()

function(_target_system_include target dep)
  get_target_property(interface_include_dirs "${dep}"
    INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories("${target}" SYSTEM PUBLIC ${interface_include_dirs})
endfunction()

function(target_system_include_directories target)
  foreach(dep IN LISTS ARGN)
    _target_system_include("${target}" "${dep}")
  endforeach()
endfunction()
