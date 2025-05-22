#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(NuriKitUtils)

if(NURI_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_or_add_package(
    NAME pybind11
    MIN_VERSION 2.13
    CPM_VERSION 2.13.6
    OPTIONS "BUILD_TESTING OFF"
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
  )

  include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
  add_compile_options(
    -include
    "${CMAKE_CURRENT_LIST_DIR}/include/nuri/python/config.h"
  )

  add_subdirectory(src)
endif()

if(NURI_BUILD_PYTHON_DOCS OR NURI_BUILD_PYTHON_DOCTEST)
  add_subdirectory(docs)
endif()
