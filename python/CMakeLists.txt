#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

if(NURI_BUILD_PYTHON)
  include(CPM)

  CPMFindPackage(
    NAME pybind11
    VERSION 2.13
    OPTIONS "BUILD_TESTING OFF" "PYBIND11_FINDPYTHON ON"
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
    EXCLUDE_FROM_ALL ON
    SYSTEM ON
  )

  configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py.in"
    "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py"
    @ONLY
  )

  if(SKBUILD)
    install(
      FILES "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py"
      DESTINATION "."
    )
  endif()

  include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
  add_compile_options(
    -include
    "${CMAKE_CURRENT_LIST_DIR}/include/nuri/python/config.h"
  )

  add_subdirectory(src)
endif()

if(NURI_BUILD_PYTHON_DOCS OR BUILD_TESTING)
  add_subdirectory(docs)
endif()
