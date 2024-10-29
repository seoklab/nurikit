#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(NuriKitUtils)
find_or_fetch_pybind11()

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py.in"
  "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py"
  @ONLY)

if(SKBUILD)
  install(FILES "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py"
    DESTINATION ".")
endif()

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
string(APPEND CMAKE_CXX_FLAGS
  " -include"
  " ${CMAKE_CURRENT_LIST_DIR}/include/nuri/python/config.h"
)

add_subdirectory(src)
add_subdirectory(docs EXCLUDE_FROM_ALL)
