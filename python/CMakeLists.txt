#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(NurikitUtils)
find_or_fetch_pybind11()

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py.in"
  "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py"
  @ONLY)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/src/nuri/_version.py"
  DESTINATION ".")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
add_compile_options(
  -include "${CMAKE_CURRENT_LIST_DIR}/include/nuri/python/config.h")

add_subdirectory(src)
add_subdirectory(docs EXCLUDE_FROM_ALL)
