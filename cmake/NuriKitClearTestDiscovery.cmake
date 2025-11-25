#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

if(NOT CMAKE_SCRIPT_MODE_FILE)
  message(WARNING "This file must be executed with CMake in script mode, not included with include() or add_subdirectory().")
  return()
endif()

file(GLOB nuri_test_discov LIST_DIRECTORIES OFF "./cmake_test_discovery*.json")

if(nuri_test_discov)
  list(LENGTH nuri_test_discov nuri_test_discov_count)
  message(NOTICE "Removing ${nuri_test_discov_count} test discovery files in ${CMAKE_SOURCE_DIR}")
  file(REMOVE ${nuri_test_discov})
endif()
