#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

if(NOT CMAKE_SCRIPT_MODE_FILE)
  message(WARNING "This file must be executed with CMake in script mode, not included with include() or add_subdirectory().")
  return()
endif()

if(NOT NURI_COVERAGE_DATA_DIR)
  message(FATAL_ERROR "No parent directory provided.")
endif()

file(GLOB_RECURSE nuri_gcda LIST_DIRECTORIES OFF "${NURI_COVERAGE_DATA_DIR}/*.gcda")

if(nuri_gcda)
  list(LENGTH nuri_gcda nuri_gcda_count)
  message(NOTICE "Removing ${nuri_gcda_count} gcda files in ${NURI_COVERAGE_DATA_DIR}")
  file(REMOVE ${nuri_gcda})
endif()

file(GLOB_RECURSE nuri_gcno LIST_DIRECTORIES OFF "${NURI_COVERAGE_DATA_DIR}/*.gcno")

foreach(gcno IN LISTS nuri_gcno)
  string(REGEX REPLACE "\\.gcno$" ".o" object "${gcno}")

  if(NOT EXISTS "${object}")
    message(STATUS "Removing stale gcno file: ${gcno}")
    file(REMOVE "${gcno}")
  endif()
endforeach()
