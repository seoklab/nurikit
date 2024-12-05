#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

if(NOT TARGET nuri_all_test)
  add_custom_target(nuri_all_test)
  clear_coverage_data(nuri_all_test)

  if(BUILD_TESTING)
    set_target_properties(nuri_all_test PROPERTIES EXCLUDE_FROM_ALL OFF)
  endif()
endif()

include(GoogleTest)

if(NURI_ENABLE_SANITIZERS AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  set(NURI_GTEST_EXTRA_ARGS DISCOVERY_MODE PRE_TEST)
endif()

function(nuri_add_test file)
  get_filename_component(test_dir ${file} DIRECTORY)
  file(RELATIVE_PATH test_dir "${PROJECT_SOURCE_DIR}/test" "${test_dir}")
  string(REPLACE "/" "_" test_prefix ${test_dir})

  get_filename_component(test_name ${file} NAME_WE)

  set(target "nuri_${test_prefix}_${test_name}")
  add_executable("${target}" "${file}")
  target_link_libraries("${target}" PRIVATE
    GTest::gtest GTest::gmock GTest::gtest_main
    absl::absl_log absl::absl_check)

  if(TARGET nuri_lib)
    target_link_libraries("${target}" PRIVATE nuri_lib)
  endif()

  gtest_discover_tests("${target}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ${NURI_GTEST_EXTRA_ARGS})

  add_dependencies(nuri_all_test "${target}")
endfunction()
