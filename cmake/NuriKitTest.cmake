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

function(_nuri_generate_test_name root file)
  get_filename_component(test_dir ${file} DIRECTORY)
  file(RELATIVE_PATH test_dir "${root}" "${test_dir}")
  string(REPLACE "/" "_" test_prefix ${test_dir})

  get_filename_component(test_name ${file} NAME_WE)

  set(NURI_TEST_TARGET "nuri_${test_prefix}_${test_name}" PARENT_SCOPE)
endfunction()

function(nuri_add_test file)
  _nuri_generate_test_name("${PROJECT_SOURCE_DIR}/test" "${file}")

  add_executable("${NURI_TEST_TARGET}" "${file}")
  target_link_libraries("${NURI_TEST_TARGET}" PRIVATE
    GTest::gtest GTest::gmock GTest::gtest_main
    absl::absl_log absl::absl_check)

  if(TARGET nuri_lib)
    target_link_libraries("${NURI_TEST_TARGET}" PRIVATE nuri_lib)
  endif()

  gtest_discover_tests("${NURI_TEST_TARGET}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ${NURI_GTEST_EXTRA_ARGS})

  add_dependencies(nuri_all_test "${NURI_TEST_TARGET}")
endfunction()

if(NOT TARGET nuri_all_fuzz)
  add_custom_target(nuri_all_fuzz)
  clear_coverage_data(nuri_all_fuzz)

  if(NURI_BUILD_FUZZING AND BUILD_TESTING)
    set_target_properties(nuri_all_fuzz PROPERTIES EXCLUDE_FROM_ALL OFF)
  endif()
endif()

function(nuri_add_fuzz file)
  _nuri_generate_test_name("${PROJECT_SOURCE_DIR}/fuzz" "${file}")

  add_executable("${NURI_TEST_TARGET}" "${file}")
  target_link_options("${NURI_TEST_TARGET}" PRIVATE -fsanitize=fuzzer)
  target_link_libraries("${NURI_TEST_TARGET}" PRIVATE absl::log_initialize)

  if(TARGET nuri_lib)
    target_link_libraries("${NURI_TEST_TARGET}" PRIVATE nuri_lib)
  endif()

  add_dependencies(nuri_all_fuzz "${NURI_TEST_TARGET}")
endfunction()
