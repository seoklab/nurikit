#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(GoogleTest)

if(NURI_ENABLE_SANITIZERS AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  set(NURI_GTEST_EXTRA_ARGS DISCOVERY_MODE PRE_TEST)
endif()

function(_nuri_generate_test_name root file)
  get_filename_component(test_dir ${file} DIRECTORY)
  file(RELATIVE_PATH test_dir "${root}" "${test_dir}")
  string(REPLACE "/" "_" test_prefix ${test_dir})

  get_filename_component(test_name ${file} NAME_WE)

  set(NURI_TEST_TARGET "NuriTest-${test_prefix}-${test_name}" PARENT_SCOPE)
  set(NURI_TEST_OUTPUT "nuri-${test_prefix}-${test_name}" PARENT_SCOPE)
endfunction()

function(nuri_add_test file)
  _nuri_generate_test_name("${PROJECT_SOURCE_DIR}/test" "${file}")

  add_executable("${NURI_TEST_TARGET}" "${file}")
  target_link_libraries(
    "${NURI_TEST_TARGET}"
    PRIVATE
    NuriLib
    GTest::gtest GTest::gmock GTest::gtest_main
    absl::absl_log absl::absl_check
  )
  set_target_properties(
    "${NURI_TEST_TARGET}"
    PROPERTIES
    OUTPUT_NAME "${NURI_TEST_OUTPUT}"
  )

  gtest_discover_tests(
    "${NURI_TEST_TARGET}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ${NURI_GTEST_EXTRA_ARGS}
  )

  add_dependencies(NuriAllTest "${NURI_TEST_TARGET}")
endfunction()

function(nuri_add_fuzz file)
  _nuri_generate_test_name("${PROJECT_SOURCE_DIR}/fuzz" "${file}")

  add_executable("${NURI_TEST_TARGET}" "${file}")
  target_link_options("${NURI_TEST_TARGET}" PRIVATE -fsanitize=fuzzer)
  target_link_libraries(
    "${NURI_TEST_TARGET}"
    PRIVATE NuriLib absl::log_initialize
  )
  set_target_properties(
    "${NURI_TEST_TARGET}"
    PROPERTIES
    OUTPUT_NAME "${NURI_TEST_OUTPUT}"
  )

  add_dependencies(NuriAllFuzz "${NURI_TEST_TARGET}")
endfunction()
