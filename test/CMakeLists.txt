#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(GoogleTest)
include(NurikitTest)

file(GLOB_RECURSE NURI_TEST_SRCS *.cpp)

foreach(nuri_test_src IN LISTS NURI_TEST_SRCS)
  get_filename_component(nuri_test_dir ${nuri_test_src} DIRECTORY)
  file(RELATIVE_PATH nuri_test_dir
    "${CMAKE_CURRENT_LIST_DIR}" "${nuri_test_dir}")
  string(REPLACE "/" "_" nuri_test_prefix ${nuri_test_dir})

  get_filename_component(nuri_test_name ${nuri_test_src} NAME_WE)

  set(nuri_test_target "nuri_${nuri_test_prefix}_${nuri_test_name}")
  add_executable("${nuri_test_target}" ${nuri_test_src})
  target_link_libraries("${nuri_test_target}" PRIVATE
    GTest::gtest GTest::gtest_main absl::absl_log absl::absl_check)

  if(TARGET nuri_lib)
    target_link_libraries("${nuri_test_target}" PRIVATE nuri_lib)
  endif()

  gtest_discover_tests("${nuri_test_target}" "" AUTO)
  add_dependencies(nuri_all_test "${nuri_test_target}")
endforeach()
