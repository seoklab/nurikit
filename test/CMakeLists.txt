#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(NuriKitTest)

add_custom_target(NuriAllTest ALL)
clear_coverage_data(NuriAllTest)

add_compile_options(-Wno-error)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wno-self-assign-overloaded)
endif()

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
add_compile_definitions("NURI_TEST_DATA_DIR=\"${CMAKE_CURRENT_LIST_DIR}/test_data\"")

file(GLOB_RECURSE NURI_TEST_SRCS CONFIGURE_DEPENDS *.cpp)

foreach(nuri_test_src IN LISTS NURI_TEST_SRCS)
  nuri_add_test("${nuri_test_src}")
endforeach()
