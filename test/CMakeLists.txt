#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(GoogleTest)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(STATUS "GNU/Clang detected, enabling coverage")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

include_directories("${PROJECT_SOURCE_DIR}/src")

add_executable(nurikit_graph_test graph_test.cpp)
target_link_libraries(nurikit_graph_test PRIVATE GTest::gtest GTest::gtest_main)

gtest_discover_tests(nurikit_graph_test "" AUTO)

add_custom_target(nurikit_test)
add_dependencies(nurikit_test nurikit_graph_test)
