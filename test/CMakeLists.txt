#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(NurikitTest)

file(GLOB_RECURSE NURI_TEST_SRCS *.cpp)

foreach(nuri_test_src IN LISTS NURI_TEST_SRCS)
  nuri_add_test(${nuri_test_src})
endforeach()
