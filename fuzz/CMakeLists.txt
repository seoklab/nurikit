#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(NuriKitTest)

add_compile_options(-Wno-error)

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
file(GLOB_RECURSE NURI_FUZZ_SRCS *.cpp)

foreach(nuri_test_src IN LISTS NURI_FUZZ_SRCS)
  nuri_add_fuzz("${nuri_test_src}")
endforeach()
