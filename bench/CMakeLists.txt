#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

include(NuriKitUtils)
include(NuriKitTest)

find_or_add_package(
  NAME benchmark
  CPM_VERSION 1.9.4
  OPTIONS
  "BENCHMARK_ENABLE_TESTING OFF"
  "BENCHMARK_ENABLE_LTO ON"
  "BENCHMARK_ENABLE_INSTALL OFF"
  GIT_REPOSITORY https://github.com/google/benchmark.git
)

add_custom_target(NuriAllBench ALL)
add_compile_options(-Wno-error)

file(GLOB_RECURSE NURI_BENCH_SRCS *.cpp)

foreach(nuri_bench_src IN LISTS NURI_BENCH_SRCS)
  nuri_add_benchmark("${nuri_bench_src}")
endforeach()
