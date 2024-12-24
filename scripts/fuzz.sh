#!/bin/bash
#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

type clang++ llvm-profdata llvm-cov c++filt

project_root="$(dirname "$(dirname "$(realpath "$0")")")"
suppression="$(printf '%q' "$project_root/.github/ubsanignore.txt")"

exe="$(realpath "$1")"
build_dir="$(dirname "$(dirname "$exe")")"
corpus="$(realpath "${2-corpus}")"

shift 2

LLVM_PROFILE_FILE='coverage.profraw' \
	LD_PRELOAD="$(clang++ -print-file-name=libclang_rt.asan-x86_64.so) /usr/lib/x86_64-linux-gnu/libubsan.so.1" \
	UBSAN_OPTIONS="suppressions=$suppression print_stacktrace=1" \
	"$exe" "$corpus" "$@"

llvm-profdata merge -sparse coverage.profraw -o coverage.profdata

llvm-cov show -format html -output-dir cov-html -Xdemangler c++filt \
	-instr-profile coverage.profdata \
	"$exe" \
	-object "$build_dir/lib/libnuri.so"
chmod -R go+rX cov-html

llvm-cov report \
	-instr-profile coverage.profdata \
	"$exe" \
	-object "$build_dir/lib/libnuri.so"
