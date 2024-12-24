#!/bin/bash
#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

project_root="$(dirname "$(dirname "$(realpath "$0")")")"
suppression="$(printf '%q' "$project_root/.github/ubsanignore.txt")"

coverage=false
corpus=corpus
while getopts ':-:' opt; do
	case "$opt" in
	-) ;;
	*) exit 1 ;;
	esac

	case "$OPTARG" in
	coverage) coverage=true ;;
	corpus)
		corpus="${!OPTIND}"
		((++OPTIND))
		;;
	*)
		echo >&2 "Unknown option: --$OPTARG"
		exit 1
		;;
	esac
done

shift $((OPTIND - 1))

exe="$(realpath "$1")"
build_dir="$(dirname "$(dirname "$exe")")"

shift 1

type clang++
if [[ $coverage == true ]]; then
	type llvm-profdata llvm-cov c++filt
fi

LLVM_PROFILE_FILE='coverage.profraw' \
	LD_PRELOAD="$(clang++ -print-file-name=libclang_rt.asan-x86_64.so) /usr/lib/x86_64-linux-gnu/libubsan.so.1" \
	UBSAN_OPTIONS="suppressions=$suppression print_stacktrace=1" \
	echo -- "$exe" "$corpus" "$@"

if [[ $coverage == true ]]; then
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
fi
