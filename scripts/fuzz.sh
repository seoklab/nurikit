#!/bin/bash
#
# Project NuriKit - Copyright 2024 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

project_root="$(dirname "$(dirname "$(realpath "$0")")")"
suppression="$project_root/.github/ubsanignore.txt"

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

if ! symbolizer="$(command -v llvm-symbolizer)"; then
	echo >&2 "$0: type: llvm-symbolizer: not found"
	exit 1
fi
printf 'llvm-symbolizer is %s\n' "$symbolizer"

libsan=(
	"$(clang++ -print-file-name=libclang_rt.asan-x86_64.so)"
	"$(clang++ -print-file-name=libubsan.so)"
)

# Force the external llvm-symbolizer instead of compiler-rt's in-process
# symbolizer. For llvm 18, both the in-process copy and llvm-symbolizer binary
# hang symbolizing on some hosts (e.g. Ubuntu 24.04); force the external
# symbolizer to allow user customize the symbolizer path (e.g. put a newer
# llvm-symbolizer in PATH).
# Quote paths with "" to avoid issues with special characters in the path.
common_opts=("external_symbolizer_path=\"$symbolizer\"")
asan_opts=("${common_opts[@]}")
ubsan_opts=(
	"${common_opts[@]}"
	"suppressions=\"$suppression\""
	"print_stacktrace=1"
)

# Append any caller-supplied options last so they override ours (e.g.
# UBSAN_OPTIONS=...:symbolize=0 to work around a hanging symbolizer).
LLVM_PROFILE_FILE='coverage.profraw' \
	LD_PRELOAD="${libsan[*]}" \
	ASAN_OPTIONS="${asan_opts[*]} ${ASAN_OPTIONS-}" \
	UBSAN_OPTIONS="${ubsan_opts[*]} ${UBSAN_OPTIONS-}" \
	"$exe" "$corpus" "$@"

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
