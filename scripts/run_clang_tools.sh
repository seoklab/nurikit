#!/bin/bash
#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

type clang-format clang-tidy

project_root="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$project_root"

if [[ ! -d build ]]; then
	echo "Build directory not found. Configure the project with cmake."
	exit 1
fi

cf_args=("-i")
nproc="$(nproc)"

ct_args=(--warnings-as-errors='*')
if command -v gcc &>/dev/null; then
	ct_args+=(
		--extra-arg="--gcc-install-dir=$(dirname "$(gcc -print-libgcc-file-name)")"
	)
fi

while getopts 'cj:x:' opt; do
	case "$opt" in
	c) cf_args=(-n --Werror) ;;
	j) nproc="$OPTARG" ;;
	x) ct_args+=(--extra-arg="$OPTARG") ;;
	*) break ;;
	esac
done

shift $((OPTIND - 1))

if [[ $# -eq 0 ]]; then
	tidy_paths=(include src python/include python/src)
	format_paths=("${tidy_paths[@]}" test)
else
	tidy_paths=("$@")
	format_paths=("$@")
fi

tmpd="$(mktemp -d)"
trap 'rm -rf "$tmpd"' INT TERM EXIT

find "${tidy_paths[@]}" \
	\( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' \) \
	-print0 >"$tmpd/tidy-checks"
find "${format_paths[@]}" \
	\( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' \) \
	-print0 >"$tmpd/format-checks"

xargs -0 -P"$nproc" -n1 clang-format "${cf_args[@]}" <"$tmpd/format-checks"
xargs -0 -P"$nproc" -n1 clang-tidy -p build "${ct_args[@]}" <"$tmpd/tidy-checks"
