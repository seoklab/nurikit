#!/bin/bash
#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
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
while getopts 'c' opt; do
	case "$opt" in
	c) cf_args=(-n --Werror) ;;
	*) break ;;
	esac
done

shift $((OPTIND - 1))

files="$(mktemp)"
trap 'rm -f "$files"' INT TERM EXIT

if [[ $# -eq 0 ]]; then
	find include src python/nuri \
		\( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' \) -print0 >"$files"
else
	printf '%s\0' "$@" >"$files"
fi

xargs -0 -P0 -n1 clang-format "${cf_args[@]}" <"$files"
xargs -0 -P0 -n1 clang-tidy -p build --warnings-as-errors='*' <"$files"
