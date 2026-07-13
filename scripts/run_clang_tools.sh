#!/bin/bash
#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

project_root="$(dirname "$(dirname "$(realpath "$0")")")"

build_dir=""
diff_base=""
nproc="$(nproc)"

format_args=(-i)

tidy_args=(--warnings-as-errors='*')
if command -v gcc &>/dev/null; then
	tidy_args+=(
		--extra-arg="--gcc-install-dir=$(dirname "$(gcc -print-libgcc-file-name)")"
	)
fi

function usage() {
	cat <<EOF
Usage: $0 [OPTIONS] [--] [FILE...]

Run clang-format and clang-tidy over the project sources. With no FILE arguments
(and without -d), the whole tree is checked.

Options:
  -b DIR        Build directory holding the compilation database.
                (default: ${project_root}/build)
  -d [REFSPEC]  Diff mode: check only files changed vs REFSPEC.
                (default REFSPEC: origin/main)
  -c            Check only: run clang-format in dry-run (--Werror) instead
                of rewriting files in place.
  -x ARG        Pass an extra --extra-arg=ARG to clang-tidy (repeatable).
  -j N          Number of parallel jobs (default: ${nproc}).
  -h            Show this help and exit.

Use -- to end option parsing (e.g. '-d --' keeps the default base without
consuming the next token).
EOF
}

while getopts 'b:dcx:j:h' opt; do
	case "$opt" in
	b)
		build_dir="$OPTARG"
		build_dir="$(realpath "$build_dir")"
		;;
	d)
		# Optional argument: '-d' alone defaults to origin/main, while
		# '-d <refspec>' overrides the base. getopts has no native optional
		# argument, so peek at the next token and consume it only if it is
		# neither another option nor the '--' terminator (both match '-*').
		diff_base="origin/main"
		if [[ -n "${!OPTIND-}" && "${!OPTIND}" != -* ]]; then
			diff_base="${!OPTIND}"
			OPTIND=$((OPTIND + 1))
		fi
		;;
	c) format_args=(-n --Werror) ;;
	x) tidy_args+=(--extra-arg="$OPTARG") ;;
	j) nproc="$OPTARG" ;;
	h)
		usage
		exit 0
		;;
	*)
		usage >&2
		exit 1
		;;
	esac
done

shift $((OPTIND - 1))

type clang-format clang-tidy

if [[ -z $build_dir ]]; then
	build_dir="${project_root}/build"
fi

if [[ ! -d $build_dir ]]; then
	echo "Build directory not found. Configure the project with cmake."
	exit 1
fi

tidy_args+=(-p "$build_dir")

if [[ -n $diff_base ]]; then
	function list-changed() {
		git -C "$project_root" diff "$diff_base" \
			--name-only -z --diff-filter=d --line-prefix="${project_root}/" \
			-- "$@"
	}

	mapfile -d '' -t tidy_paths < <(
		list-changed 'include/*' 'src/*' 'python/include/*' 'python/src/*'
	)

	mapfile -d '' -t format_paths < <(list-changed 'test/*')
	format_paths+=("${tidy_paths[@]}")
elif [[ $# -eq 0 ]]; then
	tidy_paths=("$project_root"/{include,src,python/include,python/src})
	format_paths=("${tidy_paths[@]}" "$project_root"/test)
else
	format_paths=("$@")

	tidy_paths=()
	for f in "$@"; do
		case "$f" in
		*include/* | *src/*)
			tidy_paths+=("$f")
			;;
		esac
	done
fi

function search-exec() {
	local prog="$1"
	shift

	if [[ $# -eq 0 ]]; then
		echo >&2 "No source files to check with ${prog}."
		return
	fi

	case "$prog" in
	clang-tidy)
		local -a args=("${tidy_args[@]}")
		;;
	clang-format)
		local -a args=("${format_args[@]}")
		;;
	*)
		echo >&2 "Unknown program: ${prog}"
		return 1
		;;
	esac

	find "$@" \( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' \) -print0 |
		xargs -0 -r -P"$nproc" -n1 "$prog" "${args[@]}"
}

search-exec clang-tidy "${tidy_paths[@]}"
search-exec clang-format "${format_paths[@]}"
