#!/bin/bash -eu
#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

prj_root="$(dirname "$(dirname "$(realpath "$0")")")"

cpp_test=false
py_test=false
output_args=()
output=''

while getopts ':-:' opt; do
	case "$opt" in
	-) ;;
	*) exit 1 ;;
	esac

	case "$OPTARG" in
	cpp) cpp_test=true ;;
	python) py_test=true ;;
	html)
		output_args=(--html-details)
		output="coverage/html/index.html"
		;;
	json)
		output_args=(--coveralls)
		output="coverage/json/coverage.json"
		;;
	*)
		echo >&2 "Unknown option: --$OPTARG"
		exit 1
		;;
	esac
done

shift $((OPTIND - 1))
build_dir="$(realpath "${1-build}")"

if [[ $cpp_test = false && $py_test = false ]]; then
	cpp_test=true
	py_test=true
fi

cd "$prj_root"

if [[ $cpp_test = true ]]; then
	pushd "$build_dir"
	ctest -C Debug -T test -"j$(nproc)" --output-on-failure
	popd
fi

if [[ $py_test = true ]]; then
	pytest -v -"n$(nproc)" python/test
	cmake --build "$build_dir" --target nuri_python_docs_doctest -"j$(nproc)"
fi

if [[ -n $output ]]; then
	output="$build_dir/$output"
	mkdir -p "$(dirname "$output")"
	output_args+=("$output")
fi
gcovr "${output_args[@]}" "$build_dir"
