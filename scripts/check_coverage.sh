#!/bin/bash -eu
#
# Project NuriKit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

prj_root="$(dirname "$(dirname "$(realpath "$0")")")"

cpp_test=''
py_test=''
output_args=()
output=''
jobs="$(nproc)"

usage() {
	cat <<EOF
Usage: $0 [OPTIONS] [BUILD_DIR]

Run the test suite(s) against a coverage build and report with gcovr.
BUILD_DIR is the CMake build directory (default: ${prj_root}/build).

Test selection (default: run both C++ and Python tests):
      --cpp      Run only the C++ tests (ctest).
      --python   Run only the Python tests (pytest + doctest).
      --norun    Skip running the tests, only generate the coverage report.

Report format:
      --html     Write an HTML report to
                 BUILD_DIR/coverage/html/index.html.
      --json     Write a Coveralls JSON report to
                 BUILD_DIR/coverage/json/coverage.json.

Miscellaneous:
  -j N           Number of parallel jobs (default: ${jobs}).
  -h, --help     Show this help and exit.
EOF
}

while getopts ':hj:-:' opt; do
	case "$opt" in
	h)
		usage
		exit 0
		;;
	j) jobs="$OPTARG" ;;
	-)
		case "$OPTARG" in
		help)
			usage
			exit 0
			;;
		cpp) cpp_test=true ;;
		python) py_test=true ;;
		norun)
			cpp_test=false
			py_test=false
			;;
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
			usage >&2
			exit 1
			;;
		esac
		;;
	:)
		echo >&2 "Option -$OPTARG requires an argument"
		usage >&2
		exit 1
		;;
	*)
		echo >&2 "Unknown option: -$OPTARG"
		usage >&2
		exit 1
		;;
	esac
done

shift $((OPTIND - 1))
build_dir="$(realpath "${1-${prj_root}/build}")"

if [[ -z $cpp_test && -z $py_test ]]; then
	cpp_test=true
	py_test=true
fi

cd "$prj_root"

if [[ $cpp_test = true ]]; then
	ctest --test-dir "$build_dir" -j"$jobs" --no-tests=error --output-on-failure
fi

if [[ $py_test = true ]]; then
	pytest -v -n"$jobs" python/test
	cmake --build "$build_dir" --target NuriPythonDoctest -j"$jobs"
fi

if [[ -n $output ]]; then
	output="$build_dir/$output"
	mkdir -p "$(dirname "$output")"
	output_args+=("$output")
fi
gcovr -j"$jobs" --object-directory "$build_dir" "${output_args[@]}"
