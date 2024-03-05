#!/bin/bash -eu
#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

prj_root="$(dirname "$(dirname "$(realpath "$0")")")"
build_dir="$(realpath "${1-build}")"

cd "$build_dir"
ctest -C Debug -T test -"j$(nproc)" --output-on-failure

if [[ "${3-}" == "--python" ]]; then
	cd "$prj_root/python"
	pytest -v test
	cmake --build "$build_dir" --target nuri_python_docs_doctest -"j$(nproc)"
fi

cd "$prj_root"
if [[ "${2-}" == "--html" ]]; then
	mkdir -p "$build_dir/coverage/html"
	gcovr --html-details "$build_dir/coverage/html/index.html" "$build_dir"
elif [[ "${2-}" == "--json" ]]; then
	mkdir -p "$build_dir/coverage/json"
	gcovr --coveralls "$build_dir/coverage/json/coverage.json" "$build_dir"
else
	gcovr "$build_dir"
fi
