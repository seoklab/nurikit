#!/bin/bash -eu
#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

prj_root="$(dirname "$(dirname "$(realpath "$0")")")"
build_dir="$(realpath "${1-build}")"

cd "$build_dir"
ctest -C Debug -T test -"j$(nproc)" --output-on-failure

cd "$prj_root"
if [[ "${2-}" == "--html" ]]; then
	mkdir -p "$build_dir/coverage/html"
	gcovr --html-details "$build_dir/coverage/html/index.html" "$build_dir"
	python3 -m http.server -d "$build_dir/coverage/html" "${3-8000}"
elif [[ "${2-}" == "--xml" ]]; then
mkdir -p "$build_dir/coverage/xml"
	gcovr --xml "$build_dir/coverage/xml/coverage.xml" "$build_dir"
else
	gcovr "$build_dir"
fi
