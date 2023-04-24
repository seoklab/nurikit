#!/bin/bash -eu
#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

build_dir="${1-build}"

ctest -C Debug -T test -"j$(nproc)" --output-on-failure --test-dir "$build_dir"

lcov --rc lcov_branch_coverage=1 --capture \
	--exclude '/usr/*' --exclude '*/third-party/*' --exclude "$PWD/test/*" \
	--directory "$build_dir" --output-file coverage.info
lcov --rc lcov_branch_coverage=1 --list coverage.info
