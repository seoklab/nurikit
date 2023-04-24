#!/bin/bash -eu
#
# Project nurikit - Copyright 2023 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

prj_root="$(dirname "$(dirname "$(realpath "$0")")")"

cd "${1-build}"
ctest -C Debug -T test -"j$(nproc)" --output-on-failure

lcov --rc lcov_branch_coverage=1 --capture \
	--exclude '/usr/*' --exclude '*/third-party/*' --exclude "$prj_root/test/*" \
	--directory . --output-file "$prj_root/coverage.info"
lcov --rc lcov_branch_coverage=1 --list "$prj_root/coverage.info"
