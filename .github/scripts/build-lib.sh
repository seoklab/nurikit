#!/bin/bash

#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

output="libnuri-${1}.tar.gz"
set --

if [[ $output == *linux* ]]; then
	yum -y install ninja-build
fi

cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=./install \
	'-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64' \
	-DCMAKE_VERBOSE_MAKEFILE=ON \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
	-DNURI_PREBUILT_ABSL=ON \
	-DNURI_BUILD_PYTHON=OFF \
	-DNURI_FORCE_VERSION="$NURI_VERSION" \
	-Sdist \
	-Bbuild \
	-GNinja
cmake --build build -j --target all
cmake --install build

# pypa build environment installs at lib64
if [[ ! -d install/lib ]]; then
	ln -s lib64 install/lib
fi
tar -cvzf "${output}" -C install .

ctest_args=(-M Continuous -T Test --no-tests=error --output-on-failure -j)

pushd build
if [[ $output == *linux* ]]; then
	ctest "${ctest_args[@]}"
else
	arch -arm64 ctest "${ctest_args[@]}"
	arch -x86_64 ctest "${ctest_args[@]}"
fi
popd
