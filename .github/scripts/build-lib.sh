#!/bin/bash

#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

output="libnuri-${1}.tar.gz"
set --

if [[ $output == *linux* ]]; then
	eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
fi

brew install eigen spectra boost ninja

cmake -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX=./install \
	'-DCMAKE_OSX_ARCHITECTURES=x86_64;arm64' \
	-DCMAKE_VERBOSE_MAKEFILE=ON \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
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

pushd build
if [[ $output == *linux* ]]; then
	ctest -T test --output-on-failure -j
else
	arch -arm64 ctest -T test --output-on-failure -j
	arch -x86_64 ctest -T test --output-on-failure -j
fi
popd
