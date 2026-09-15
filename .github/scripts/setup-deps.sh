#!/bin/bash

#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1

if [[ -x /home/linuxbrew/.linuxbrew/bin/brew ]]; then
	eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
fi

brew update || true

brew install eigen@3 boost spectra
brew link --force --overwrite eigen@3

echo "CMAKE_PREFIX_PATH=$HOMEBREW_PREFIX" >>"$GITHUB_ENV"
