#!/bin/bash

#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

# Print the wheel build matrix as a JSON array of {"python": "<pyver>-<os>"}.
#
# With MINIMAL=true, emit only the cp38 smoke-test set plus any targets
# requested via a `[wheel <prefix> ...]` token in the last commit message;
# each prefix is matched against the full target list. Otherwise emit the
# full target list.

set -euo pipefail

pyvers=(cp38 cp39 cp310 cp311 cp312 cp313 cp314 cp314t)
oses=(manylinux_x86_64 macosx_x86_64 macosx_arm64)

targets=()
for p in "${pyvers[@]}"; do
	for o in "${oses[@]}"; do
		targets+=("$p-$o")
	done
done

declare -a selected

function select-targets() {
	if [[ ${MINIMAL-} != true ]]; then
		selected=("${targets[@]}")
		return
	fi

	if [[ ${GITHUB_EVENT_NAME-} = pull_request ]]; then
		msg="$(git log --no-merges --format=%B -n 1 HEAD)"
	else
		msg="$(git log --format=%B -n 1 HEAD)"
	fi

	prefix=(cp38)
	while read -r -a words; do
		prefix+=("${words[@]}")
	done < <(grep -oE '\[wheel[^]]*\]' <<<"$msg" | sed -E 's/\[wheel//g; s/\]//g')

	for tgt in "${targets[@]}"; do
		for pfx in "${prefix[@]}"; do
			if [[ $pfx = all ]]; then
				selected=("${targets[@]}")
				return
			fi

			if [[ $tgt = "$pfx"* ]]; then
				selected+=("$tgt")
				break
			fi
		done
	done
}

select-targets

jq -nc '$ARGS.positional | map({python: .})' --args "${selected[@]}"
