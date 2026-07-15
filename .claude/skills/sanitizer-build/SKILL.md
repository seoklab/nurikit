---
name: sanitizer-build
description: >-
  Configure and build the `sanitizer` (ASan/UBSan) CMake preset of nurikit.
  Use when building, configuring, or running the sanitizer build — or when a
  sanitizer build fails to link/run under GCC. Covers the clang++ requirement
  (active GCC bug), the preset, and the build-time environment variables.
---

# Sanitizer builds

Sanitizer builds (the `sanitizer` CMake preset) must use `clang++` due to an
active GCC bug. Building the sanitizer variant with GCC will fail to link or run
correctly.

## Configure

Use the `sanitizer` preset. Append the `--gcc-install-dir` flag to support mixed GCC runtime:

```bash
cmake --preset sanitizer \
  -DCMAKE_CXX_FLAGS="--gcc-install-dir=$(dirname "$(gcc -print-libgcc-file-name)")"
```

## Build

Set these environment variables at build time:

```bash
LD_PRELOAD="$(clang++ -print-file-name=libubsan.so)" \
  ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0" \
  cmake --build build/sanitizer -j"$(nproc)"
```

## Run the tests

The extensions are built with `-shared-libasan`, so the ASan runtime must be
preloaded **before** libubsan (`LD_PRELOAD` order matters — ASan must come
first, or you get `ASan runtime does not come first in initial library list`).
Use the **same** `clang++` that built the tree.

```bash
export LD_PRELOAD="$(clang++ -print-file-name=libclang_rt.asan-x86_64.so) $(clang++ -print-file-name=libubsan.so)"
export ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0"
```

C++ (`NURI_BUILD_TESTING` is ON by default):

```bash
ctest --test-dir build/sanitizer -j"$(nproc)" --output-on-failure
```

Python (`NURI_BUILD_PYTHON` is ON by default; the dev build stages the
extensions into `python/src/nuri/`):

```bash
PYTHONPATH="$PWD/python/src" python -m pytest -v python/test
```

`detect_leaks=0` is required — the Python interpreter itself leaks and would
otherwise drown the report. A clean run (no `heap-use-after-free`,
`stack-use-after-scope`, or UB) across both suites is the pass condition.
