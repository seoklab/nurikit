---
name: sanitizer-build
description: >-
  Configure and build the `sanitizers` (ASan/UBSan) CMake variant of nurikit.
  Use when building, configuring, or running the sanitizer build — or when a
  sanitizer build fails to link/run under GCC. Covers the clang++ requirement
  (active GCC bug), the required CMake flags, and the build-time environment
  variables.
---

# Sanitizer builds

Sanitizer builds (the `sanitizers` CMake variant) must use `clang++` due to an
active GCC bug. Building the sanitizer variant with GCC will fail to link or run
correctly.

## Configure

Use `clang++` and point it at the GCC install so the correct runtime is found:

```bash
cmake -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="--gcc-install-dir=$(dirname "$(gcc -print-libgcc-file-name)")" \
  ...
```

## Build

Set these environment variables at build time:

```bash
LD_PRELOAD="$(clang++ -print-file-name=libubsan.so)" \
  ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0" \
  cmake --build ...
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

C++ (build with `-DNURI_BUILD_TESTING=ON`):

```bash
ctest --test-dir <build> -j"$(nproc)" --output-on-failure
```

Python (build with `-DNURI_BUILD_PYTHON=ON`; the dev build stages the extensions
into `python/src/nuri/`):

```bash
PYTHONPATH="$PWD/python/src" python -m pytest -v python/test
```

`detect_leaks=0` is required — the Python interpreter itself leaks and would
otherwise drown the report. A clean run (no `heap-use-after-free`,
`stack-use-after-scope`, or UB) across both suites is the pass condition.
