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
