# Agent Instructions

## Operating rules

**Always**

- **Tooling** - Assume tools like `cmake`, C++ compiler(s),
  `clang-format`/`clang-tidy`, `pre-commit`, and `python` are installed
  globally; don't check. C++ deps are fetched by CPM at configure time. If a
  tool is genuinely missing, surface the error instead of working around it.
- **Build dir** - Drive the project CMake-first through `build/<variant>` at the
  repo root; see `cmake-variants.json` for variants.
- **Reconfigure** - Re-run the configure step after pulling or switching
  branches.

**When you change...**

- **C++ code** - Follow the style guide (Google C++ with tweaks; see
  `CONTRIBUTING.md`) and the surrounding code. Notably: `lower_case` names for
  all functions; `.cpp` sources; exceptions never escape the public API;
  non-trivially-destructible static-storage objects must be `const`/`constexpr`.
  Format `clang-format`, lint `clang-tidy`.
- **Python code** - Follow PEP 8; `ruff` handles format and import order.
- **Commits** - Use Conventional Commits (`gitlint`). Commit on a feature
  branch, never `main` or `release/*`.
- **Tests** - Add tests for new features. C++ under `test/` (GoogleTest, via
  `nuri_add_test()`); Python under `python/test/` (pytest).
- **Docs** - C++ is Doxygen (`docs/`), Python is Sphinx (`python/docs/`); keep
  docstrings and doc sources in sync. For Python docstrings, always use
  Sphinx-compatible reST.
- **Format & lint** - C++ with `scripts/run_clang_tools.sh` (slow, run only when
  needed). `ruff` and other checks run via `pre-commit`.

### Caveats

- pre-commit hooks must run at every commit. If not, run
  `pre-commit install --install-hooks --overwrite` to fix.

## Notable dev commands

Use `ninja` generator for faster compilation.

Below, `<build>` is `build/<variant>`; `<variant>` is one defined in
`cmake-variants.json`: `debug`, `release`, `minsize`, `reldeb`, `coverage`,
`sanitizers`, `fuzzer`. Prefer:

- `coverage` for local development and debugging,
- `sanitizers` for memory and UB safety, and
- `reldeb` for performance profiling.

Avoid using others unless explicitly requested.

| Task              | Command                                                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Configure         | `cmake -G Ninja -S . -B <build> -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNURI_BUILD_DEV_MODE=ON && cp <build>/compile_commands.json build/` |
| Build             | `cmake --build <build> -j"$(nproc)"`                                                                                                    |
| C++ tests         | `ctest -j"$(nproc)" --test-dir <build> --output-on-failure`                                                                             |
| Python tests      | `PYTHONPATH="$PWD/python/src" pytest -vs python/test`                                                                                   |
| Python doctest    | `PYTHONPATH="$PWD/python/src" cmake --build <build> --target NuriPythonDoctest`                                                         |
| Format & lint C++ | `scripts/run_clang_tools.sh -j"$(nproc)" [files...]`                                                                                    |
| Coverage          | `scripts/check_coverage.sh <build>`                                                                                                     |
| C++ API docs      | `cmake --build <build> --target NuriDocs`                                                                                               |
