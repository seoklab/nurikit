---
name: fuzzer-build
description: >-
  Configure, build, and run the `fuzzer` (libFuzzer + ASan/UBSan) CMake variant
  of nurikit, then triage crashes and report coverage. Use when asked to
  fuzz-test a format parser (cif, mol2, pdb, sdf, smi), investigate a fuzz
  crash, or run the build/fuzz/triage/report loop. Clang-only; all fuzz output
  lives in a user-chosen directory outside git.
---

# Fuzzer builds

The `fuzzer` CMake variant is libFuzzer + ASan/UBSan and is **Clang-only**
(`CMakeLists.txt` hard-errors otherwise, and force-enables sanitizers). It
exercises the format parsers via `LLVMFuzzerTestOneInput` harnesses in
`fuzz/fmt/`. See the `sanitizer-build` skill for the clang++ build rationale.

**All fuzz output (corpus growth, `crash-*`, coverage) MUST stay outside git** —
run everything from a user-chosen run directory, never inside the repo.

Resolve the repo root once; use it for every repo path:

```bash
REPO="$(git rev-parse --show-toplevel)"
```

## 1. Formats and targets

The user picks the format(s). If told to run **all**, loop over all five.

| Format | Seed corpus (in repo) | Fuzz binary |
| ------ | --------------------- | ----------- |
| `cif`  | `fuzz/corpus-min/cif`  | `nuri-fmt-cif_fuzz`    |
| `mol2` | `fuzz/corpus-min/mol2` | `nuri-fmt-mol2_fuzz`   |
| `pdb`  | `fuzz/corpus-min/pdb`  | `nuri-fmt-pdb_fuzz`    |
| `sdf`  | `fuzz/corpus-min/sdf`  | `nuri-fmt-sdf_fuzz`    |
| `smi`  | `fuzz/corpus-min/smi`  | `nuri-fmt-smiles_fuzz` |

Note `smi` maps to the `smiles` binary. Binaries land at
`$REPO/build/fuzzer/fuzz/<binary>` after the build.

## 2. Run directory (persisted in memory)

The run directory holds per-format corpora that persist across runs. Do **not**
hardcode it.

1. Check Claude memory for `fuzz-run-dir` (see §7). If present, reuse it —
   don't re-ask.
2. Otherwise ask the user for an absolute path, then save it to memory.
3. If the user names a different directory, use it and update the memory.

```bash
RUNDIR="<absolute path from memory or the user>"
```

## 3. Build the `fuzzer` variant (clang++)

Append the `--gcc-install-dir` flag to support mixed GCC runtime:

```bash
cmake --preset fuzzer \
  -DCMAKE_CXX_FLAGS="--gcc-install-dir=$(dirname "$(gcc -print-libgcc-file-name)")"

LD_PRELOAD="$(clang++ -print-file-name=libubsan.so)" \
  ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0" \
  cmake --build "$REPO/build/fuzzer" --target NuriAllFuzz -j"$(nproc)"
```

`NuriAllFuzz` builds all five binaries. For a single format, use its target,
e.g. `--target NuriFuzz-fmt-pdb_fuzz`.

## 4. Run the loop (local, time-boxed — default)

`scripts/fuzz.sh` preloads the ASan/UBSan runtimes and UBSan suppressions
itself — call it directly. Seed the persistent corpus from the in-repo minimized
corpus only when empty. libFuzzer reads *and* writes the corpus dir, and drops
`crash-*` / timeout / oom files into the cwd — all inside `$RUNDIR`, outside git.

```bash
fmt=pdb; bin=nuri-fmt-pdb_fuzz   # smi -> nuri-fmt-smiles_fuzz
mkdir -p "$RUNDIR/$fmt/corpus"
[ -z "$(ls -A "$RUNDIR/$fmt/corpus")" ] && \
  cp -rn "$REPO"/fuzz/corpus-min/"$fmt"/* "$RUNDIR/$fmt/corpus/"

cd "$RUNDIR/$fmt"
"$REPO/scripts/fuzz.sh" "$REPO/build/fuzzer/fuzz/$bin" \
  -only_ascii=1 -max_total_time=300 -ignore_crashes=1 -fork="$(nproc)"
```

`-fork` = `nproc` (one CPU knob, same as SLURM). `-max_total_time` is the time
box — raise it for longer campaigns. Run-all: loop over the five formats.

## 5. Triage a crash

Reproduce a single crashing input with no fork for a clean, symbolized stack
(RelWithDebInfo carries debug info):

```bash
cd "$RUNDIR/$fmt"
"$REPO/scripts/fuzz.sh" "$REPO/build/fuzzer/fuzz/$bin" crash-<sha>
```

Minimize the reproducer:

```bash
"$REPO/scripts/fuzz.sh" "$REPO/build/fuzzer/fuzz/$bin" \
  -minimize_crash=1 -runs=100000 crash-<sha>
```

Report the top `nuri` frame as `file:line`; the parser lives in
`nuri/fmt/<fmt>.h` (+ its `.cpp`). New regressions get a fixed-input test under
`test/fmt/` — see the `// Found with fuzzing` precedent there.

## 6. Coverage report (on request)

`--coverage` must come **first** — `fuzz.sh` parses it with `getopts`, which
stops at the first non-dash token (the binary path). Pass `-runs=0` so libFuzzer
replays the corpus once and exits; without it (or `-max_total_time`) it replays
then fuzzes forever. Don't pass the corpus dir positionally — the script already
appends the default `corpus`.

```bash
cd "$RUNDIR/$fmt"
"$REPO/scripts/fuzz.sh" --coverage "$REPO/build/fuzzer/fuzz/$bin" -runs=0
```

Emits `cov-html/` + a text summary in the run dir. Report line-coverage % and
notably cold regions.

## 7. Persist the run dir (memory)

On the first run — or whenever the user changes it — save the run directory to
Claude memory as a file `fuzz-run-dir.md` (`type: project`) holding the absolute
path plus the host/user it applies to, and add a one-line pointer to `MEMORY.md`.
On later runs, read it and reuse without re-asking.

## 8. SLURM (opt-in)

Only when the user asks for SLURM. Ask for the batch args first:

- **CPUs `N`** — `-fork` equals `-c` (one knob; fork workers = allocated CPUs).
- **`-max_total_time`** (seconds).
- **Extra sbatch flags** — partition `-p`, node `-w`, etc., passed through verbatim.

Submit a generated job from the run dir (no dependency on any out-of-tree
script). `$REPO` must be absolute — the job's cwd is the submit dir.

```bash
cd "$RUNDIR/$fmt"
sbatch -c <N> -J "fuzz-$fmt" <extra sbatch flags> --wrap \
  "'$REPO/scripts/fuzz.sh' '$REPO/build/fuzzer/fuzz/$bin' \
   -only_ascii=1 -max_total_time=<T> -ignore_crashes=1 -fork=<N>"
```

Run-all → one job per format. This mirrors the historical `fuzz-slurm.sh`
pattern (`-c 24`, `-fork=24`, `-max_total_time=3600`).

## 9. Troubleshooting: symbolizer hang

If a run produces **no output and never finishes** (wedged during corpus load),
the symbolizer is hanging. Root cause: **LLVM 18's symbolizer deadlocks**
symbolizing these binaries on some hosts (e.g. Ubuntu 24.04) — both the
in-process symbolizer baked into the binary by compiler-rt and the external
`llvm-symbolizer` 18. LLVM ≥ 19 fixed it.

`fuzz.sh` already forces the external symbolizer via
`external_symbolizer_path="$(command -v llvm-symbolizer)"` (this bypasses the
broken in-process one). So the fix is to make the first `llvm-symbolizer` on
`PATH` a **≥ 19** build:

```bash
command -v llvm-symbolizer            # which one fuzz.sh will use
llvm-symbolizer --version             # must be >= 19; 18.x hangs
```

If it's 18.x, put a newer one first on `PATH` (e.g. from a Homebrew/apt LLVM ≥
19 install) and re-run.

Last resort, sacrificing symbolized crash stacks: `fuzz.sh` merges any
caller-supplied `ASAN_OPTIONS`/`UBSAN_OPTIONS` after its own (caller wins), so
set `symbolize=0` on the invocation — no script edit:

```bash
cd "$RUNDIR/$fmt"
UBSAN_OPTIONS=symbolize=0 ASAN_OPTIONS=symbolize=0 \
  "$REPO/scripts/fuzz.sh" "$REPO/build/fuzzer/fuzz/$bin" corpus \
  -only_ascii=1 -max_total_time=<seconds> -ignore_crashes=1 -fork="$(nproc)"
```

The same merge lets you pass any other sanitizer flag (e.g.
`UBSAN_OPTIONS=halt_on_error=1`). Triage any reproducer later on a host with a
working symbolizer.

## 10. Report format

Per run report: format, wall time, exec/s, corpus growth, crash count with a
one-line cause each (or "no crashes"), and coverage % if run.
