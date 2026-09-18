#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

"""Black-box driver for the Rosetta ``sc`` application (reference values)."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BINARY = Path(
    os.environ.get(
        "ROSETTA_SC",
        "/applic/rosetta/current/main/source/bin/sc.linuxgccrelease",
    )
)


@dataclass
class RosettaSide:
    n_atoms: int
    n_buried_atoms: int
    n_blocked_atoms: int
    n_dots: int
    n_trimmed: int
    trimmed_area: float
    d_mean: float
    d_median: float
    s_mean: float
    s_median: float


@dataclass
class RosettaResult:
    sc: float
    distance: float
    area: float
    sides: tuple[RosettaSide, RosettaSide]
    log: str


def available(binary: Path = DEFAULT_BINARY) -> bool:
    return binary.exists() and os.access(binary, os.X_OK)


def _ints(pattern: str, text: str) -> list[int]:
    return [int(x) for x in re.findall(pattern, text)]


def _floats(pattern: str, text: str) -> list[float]:
    return [float(x) for x in re.findall(pattern, text)]


def parse_verbose(log: str) -> RosettaResult:
    n_atoms = _ints(r"Total Atoms:\s+(\d+)", log)
    n_buried = _ints(r"Buried Atoms:\s+(\d+)", log)
    n_blocked = _ints(r"Blocked Atoms:\s+(\d+)", log)
    n_dots = _ints(r"Total Dots:\s+(\d+)", log)
    n_trimmed = _ints(r"Trimmed Surface Dots:\s+(\d+)", log)
    trimmed_area = _floats(r"Trimmed Area:\s+(\S+)", log)
    d_mean = _floats(r"Mean Separation:\s+(\S+)", log)
    d_median = _floats(r"Median Separation:\s+(\S+)", log)
    s_mean = _floats(r"Mean Shape Compl\.:\s+(\S+)", log)
    s_median = _floats(r"Median Shape Compl\.:\s+(\S+)", log)
    if len(n_atoms) < 2 or len(s_median) < 2:
        raise RuntimeError(f"unparsable rosetta output:\n{log[-2000:]}")

    sides = tuple(
        RosettaSide(
            n_atoms[i],
            n_buried[i],
            n_blocked[i],
            n_dots[i],
            n_trimmed[i],
            trimmed_area[i],
            d_mean[i],
            d_median[i],
            s_mean[i],
            s_median[i],
        )
        for i in range(2)
    )
    sc = _floats(r"Shape Complementarity:\s+(\S+)", log)
    dist = _floats(r"Interface separation \(A\):\s+(\S+)", log)
    area = _floats(r"Area buried in interface \(A\^2\):\s+(\S+)", log)
    return RosettaResult(sc[-1], dist[-1], area[-1], sides, log)


def run_sc(
    pdb: str | Path,
    chains_a: str,
    chains_b: str,
    density: float = 15.0,
    rp: float = 1.7,
    band: float = 1.5,
    sep: float = 8.0,
    weight: float = 0.5,
    binary: Path = DEFAULT_BINARY,
    timeout: float = 900.0,
) -> RosettaResult:
    cmd = [
        str(binary),
        "-s",
        str(Path(pdb).resolve()),
        "-ignore_unrecognized_res",
        "-sc:molecule_1",
        chains_a,
        "-sc:molecule_2",
        chains_b,
        "-sc:verbose",
        "-sc:density",
        str(density),
        "-sc:rp",
        str(rp),
        "-sc:trim",
        str(band),
        "-sc:sec",
        str(sep),
        "-sc:weight",
        str(weight),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"rosetta sc failed ({proc.returncode}):\n{log}")
    return parse_verbose(log)
