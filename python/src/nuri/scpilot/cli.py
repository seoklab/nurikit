#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import time
from pathlib import Path

import typer

from . import rosetta
from .io import Structure, load_structure
from .radii import united_atom_radii
from .sc import ScParams, ScResult, shape_complementarity

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _pairs(spec: str) -> list[tuple[str, str]]:
    out = []
    for item in spec.split(","):
        a, b = item.split(":")
        out.append((a.strip(), b.strip()))
    return out


def _fmt_ours(r: ScResult, elapsed: float) -> str:
    a, b = r.sides
    return (
        f"ours    sc={r.sc:.4f} d={r.distance:.4f} area={r.area:7.1f}"
        f"  sides sc=({a.s_median:.4f},{b.s_median:.4f})"
        f" d=({a.d_median:.3f},{b.d_median:.3f})"
        f" area=({a.trimmed_area:.1f},{b.trimmed_area:.1f})"
        f" active=({a.n_active},{b.n_active})"
        f" dots=({a.n_dots},{b.n_dots})"
        f" trimmed=({a.n_trimmed},{b.n_trimmed})  [{elapsed:.1f}s]"
    )


def _fmt_rosetta(r: rosetta.RosettaResult, elapsed: float) -> str:
    a, b = r.sides
    return (
        f"rosetta sc={r.sc:.4f} d={r.distance:.4f} area={r.area:7.1f}"
        f"  sides sc=({a.s_median:.4f},{b.s_median:.4f})"
        f" d=({a.d_median:.3f},{b.d_median:.3f})"
        f" area=({a.trimmed_area:.1f},{b.trimmed_area:.1f})"
        f" active=({a.n_buried_atoms},{b.n_buried_atoms})"
        f" dots=({a.n_dots},{b.n_dots})"
        f" trimmed=({a.n_trimmed},{b.n_trimmed})  [{elapsed:.1f}s]"
    )


def _radii(st: Structure, scheme: str, shift: float, scale: float):
    if scheme == "vdw":
        base = st.vdw_radii
    elif scheme == "united":
        base, fallback = united_atom_radii(st)
        if fallback.any():
            typer.echo(
                f"   {int(fallback.sum())} atoms fell back to vdW radii"
            )
    else:
        raise typer.BadParameter(f"unknown radii scheme {scheme!r}")
    return base * scale + shift


@app.command()
def run(
    pdb: Path,
    pairs: str = typer.Option("H:L", help="chain pairs, e.g. 'H:L,A:HL'"),
    density: float = 15.0,
    rp: float = 1.7,
    weight: float = 0.5,
    band: float = 1.5,
    sep: float = 8.0,
    radii: str = typer.Option("united", help="'united' or 'vdw'"),
    radii_shift: float = typer.Option(0.0, help="added to every radius"),
    radii_scale: float = typer.Option(1.0, help="multiplies every radius"),
    reference: bool = typer.Option(True, help="also run Rosetta sc"),
):
    """Compute Sc for chain pairs of one PDB and compare with Rosetta."""
    st = load_structure(pdb)
    params = ScParams(
        rp=rp, density=density, weight=weight, band=band, sep=sep
    )
    for a, b in _pairs(pairs):
        typer.echo(f"== {pdb.name} {a} | {b}")
        sa, sb = st.chains(a), st.chains(b)
        t0 = time.perf_counter()
        ours = shape_complementarity(
            sa.coords,
            _radii(sa, radii, radii_shift, radii_scale),
            sb.coords,
            _radii(sb, radii, radii_shift, radii_scale),
            params,
        )
        typer.echo(_fmt_ours(ours, time.perf_counter() - t0))
        if reference and rosetta.available():
            t0 = time.perf_counter()
            ref = rosetta.run_sc(
                pdb,
                a,
                b,
                density=density,
                rp=rp,
                band=band,
                sep=sep,
                weight=weight,
            )
            typer.echo(_fmt_rosetta(ref, time.perf_counter() - t0))


@app.command()
def sweep(
    pdb: Path,
    pairs: str = typer.Option("H:L"),
    shifts: str = typer.Option("0,0.05,0.1,0.15,0.2"),
    density: float = 15.0,
    radii: str = typer.Option("united", help="'united' or 'vdw'"),
):
    """Radii-shift sweep against Rosetta for the given chain pairs."""
    st = load_structure(pdb)
    for a, b in _pairs(pairs):
        ref = rosetta.run_sc(pdb, a, b, density=density)
        typer.echo(
            f"== {pdb.name} {a}|{b}  rosetta sc={ref.sc:.4f}"
            f" d={ref.distance:.4f} area={ref.area:.1f}"
        )
        sa, sb = st.chains(a), st.chains(b)
        for shift in (float(x) for x in shifts.split(",")):
            r = shape_complementarity(
                sa.coords,
                _radii(sa, radii, shift, 1.0),
                sb.coords,
                _radii(sb, radii, shift, 1.0),
                ScParams(density=density),
            )
            typer.echo(
                f"  shift {shift:+.2f}: sc={r.sc:.4f} ({r.sc - ref.sc:+.4f})"
                f" d={r.distance:.4f} ({r.distance - ref.distance:+.4f})"
                f" area={r.area:.1f} ({r.area / ref.area - 1:+.1%})"
            )


if __name__ == "__main__":
    app()
