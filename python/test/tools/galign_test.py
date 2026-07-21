#
# Project NuriKit - Copyright 2025 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import nuri
from nuri.core import Molecule
from nuri.core.geometry import transform
from nuri.tools.galign import galign


@pytest.fixture()
def templ():
    mol = next(
        nuri.readstring(
            "smi", "O=C(C1CCC(CC(C)OC)CC1)c2cc3c(CC)ccc(C(C)C)c3cc2.Cl.[H][H]"
        )
    )
    mol.add_conf(
        [
            [-2.2395, -1.8709, 7.9552],
            [-2.7480, -1.6062, 6.8866],
            [-4.1759, -2.0027, 6.6126],
            [-5.1017, -0.8271, 6.9314],
            [-6.5514, -1.2296, 6.6533],
            [-6.9257, -2.4261, 7.5305],
            [-8.3753, -2.8286, 7.2523],
            [-8.7951, -3.9296, 8.2283],
            [-10.2124, -4.3969, 7.8911],
            [-8.7666, -3.4195, 9.5628],
            [-8.4691, -4.4037, 10.5551],
            [-5.9998, -3.6016, 7.2117],
            [-4.5502, -3.1991, 7.4898],
            [-1.9712, -0.8982, 5.8579],
            [-0.6559, -0.5350, 6.1151],
            [0.0776, 0.1400, 5.1294],
            [1.4126, 0.5242, 5.3522],
            [2.0811, 0.2130, 6.6665],
            [2.6519, -1.2059, 6.6249],
            [2.0973, 1.1779, 4.3745],
            [1.4970, 1.4690, 3.1509],
            [0.2053, 1.1173, 2.9018],
            [-0.4257, 1.4467, 1.5735],
            [0.4336, 0.8738, 0.4446],
            [-0.5247, 2.9656, 1.4185],
            [-0.5347, 0.4413, 3.8857],
            [-1.8695, 0.0601, 3.6547],
            [-2.5665, -0.5926, 4.6184],
            [7.6519, 7.9656, 15.555],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return mol


@pytest.fixture()
def query(templ: Molecule):
    mol = templ.copy()
    with mol.mutator() as mut:
        mut.mark_atom_erase(29)
        mut.mark_atom_erase(30)
    return mol


@pytest.fixture()
def xform():
    gen = np.random.default_rng(42)

    r = R.random(random_state=gen).as_matrix()
    t = gen.standard_normal(3) * 10.0

    T = np.zeros((4, 4))
    T[:3, :3] = r
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T


def test_rigid_galign(
    templ: Molecule,
    query: Molecule,
    xform: np.ndarray,
) -> None:
    templ.set_conf(transform(xform, templ.get_conf()))

    results = galign(query, templ, flexible=False)
    assert len(results) == 1

    result = results[0]
    np.testing.assert_allclose(
        result.pos,
        templ.get_conf()[: len(query)],
        atol=1e-6,
    )
    assert result.score >= 0.95


def test_flexible_galign(
    templ: Molecule,
    query: Molecule,
    xform: np.ndarray,
) -> None:
    templ.set_conf(transform(xform, templ.get_conf()))

    results = galign(
        query,
        templ,
        flexible=True,
        pool_size=2,
        sample_size=4,
        max_generations=5,
        patience=2,
        opt_ftol=0.1,
        opt_max_iters=50,
    )
    assert len(results) == 1

    result = results[0]
    np.testing.assert_allclose(
        result.pos,
        templ.get_conf()[: len(query)],
        atol=1e-6,
    )
    assert result.score >= 0.95


def test_galign_nonfinite(templ: Molecule, query: Molecule) -> None:
    with pytest.raises(ValueError, match="finite"):
        galign(query, templ, vdw_scale=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        galign(query, templ, flexible=True, opt_ftol=float("nan"))


def test_galign_sampling_args_validation(
    templ: Molecule, query: Molecule
) -> None:
    # These reach GASamplingArgs even in rigid mode, so they are validated
    # regardless of `flexible`.
    with pytest.raises(ValueError, match="pool_size must be positive"):
        galign(query, templ, pool_size=0)
    with pytest.raises(ValueError, match="patience must be positive"):
        galign(query, templ, patience=0)
    with pytest.raises(ValueError, match="rigid_max_confs must be positive"):
        galign(query, templ, rigid_max_confs=0)
    with pytest.raises(ValueError, match="rigid_min_msd must be non-negative"):
        galign(query, templ, rigid_min_msd=-1.0)

    # Lower bound is inclusive here, unlike chimera's global_ratio
    assert galign(query, templ, flexible=False, p_mutation=0.0, n_mutation=0)
