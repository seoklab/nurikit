#
# Project NuriKit - Copyright 2026 SNU Compbio Lab.
# SPDX-License-Identifier: Apache-2.0
#

from collections import Counter
from queue import SimpleQueue
from threading import Barrier, Thread

import pytest

import nuri


def _consume(reader):
    results = []
    while True:
        try:
            mol = next(reader)
        except StopIteration:
            break
        except ValueError as exc:
            results.append(("error", str(exc)))
        else:
            results.append(
                (
                    "mol",
                    mol.name,
                    mol.props.get("model", ""),
                    tuple(atom.atomic_number for atom in mol),
                    mol.num_bonds(),
                )
            )
    for _ in range(2):
        with pytest.raises(StopIteration):
            next(reader)
    return results


def _concurrent_results(readers):
    barrier = Barrier(len(readers), timeout=10)
    completed = SimpleQueue()

    def consume(reader):
        try:
            barrier.wait()
            completed.put(_consume(reader))
        except BaseException as exc:
            completed.put(exc)

    workers = [
        Thread(target=consume, args=(reader,), daemon=True)
        for reader in readers
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert not worker.is_alive(), "reader worker did not finish"
    results = []
    for _ in workers:
        result = completed.get_nowait()
        if isinstance(result, BaseException):
            raise result
        results.extend(result)
    return Counter(results)


@pytest.fixture(params=["smi", "sdf", "mmcif"])
def threaded_input(request):
    fmt = request.param
    if fmt == "smi":
        text = "".join(
            f"CCO mol{i}\nerror\nC(C)(C)(C)(C)(C)\n" for i in range(100)
        )
    elif fmt == "sdf":
        text = "".join(
            f"mol{i}\n\n\n  0  0  0     0  0  0  0  0  0999 V2000\n"
            "M  END\n$$$$\n$$$$\n"
            for i in range(100)
        )
    else:
        text = "".join(
            f"""data_mol{i}
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
1 C CA ALA A 1 1 2 3 1
2 N N ALA A 1 4 5 6 2
data_empty{i}
_x.y 1
data_bad{i}
loop_
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
1 N N ALA A invalid 1 2 3
"""
            for i in range(40)
        )
    return fmt, text


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("skip_on_error", [False, True])
def test_concurrent_readers(threaded_input, shared, skip_on_error):
    fmt, text = threaded_input

    def reader():
        return nuri.readstring(
            fmt, text, sanitize=fmt != "mmcif", skip_on_error=skip_on_error
        )

    expected = Counter(_consume(reader()))
    assert any(key[0] == "mol" for key in expected)
    assert any(key[0] == "error" for key in expected) != skip_on_error
    if shared:
        readers = [reader()] * 4
    else:
        readers = [reader() for _ in range(4)]
        expected = Counter({key: count * 4 for key, count in expected.items()})
    assert _concurrent_results(readers) == expected
