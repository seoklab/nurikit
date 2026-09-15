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


def _model_batch(models, atoms=1, first_x="1"):
    header = """data_batch
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
"""
    rows = []
    for model in range(1, models + 1):
        for atom in range(1, atoms + 1):
            x = first_x if model == 1 else "1"
            rows.append(
                f"{(model - 1) * atoms + atom} C C{atom} ALA A 1 "
                f"{x} 2 3 {model}\n"
            )
    return header + "".join(rows)


def test_shared_reader_final_batch():
    text = _model_batch(32, atoms=32)
    expected = Counter(_consume(nuri.readstring("cif", text, sanitize=False)))
    reader = nuri.readstring("cif", text, sanitize=False)
    assert _concurrent_results([reader] * 4) == expected


def test_batch_outlives_consuming_thread():
    reader = nuri.readstring("cif", _model_batch(8), sanitize=False)
    completed = SimpleQueue()

    def take_one():
        try:
            completed.put(next(reader))
        except BaseException as exc:
            completed.put(exc)

    worker = Thread(target=take_one, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    first = completed.get_nowait()
    if isinstance(first, BaseException):
        raise first
    assert first.props["model"] == "1"
    assert [mol.props["model"] for mol in reader] == [
        str(i) for i in range(2, 9)
    ]


@pytest.mark.parametrize("skip_on_error", [False, True])
def test_invalid_model_preserves_batch(skip_on_error):
    reader = nuri.readstring(
        "cif",
        _model_batch(4, first_x="nan"),
        sanitize=False,
        skip_on_error=skip_on_error,
    )
    if not skip_on_error:
        with pytest.raises(ValueError, match="non-finite"):
            next(reader)
    assert [mol.props["model"] for mol in reader] == ["2", "3", "4"]

    def fresh_reader():
        return nuri.readstring(
            "cif",
            _model_batch(4, first_x="nan"),
            sanitize=False,
            skip_on_error=skip_on_error,
        )

    expected = Counter(_consume(fresh_reader()))
    shared = fresh_reader()
    assert _concurrent_results([shared] * 4) == expected


@pytest.mark.parametrize("skip_on_error", [False, True])
def test_empty_batches_before_final_batch(skip_on_error):
    text = "".join(f"data_empty{i}\n_x.y 1\n" for i in range(20))
    text += _model_batch(8)
    expected = Counter(_consume(nuri.readstring("cif", text, sanitize=False)))
    reader = nuri.readstring(
        "cif",
        text,
        sanitize=False,
        skip_on_error=skip_on_error,
    )
    assert _concurrent_results([reader] * 4) == expected
