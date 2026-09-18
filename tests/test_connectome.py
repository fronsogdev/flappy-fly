"""Phase 1 tests: synthetic GF generator, fixture I/O, validation, FlyWire gate.

No network is ever used: the FlyWire branch is exercised through a fake client
and the auth gate is checked with credentials forced absent.
"""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest

import flappy_fly.connectome as connectome
from flappy_fly.connectome import (
    AuthRequiredError,
    AdjacencyData,
    _validate_adjacency,
    load_adjacency,
    make_synthetic_gf,
    population_slices,
    save_adjacency,
)


def test_synthetic_is_deterministic() -> None:
    a = make_synthetic_gf(64, seed=7)
    b = make_synthetic_gf(64, seed=7)
    assert np.array_equal(a.W, b.W)
    assert np.array_equal(a.ids, b.ids)
    assert np.array_equal(a.sign, b.sign)

    c = make_synthetic_gf(64, seed=8)
    assert not np.array_equal(a.W, c.W)
    assert not np.array_equal(a.sign, c.sign)


def test_synthetic_shapes_and_dtypes() -> None:
    data = make_synthetic_gf(64, seed=0)
    assert data.W.shape == (64, 64)
    assert data.ids.shape == (64,)
    assert data.sign.shape == (64,)
    assert data.W.dtype == np.float32
    assert data.ids.dtype == np.int64
    assert data.sign.dtype == np.int8
    assert np.unique(data.ids).size == data.ids.size
    assert np.all(data.W >= 0)
    assert set(np.unique(data.sign)).issubset({-1, 0, 1})


def test_validation_rejects_bad_data() -> None:
    good = make_synthetic_gf(64, seed=0)
    _validate_adjacency(good)  # does not raise

    n = 64
    zeros_W = np.zeros((n, n), dtype=np.float32)
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    sign = np.zeros(n, dtype=np.int8)

    with pytest.raises(ValueError):
        _validate_adjacency(AdjacencyData(W=zeros_W.astype(np.float64), ids=ids, sign=sign))
    with pytest.raises(ValueError):
        _validate_adjacency(AdjacencyData(W=zeros_W, ids=ids.astype(np.int32), sign=sign))
    with pytest.raises(ValueError):
        _validate_adjacency(AdjacencyData(W=zeros_W, ids=ids, sign=sign.astype(np.int64)))

    dup_ids = ids.copy()
    dup_ids[1] = dup_ids[0]
    with pytest.raises(ValueError):
        _validate_adjacency(AdjacencyData(W=zeros_W, ids=dup_ids, sign=sign))

    bad_sign = sign.copy()
    bad_sign[0] = 2
    with pytest.raises(ValueError):
        _validate_adjacency(AdjacencyData(W=zeros_W, ids=ids, sign=bad_sign))

    neg_W = zeros_W.copy()
    neg_W[0, 0] = -1.0
    with pytest.raises(ValueError):
        _validate_adjacency(AdjacencyData(W=neg_W, ids=ids, sign=sign))

    # N outside the GF sub-circuit scope.
    small = np.zeros((4, 4), dtype=np.float32)
    small_ids = np.arange(4, dtype=np.int64)
    small_sign = np.zeros(4, dtype=np.int8)
    with pytest.raises(ValueError):
        _validate_adjacency(AdjacencyData(W=small, ids=small_ids, sign=small_sign))


def test_population_slices_cover_all_neurons() -> None:
    for n in (20, 64, 100):
        sl = population_slices(n)
        assert set(sl) == {"sensory", "interneuron", "gf", "motor"}
        covered = np.zeros(n, dtype=bool)
        for block in sl.values():
            covered[block] = True
        assert covered.all()
        assert sl["sensory"].start == 0
        assert sl["motor"].stop == n
        assert sl["gf"].stop == sl["motor"].start
        assert sl["interneuron"].stop == sl["gf"].start


@pytest.mark.parametrize("w_max", [1.0, 2.5])
def test_normalization_bounds(w_max: float) -> None:
    data = make_synthetic_gf(64, seed=0, w_max=w_max)
    rowsums = data.W.sum(axis=1)
    assert rowsums.max() <= w_max + 1e-6
    assert np.any(rowsums > 0)


def test_biological_sanity() -> None:
    data = make_synthetic_gf(64, seed=0)
    sl = population_slices(64)
    gf, motor, sensory = sl["gf"], sl["motor"], sl["sensory"]

    gf_total = data.W[gf].sum()
    gf_to_motor = data.W[gf][:, motor].sum()
    assert gf_to_motor / gf_total > 0.6
    assert data.W[sensory][:, gf].sum() > 0
    assert data.W[gf][:, sensory].sum() == 0


def test_sign_split_and_gf_excitatory() -> None:
    data = make_synthetic_gf(64, seed=0)
    excitatory_fraction = float(np.mean(data.sign == 1))
    assert 0.6 <= excitatory_fraction <= 0.9
    assert np.all(data.sign[population_slices(64)["gf"]] == 1)


def test_fixture_roundtrip(tmp_path: Path) -> None:
    data = make_synthetic_gf(64, seed=3)
    path = save_adjacency(data, tmp_path / "roundtrip.npz")
    assert isinstance(path, Path)
    loaded = load_adjacency(source="fixture", fixture_path=path)
    assert np.array_equal(loaded.W, data.W)
    assert np.array_equal(loaded.ids, data.ids)
    assert np.array_equal(loaded.sign, data.sign)


def test_default_fixture_loads() -> None:
    data = load_adjacency(source="fixture")
    assert data.W.shape[1] == data.W.shape[0]
    assert 20 <= data.W.shape[0] <= 100
    assert data.W.dtype == np.float32
    assert data.ids.dtype == np.int64
    assert data.sign.dtype == np.int8


def _forbid_socket(*args: object, **kwargs: object) -> None:
    raise AssertionError("no network in tests")


def test_flywire_auth_gate_blocks_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(connectome, "_caveclient_authenticated", lambda: False)
    monkeypatch.setattr(socket, "create_connection", _forbid_socket)
    with pytest.raises(AuthRequiredError):
        load_adjacency(source="flywire")


def test_flywire_builds_normalized_adjacency(monkeypatch: pytest.MonkeyPatch) -> None:
    root_ids = np.arange(1000, 1030, dtype=np.int64)
    table = {
        "pre_pt_root_id": root_ids,
        "post_pt_root_id": np.roll(root_ids, -1),
    }

    class _FakeMaterialize:
        def synapse_query(self, *args: object, **kwargs: object) -> dict:
            return table

    class _FakeClient:
        materialize = _FakeMaterialize()

    monkeypatch.setattr(connectome, "_caveclient_authenticated", lambda: True)
    monkeypatch.setattr(connectome, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(socket, "create_connection", _forbid_socket)

    data = load_adjacency(source="flywire", n_neurons=20)
    assert data.W.shape == (20, 20)
    assert data.W.dtype == np.float32
    assert data.ids.dtype == np.int64
    assert data.sign.dtype == np.int8
    rowsums = data.W.sum(axis=1)
    assert rowsums.max() <= 1.0 + 1e-6
    assert np.any(rowsums > 0)
    # Sign annotation is deferred to post-migration; unknown (0) for now.
    assert np.all(data.sign == 0)
