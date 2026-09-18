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


def test_no_autapses() -> None:
    for n, seed in ((20, 0), (64, 1), (64, 7), (100, 42)):
        W = make_synthetic_gf(n, seed=seed).W
        assert np.array_equal(W.diagonal(), np.zeros(n, dtype=W.dtype))

    fixture = load_adjacency(source="fixture")
    assert np.array_equal(
        fixture.W.diagonal(), np.zeros(fixture.W.shape[0], dtype=fixture.W.dtype)
    )


def test_fixture_row_sum_validation(tmp_path: Path) -> None:
    # The committed fixture still loads under the new row-sum check.
    assert load_adjacency(source="fixture").W.shape[0] > 0

    data = make_synthetic_gf(64, seed=0)
    bad_W = data.W.copy()
    bad_W[0, 1] = 5.0  # row 0 now sums well above w_max=1.0
    assert bad_W[0].sum() > 1.0 + 1e-6
    bad = AdjacencyData(W=bad_W, ids=data.ids, sign=data.sign)
    bad_path = save_adjacency(bad, tmp_path / "malformed.npz", w_max=1.0)

    with pytest.raises(ValueError):
        load_adjacency(source="fixture", fixture_path=bad_path)

    # An npz without a w_max key is treated as w_max=1.0 (backward compatible).
    np.savez(
        tmp_path / "legacy.npz",
        W=data.W,
        ids=data.ids,
        sign=data.sign,
    )
    legacy = load_adjacency(source="fixture", fixture_path=tmp_path / "legacy.npz")
    assert np.array_equal(legacy.W, data.W)


def test_fixture_rejects_invalid_w_max(tmp_path: Path) -> None:
    data = make_synthetic_gf(64, seed=0)

    # NaN would bypass `rowsums > row_w_max`; it must be rejected explicitly.
    nan_path = save_adjacency(data, tmp_path / "nan_w_max.npz", w_max=float("nan"))
    with pytest.raises(ValueError):
        load_adjacency(source="fixture", fixture_path=nan_path)

    # Negative bounds are nonsensical and must be rejected too.
    neg_path = save_adjacency(data, tmp_path / "neg_w_max.npz", w_max=-1.0)
    with pytest.raises(ValueError):
        load_adjacency(source="fixture", fixture_path=neg_path)

    # The programmatic path (no npz involved) must enforce the same rule.
    with pytest.raises(ValueError):
        _validate_adjacency(data, row_w_max=float("nan"))

    # Sanity: a valid bound still loads when row sums respect it.
    valid_data = make_synthetic_gf(64, seed=0, w_max=0.5)
    valid_path = save_adjacency(valid_data, tmp_path / "valid_w_max.npz", w_max=0.5)
    loaded = load_adjacency(source="fixture", fixture_path=valid_path)
    assert loaded.W.sum(axis=1).max() <= 0.5 + 1e-6


@pytest.mark.parametrize(
    "bad_w_max",
    [
        np.array([0.5]),            # genuine array, not an np.savez 0-d scalar
        np.complex128(1.0 + 0.5j),  # complex is not a real scalar
        "0.5",                      # string metadata
    ],
)
def test_fixture_rejects_non_scalar_w_max(tmp_path: Path, bad_w_max: object) -> None:
    data = make_synthetic_gf(64, seed=0)
    path = tmp_path / "bad_w_max.npz"
    np.savez(path, W=data.W, ids=data.ids, sign=data.sign, w_max=bad_w_max)

    # The fixture loader must surface the clear metadata ValueError, not the
    # TypeError that a pre-validation float() coercion would raise.
    with pytest.raises(ValueError, match="invalid w_max in fixture metadata"):
        load_adjacency(source="fixture", fixture_path=path)


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
