"""Phase 1: FlyWire adjacency extraction and loading (fixture-first, auth-gated).

Conventions
-----------
* ``W[i, j]`` is the synapse from neuron ``i`` to neuron ``j``. **Rows are
  presynaptic**: row ``i`` holds the output weights of neuron ``i``, so a
  postsynaptic aggregation is ``W_signed.T @ spikes`` (Phase 2).
* ``W`` is unsigned (magnitudes). Synaptic sign is carried separately in
  ``sign``, the presynaptic sign of neuron ``i``: ``+1`` excitatory, ``-1``
  inhibitory, ``0`` unknown.
* ``ids`` are FlyWire root ids; in fixture/synthetic mode they are synthetic
  placeholders. ``W`` is ``float32``, ``ids`` is ``int64``, ``sign`` is
  ``int8``.

Population layout (see :func:`population_slices`)
--------------------------------------------------
Neurons are laid out in contiguous blocks, in this order::

    [ sensory | interneurons | giant fiber (GF) | motor ]

with ``n_sensory = max(2, round(0.15 n))``,
``n_motor = max(2, round(0.20 n))`` and ``n_gf = max(2, round(0.05 n))``.
Interneurons fill whatever remains in the middle. The GF block sits
immediately before the motor block, mirroring the escape pathway
sensory -> GF -> motor.

Normalization
-------------
Rows are presynaptic output. Each row is normalized by its own sum, floors at
one so that all-zero rows stay zero, then scaled to ``w_max``::

    W[i, :] /= max(rowsum(W[i, :]), 1);  W *= w_max

Every row sum is therefore ``<= w_max`` (exactly ``w_max`` for nonzero rows).

Autapses
--------
Self-synapses (``W[i, i]``) are excluded: :func:`make_synthetic_gf` zeroes the
diagonal before normalization, so a neuron never projects onto itself.

Fixture-first policy
--------------------
The default ``source`` is ``"fixture"``: a committed
``data/fixtures/*.npz`` file, or the newest one when no path is given. Tests
and CI never touch the network. ``source="flywire"`` is opt-in, gated on local
CAVE credentials, and raises :class:`AuthRequiredError` immediately when they
are absent -- before ``caveclient`` is imported and before any network call.
Real FlyWire sign annotation is deferred until credentials exist;
:func:`_sign_from_annotations` returns unknown (``0``) for now and is the one
place to fill in later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


class AuthRequiredError(RuntimeError):
    """Raised when a live FlyWire query is attempted without credentials."""


@dataclass(frozen=True)
class AdjacencyData:
    W: np.ndarray      # (N, N) float32, W[i,j] = synapse i->j (row = presynaptic), row-normalized, unsigned magnitudes
    ids: np.ndarray    # (N,) int64, neuron/root ids (fixture: synthetic ids)
    sign: np.ndarray   # (N,) int8: +1 excitatory, -1 inhibitory, 0 unknown (presynaptic sign of neuron i)


#: Repo-relative default fixture directory (fixture-first policy).
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"

#: Synthetic ids start here so they never collide with small fixture ids.
SYNTHETIC_ID_BASE = 1_000_000_000

#: GF sub-circuit scope, enforced by :func:`_validate_adjacency`.
MIN_NEURONS = 20
MAX_NEURONS = 100


def population_slices(n: int) -> dict[str, slice]:
    """Return the contiguous population blocks of an ``n``-neuron layout.

    Keys are ``"sensory"``, ``"interneuron"``, ``"gf"`` and ``"motor"``, in
    that anatomical order. Sizes are ``max(2, round(0.15 n))`` sensory,
    ``max(2, round(0.20 n))`` motor and ``max(2, round(0.05 n))`` GF;
    interneurons fill the remaining middle block.
    """
    if n < 4:
        raise ValueError(f"n must be >= 4 to define populations, got {n}")
    n_sensory = max(2, round(0.15 * n))
    n_motor = max(2, round(0.20 * n))
    n_gf = max(2, round(0.05 * n))
    return {
        "sensory": slice(0, n_sensory),
        "interneuron": slice(n_sensory, n - n_motor - n_gf),
        "gf": slice(n - n_motor - n_gf, n - n_motor),
        "motor": slice(n - n_motor, n),
    }


def _assign_sign(sign: np.ndarray, block: slice, p_exc: float, rng: np.random.Generator) -> None:
    """Set ``block`` of a default-inhibitory sign vector to +1 with prob ``p_exc``."""
    n = block.stop - block.start
    sign[block] = np.where(rng.random(n) < p_exc, np.int8(1), np.int8(-1))


def _overlay(W: np.ndarray, rows: slice, cols: slice, rng: np.random.Generator,
             p: float, lo: float, hi: float) -> None:
    """Overwrite a random ``p`` fraction of a block with uniform ``[lo, hi)`` weights."""
    block = W[rows, cols]
    mask = rng.random(block.shape) < p
    W[rows, cols] = np.where(mask, rng.uniform(lo, hi, size=block.shape), block)


def _normalize(W: np.ndarray, w_max: float) -> np.ndarray:
    """Row-normalize presynaptic output and scale to ``w_max`` (returns float32)."""
    rowsums = W.sum(axis=1, keepdims=True)
    W = W / np.maximum(rowsums, 1.0) * w_max
    return W.astype(np.float32)


def make_synthetic_gf(
    n_neurons: int = 64,
    seed: int = 0,
    w_max: float = 1.0,
    p_base: float = 0.08,
) -> AdjacencyData:
    """Build a deterministic GF-like synthetic connectome.

    Structure: a sparse Erdős–Rényi base over all ordered pairs (probability
    ``p_base``, uniform ``[0, 1]`` weights), overlaid with the escape pathway:
    sensory->GF strong (weights 3-6, p=0.5), GF->motor strong (4-8, p=0.7),
    sensory->motor weak/sparse (0.2-1, p=0.15). GF->sensory is explicitly
    silenced. Autapses (self-synapses) are excluded: the matrix diagonal is
    zeroed before normalization, so ``W[i, i] == 0`` for every neuron. Signs
    are presynaptic: sensory ~90% excitatory, GF 100%, interneurons ~70%,
    motor ~80%. Fully deterministic from ``seed`` via
    ``np.random.default_rng``; ids are ``SYNTHETIC_ID_BASE + arange(N)``.
    """
    if n_neurons < MIN_NEURONS or n_neurons > MAX_NEURONS:
        raise ValueError(f"n_neurons must be in [{MIN_NEURONS}, {MAX_NEURONS}], got {n_neurons}")
    rng = np.random.default_rng(seed)
    sl = population_slices(n_neurons)

    W = (rng.random((n_neurons, n_neurons)) < p_base) * rng.random((n_neurons, n_neurons))
    _overlay(W, sl["sensory"], sl["gf"], rng, p=0.5, lo=3.0, hi=6.0)
    _overlay(W, sl["gf"], sl["motor"], rng, p=0.7, lo=4.0, hi=8.0)
    _overlay(W, sl["sensory"], sl["motor"], rng, p=0.15, lo=0.2, hi=1.0)
    W[sl["gf"], sl["sensory"]] = 0.0  # GF does not project back to sensory
    W[np.diag_indices(n_neurons)] = 0.0  # no autapses (self-synapses)

    sign = np.full(n_neurons, -1, dtype=np.int8)
    _assign_sign(sign, sl["sensory"], 0.9, rng)
    _assign_sign(sign, sl["interneuron"], 0.7, rng)
    _assign_sign(sign, sl["gf"], 1.0, rng)
    _assign_sign(sign, sl["motor"], 0.8, rng)

    data = AdjacencyData(
        W=_normalize(W, w_max),
        ids=(SYNTHETIC_ID_BASE + np.arange(n_neurons, dtype=np.int64)),
        sign=sign,
    )
    _validate_adjacency(data, row_w_max=w_max)
    return data


def _validate_adjacency(data: AdjacencyData, row_w_max: float = 1.0) -> None:
    """Raise :class:`ValueError` if ``data`` violates the adjacency contract.

    ``row_w_max`` bounds every presynaptic row sum: ``rowsum(W[i, :]) <=
    row_w_max + 1e-6``. The synthetic and FlyWire paths pass their own
    ``w_max``; the fixture path passes the value stored in the npz (default
    ``1.0`` for old fixtures that predate the key).
    """
    W, ids, sign = data.W, data.ids, data.sign
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square (N, N), got shape {W.shape}")
    n = W.shape[0]
    if ids.shape != (n,):
        raise ValueError(f"ids must have shape ({n},), got {ids.shape}")
    if sign.shape != (n,):
        raise ValueError(f"sign must have shape ({n},), got {sign.shape}")
    if not (MIN_NEURONS <= n <= MAX_NEURONS):
        raise ValueError(f"N must be in [{MIN_NEURONS}, {MAX_NEURONS}], got {n}")
    if W.dtype != np.float32:
        raise ValueError(f"W must be float32, got {W.dtype}")
    if ids.dtype != np.int64:
        raise ValueError(f"ids must be int64, got {ids.dtype}")
    if sign.dtype != np.int8:
        raise ValueError(f"sign must be int8, got {sign.dtype}")
    if not np.all(np.isfinite(W)):
        raise ValueError("W must contain only finite values")
    if np.any(W < 0):
        raise ValueError("W must be non-negative (unsigned magnitudes)")
    if np.unique(ids).size != ids.size:
        raise ValueError("ids must be unique")
    if np.any((sign < -1) | (sign > 1)):
        raise ValueError("sign entries must be in {-1, 0, 1}")
    if (
        not isinstance(row_w_max, (int, float, np.integer, np.floating))
        or not np.isfinite(row_w_max)
        or row_w_max < 0
    ):
        raise ValueError(
            f"invalid w_max in fixture metadata: expected a finite value >= 0, "
            f"got {row_w_max!r}"
        )
    rowsums = W.sum(axis=1)
    if np.any(rowsums > row_w_max + 1e-6):
        raise ValueError(
            f"row sums must be <= w_max ({row_w_max}) + 1e-6, got max {rowsums.max()}"
        )


def save_adjacency(data: AdjacencyData, path: Path, w_max: float = 1.0) -> Path:
    """Write ``W``, ``ids``, ``sign`` and ``w_max`` to ``path`` (``np.savez``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=data.W, ids=data.ids, sign=data.sign, w_max=np.float64(w_max))
    return path


def _default_fixture_path() -> Path:
    """Return the newest ``data/fixtures/*.npz`` fixture, or raise if none exist."""
    candidates = sorted(FIXTURE_DIR.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"no adjacency fixtures found in {FIXTURE_DIR}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _load_fixture(path: Path) -> AdjacencyData:
    with np.load(path) as npz:
        # ``w_max`` is absent in fixtures saved before it was recorded.
        w_max = float(npz["w_max"]) if "w_max" in npz else 1.0
        data = AdjacencyData(
            W=np.asarray(npz["W"], dtype=np.float32).copy(),
            ids=np.asarray(npz["ids"], dtype=np.int64).copy(),
            sign=np.asarray(npz["sign"], dtype=np.int8).copy(),
        )
    _validate_adjacency(data, row_w_max=w_max)
    return data


def _caveclient_authenticated() -> bool:
    """True only if local CAVE credentials exist. Performs no network I/O."""
    import os

    if os.environ.get("CAVECLIENT_TOKEN"):
        return True
    home = Path.home()
    return (home / ".cloudvolume" / "secrets.json").is_file() or (
        home / ".caveclient" / "auth.json"
    ).is_file()


def _get_client():
    """Return an authenticated ``caveclient.CAVEclient`` (lazy import, no top-level)."""
    import caveclient  # local import: only reachable once credentials are present

    return caveclient.CAVEclient()


def _fetch_synapse_table(client):
    """Fetch the synapse table to map.

    Placeholder for the real bounded query: post-migration this restricts the
    lookup to GF sub-circuit members (cell-type/candidate-id query) instead of
    asking the client for its default table.
    """
    return client.materialize.synapse_query()


def _table_root_ids(table) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(pre, post)`` root-id arrays from a synapse table."""
    pre = np.asarray(table["pre_pt_root_id"], dtype=np.int64)
    post = np.asarray(table["post_pt_root_id"], dtype=np.int64)
    return pre, post


def _map_ids(values: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map root ids to row/column indices of sorted ``selected``; return (idx, found)."""
    pos = np.searchsorted(selected, values)
    pos = np.clip(pos, 0, selected.size - 1)
    return pos, selected[pos] == values


def _sign_from_annotations(client, selected: np.ndarray) -> np.ndarray:
    """Presynaptic signs for ``selected``; unknown (0) until real annotations.

    Real FlyWire neurotransmitter/sign annotation is deferred until credentials
    exist. When available, fill this in from the client's annotation tables and
    keep the surrounding code in :func:`_load_flywire` unchanged.
    """
    return np.zeros(selected.size, dtype=np.int8)


def _load_flywire(n_neurons: int, w_max: float = 1.0) -> AdjacencyData:
    """Build adjacency from a live FlyWire query (caller has authenticated)."""
    client = _get_client()
    pre, post = _table_root_ids(_fetch_synapse_table(client))

    all_ids = np.unique(np.concatenate([pre, post]))
    if all_ids.size < n_neurons:
        raise ValueError(
            f"flywire table exposes {all_ids.size} root ids, need >= {n_neurons}"
        )
    # Selection placeholder: the frozen signature has no id argument, so take
    # the lowest root ids. Real GF-membership selection is refined post-migration.
    selected = all_ids[:n_neurons]

    ip, pre_ok = _map_ids(pre, selected)
    jp, post_ok = _map_ids(post, selected)
    valid = pre_ok & post_ok
    W = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    np.add.at(W, (ip[valid], jp[valid]), 1.0)

    data = AdjacencyData(
        W=_normalize(W, w_max),
        ids=selected.astype(np.int64),
        sign=_sign_from_annotations(client, selected).astype(np.int8),
    )
    _validate_adjacency(data, row_w_max=w_max)
    return data


def load_adjacency(
    source: Literal["fixture", "synthetic", "flywire"] = "fixture",
    n_neurons: int = 64,               # GF sub-circuit scope: 20–100
    fixture_path: Path | None = None,  # default: newest data/fixtures/*.npz
    seed: int | None = 0,              # synthetic determinism
) -> AdjacencyData:
    """Load adjacency data from a fixture, a synthetic generator, or FlyWire.

    ``flywire`` raises :class:`AuthRequiredError` immediately when local
    credentials are absent, before importing ``caveclient`` or opening any
    network connection. ``seed=None`` falls back to ``0`` for deterministic
    synthetic output.
    """
    if source == "fixture":
        return _load_fixture(Path(fixture_path) if fixture_path is not None else _default_fixture_path())
    if source == "synthetic":
        return make_synthetic_gf(n_neurons, seed=0 if seed is None else seed)
    if source == "flywire":
        if not _caveclient_authenticated():
            raise AuthRequiredError(
                "FlyWire credentials not found; run in fixture/synthetic mode or "
                "configure ~/.cloudvolume/secrets.json, ~/.caveclient/auth.json, "
                "or CAVECLIENT_TOKEN"
            )
        return _load_flywire(n_neurons)
    raise ValueError(f"unknown source {source!r}; expected fixture, synthetic or flywire")
