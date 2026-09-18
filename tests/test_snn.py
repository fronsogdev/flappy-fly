"""Phase 2 tests: vectorized LIF engine (thresholds, delays, vectorization, biology).

Pure NumPy, no network, no mujoco/flygym. The hand-built networks use the
frozen contract defaults (``dt=1e-4``, ``r_m=1e8``, ``v_thresh-v_rest=10 mV``),
so ``I_th = (v_thresh - v_rest) / r_m = 1e-10`` A.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest

from flappy_fly.connectome import make_synthetic_gf, population_slices
from flappy_fly.snn import LIFNetwork

REPO_ROOT = Path(__file__).resolve().parents[1]
SNN_PATH = REPO_ROOT / "flappy_fly" / "snn.py"

#: Threshold current: drive needed to reach ``v_thresh`` from ``v_rest`` at rest.
I_TH = (-0.050 - -0.060) / 1e8

#: GF connectome weight multiplier tuned so the synthetic sensory->GF->motor
#: pathway fires with a clear causal order (see ``test_gf_escape_latency``).
W_SCALE_GF = 20.0


def _first_spike_step(spikes: np.ndarray, neuron: int) -> int | None:
    """Return the first step index at which ``neuron`` spikes, or ``None``."""
    idx = np.nonzero(spikes[:, neuron])[0]
    return int(idx[0]) if idx.size else None


def _first_population_step(spikes: np.ndarray, block: slice) -> int | None:
    """Return the first step index at which any neuron in ``block`` spikes."""
    steps = np.nonzero(spikes[:, block].any(axis=1))[0]
    return int(steps[0]) if steps.size else None


def test_single_neuron_threshold() -> None:
    v_rest, v_reset, v_thresh = -0.060, -0.060, -0.050
    net = LIFNetwork(np.zeros((1, 1)))
    T = 2000  # 200 ms
    I_ext = np.full((T, 1), 1.5 * I_TH)
    spikes, V = net.run(I_ext)

    assert not spikes[0, 0], "neuron must not spike on the first subthreshold step"
    idx = np.nonzero(spikes[:, 0])[0]
    assert idx.size >= 2, "constant 1.5x threshold drive must evoke repeated spikes"

    first = int(idx[0])
    # Euler charging from rest reaches threshold at ~11 ms; allow 3-20 ms.
    assert 30 <= first <= 200, f"first spike at step {first} outside expected window"
    assert V[first, 0] == v_reset, "recorded spike-step voltage must be the reset value"

    # No second spike while the refractory clamp is active.
    n_refrac = net.n_steps_refrac
    assert not spikes[first + 1 : first + n_refrac, 0].any()

    # And the first ISI respects the refractory period.
    isi = (int(idx[1]) - first) * net.dt
    assert isi >= net.t_refrac - net.dt, f"ISI {isi} shorter than refractory {net.t_refrac}"


def test_dt_convergence() -> None:
    rng = np.random.default_rng(0)
    N = 8
    W_signed = np.zeros((N, N))
    for i in range(4):  # excitatory rows 0-3
        for j in range(N):
            if i != j and rng.random() < 0.5:
                W_signed[i, j] = rng.uniform(2.0, 6.0)
    for i in range(4, 8):  # inhibitory rows 4-7
        for j in range(N):
            if i != j and rng.random() < 0.5:
                W_signed[i, j] = -rng.uniform(2.0, 6.0)

    total = 0.05  # 50 ms of identical simulated time in both runs

    def first_spike_times(dt: float) -> list[float | None]:
        T = int(round(total / dt))
        I_ext = np.zeros((T, N))
        I_ext[:, :4] = 1.5 * I_TH
        spikes, _ = LIFNetwork(W_signed, dt=dt).run(I_ext)
        times: list[float | None] = []
        for n in range(N):
            idx = np.nonzero(spikes[:, n])[0]
            times.append(float(idx[0]) * dt if idx.size else None)
        return times

    coarse = first_spike_times(1e-4)
    fine = first_spike_times(5e-5)
    common = [(a, b) for a, b in zip(coarse, fine) if a is not None and b is not None]
    assert common, "no neuron spikes in both runs; stimulus is not driving the network"

    diffs = np.array([abs(a - b) for a, b in common])
    assert diffs.max() < 0.5e-3, f"max first-spike shift {diffs.max() * 1e3:.3f} ms >= 0.5 ms"
    assert diffs.mean() < 0.5e-3, f"mean first-spike shift {diffs.mean() * 1e3:.3f} ms >= 0.5 ms"


def test_step_has_no_loops() -> None:
    tree = ast.parse(SNN_PATH.read_text(encoding="utf-8"), filename=str(SNN_PATH))
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in {"step", "run"}
    }
    assert "step" in methods and "run" in methods

    step_loops = [
        node
        for node in ast.walk(methods["step"])
        if isinstance(node, (ast.For, ast.While))
    ]
    assert not step_loops, "step() must be loop-free over neurons and time"

    run_whiles = [n for n in ast.walk(methods["run"]) if isinstance(n, ast.While)]
    run_fors = [n for n in ast.walk(methods["run"]) if isinstance(n, ast.For)]
    assert not run_whiles, "run() may only loop over time, not while-loop"
    assert len(run_fors) == 1, "run() must contain exactly one loop, over time T"


def test_gf_escape_latency() -> None:
    data = make_synthetic_gf(64, seed=0)
    sl = population_slices(64)
    W_signed = data.W * data.sign[:, None] * W_SCALE_GF

    net = LIFNetwork(W_signed, dt=1e-4)
    T = 600  # 60 ms
    I_ext = np.zeros((T, 64))
    onset = 200  # 20 ms
    I_ext[onset:, sl["sensory"]] = 2.0 * I_TH
    spikes, _ = net.run(I_ext)

    assert not spikes[:onset].any(), "network fired before the stimulus onset"

    sensory = _first_population_step(spikes, sl["sensory"])
    gf = _first_population_step(spikes, sl["gf"])
    motor = _first_population_step(spikes, sl["motor"])
    assert sensory is not None and gf is not None and motor is not None

    # (a) sensory responds within 10 ms of onset.
    assert (sensory - onset) * net.dt <= 10e-3
    # (b) causal order: sensory -> GF -> motor.
    assert sensory < gf < motor
    # (c) GF -> motor monosynaptic latency <= 5 ms.
    motor_steps = np.nonzero(spikes[:, sl["motor"]].any(axis=1))[0]
    after_gf = motor_steps[motor_steps >= gf]
    assert after_gf.size
    latency = (int(after_gf[0]) - gf) * net.dt
    assert latency <= 5e-3, f"GF->motor latency {latency * 1e3:.3f} ms > 5 ms"


def test_inhibitory_suppression() -> None:
    w = 15.0
    W_signed = np.zeros((3, 3))
    W_signed[0, 2] = w    # neuron 0 tonically drives neuron 2
    W_signed[1, 2] = -w   # neuron 1 inhibits neuron 2

    T = 1500  # 150 ms
    onset, offset = 500, 1000  # inhibition window: 50-100 ms

    def spikes_in_window(with_inhibition: bool) -> int:
        I_ext = np.zeros((T, 3))
        I_ext[:, 0] = 3.0 * I_TH
        if with_inhibition:
            I_ext[onset:offset, 1] = 3.0 * I_TH
        spikes, _ = LIFNetwork(W_signed).run(I_ext)
        return int(spikes[onset:offset, 2].sum())

    without = spikes_in_window(False)
    with_inh = spikes_in_window(True)
    assert without > 0, "neuron 2 must fire regularly without inhibition"
    assert with_inh < without, f"inhibition failed to suppress: {with_inh} vs {without}"


def test_refractory_isi() -> None:
    net = LIFNetwork(np.zeros((1, 1)))
    T = 2000  # 200 ms
    I_ext = np.full((T, 1), 5.0 * I_TH)
    spikes, _ = net.run(I_ext)

    idx = np.nonzero(spikes[:, 0])[0]
    assert idx.size >= 2, "strong drive should elicit many spikes"
    isis = np.diff(idx) * net.dt
    assert np.all(isis >= net.t_refrac - net.dt), (
        f"minimum ISI {isis.min() * 1e3:.3f} ms < t_refrac - dt"
    )


def test_determinism() -> None:
    rng = np.random.default_rng(7)
    N = 16
    W_signed = rng.normal(0.0, 0.05, (N, N))
    I_ext_seq = rng.normal(0.0, 1.0e-10, (300, N))

    net_a = LIFNetwork(W_signed)
    net_b = LIFNetwork(W_signed)
    spikes_a, V_a = net_a.run(I_ext_seq)
    spikes_b, V_b = net_b.run(I_ext_seq)
    assert np.array_equal(spikes_a, spikes_b)
    assert np.array_equal(V_a, V_b)

    # A second call on the same network must reset and reproduce the first.
    spikes_c, V_c = net_a.run(I_ext_seq)
    assert np.array_equal(spikes_a, spikes_c)
    assert np.array_equal(V_a, V_c)


def test_delay_ring_buffer() -> None:
    dt = 1e-4
    T = 2000  # 200 ms

    def first_spikes(delay_steps: int) -> tuple[int, int]:
        W_signed = np.zeros((2, 2))
        W_signed[0, 1] = 30.0  # strong monosynaptic drive, no feedback
        net = LIFNetwork(W_signed, dt=dt, delay_steps=delay_steps)
        I_ext = np.zeros((T, 2))
        I_ext[:, 0] = 2.0 * I_TH
        spikes, _ = net.run(I_ext)
        a = _first_spike_step(spikes, 0)
        b = _first_spike_step(spikes, 1)
        assert a is not None and b is not None
        return a, b

    a3, b3 = first_spikes(3)
    a0, b0 = first_spikes(0)
    offset3 = (b3 - a3) * dt
    offset0 = (b0 - a0) * dt

    assert offset3 >= 3 * dt, f"delay shorter than 3 steps: {offset3 * 1e3:.3f} ms"
    assert offset3 <= 3 * dt + 2e-3, f"delay too long: {offset3 * 1e3:.3f} ms"
    assert offset0 <= 2e-3, f"zero-delay offset too long: {offset0 * 1e3:.3f} ms"
    assert offset0 < offset3, "delay_steps=0 must reach the target sooner than delay_steps=3"


def test_performance() -> None:
    N, T = 100, 10_000
    rng = np.random.default_rng(0)
    W_signed = rng.normal(0.0, 0.03, (N, N))
    bias = rng.uniform(0.6 * I_TH, 1.4 * I_TH, size=N)
    I_ext_seq = bias[None, :] + rng.normal(0.0, 0.05 * I_TH, (T, N))

    net = LIFNetwork(W_signed)
    t0 = time.perf_counter()
    spikes, V = net.run(I_ext_seq)
    elapsed = time.perf_counter() - t0
    print(f"\nLIF run N={N} T={T}: {elapsed:.3f}s ({spikes.sum()} spikes)")

    assert spikes.shape == (T, N)
    assert V.shape == (T, N)
    assert np.isfinite(V).all()
    assert elapsed < 10.0, f"run took {elapsed:.3f}s, exceeds 10s bound"


def test_validation() -> None:
    good = np.zeros((2, 2))

    # W_signed shape / finiteness.
    with pytest.raises(ValueError):
        LIFNetwork(np.zeros(3))
    with pytest.raises(ValueError):
        LIFNetwork(np.zeros((2, 3)))
    non_finite = good.copy()
    non_finite[0, 0] = np.inf
    with pytest.raises(ValueError):
        LIFNetwork(non_finite)

    # Positive dynamic parameters.
    for kwargs in ({"dt": 0.0}, {"dt": -1e-4}, {"tau_m": 0.0}, {"tau_syn": -1.0},
                   {"t_refrac": 0.0}, {"r_m": -1.0}):
        with pytest.raises(ValueError):
            LIFNetwork(good, **kwargs)

    # Threshold must exceed reset.
    with pytest.raises(ValueError):
        LIFNetwork(good, v_thresh=-0.060, v_reset=-0.060)

    # delay_steps must be a non-negative integer.
    with pytest.raises(ValueError):
        LIFNetwork(good, delay_steps=-1)
    with pytest.raises(ValueError):
        LIFNetwork(good, delay_steps=1.5)

    # I_ext / I_ext_seq shape and finiteness.
    net = LIFNetwork(good)
    with pytest.raises(ValueError):
        net.step(np.zeros(3))
    with pytest.raises(ValueError):
        net.step(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        net.run(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        net.run(np.array([[np.inf, 0.0]]))
