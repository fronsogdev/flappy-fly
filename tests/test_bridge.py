"""Phase 3 tests: sensory-motor bridge (monotonic encoding, flap latch, agnostic loop).

Pure NumPy; the bridge only sees plain arrays and the protocols, so it is tested
independently of any concrete physics backend.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from flappy_fly.bridge import FlapLatch, sensors_to_I_ext, spikes_to_flaps
from flappy_fly.connectome import make_synthetic_gf, population_slices
from flappy_fly.physics.base import Obs, PhysicsBackend
from flappy_fly.physics.mujoco_env import MujocoPhysics
from flappy_fly.physics.numpy2d import NumPyPhysics
from flappy_fly.snn import LIFNetwork

#: Same convention as ``tests/test_snn.py``: the synthetic GF connectome needs
#: this multiplier for the sensory -> GF -> motor pathway to fire.
W_SCALE_GF = 20.0


def test_sensors_to_i_ext_monotonic() -> None:
    rng = np.random.default_rng(0)
    n, k = 6, 4
    G = rng.uniform(0.1, 1.0, (n, k))  # strictly positive gains

    near = np.full(k, 0.1)
    far = np.full(k, 0.9)
    i_near = sensors_to_I_ext(near, G)
    i_far = sensors_to_I_ext(far, G)

    assert np.all(i_near >= i_far), "closer obstacles must not reduce excitation"
    assert np.all(i_near > i_far), "positive gains must strictly increase drive"

    baseline = rng.normal(size=n)
    with_baseline = sensors_to_I_ext(near, G, baseline=baseline)
    assert np.allclose(with_baseline, i_near + baseline), "baseline must add exactly"

    assert np.allclose(sensors_to_I_ext(near, G, gain=2.0), 2.0 * i_near)

    # Distances outside [0, 1] clamp into the threat window.
    assert np.allclose(sensors_to_I_ext(np.array([-1.0, 0.0, 1.0, 2.0]), G),
                       sensors_to_I_ext(np.array([0.0, 0.0, 1.0, 1.0]), G))


def test_sensors_shape_validation() -> None:
    good_distances = np.zeros(4)
    good_G = np.zeros((3, 4))

    with pytest.raises(ValueError):
        sensors_to_I_ext(np.zeros((1, 4)), good_G)
    with pytest.raises(ValueError):
        sensors_to_I_ext(good_distances, np.zeros((3, 5)))
    with pytest.raises(ValueError):
        sensors_to_I_ext(good_distances, np.zeros(4))
    with pytest.raises(ValueError):
        sensors_to_I_ext(good_distances, good_G, baseline=np.zeros(4))
    with pytest.raises(ValueError):
        sensors_to_I_ext(np.array([np.nan, 0.0, 0.0, 0.0]), good_G)
    with pytest.raises(ValueError):
        sensors_to_I_ext(good_distances, np.full((3, 4), np.inf))
    with pytest.raises(ValueError):
        sensors_to_I_ext(good_distances, good_G, gain=np.nan)


def test_flap_latch_hold_down() -> None:
    state = FlapLatch()
    assert state.t_hold == pytest.approx(2e-2)
    assert np.all(state.is_open)

    motor_ids = np.array([[0, 1], [2, 3]])
    left = np.zeros(4, dtype=bool)
    left[0] = True
    right = np.zeros(4, dtype=bool)
    right[2] = True

    assert np.array_equal(spikes_to_flaps(left, motor_ids, state), [1.0, 0.0])
    # The same side is held down on an immediate repeat...
    assert np.array_equal(spikes_to_flaps(left, motor_ids, state), [0.0, 0.0])
    # ...while the other side remains independent.
    assert np.array_equal(spikes_to_flaps(right, motor_ids, state), [0.0, 1.0])
    assert np.array_equal(spikes_to_flaps(left, motor_ids, state), [0.0, 0.0])
    assert np.array_equal(spikes_to_flaps(right, motor_ids, state), [0.0, 0.0])

    state.advance(2e-2)
    assert np.all(state.is_open)
    assert np.array_equal(spikes_to_flaps(left, motor_ids, state), [1.0, 0.0])

    # No spike => no flap, even with an open latch.
    state.advance(1.0)
    assert np.array_equal(spikes_to_flaps(np.zeros(4, dtype=bool), motor_ids, state),
                          [0.0, 0.0])


class _FakeBackend:
    """Minimal in-memory backend; records every flap vector it receives."""

    name = "fake"
    K = 4

    def __init__(self, obs: Obs) -> None:
        self._obs = obs
        self.received: list[np.ndarray] = []

    @staticmethod
    def available() -> bool:
        return True

    def reset(self, seed: int | None = None) -> Obs:
        return self._obs

    def step(self, flaps: np.ndarray) -> Obs:
        self.received.append(np.array(flaps, dtype=float, copy=True))
        return self._obs


def test_backend_agnostic_bridge() -> None:
    obs = Obs(distances=np.array([0.1, 0.4, 0.7, 1.0]), altitude=5.0, vy=0.0,
              alive=True)
    backend = _FakeBackend(obs)
    assert backend.name == "fake" and backend.K == 4

    G = np.ones((4, 4))
    motor_ids = np.array([[0], [1]])
    latch = FlapLatch()
    spike_seq = [
        np.array([True, False, False, False]),
        np.array([False, False, False, False]),
        np.array([False, True, False, False]),
        np.array([True, True, False, False]),
    ]

    expected: list[np.ndarray] = []
    current = backend.reset(seed=0)
    for spikes in spike_seq:
        sensors_to_I_ext(current.distances, G)  # bridge consumes only the Obs
        flaps = spikes_to_flaps(spikes, motor_ids, latch)
        current = backend.step(flaps)
        expected.append(flaps)

    assert len(backend.received) == len(expected)
    for got, want in zip(backend.received, expected):
        assert np.array_equal(got, want), "backend must receive the bridge's flaps"
    assert np.array_equal(expected[0], [1.0, 0.0])
    assert np.array_equal(expected[1], [0.0, 0.0])
    assert np.array_equal(expected[2], [0.0, 1.0])


def _run_closed_loop(
    backend: PhysicsBackend,
    net: LIFNetwork,
    G: np.ndarray,
    motor_ids: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive ``backend`` through the bridge for ``steps`` SNN ticks.

    Touches only the ``PhysicsBackend`` protocol (``reset``/``step``/``K``), so a
    concrete backend and a delegated mock are exercised identically. Returns the
    ``(steps, 2)`` flap action sequence and the ``(steps, K + 3)`` observation
    sequence (distances, altitude, vy, alive) it produced.
    """
    latch = FlapLatch()
    net.reset()
    obs = backend.reset(seed=0)
    flaps_seq = np.zeros((steps, 2), dtype=float)
    obs_seq = np.zeros((steps, backend.K + 3), dtype=float)
    for t in range(steps):
        I_ext = sensors_to_I_ext(obs.distances, G)
        spikes = net.step(I_ext)
        flaps = spikes_to_flaps(spikes, motor_ids, latch)
        latch.advance(net.dt)
        flaps_seq[t] = flaps
        obs_seq[t, : backend.K] = obs.distances
        obs_seq[t, backend.K] = obs.altitude
        obs_seq[t, backend.K + 1] = obs.vy
        obs_seq[t, backend.K + 2] = float(obs.alive)
        obs = backend.step(flaps)
    return flaps_seq, obs_seq


def test_backend_swap_action_sequence() -> None:
    """Same SNN, seed and bridge must behave identically through both backends.

    ``MujocoPhysics`` cannot run on this CPU, so its interface is exercised with
    a ``MagicMock(spec=MujocoPhysics)`` whose ``reset``/``step`` delegate to a
    real ``NumPyPhysics``. That tests backend-agnosticism of the game loop -- the
    action sequence and observations must not depend on the backend -- without
    importing mujoco.
    """
    data = make_synthetic_gf(64, seed=0)
    sl = population_slices(64)
    W_signed = data.W * data.sign[:, None] * W_SCALE_GF
    net = LIFNetwork(W_signed)

    G = np.zeros((net.N, NumPyPhysics.K), dtype=float)
    # ~2x threshold current per unit threat, so a near obstacle drives the
    # sensory block and the sensory -> GF -> motor path emits flaps.
    G[sl["sensory"], :] = 2.0e-10

    motor_idx = np.arange(sl["motor"].start, sl["motor"].stop)
    half = motor_idx.size // 2  # floor-split so both rows share a width
    motor_ids = np.array([motor_idx[:half], motor_idx[half : 2 * half]])

    # 2000 SNN ticks (~0.2 s of SNN time); each call advances the physics tick.
    steps = 2000

    flaps_numpy, obs_numpy = _run_closed_loop(NumPyPhysics(), net, G, motor_ids, steps)

    inner = NumPyPhysics()
    backend_mujoco = MagicMock(spec=MujocoPhysics)
    backend_mujoco.name = "mujoco"
    backend_mujoco.K = inner.K
    backend_mujoco.reset.side_effect = inner.reset
    backend_mujoco.step.side_effect = inner.step
    flaps_mujoco, obs_mujoco = _run_closed_loop(
        backend_mujoco, net, G, motor_ids, steps
    )

    # (a) identical wing-command sequence...
    assert np.array_equal(flaps_numpy, flaps_mujoco)
    # (b) ...and identical observation sequence.
    assert np.array_equal(obs_numpy, obs_mujoco)
    # (c) the closed loop actually produced output on both backends.
    assert int(flaps_numpy.any(axis=1).sum()) >= 1
    assert int(flaps_mujoco.any(axis=1).sum()) >= 1
    # (d) sanity: the real MuJoCo backend is unavailable on this CPU.
    assert MujocoPhysics.available() is False
