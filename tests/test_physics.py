"""Phase 3 tests: physics backends (NumPy world, availability probe, contract).

Pure NumPy; never imports or probes ``mujoco``/``flygym``. The MuJoCo probe is
monkeypatched with an import sentinel so a regression that reaches for MuJoCo on
this CPU fails loudly instead of SIGILL-ing the interpreter.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from flappy_fly.physics import available_backends, get_backend
from flappy_fly.physics import mujoco_env
from flappy_fly.physics.mujoco_env import MujocoPhysics
from flappy_fly.physics.numpy2d import NumPyPhysics


def _hold(obs, target: float) -> np.ndarray:
    """Cheap bang-bang autopilot holding altitude near ``target``.

    A single-wing impulse is ``+3.0`` game units/s, so one flap when slow and
    falling keeps the flyer within a small band of the target.
    """
    if obs.altitude < target and obs.vy < -0.5:
        return np.array([1.0, 0.0])
    return np.zeros(2)


def test_numpy2d_available_and_contract() -> None:
    assert NumPyPhysics.available() is True

    backend = NumPyPhysics()
    assert backend.name == "numpy2d"
    assert backend.K == 4

    obs = backend.reset(seed=0)
    assert isinstance(obs.distances, np.ndarray)
    assert obs.distances.shape == (backend.K,)
    assert obs.distances.dtype == np.float64
    assert np.all(obs.distances >= 0.0) and np.all(obs.distances <= 1.0)
    assert isinstance(obs.altitude, float)
    assert isinstance(obs.vy, float)
    assert isinstance(obs.alive, bool)


def test_mujoco_available_false_without_import(monkeypatch) -> None:
    real_import = builtins.__import__
    attempts: list[str] = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"mujoco", "flygym"}:
            attempts.append(name)
            raise AssertionError(f"forbidden import attempted: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(mujoco_env, "_AVAILABILITY", None)

    assert MujocoPhysics.available() is False
    assert attempts == [], f"available() tried to import {attempts}"

    # Registry still resolves the class, but it is not advertised as available.
    assert get_backend("mujoco") is MujocoPhysics
    assert available_backends() == ["numpy2d"]
    assert attempts == []

    with pytest.raises(RuntimeError):
        MujocoPhysics()


def test_gravity_and_ground() -> None:
    backend = NumPyPhysics()
    obs = backend.reset(seed=0)
    altitudes = [obs.altitude]

    while obs.alive:
        obs = backend.step(np.zeros(2))
        altitudes.append(obs.altitude)

    diffs = np.diff(altitudes)
    assert np.all(diffs < 0.0), "altitude must strictly decrease with no flaps"
    assert obs.alive is False
    assert obs.altitude == 0.0


def test_flap_impulse_and_ceiling() -> None:
    backend = NumPyPhysics()
    obs = backend.reset(seed=0)
    start_altitude = obs.altitude

    obs = backend.step(np.array([1.0, 0.0]))
    assert obs.vy > 0.0, "a flap must give positive vertical velocity"
    assert obs.altitude > start_altitude

    # Sustained two-wing flapping drives the flyer into the ceiling.
    backend = NumPyPhysics()
    obs = backend.reset(seed=0)
    for _ in range(1000):
        obs = backend.step(np.array([1.0, 1.0]))
        if not obs.alive:
            break
    assert obs.alive is False
    assert obs.altitude == NumPyPhysics.ceiling


def test_pipe_collision_kills() -> None:
    # Collision when the flyer's altitude is held far outside the pipe gap.
    backend = NumPyPhysics()
    obs = backend.reset(seed=0)
    obs = backend.step(np.zeros(2))  # spawn the first pipe
    pipe = backend.pipes[0]
    gap = pipe.gap_center
    target = gap + 2.0 if gap <= 5.0 else gap - 2.0

    for _ in range(3000):
        if not obs.alive or pipe.x_front <= -0.3:
            break
        obs = backend.step(_hold(obs, target))

    assert obs.alive is False, "flying outside the gap must be fatal"
    assert abs(pipe.x_front) < NumPyPhysics.flyer_half_width

    # Surviving the same geometry when the gap is aligned with the flyer.
    backend = NumPyPhysics()
    obs = backend.reset(seed=0)
    obs = backend.step(np.zeros(2))
    pipe = backend.pipes[0]
    gap = pipe.gap_center

    for _ in range(3000):
        if not obs.alive or pipe.x_front <= -0.3:
            break
        obs = backend.step(_hold(obs, gap))

    assert obs.alive is True, "a gap aligned with the flyer must be survivable"
    assert pipe.x_front <= -0.3


def test_distances_monotone_as_pipe_approaches() -> None:
    backend = NumPyPhysics()
    obs = backend.reset(seed=0)
    assert np.allclose(obs.distances, 1.0), "no pipes => far field must be clear"

    obs = backend.step(np.zeros(2))  # spawn the first pipe
    pipe = backend.pipes[0]
    gap = pipe.gap_center
    target = gap + 2.0 if gap <= 5.0 else gap - 2.0

    previous_bin: int | None = None
    previous_read: float | None = None
    per_bin_first: dict[int, float] = {}
    per_bin_last: dict[int, float] = {}

    while obs.alive and pipe.x_front >= 0.0:
        obs = backend.step(_hold(obs, target))
        if pipe.x_front < 0.0:
            break
        k = int(pipe.x_front // NumPyPhysics.bin_width)
        if not 0 <= k < backend.K:
            continue
        read = obs.distances[k]
        if k == previous_bin and previous_read is not None:
            assert read <= previous_read + 1e-9, "distance increased within a bin"
        per_bin_first.setdefault(k, read)
        per_bin_last[k] = read
        previous_bin, previous_read = k, read

    assert obs.alive is False, "the tracked pipe must eventually collide"
    # Each occupied bin starts far away (~1.0) and sweeps down to adjacent (~0).
    for k in per_bin_first:
        assert per_bin_first[k] >= 0.9, f"bin {k} did not start clear"
        assert per_bin_last[k] <= 0.2, f"bin {k} did not sweep toward adjacent"
    # Bins fully traversed down to the flyer reach ~0; the near bin stops at the
    # collision x (|x_front| < half-width), so it is small but not exactly 0.
    for k in per_bin_last:
        if k > 0:
            assert per_bin_last[k] <= 0.05, f"bin {k} did not reach adjacent"
    assert obs.distances[0] <= 0.2, "the near bin must be small at collision"


def _observation_vector(obs) -> np.ndarray:
    return np.concatenate(
        [obs.distances, np.array([obs.altitude, obs.vy, float(obs.alive)])]
    )


def _run_script(seed: int, flaps_seq: list[np.ndarray]) -> np.ndarray:
    backend = NumPyPhysics()
    obs = backend.reset(seed=seed)
    rows = [_observation_vector(obs)]
    for flaps in flaps_seq:
        rows.append(_observation_vector(backend.step(flaps)))
    return np.array(rows)


def test_reset_determinism() -> None:
    # Build a deterministic flight script from a reference run.
    backend = NumPyPhysics()
    obs = backend.reset(seed=0)
    flaps_seq: list[np.ndarray] = []
    for _ in range(700):
        flaps = _hold(obs, 5.0)
        flaps_seq.append(flaps)
        obs = backend.step(flaps)

    same_a = _run_script(0, flaps_seq)
    same_b = _run_script(0, flaps_seq)
    assert np.array_equal(same_a, same_b), "same seed + same flaps must be identical"

    # Find one seed whose first gap aligns with the held altitude and one whose
    # gap does not; the obstacle readings must then differ.
    aligned_seed: int | None = None
    threat_seed: int | None = None
    for seed in range(40):
        probe = NumPyPhysics()
        probe.reset(seed=seed)
        probe.step(np.zeros(2))
        gap = probe.pipes[0].gap_center
        if aligned_seed is None and abs(gap - 5.0) <= 1.25:
            aligned_seed = seed
        if threat_seed is None and abs(gap - 5.0) > 1.5:
            threat_seed = seed

    assert aligned_seed is not None and threat_seed is not None
    other = _run_script(threat_seed, flaps_seq)
    assert not np.array_equal(same_a, other), "different seed must diverge"


def test_step_validation() -> None:
    backend = NumPyPhysics()
    backend.reset(seed=0)

    for bad in (np.zeros(3), np.zeros((2, 2)), 0.5):
        with pytest.raises(ValueError):
            backend.step(bad)
    for value in (-0.1, 1.5):
        with pytest.raises(ValueError):
            backend.step(np.array([value, 0.0]))
    with pytest.raises(ValueError):
        backend.step(np.array([np.nan, 0.0]))
