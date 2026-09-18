"""Phase 3: ``MujocoPhysics`` — import-guarded MuJoCo backend skeleton.

**This module never imports MuJoCo or FlyGym at import time.** Both are optional
physics extras, and on a CPU without the ``avx`` flag ``import mujoco`` is not
merely unavailable but fatal (SIGILL / core dump). :meth:`MujocoPhysics.available`
therefore probes the CPU *first* and only attempts a lazy import when the
hardware can host it.

The class is a deliberately minimal skeleton: the XML model, reset/step wiring,
and the ``K = 4`` distance convention mirror :class:`NumPyPhysics`, but the
physics is only exercised after migrating to an AVX-capable machine. It still
satisfies the :class:`~flappy_fly.physics.base.PhysicsBackend` protocol
structurally so the game loop can select it by name.
"""

from __future__ import annotations

import platform
import sys

import numpy as np

from .base import Obs

#: Cached availability probe. ``None`` means "not probed yet"; once set, later
#: calls short-circuit and (on non-AVX hardware) never reach an import.
_AVAILABILITY: bool | None = None

#: Minimal inline model: a vertical slide joint for the flyer, two hinge wing
#: actuators, a static ground plane, and two mocap pipe boxes. Positions are in
#: game units and the timestep matches ``NumPyPhysics``.
_XML = """
<mujoco model="flappy_fly">
  <option timestep="0.005" gravity="0 0 -9.8"/>
  <worldbody>
    <geom name="ground" type="plane" size="50 1 0.1" pos="0 0 0"
          rgba="0.3 0.3 0.3 1"/>
    <body name="flyer" pos="0 0 5">
      <joint name="flyer_y" type="slide" axis="0 0 1" range="0 10"
             limited="true" damping="0"/>
      <geom name="body" type="box" size="0.3 0.1 0.1"
            rgba="1 0.7 0 1"/>
      <body name="wing_l" pos="0 -0.1 0">
        <joint name="wing_l" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="wing_l_geom" type="box" size="0.15 0.3 0.02"
              rgba="0.6 0.8 1 1"/>
      </body>
      <body name="wing_r" pos="0 0.1 0">
        <joint name="wing_r" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="wing_r_geom" type="box" size="0.15 0.3 0.02"
              rgba="0.6 0.8 1 1"/>
      </body>
    </body>
    <body name="pipe_0" mocap="true" pos="12 0 0">
      <geom name="pipe_0_geom" type="box" size="0.2 0.05 2" pos="0 0 0"
            rgba="0.1 0.8 0.1 1"/>
    </body>
    <body name="pipe_1" mocap="true" pos="12 0 0">
      <geom name="pipe_1_geom" type="box" size="0.2 0.05 2" pos="0 0 0"
            rgba="0.1 0.8 0.1 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="wing_l_act" joint="wing_l" kp="10"/>
    <position name="wing_r_act" joint="wing_r" kp="10"/>
  </actuator>
</mujoco>
"""


def _cpu_supports_avx() -> bool:
    """Return whether this CPU can host MuJoCo.

    Linux x86_64 is checked against ``/proc/cpuinfo`` flags; macOS arm64 is
    assumed capable. Any other platform is treated as unsupported.
    """
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return True
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    key, _, value = line.partition(":")
                    if key.strip() in {"flags", "Features"}:
                        return "avx" in value.split()
        except OSError:
            return False
        return False
    return False


class MujocoPhysics:
    """MuJoCo-backed physics; unavailable on non-AVX CPUs (see module docstring)."""

    name = "mujoco"
    K = 4

    def __init__(self) -> None:
        if not self.available():
            raise RuntimeError("MujocoPhysics unavailable on this CPU")

        # Only reached on AVX-capable hardware.
        import mujoco

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_string(_XML)
        self._data = mujoco.MjData(self._model)
        self._dt = float(self._model.opt.timestep)
        self._rng = np.random.default_rng(0)
        self._pipes: list[tuple[float, float]] = []
        self._alive = False
        self.reset()

    @staticmethod
    def available() -> bool:
        """Probe CPU support, then lazily import MuJoCo; result is cached."""
        global _AVAILABILITY
        if _AVAILABILITY is None:
            if not _cpu_supports_avx():
                _AVAILABILITY = False
                return _AVAILABILITY
            try:
                import mujoco  # noqa: F401
            except Exception:
                _AVAILABILITY = False
            else:
                _AVAILABILITY = True
        return _AVAILABILITY

    def reset(self, seed: int | None = None) -> Obs:
        """Reset the MuJoCo state and obstacle bookkeeping."""
        import mujoco  # local: keeps the import guard intact

        self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self._model, self._data)
        # Slide joint starts at mid-altitude.
        self._data.qpos[0] = 5.0
        self._data.qvel[0] = 0.0
        self._pipes = []
        self._alive = True
        mujoco.mj_forward(self._model, self._data)
        return self._obs()

    def step(self, flaps: np.ndarray) -> Obs:
        """Advance one MuJoCo timestep (mirrors ``NumPyPhysics`` semantics)."""
        import mujoco  # local: keeps the import guard intact

        flaps = np.asarray(flaps, dtype=float)
        if flaps.shape != (2,):
            raise ValueError(f"flaps must have shape (2,), got {flaps.shape}")
        if not np.all(np.isfinite(flaps)) or np.any(flaps < 0.0) or np.any(flaps > 1.0):
            raise ValueError("flaps must be finite and lie in [0, 1]")

        # Wing hinges track the commanded flap amplitude.
        self._data.ctrl[0] = float(flaps[0])
        self._data.ctrl[1] = float(flaps[1])
        mujoco.mj_step(self._model, self._data)

        altitude = float(self._data.qpos[0])
        vy = float(self._data.qvel[0])

        # Scroll, cull, and spawn pipes exactly like NumPyPhysics.
        self._pipes = [(x - 2.0 * self._dt, gap) for x, gap in self._pipes]
        self._pipes = [p for p in self._pipes if p[0] >= -2.0]
        if not self._pipes or (12.0 - max(x for x, _ in self._pipes)) > 6.0:
            gap_center = float(self._rng.uniform(1.5, 8.5))
            self._pipes.append((12.0, gap_center))
        # Push pipe positions into the mocap bodies (visual/debug mirror).
        for i, (x_front, _gap) in enumerate(self._pipes[:2]):
            self._data.mocap_pos[i] = (x_front, 0.0, 0.0)

        # Collision / ground / ceiling.
        for x_front, gap_center in self._pipes:
            if abs(x_front) < 0.3:
                if not (gap_center - 1.25 <= altitude <= gap_center + 1.25):
                    self._alive = False
                    break
        if altitude <= 0.0 or altitude >= 10.0:
            self._alive = False

        return self._obs()

    def _obs(self) -> Obs:
        return Obs(
            distances=self._distances(),
            altitude=float(self._data.qpos[0]),
            vy=float(self._data.qvel[0]),
            alive=bool(self._alive),
        )

    def _distances(self) -> np.ndarray:
        distances = np.ones(self.K, dtype=float)
        altitude = float(self._data.qpos[0])
        for k in range(self.K):
            bin_lo = k * 2.0
            bin_hi = bin_lo + 2.0
            nearest: float | None = None
            for x_front, gap_center in self._pipes:
                if not (bin_lo <= x_front < bin_hi):
                    continue
                if gap_center - 1.25 <= altitude <= gap_center + 1.25:
                    continue
                if nearest is None or x_front < nearest:
                    nearest = x_front
            if nearest is not None:
                distances[k] = float(np.clip((nearest - bin_lo) / 2.0, 0.0, 1.0))
        return distances
