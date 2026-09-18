"""Phase 3: ``Obs`` dataclass and the ``PhysicsBackend`` protocol.

Every physics backend speaks the same small language so that the game loop
(Phase 4) stays backend-agnostic:

* ``reset(seed)`` starts a fresh episode and returns the first observation.
* ``step(flaps)`` advances the world by one physics tick and returns the next
  observation. ``flaps`` is ``(2,)`` in ``[0, 1]``: one command per wing.
* ``Obs.distances`` is the only sensory channel, a ``(K,)`` vector of forward
  range samples with a fixed convention:

  * ``1.0`` — clear / no obstacle within the sampled range (maximum range),
  * ``0.0`` — obstacle surface adjacent to the flyer.

  The convention is deliberately "distance-like": the bridge converts it to a
  depolarizing drive with ``threat = 1 - distances``, so a smaller reading
  produces a larger current.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Obs:
    """Observation returned by every physics backend.

    Attributes
    ----------
    distances:
        ``(K,)`` float array of forward range samples normalized to ``[0, 1]``.
        ``1.0`` means clear at maximum range, ``0.0`` means an obstacle surface
        is adjacent to the flyer. ``K`` is backend-defined (``4`` for both
        shipped backends).
    altitude:
        Flyer altitude in game units; ``0.0`` is the ground.
    vy:
        Vertical velocity in game units per second.
    alive:
        ``False`` once the episode has ended (collision, ground, or ceiling).
        Termination is sticky: once ``False`` it stays ``False`` until reset.
    """

    distances: np.ndarray
    altitude: float
    vy: float
    alive: bool


class PhysicsBackend(Protocol):
    """Structural protocol implemented by every physics backend.

    ``name`` and ``K`` are class-level attributes, not per-instance state, so a
    backend class can advertise its identity and sensor width without being
    instantiated (important because instantiating the MuJoCo backend SIGILLs on
    non-AVX CPUs).
    """

    name: str
    K: int

    def reset(self, seed: int | None = None) -> Obs:
        """Start a new episode; returns the first observation."""
        ...

    def step(self, flaps: np.ndarray) -> Obs:
        """Advance one physics tick; ``flaps`` is ``(2,)`` in ``[0, 1]``."""
        ...

    @staticmethod
    def available() -> bool:
        """Whether this backend can run on the current machine."""
        ...
