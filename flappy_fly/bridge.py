"""Phase 3: sensory-motor bridge between physics observations and the LIF SNN.

Closed-loop data flow::

    Obs.distances (K,)
        -> sensors_to_I_ext(distances, G)      # threat -> current
        -> LIFNetwork.step(I_ext)              # (N,) spikes
        -> spikes_to_flaps(spikes, motor_ids, latch)   # (2,) wing commands
        -> PhysicsBackend.step(flaps)

Sensory encoding
----------------
``sensors_to_I_ext`` turns the distance convention (``1.0`` clear, ``0.0``
adjacent) into a depolarizing drive. ``threat = clip(1 - distances, 0, 1)`` so a
closer obstacle yields a larger threat; the ``(N, K)`` gain matrix ``G`` maps
that threat onto the ``N`` sensory rows of the network. A positive ``G`` entry
therefore means "neuron ``n`` is excited by obstacles in sensor bin ``k``". An
optional ``baseline`` adds a tonic drive.

Motor decoding
--------------
``spikes_to_flaps`` reads two populations of motor neuron indices (row 0 left,
row 1 right). A spike in a population requests a flap on that side, but a
:class:`FlapLatch` holds each side down for ``t_hold`` seconds after a flap so
the wings cannot flap faster than the refractory window: the caller advances the
latch once per SNN tick with ``state.advance(dt)``.
"""

from __future__ import annotations

import numpy as np

#: Default per-side flap refractory window in seconds (frozen contract).
T_HOLD_DEFAULT = 2e-2


class FlapLatch:
    """Per-side refractory timer enforcing a minimum interval between flaps.

    ``timer[i]`` counts down to zero; ``is_open[i]`` is True when the side is
    free to flap. ``spikes_to_flaps`` reloads ``timer[i] = t_hold`` on a trigger.
    The caller owns the clock: advance the latch every SNN tick.
    """

    def __init__(self, t_hold: float = T_HOLD_DEFAULT) -> None:
        if not np.isfinite(t_hold) or t_hold < 0.0:
            raise ValueError(f"t_hold must be a finite value >= 0, got {t_hold}")
        self.t_hold = float(t_hold)
        self.timer = np.zeros(2, dtype=float)

    def advance(self, dt: float) -> None:
        """Decrement both timers by ``dt`` seconds, saturating at zero."""
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError(f"dt must be a finite value >= 0, got {dt}")
        self.timer = np.maximum(self.timer - dt, 0.0)

    @property
    def is_open(self) -> np.ndarray:
        """``(2,)`` bool: True where the side's refractory timer has elapsed."""
        return self.timer <= 0.0


def sensors_to_I_ext(
    distances: np.ndarray,
    G: np.ndarray,
    *,
    gain: float = 1.0,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Map ``(K,)`` distances through ``(N, K)`` gains to ``(N,)`` currents.

    ``I_ext = gain * (G @ threat)`` with ``threat = clip(1 - distances, 0, 1)``.
    A smaller distance (closer obstacle) produces a larger ``I_ext`` wherever the
    corresponding ``G`` row is positive. ``baseline``, if given, is added
    elementwise.
    """
    distances = np.asarray(distances, dtype=float)
    G = np.asarray(G, dtype=float)
    if distances.ndim != 1:
        raise ValueError(f"distances must have shape (K,), got {distances.shape}")
    if G.ndim != 2 or G.shape[1] != distances.shape[0]:
        raise ValueError(
            f"G must have shape (N, {distances.shape[0]}), got {G.shape}"
        )
    if not np.all(np.isfinite(distances)):
        raise ValueError("distances must contain only finite values")
    if not np.all(np.isfinite(G)):
        raise ValueError("G must contain only finite values")
    if not np.isfinite(gain):
        raise ValueError(f"gain must be finite, got {gain}")

    threat = np.clip(1.0 - distances, 0.0, 1.0)
    I_ext = gain * (G @ threat)

    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float)
        if baseline.shape != (G.shape[0],):
            raise ValueError(
                f"baseline must have shape ({G.shape[0]},), got {baseline.shape}"
            )
        if not np.all(np.isfinite(baseline)):
            raise ValueError("baseline must contain only finite values")
        I_ext = I_ext + baseline

    return I_ext


def spikes_to_flaps(
    spikes: np.ndarray,
    motor_ids: np.ndarray,
    state: FlapLatch,
) -> np.ndarray:
    """Decode ``(N,)`` motor spikes into ``(2,)`` wing commands in ``{0, 1}``.

    ``motor_ids`` has shape ``(2, M)``: row 0 lists left motor neuron indices,
    row 1 right. A side flaps when any of its motor neurons spiked *and* its
    latch is open; triggering reloads that side's hold timer. Indices must lie
    within ``[0, N)``.
    """
    spikes = np.asarray(spikes)
    if spikes.ndim != 1:
        raise ValueError(f"spikes must have shape (N,), got {spikes.shape}")
    n = spikes.shape[0]

    motor_ids = np.asarray(motor_ids)
    if motor_ids.ndim != 2 or motor_ids.shape[0] != 2:
        raise ValueError(f"motor_ids must have shape (2, M), got {motor_ids.shape}")
    if not np.issubdtype(motor_ids.dtype, np.integer):
        raise ValueError("motor_ids must have an integer dtype")
    if motor_ids.size and (np.any(motor_ids < 0) or np.any(motor_ids >= n)):
        raise ValueError(f"motor_ids must lie in [0, {n})")

    flaps = np.zeros(2, dtype=float)
    for side in range(2):
        ids = motor_ids[side]
        triggered = bool(spikes[ids].any()) if ids.size else False
        if triggered and state.is_open[side]:
            flaps[side] = 1.0
            state.timer[side] = state.t_hold
    return flaps
