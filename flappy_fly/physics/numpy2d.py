"""Phase 3: ``NumPyPhysics`` — pure-NumPy 2D point-mass flight backend.

This is the backend that runs on every machine (including the SSE4.2-only
development box). It is a side-scrolling Flappy-Bird world:

* The flyer sits at ``x = 0`` and only moves vertically (altitude ``y``,
  vertical velocity ``vy``). ``y = 0`` is the ground, ``y = CEILING`` is the
  roof.
* Gravity pulls ``vy`` down by ``GRAVITY * dt_phys`` every tick.
* A flap command is an impulse: ``vy += FLAP_STRENGTH * (flaps[0] + flaps[1])``
  (both wings together give twice the impulse of one), clamped to
  ``|vy| <= VY_MAX``.
* Pipes scroll leftwards from ``SPAWN_X``. A new pipe is spawned whenever the
  rightmost pipe has travelled more than ``PIPE_SPACING`` away from the spawn
  column, so obstacle gaps are roughly evenly spaced. Each pipe is a vertical
  pair of surfaces with a gap of ``GAP_SIZE`` centred on a random altitude.
* The episode ends on ground contact, ceiling contact, or when a pipe's front
  face overlaps the flyer's x-extent while the flyer is outside that pipe's gap.

``Obs.distances`` (``K = 4``) partitions the forward range into 2-unit bins
``[k * BIN_WIDTH, (k + 1) * BIN_WIDTH)``. For each bin the nearest threatening
pipe front face is normalized within the bin: a surface at the far edge reads
``1.0`` (clear) and a surface at the near edge reads ``0.0`` (adjacent)::

    distances[k] = clip((x_front - bin_lo) / BIN_WIDTH, 0, 1)

A pipe only threatens when the flyer's current altitude is *outside* its gap; a
gap aligned with the flyer is not counted. When a pipe crosses a bin boundary
the newly occupied bin restarts its own reading (per-bin range sensor handoff),
so monotonicity holds while the pipe stays within one bin.

Determinism: the only randomness is the pipe gap draw from
``np.random.default_rng(seed)``. Same seed + same flap sequence ⇒ identical
observation sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Obs

#: World geometry and dynamics (game units and seconds).
CEILING = 10.0
GRAVITY = 9.8
FLAP_STRENGTH = 3.0
DT_PHYS = 5e-3
SPAWN_X = 12.0
PIPE_SPACING = 6.0
GAP_SIZE = 2.5
PIPE_SPEED = 2.0
FLYER_HALF_WIDTH = 0.3
DESPAWN_X = -2.0
VY_MAX = 8.0
BIN_WIDTH = 2.0


@dataclass
class Pipe:
    """A scrolling obstacle column: front face x, gap centre, gap size."""

    x_front: float
    gap_center: float
    gap_size: float


class NumPyPhysics:
    """2D point-mass flight backend; see the module docstring for the model."""

    name = "numpy2d"
    K = 4

    #: Constants are class attributes so callers/tests can introspect or
    #: override them without touching the code.
    ceiling = CEILING
    gravity = GRAVITY
    flap_strength = FLAP_STRENGTH
    dt_phys = DT_PHYS
    spawn_x = SPAWN_X
    pipe_spacing = PIPE_SPACING
    gap_size = GAP_SIZE
    pipe_speed = PIPE_SPEED
    flyer_half_width = FLYER_HALF_WIDTH
    despawn_x = DESPAWN_X
    vy_max = VY_MAX
    bin_width = BIN_WIDTH

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.y = 5.0
        self.vy = 0.0
        self.alive = False
        self.pipes: list[Pipe] = []
        self.reset()

    @staticmethod
    def available() -> bool:
        """Pure NumPy — always available."""
        return True

    def reset(self, seed: int | None = None) -> Obs:
        """Begin a new episode at mid-altitude with no obstacles."""
        self.rng = np.random.default_rng(seed)
        self.y = 0.5 * self.ceiling
        self.vy = 0.0
        self.alive = True
        self.pipes = []
        return self._obs()

    def step(self, flaps: np.ndarray) -> Obs:
        """Advance exactly one ``dt_phys`` tick; returns the new observation."""
        flaps = self._validate_flaps(flaps)

        # ``alive`` is only ever cleared below, never set back to True, so the
        # terminal flag is sticky while the world keeps scrolling.
        # Impulse from the wings, then gravity; semi-implicit Euler.
        flap_sum = float(flaps[0] + flaps[1])
        self.vy = float(
            np.clip(
                self.vy + self.flap_strength * flap_sum - self.gravity * self.dt_phys,
                -self.vy_max,
                self.vy_max,
            )
        )
        self.y += self.vy * self.dt_phys

        # Scroll, cull, and spawn obstacles.
        for pipe in self.pipes:
            pipe.x_front -= self.pipe_speed * self.dt_phys
        self.pipes = [p for p in self.pipes if p.x_front >= self.despawn_x]
        self._maybe_spawn()

        # Collision with pipe bodies.
        for pipe in self.pipes:
            if abs(pipe.x_front) < self.flyer_half_width:
                gap_lo = pipe.gap_center - 0.5 * pipe.gap_size
                gap_hi = pipe.gap_center + 0.5 * pipe.gap_size
                if not (gap_lo <= self.y <= gap_hi):
                    self.alive = False
                    break

        # Ground and ceiling.
        if self.y <= 0.0:
            self.y = 0.0
            self.alive = False
        elif self.y >= self.ceiling:
            self.y = self.ceiling
            self.alive = False

        return self._obs()

    # -- internals ---------------------------------------------------------

    def _maybe_spawn(self) -> None:
        rightmost = max((p.x_front for p in self.pipes), default=None)
        if rightmost is None or (self.spawn_x - rightmost) > self.pipe_spacing:
            gap_center = float(self.rng.uniform(1.5, self.ceiling - 1.5))
            self.pipes.append(Pipe(self.spawn_x, gap_center, self.gap_size))

    def _validate_flaps(self, flaps: np.ndarray) -> np.ndarray:
        flaps = np.asarray(flaps, dtype=float)
        if flaps.shape != (2,):
            raise ValueError(f"flaps must have shape (2,), got {flaps.shape}")
        if not np.all(np.isfinite(flaps)):
            raise ValueError("flaps must contain only finite values")
        if np.any(flaps < 0.0) or np.any(flaps > 1.0):
            raise ValueError("flaps must lie in [0, 1]")
        return flaps

    def _obs(self) -> Obs:
        return Obs(
            distances=self._distances(),
            altitude=float(self.y),
            vy=float(self.vy),
            alive=bool(self.alive),
        )

    def _distances(self) -> np.ndarray:
        """``(K,)`` per-bin normalized range to the nearest threatening pipe."""
        distances = np.ones(self.K, dtype=float)
        for k in range(self.K):
            bin_lo = k * self.bin_width
            bin_hi = bin_lo + self.bin_width
            nearest: float | None = None
            for pipe in self.pipes:
                if not (bin_lo <= pipe.x_front < bin_hi):
                    continue
                gap_lo = pipe.gap_center - 0.5 * pipe.gap_size
                gap_hi = pipe.gap_center + 0.5 * pipe.gap_size
                if gap_lo <= self.y <= gap_hi:
                    # Gap aligned with the flyer: not a threat.
                    continue
                if nearest is None or pipe.x_front < nearest:
                    nearest = pipe.x_front
            if nearest is not None:
                read = (nearest - bin_lo) / self.bin_width
                distances[k] = float(np.clip(read, 0.0, 1.0))
        return distances
