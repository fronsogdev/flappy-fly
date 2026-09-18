"""Phase 2: vectorized leaky integrate-and-fire network in pure NumPy.

Conventions
-----------
* ``W_signed[i, j]`` is the signed synapse from neuron ``i`` (presynaptic, row)
  to neuron ``j`` (postsynaptic, column), built as
  ``W_signed = W * sign[:, None]`` where ``sign[i]`` is the presynaptic sign of
  neuron ``i`` (from Phase 1). Postsynaptic aggregation is therefore a single
  ``W_signed.T @ spikes`` matmul per step: row ``i`` is neuron ``i``'s output.
* All dynamics are SI: seconds, volts, amperes. Defaults come from the frozen
  contract (``dt`` 1e-4 s, ``tau_m`` 1e-2 s, ``tau_syn`` 2e-3 s, resting
  ``v_rest`` and reset ``v_reset`` -60 mV, threshold ``v_thresh`` -50 mV,
  refractory ``t_refrac`` 2e-3 s, ``r_m`` 1e8 ohm).
* ``g_syn`` is a dimensionless conductance *scale* in units of 1 nS. The
  synaptic current is ``I_syn = g_syn * 1e-9 * (e_syn - V)``: one unit of
  ``g_syn`` is a 1 nS conductance, so a single unit of drive produces roughly a
  6 mV steady-state EPSP through ``r_m`` (``r_m * 1 nS = 0.1``, and with
  ``e_syn = 0`` the resting -60 mV settles near -54.5 mV). Keeping the scale
  separate from physical amperes makes connectome weights (order 1) directly
  usable as synaptic strengths.

Per-step order (frozen for reproducibility)
-------------------------------------------
Each :meth:`LIFNetwork.step` performs exactly the following, in order:

1. ``refractory = max(refractory - dt, 0)``.
2. Shift the spike delay ring buffer: read the delayed spikes arriving now from
   ``buffer[delay_steps]``, then shift and push the *previous* step's spikes
   into ``buffer[0]`` (at ``t = 0`` there are no previous spikes).
3. ``g_syn = g_syn * exp(-dt / tau_syn) + W_signed.T @ delayed_spikes``.
4. ``I_syn = g_syn * 1e-9 * (e_syn - V)`` (amperes).
5. Euler membrane update:
   ``V = V + (dt / tau_m) * (v_rest - V + r_m * (I_syn + I_ext))``.
6. Refractory clamp: ``V[in_refrac] = v_reset`` for ``in_refrac = refractory > 0``.
7. ``spikes = (V >= v_thresh) & ~in_refrac``; then ``V[spikes] = v_reset`` and
   ``refractory[spikes] = t_refrac``.

The two ``V`` assignments in steps 6-7 mean a spiking neuron's recorded voltage
is the reset value in the same step it fires.

Determinism
-----------
:meth:`LIFNetwork.run` calls :meth:`reset` first, so every call starts from a
clean state and is deterministic given the same ``W_signed`` and ``I_ext_seq``.
It loops over time ``T`` only (a Python time loop is allowed and required for
vectorization); there are **zero** Python loops over neurons ``N`` anywhere.
"""

from __future__ import annotations

import math

import numpy as np


class LIFNetwork:
    """Vectorized conductance-based leaky integrate-and-fire network.

    See the module docstring for the neuron ordering convention, the exact
    per-step update order, SI units, and the 1 nS conductance factor.
    """

    def __init__(self, W_signed: np.ndarray, *, dt: float = 1e-4,
                 tau_m: float = 1e-2, tau_syn: float = 2e-3,
                 v_rest: float = -0.060, v_thresh: float = -0.050,
                 v_reset: float = -0.060, t_refrac: float = 2e-3,
                 r_m: float = 1e8, e_syn: float = 0.0,
                 delay_steps: int = 1) -> None:
        W = np.asarray(W_signed, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W_signed must be square (N, N), got shape {W.shape}")
        if not np.all(np.isfinite(W)):
            raise ValueError("W_signed must contain only finite values")

        for name, value in (
            ("dt", dt),
            ("tau_m", tau_m),
            ("tau_syn", tau_syn),
            ("t_refrac", t_refrac),
            ("r_m", r_m),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite value > 0, got {value}")

        if not np.isfinite(v_thresh) or not np.isfinite(v_reset) or v_thresh <= v_reset:
            raise ValueError(
                f"v_thresh must be > v_reset, got v_thresh={v_thresh}, v_reset={v_reset}"
            )

        if isinstance(delay_steps, bool) or not isinstance(delay_steps, (int, np.integer)):
            raise ValueError(f"delay_steps must be an integer >= 0, got {delay_steps!r}")
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be an integer >= 0, got {delay_steps}")

        self.W_signed = W
        self.N = W.shape[0]
        self.dt = float(dt)
        self.tau_m = float(tau_m)
        self.tau_syn = float(tau_syn)
        self.v_rest = float(v_rest)
        self.v_thresh = float(v_thresh)
        self.v_reset = float(v_reset)
        self.t_refrac = float(t_refrac)
        self.r_m = float(r_m)
        self.e_syn = float(e_syn)
        self.delay_steps = int(delay_steps)
        # Steps a neuron stays clamped after a spike. State is tracked in
        # seconds; this is kept for introspection/documentation of the contract.
        self.n_steps_refrac = max(1, math.ceil(self.t_refrac / self.dt))

        self.V = np.empty(self.N, dtype=float)
        self.g_syn = np.empty(self.N, dtype=float)
        self.refractory = np.empty(self.N, dtype=float)
        self._spike_buffer = np.zeros((self.delay_steps + 1, self.N), dtype=bool)
        self._prev_spikes = np.zeros(self.N, dtype=bool)
        self.reset()

    def reset(self, v0: float | None = None) -> None:
        """Re-initialize all state; ``v0`` sets the initial voltage (default rest)."""
        self.V[:] = self.v_rest if v0 is None else float(v0)
        self.g_syn[:] = 0.0
        self.refractory[:] = 0.0
        self._spike_buffer[:] = False
        self._prev_spikes[:] = False

    def step(self, I_ext: np.ndarray) -> np.ndarray:
        """Advance one step; ``I_ext`` is ``(N,)`` amperes, returns ``(N,)`` bool spikes."""
        I_ext = np.asarray(I_ext, dtype=float)
        if I_ext.shape != (self.N,):
            raise ValueError(f"I_ext must have shape ({self.N},), got {I_ext.shape}")
        if not np.all(np.isfinite(I_ext)):
            raise ValueError("I_ext must contain only finite values")

        # 1. Decay the refractory countdown.
        self.refractory = np.maximum(self.refractory - self.dt, 0.0)

        # 2. Read the delayed spikes, then shift and push the previous spikes.
        delayed_spikes = self._spike_buffer[self.delay_steps].copy()
        self._spike_buffer[1:] = self._spike_buffer[:-1]
        self._spike_buffer[0] = self._prev_spikes

        # 3. Postynaptic conductance aggregation (one matmul per step).
        self.g_syn = self.g_syn * np.exp(-self.dt / self.tau_syn) + self.W_signed.T @ delayed_spikes

        # 4. Synaptic current in amperes (g_syn is in units of 1 nS).
        i_syn = self.g_syn * 1e-9 * (self.e_syn - self.V)

        # 5. Euler membrane update.
        self.V = self.V + (self.dt / self.tau_m) * (
            self.v_rest - self.V + self.r_m * (i_syn + I_ext)
        )

        # 6. Refractory clamp.
        in_refrac = self.refractory > 0
        self.V[in_refrac] = self.v_reset

        # 7. Spike detection and reset.
        spikes = (self.V >= self.v_thresh) & ~in_refrac
        self.V[spikes] = self.v_reset
        self.refractory[spikes] = self.t_refrac
        self._prev_spikes = spikes
        return spikes

    def run(self, I_ext_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run ``(T, N)`` current input; returns ``(spikes (T, N) bool, V (T, N))``.

        Calls :meth:`reset` first, so each call is independently deterministic.
        Recorded ``V`` is the post-spike-reset voltage (spike steps show
        ``v_reset``). The only Python loop is over time ``T``.
        """
        I_ext_seq = np.asarray(I_ext_seq, dtype=float)
        if I_ext_seq.ndim != 2 or I_ext_seq.shape[1] != self.N:
            raise ValueError(
                f"I_ext_seq must have shape (T, {self.N}), got {I_ext_seq.shape}"
            )
        if not np.all(np.isfinite(I_ext_seq)):
            raise ValueError("I_ext_seq must contain only finite values")

        T = I_ext_seq.shape[0]
        self.reset()
        spikes = np.zeros((T, self.N), dtype=bool)
        V = np.zeros((T, self.N), dtype=float)
        for t in range(T):
            spikes[t] = self.step(I_ext_seq[t])
            V[t] = self.V
        return spikes, V
