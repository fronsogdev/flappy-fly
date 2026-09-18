"""Phase 3: physics backend registry and availability probes.

Usage
-----
Always probe before instantiating, and prefer selecting through the registry::

    from flappy_fly.physics import available_backends, get_backend

    names = available_backends()          # e.g. ["numpy2d"]
    backend_cls = get_backend(names[0])   # -> NumPyPhysics
    backend = backend_cls()               # safe: availability already checked

``get_backend`` only resolves the *class*; it does not instantiate or probe.
``MujocoPhysics.__init__`` raises ``RuntimeError`` on hardware that cannot host
MuJoCo, so ``available()`` (directly or via ``available_backends``) is the
guard.
"""

from __future__ import annotations

from .base import Obs, PhysicsBackend
from .mujoco_env import MujocoPhysics
from .numpy2d import NumPyPhysics

#: Registry of backend name -> implementation class, in preference order.
BACKENDS: dict[str, type[PhysicsBackend]] = {
    "numpy2d": NumPyPhysics,
    "mujoco": MujocoPhysics,
}

__all__ = ["BACKENDS", "Obs", "PhysicsBackend", "available_backends", "get_backend"]


def get_backend(name: str) -> type[PhysicsBackend]:
    """Return the backend class registered under ``name``.

    Raises ``ValueError`` listing the valid names for an unknown key.
    """
    try:
        return BACKENDS[name]
    except KeyError:
        valid = ", ".join(sorted(BACKENDS))
        raise ValueError(f"unknown physics backend {name!r}; valid names: {valid}") from None


def available_backends() -> list[str]:
    """Return the names of backends whose ``available()`` probe is ``True``."""
    return [name for name, cls in BACKENDS.items() if cls.available()]
