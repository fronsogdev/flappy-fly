#!/usr/bin/env bash
#
# Idempotent environment bootstrap for flappy-fly.
# Doubles as the migration script for a future AVX-capable machine.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> repo root: $ROOT"

# 1. Virtual environment (create only if missing).
if [ ! -d "$ROOT/.venv" ]; then
    echo "==> creating virtualenv at $ROOT/.venv"
    python3 -m venv "$ROOT/.venv"
else
    echo "==> reusing existing virtualenv at $ROOT/.venv"
fi

VENV_PY="$ROOT/.venv/bin/python"
VENV_PIP="$ROOT/.venv/bin/pip"

# 2. Editable install with dev extras. Physics extra is intentionally omitted:
#    mujoco/flygym are not installable/runnable on non-AVX hardware.
echo "==> upgrading pip"
"$VENV_PIP" install --upgrade pip

echo "==> installing editable package with [dev] extra"
"$VENV_PIP" install -e "$ROOT[dev]"

# 3. Import probe for the optional physics backend. Never let the probe abort
#    the script: on CPUs without AVX, `import mujoco` dies with SIGILL (exit
#    132). Core files are disabled inside the probe shell so nothing is left
#    behind, and the shell's own signal notice is silenced.
set +e
MUJOCO_EXIT=$(bash -c '
    ulimit -c 0 2>/dev/null || true
    "$1" -c "import mujoco"
    printf "%s" "$?"
' _ "$VENV_PY" 2>/dev/null)
set -e

if [ "$MUJOCO_EXIT" -eq 0 ]; then
    MUJOCO_STATUS="AVAILABLE (exit 0)"
else
    MUJOCO_STATUS="UNAVAILABLE (exit $MUJOCO_EXIT) — expected on non-AVX CPUs"
fi
echo "mujoco: $MUJOCO_STATUS"

# 4. Final summary.
echo "==> summary"
echo "python: $("$VENV_PY" --version 2>&1)"
if "$VENV_PY" -c "import flappy_fly" >/dev/null 2>&1; then
    echo "flappy_fly: import OK (version $("$VENV_PY" -c 'import flappy_fly; print(flappy_fly.__version__)'))"
else
    echo "flappy_fly: import FAILED"
fi
echo "backend numpy2d: AVAILABLE (pure NumPy)"
echo "backend mujoco: $MUJOCO_STATUS"

exit 0
