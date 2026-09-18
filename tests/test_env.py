"""Phase 0 smoke tests: package import, import guard, and manifest extras."""

from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "flappy_fly"
FORBIDDEN_IMPORTS = {"mujoco", "flygym"}

SUBMODULES = [
    "flappy_fly",
    "flappy_fly.connectome",
    "flappy_fly.snn",
    "flappy_fly.bridge",
    "flappy_fly.physics",
    "flappy_fly.physics.base",
    "flappy_fly.physics.numpy2d",
    "flappy_fly.physics.mujoco_env",
    "flappy_fly.game",
    "flappy_fly.play",
]


def test_package_imports() -> None:
    for name in SUBMODULES:
        module = importlib.import_module(name)
        assert module is not None


def _top_level_import_nodes(tree: ast.Module) -> list[ast.stmt]:
    """Return import nodes reachable at module scope.

    Imports nested inside function or class bodies are deliberately excluded:
    the architecture requires lazy, guarded imports of the optional physics
    dependencies, while a module-scope import would execute on package load.
    """
    nodes: list[ast.stmt] = []
    stack: list[ast.stmt] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
            continue
        stack.extend(ast.iter_child_nodes(node))
    return nodes


def _imported_names(node: ast.stmt) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            names.add(alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom) and node.module:
        names.add(node.module.split(".")[0])
    return names


def test_no_top_level_mujoco_or_flygym_imports() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _top_level_import_nodes(tree):
            hit = _imported_names(node) & FORBIDDEN_IMPORTS
            if hit:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{node.lineno} imports {sorted(hit)}")
    assert not offenders, "top-level physics imports found: " + "; ".join(offenders)


def test_pyproject_declares_extras() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)

    project = data["project"]
    assert project["name"] == "flappy-fly"

    base_deps = [d.lower() for d in project["dependencies"]]
    assert any("numpy" in d for d in base_deps), base_deps
    assert any("scipy" in d for d in base_deps), base_deps

    extras = {
        name: [d.lower() for d in deps]
        for name, deps in project["optional-dependencies"].items()
    }
    assert "physics" in extras
    assert any("mujoco" in d for d in extras["physics"])
    assert any("flygym" in d for d in extras["physics"])

    assert "dev" in extras
    assert any("pytest" in d for d in extras["dev"])
