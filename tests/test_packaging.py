"""Plugin packaging metadata (pyproject.toml, requirements.txt)."""
import os
import re

import pytest

tomllib = pytest.importorskip("tomllib")
from packaging.requirements import Requirement  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _project():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]


def test_license_and_authors():
    project = _project()
    assert project["license"] == {"text": "MIT"}
    assert project["authors"] == [{"name": "Benjamin Coupry"}]


def test_pinned_dependencies():
    requirements = {Requirement(r).name: Requirement(r) for r in _project()["dependencies"]}
    assert re.search(r"LED-getter\.git@[0-9a-f]{40}$", requirements["ledgetter"].url)  # full commit sha
    assert requirements["jax"].extras == {"cuda12"} and str(requirements["jax"].specifier).startswith("==")
    with open(os.path.join(ROOT, "requirements.txt")) as f:
        lines = [line.strip() for line in f if line.strip()]
    listed = {Requirement(line).name: str(Requirement(line)) for line in lines}
    assert listed == {name: str(r) for name, r in requirements.items()}
