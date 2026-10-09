"""Contract of ledBackend.py, the only module calling JAX / LED-getter (not available in the CPU tests)."""
import inspect
import sys

import helpers

API = ("selectDevice", "loadMesh", "castRays", "calibrateLights", "solvePhotometricStereo")


def test_backend_imports_without_jax():
    """Meshroom loads every module of the plugin to find the nodes: JAX must only be imported at compute time."""
    backend = helpers.loadModule("ledBackend")
    assert "jax" not in sys.modules and "ledgetter" not in sys.modules
    assert backend.CALIBRATION_STAGES == ("directional", "punctual", "LED")


def test_fake_backend_matches_backend_api():
    """The node tests run with helpers.FakeBackend: it must expose the same functions and parameters."""
    backend = helpers.loadModule("ledBackend")
    fake = helpers.FakeBackend()
    for name in API:
        expected = list(inspect.signature(getattr(backend, name)).parameters)
        assert list(inspect.signature(getattr(fake, name)).parameters) == expected, name
