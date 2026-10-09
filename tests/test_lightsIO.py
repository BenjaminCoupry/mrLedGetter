"""Unit tests of lightsIO.py: the lights JSON written by LedGetterLights and read by LedGetterPS (CPU only)."""
import json

import numpy as np
import pytest

import helpers
from helpers import LOGGER

lc = helpers.loadModule("ledCommon")
lio = helpers.loadModule("lightsIO")


def _camera(angle=0.3, center=(1.0, 2.0, 3.0), width=40, height=30):
    K = [[50.0, 0, 20.0], [0, 50.0, 15.0], [0, 0, 1]]
    return lc.Camera(K, helpers.rotationZ(angle), center, width, height)


def _values(nbLights=3, nbPixels=5):
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(nbLights, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return {
        "light_locations": rng.normal(size=(nbLights, 3)).astype(np.float32),
        "light_power": rng.uniform(1, 2, nbLights).astype(np.float32),
        "light_principal_direction": directions,
        "mu": np.array([0.5, 0.6, 0.7], np.float32),
        "rho": rng.uniform(0, 1, (nbPixels, 3)).astype(np.float32),
        "pixels": np.array([[0, 0], [10, 5], [39, 29], [20, 15], [3, 4]], np.int32)[:nbPixels],
        "points": rng.normal(size=(nbPixels, 3)).astype(np.float32),
    }


def _views(nbLights=3):
    return [{"viewId": str(100 + i), "path": "/img/{}.exr".format(100 + i)} for i in range(nbLights)]


def test_pose_lights_frames():
    camera, values = _camera(), _values()
    lights = lio.poseLights("7", camera, _views(), values, downscale=2)
    assert lights["format"] == lio.LIGHTS_FORMAT and lights["model"] == "LED" and lights["poseId"] == "7"
    assert lights["anisotropy"] == pytest.approx([0.5, 0.6, 0.7])
    assert lights["camera"]["downscale"] == 2 and lights["camera"]["width"] == 40
    assert [light["viewId"] for light in lights["lights"]] == ["100", "101", "102"]
    for index, light in enumerate(lights["lights"]):
        position = values["light_locations"][index]
        direction = values["light_principal_direction"][index]
        assert light["positionCamera"] == pytest.approx(position.tolist(), abs=1e-6)
        assert light["principalDirectionCamera"] == pytest.approx(direction.tolist(), abs=1e-6)
        assert light["power"] == pytest.approx(float(values["light_power"][index]))
        # world = R_cameraToWorld camera + center
        assert np.allclose(light["positionWorld"], helpers.rotationZ(0.3) @ position + [1.0, 2.0, 3.0], atol=1e-6)
        assert np.allclose(light["principalDirectionWorld"], helpers.rotationZ(0.3) @ direction, atol=1e-6)
        assert np.allclose(camera.worldToCamera([light["positionWorld"]])[0], position, atol=1e-5)
    json.dumps(lights)  # plain JSON types only


def test_pose_lights_requires_led_values():
    values = _values()
    del values["mu"]
    with pytest.raises(RuntimeError, match="mu"):
        lio.poseLights("7", _camera(), _views(), values, downscale=1)
    with pytest.raises(RuntimeError, match="lights"):
        lio.poseLights("7", _camera(), _views(2), _values(), downscale=1)


def test_write_and_read_pose_lights(tmp_path):
    camera, values = _camera(), _values()
    lights = lio.poseLights("7", camera, _views(), values, downscale=1)
    lio.writePoseLights(str(tmp_path), lights, values)
    assert (tmp_path / "7" / "lights.json").is_file() and (tmp_path / "7" / "calibration.npz").is_file()
    read = lio.readPoseLights(str(tmp_path), "7")
    assert read == json.loads(json.dumps(lights))
    with pytest.raises(RuntimeError, match="LedGetterLights"):
        lio.readPoseLights(str(tmp_path), "8")
    lio.writeSummary(str(tmp_path / "lights.json"), {"7": lights})
    summary = json.loads((tmp_path / "lights.json").read_text())
    assert summary["format"] == lio.LIGHTS_FORMAT and list(summary["poses"]) == ["7"]


def test_read_pose_lights_checks_format(tmp_path):
    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "lights.json").write_text(json.dumps({"format": "other", "poseId": "7"}))
    with pytest.raises(RuntimeError, match="format"):
        lio.readPoseLights(str(tmp_path), "7")


def test_led_values_for_selected_views():
    values = _values()
    lights = json.loads(json.dumps(lio.poseLights("7", _camera(), _views(), values, downscale=1)))
    selected = lio.ledValues(lights, ["102", "100"])
    assert set(selected) == {"light_locations", "light_power", "light_principal_direction", "mu"}
    assert all(v.dtype == np.float32 for v in selected.values())
    assert np.allclose(selected["light_locations"], values["light_locations"][[2, 0]])
    assert np.allclose(selected["light_power"], values["light_power"][[2, 0]])
    assert np.allclose(selected["light_principal_direction"], values["light_principal_direction"][[2, 0]])
    assert np.allclose(selected["mu"], values["mu"])
    with pytest.raises(RuntimeError, match="999"):
        lio.ledValues(lights, ["999"])


def test_calibrated_view_ids():
    lights = lio.poseLights("7", _camera(), _views(), _values(), downscale=1)
    assert lio.calibratedViewIds(lights) == {"100", "101", "102"}


def test_initialization_rescaled_to_solve_resolution(tmp_path):
    values = _values()
    lights = lio.poseLights("7", _camera(width=40, height=30), _views(), values, downscale=1)
    lio.writePoseLights(str(tmp_path), lights, values)
    same = lio.readInitialization(str(tmp_path), lights, (40, 30), LOGGER)
    assert set(same) == {"rho", "pixels"}  # pixelwise values used to initialize the solve
    assert np.allclose(same["rho"], values["rho"]) and np.allclose(same["pixels"], values["pixels"])
    double = lio.readInitialization(str(tmp_path), lights, (80, 60), LOGGER)
    # pixel centers: (p + 0.5) * 2 - 0.5
    assert np.allclose(double["pixels"], (values["pixels"] + 0.5) * 2.0 - 0.5)
    assert lio.readInitialization(str(tmp_path / "missing"), lights, (40, 30), LOGGER) == {}


def test_same_camera_warning(caplog):
    lights = lio.poseLights("7", _camera(), _views(), _values(), downscale=1)
    with caplog.at_level("WARNING"):
        lio.checkSameCamera(lights, _camera(), LOGGER)
        assert not caplog.records
        lio.checkSameCamera(lights, _camera(center=(1.0, 2.0, 3.5)), LOGGER)
        assert "camera" in caplog.text
