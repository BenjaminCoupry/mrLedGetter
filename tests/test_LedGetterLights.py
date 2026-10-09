"""LedGetterLights node run with the stand-in backend (CPU only)."""
import json
import os

import numpy as np
import pytest

import helpers
from helpers import FakeBackend, FakeChunk, FakeNode

module = helpers.loadModule("LedGetterLights")
lc = helpers.loadModule("ledCommon")


def _node(tmp_path, **values):
    sfmPath = values.pop("inputSfm", None) or helpers.writeScene(tmp_path / "images")
    values.setdefault("samplingStep", 1)
    return FakeNode(module.LedGetterLights, tmp_path / "cache", inputSfm=sfmPath, inputMesh="/data/mesh.obj",
                    **values)


def _expectedPixels(step=1):
    """Pixels of the alpha mask (rows 4:20, columns 6:26) on the grid, inside the disc seen by the camera."""
    rows, cols = lc.gridIndices(helpers.HEIGHT, helpers.WIDTH, step)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    inMask = (rr >= 4) & (rr < 20) & (cc >= 6) & (cc < 26)
    radius = np.hypot(cc - helpers.WIDTH / 2.0, rr - helpers.HEIGHT / 2.0) * helpers.PLANE_DEPTH / helpers.FOCAL
    return inMask & (radius <= helpers.PLANE_RADIUS)


def test_calibrates_each_pose(tmp_path):
    backend = FakeBackend()
    node = _node(tmp_path, iterations=500, learningRate=0.002, delta=0.02, meshAliceVisionAxes=False)
    module.run(FakeChunk(node), backend)
    cache = tmp_path / "cache"
    calls = [c for c in backend.calls if c[0] == "calibrateLights"]
    assert len(calls) == 2
    _, data, settings, exportFolder, lightNames = calls[0]
    assert settings == {"device": "cpu", "iterations": 500, "learningRate": 0.002, "delta": 0.02,
                        "exportImages": False}
    assert lightNames == ["100", "101", "102", "103"]  # the viewIds, in the order of the images
    assert exportFolder == str(cache / "100" / "ledgetter")
    expected = _expectedPixels()
    assert data["gridMask"].tolist() == expected.tolist()
    assert len(data["pixels"]) == expected.sum() and data["images"].shape == (expected.sum(), 3, 4)
    # image values of the light i: intensity i (blue: half) at every pixel
    assert np.allclose(data["images"][:, 0, :], helpers.lightIntensities(4))
    assert np.allclose(data["images"][:, 2, :], np.array(helpers.lightIntensities(4)) * 0.5)
    meshCalls = [c for c in backend.calls if c[0] == "loadMesh"]
    assert meshCalls[0][1:3] == ("/data/mesh.obj", False)
    assert np.allclose(meshCalls[1][3], helpers.rotationZ(0.3).T) and np.allclose(meshCalls[1][4], [0.5, 0, 0])

    lights = json.loads((cache / "200" / "lights.json").read_text())
    assert [light["viewId"] for light in lights["lights"]] == ["200", "201", "202", "203"]
    first = lights["lights"][1]
    position = np.array([np.cos(1), np.sin(1), -1.0])
    assert np.allclose(first["positionCamera"], position, atol=1e-6)
    assert np.allclose(first["positionWorld"], helpers.rotationZ(0.3).T @ position + [0.5, 0, 0], atol=1e-6)
    assert (cache / "200" / "calibration.npz").is_file()
    assert (cache / "200" / "ledgetter" / "LED" / "light_plot.html").is_file()
    summary = json.loads((cache / "lights.json").read_text())
    assert sorted(summary["poses"]) == ["100", "200"]
    assert not any(os.path.islink(os.path.join(root, f)) for root, _, files in os.walk(cache) for f in files)


def test_sampling_step_and_selection(tmp_path):
    backend = FakeBackend()
    node = _node(tmp_path, samplingStep=2, nbImages=3, imageSelection="first", excludePoseReferenceImage=True)
    module.run(FakeChunk(node), backend)
    _, data, _, _, lightNames = [c for c in backend.calls if c[0] == "calibrateLights"][0]
    assert lightNames == ["101", "102", "103"]
    assert data["gridMask"].shape == (helpers.HEIGHT // 2, helpers.WIDTH // 2)
    assert data["gridMask"].tolist() == _expectedPixels(2).tolist()
    assert (data["pixels"] % 2 == 0).all()


def test_rejects_non_linear_images(tmp_path, caplog):
    sfmPath = helpers.writeScene(tmp_path / "images", fileType="png")
    with pytest.raises(RuntimeError, match="could not be processed"):
        module.run(FakeChunk(_node(tmp_path, inputSfm=sfmPath)), FakeBackend())
    assert "outputFileType=exr" in caplog.text


def test_linearize_input_option(tmp_path):
    sfmPath = helpers.writeScene(tmp_path / "images", fileType="png")
    backend = FakeBackend()
    module.run(FakeChunk(_node(tmp_path, inputSfm=sfmPath, linearizeInput=True)), backend)
    _, data, _, _, _ = [c for c in backend.calls if c[0] == "calibrateLights"][0]
    stored = np.round(np.array(helpers.lightIntensities(4), np.float32) * 255) / 255  # as written
    linear = np.where(stored <= 0.04045, stored / 12.92, ((stored + 0.055) / 1.055) ** 2.4)
    assert np.allclose(data["images"][0, 0, :], linear, atol=1e-6)


def test_mesh_not_visible(tmp_path, caplog):
    class NoHitBackend(FakeBackend):
        def castRays(self, scene, K, pixels):
            hit, points, normals, ids = super().castRays(scene, K, pixels)
            return np.zeros_like(hit), points, normals, ids
    with pytest.raises(RuntimeError, match="could not be processed"):
        module.run(FakeChunk(_node(tmp_path)), NoHitBackend())
    assert "AliceVision Mesh Axes" in caplog.text


def test_failure_policy_never(tmp_path):
    class FailingBackend(FakeBackend):
        def calibrateLights(self, *args):
            raise MemoryError("simulated out of memory")
    module.run(FakeChunk(_node(tmp_path, failurePolicy="never")), FailingBackend())
    assert json.loads((tmp_path / "cache" / "lights.json").read_text())["poses"] == {}
