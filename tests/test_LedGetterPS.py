"""LedGetterPS node run with the stand-in backend, after LedGetterLights (CPU only)."""
import json
import os
import shutil

import cv2
import numpy as np
import pytest

import helpers
from helpers import FakeBackend, FakeChunk, FakeNode

lightsModule = helpers.loadModule("LedGetterLights")
psModule = helpers.loadModule("LedGetterPS")
lc = helpers.loadModule("ledCommon")


def _calibrate(tmp_path, sfmPath, **values):
    cache = tmp_path / "lights"
    node = FakeNode(lightsModule.LedGetterLights, cache, inputSfm=sfmPath, inputMesh="/data/mesh.obj",
                    samplingStep=1, **values)
    lightsModule.run(FakeChunk(node), FakeBackend())
    return str(cache)


def _solve(tmp_path, backend=None, lightsValues=None, alphaMask=True, **values):
    sfmPath = helpers.writeScene(tmp_path / "images", alphaMask=alphaMask)
    lights = _calibrate(tmp_path, sfmPath, **(lightsValues or {}))
    node = FakeNode(psModule.LedGetterPS, tmp_path / "cache", inputSfm=sfmPath, inputMesh="/data/mesh.obj",
                    inputLights=lights, **values)
    backend = backend or FakeBackend()
    psModule.run(FakeChunk(node), backend)
    return tmp_path / "cache", backend


def _support(downscale=1, alphaMask=True):
    """Expected support: alpha mask (rows 4:20, columns 6:26) inside the disc seen by the camera."""
    height, width = helpers.HEIGHT // downscale, helpers.WIDTH // downscale
    rr, cc = np.mgrid[0:height, 0:width]
    focal = helpers.FOCAL * (width / float(helpers.WIDTH))
    center = ((helpers.WIDTH / 2.0 + 0.5) / downscale - 0.5, (helpers.HEIGHT / 2.0 + 0.5) / downscale - 0.5)
    disc = np.hypot(cc - center[0], rr - center[1]) * helpers.PLANE_DEPTH / focal <= helpers.PLANE_RADIUS
    if not alphaMask:
        return disc
    alpha = np.zeros((helpers.HEIGHT, helpers.WIDTH), bool)
    alpha[4:20, 6:26] = True
    return disc & helpers.loadModule("psCommon").resizeMask(alpha, (width, height))


def _decode(path):
    stored = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    return stored.astype(np.float32) / 65535.0


def test_outputs_like_the_other_ps_nodes(tmp_path):
    cache, backend = _solve(tmp_path, iterations=300, learningRate=0.005, delta=0.02, gpuChunks=7)
    normalSfm = json.loads((cache / "normalMaps.sfm").read_text())
    assert sorted(v["viewId"] for v in normalSfm["views"]) == ["100", "200"]
    assert {v["path"] for v in normalSfm["views"]} == {str(cache / "100.png"), str(cache / "200.png")}
    albedoSfm = json.loads((cache / "albedoMaps.sfm").read_text())
    assert {v["path"] for v in albedoSfm["views"]} == {str(cache / "albedo" / "100.png"),
                                                       str(cache / "albedo" / "200.png")}
    support = _support()
    mask = cv2.imread(str(cache / "masks" / "100.png"), cv2.IMREAD_UNCHANGED) > 0
    assert mask.tolist() == support.tolist()
    normal = _decode(cache / "100.png") * 2.0 - 1.0
    # LED-getter normals (0, 0.6, -0.8) in the OpenCV camera frame -> OpenGL camera frame (RNb-NeuS2)
    assert np.allclose(normal[support], [0.0, -0.6, 0.8], atol=1e-4)
    assert np.allclose(normal[~support], 0.0, atol=1e-4)
    albedo = _decode(cache / "albedo" / "100.png")
    assert np.allclose(albedo[support], [0.1, 0.2, 0.3], atol=1e-4) and not albedo[~support].any()
    solves = [c for c in backend.calls if c[0] == "solvePhotometricStereo"]
    assert len(solves) == 2 * 4  # 2 poses x 2 x 2 pixel subsets (split 2)
    _, data, lights, settings = solves[0]
    assert settings == {"device": "cpu", "iterations": 300, "learningRate": 0.005, "delta": 0.02, "chunks": 7}
    assert set(lights) == {"light_locations", "light_power", "light_principal_direction", "mu", "rho", "pixels"}
    assert np.allclose(lights["light_power"], [1, 2, 3, 4])  # the calibrated lights of the 4 images
    assert data["images"].shape[2] == 4 and (data["pixels"] % 2 == 0).all()
    assert not any(os.path.islink(os.path.join(root, f)) for root, _, files in os.walk(cache) for f in files)


def test_split_factor_one(tmp_path):
    _, backend = _solve(tmp_path, splitFactor=1)
    solves = [c for c in backend.calls if c[0] == "solvePhotometricStereo"]
    assert len(solves) == 2 and len(solves[0][1]["pixels"]) == _support().sum()


def test_without_mask_the_mesh_defines_the_support(tmp_path):
    cache, _ = _solve(tmp_path, alphaMask=False)
    mask = cv2.imread(str(cache / "masks" / "200.png"), cv2.IMREAD_UNCHANGED) > 0
    assert mask.tolist() == _support(alphaMask=False).tolist()


def test_exr_and_opencv_convention(tmp_path):
    cache, _ = _solve(tmp_path, outputFormat="exr", normalConvention="opencv")
    normal = cv2.imread(str(cache / "100.exr"), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    assert np.allclose(normal[_support()], [0.0, 0.6, -0.8], atol=1e-6)
    albedo = cv2.imread(str(cache / "albedo" / "100.exr"), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    assert np.allclose(albedo[_support()], [0.1, 0.2, 0.3], atol=1e-6)


def test_downscale_between_the_nodes(tmp_path):
    cache, backend = _solve(tmp_path, downscale=2)
    normal = cv2.imread(str(cache / "100.png"), cv2.IMREAD_UNCHANGED)
    assert normal.shape == (helpers.HEIGHT // 2, helpers.WIDTH // 2, 3)
    mask = cv2.imread(str(cache / "masks" / "100.png"), cv2.IMREAD_UNCHANGED) > 0
    assert mask.tolist() == _support(2).tolist()
    _, _, lights, _ = [c for c in backend.calls if c[0] == "solvePhotometricStereo"][0]
    calibration = np.load(str(tmp_path / "lights" / "100" / "calibration.npz"))
    assert np.allclose(lights["pixels"], (calibration["pixels"] + 0.5) * 0.5 - 0.5)
    assert json.loads((cache / "normalMaps.sfm").read_text())["intrinsics"][0]["width"] == str(helpers.WIDTH // 2)


def test_views_without_calibrated_light(tmp_path, caplog):
    _, backend = _solve(tmp_path, lightsValues={"nbImages": 3, "imageSelection": "first"})
    _, data, lights, _ = [c for c in backend.calls if c[0] == "solvePhotometricStereo"][0]
    assert data["images"].shape[2] == 3 and np.allclose(lights["light_power"], [1, 2, 3])
    assert "without calibrated light" in caplog.text


def test_pose_without_lights(tmp_path, caplog):
    sfmPath = helpers.writeScene(tmp_path / "images")
    lights = _calibrate(tmp_path, sfmPath)
    shutil.rmtree(os.path.join(lights, "200"))
    node = FakeNode(psModule.LedGetterPS, tmp_path / "cache", inputSfm=sfmPath, inputMesh="/m.obj", inputLights=lights)
    psModule.run(FakeChunk(node), FakeBackend())
    assert [v["viewId"] for v in json.loads((tmp_path / "cache" / "normalMaps.sfm").read_text())["views"]] == ["100"]
    assert "LedGetterLights" in caplog.text
    with pytest.raises(RuntimeError, match="could not be processed"):
        node.failurePolicy.value = "anyPose"
        psModule.run(FakeChunk(node), FakeBackend())


def test_rejects_non_linear_images(tmp_path, caplog):
    sfmPath = helpers.writeScene(tmp_path / "images")
    lights = _calibrate(tmp_path, sfmPath)
    pngScene = helpers.writeScene(tmp_path / "png", fileType="png")
    node = FakeNode(psModule.LedGetterPS, tmp_path / "cache", inputSfm=pngScene, inputMesh="/m.obj", inputLights=lights)
    with pytest.raises(RuntimeError, match="could not be processed"):
        psModule.run(FakeChunk(node), FakeBackend())
    assert "outputFileType=exr" in caplog.text


def test_excluded_reference_image_is_not_reported_as_uncalibrated(tmp_path, caplog):
    _, backend = _solve(tmp_path, lightsValues={"excludePoseReferenceImage": True}, excludePoseReferenceImage=True)
    _, data, lights, _ = [c for c in backend.calls if c[0] == "solvePhotometricStereo"][0]
    assert data["images"].shape[2] == 3 and np.allclose(lights["light_power"], [1, 2, 3])
    assert "without calibrated light" not in caplog.text
