"""Unit tests of ledCommon.py: SfMData cameras, frames, linear images and pixel sampling (CPU only)."""
import copy
import types

import cv2
import numpy as np
import pytest

import helpers
from helpers import LOGGER

lc = helpers.loadModule("ledCommon")
ps = helpers.loadModule("psCommon")


# ---------------------------------------------------------------------------------------------- intrinsics

def _intrinsic(**values):
    intrinsic = {"intrinsicId": "1", "width": "7200", "height": "4800", "sensorWidth": "36", "sensorHeight": "24",
                 "type": "pinhole", "distortionType": "none", "distortionParams": "", "focalLength": "50",
                 "pixelRatio": "2", "principalPoint": ["10.5", "-3"]}
    intrinsic.update(values)
    return intrinsic


@pytest.mark.parametrize("version,fx,fy", [
    ((1, 2, 14), 5000.0, 10000.0),  # focal in mm along y, fx = fy / pixelRatio
    ((1, 2, 8), 10000.0, 20000.0),  # compatibility: focal in mm along x, fy = fx * pixelRatio
    ((1, 2, 3), 10000.0, 5000.0),  # "pixelRatio" stored the focal ratio fx / fy
])
def test_intrinsic_matrix_focal_by_version(version, fx, fy):
    K = lc.intrinsicMatrix(_intrinsic(), version)
    assert np.allclose(K, [[fx, 0, 3610.5], [0, fy, 2397.0], [0, 0, 1]])


def test_intrinsic_matrix_ignores_sensor_height():
    """AliceVision derives both focals from the width and the sensor width (not from the sensor height)."""
    K = lc.intrinsicMatrix(_intrinsic(pixelRatio="1", sensorHeight="30"), (1, 2, 14))
    assert np.isclose(K[0, 0], 10000.0) and np.isclose(K[1, 1], 10000.0)


def test_intrinsic_matrix_old_versions_rejected():
    with pytest.raises(RuntimeError, match="version"):
        lc.intrinsicMatrix(_intrinsic(pxFocalLength="1000"), (1, 1, 0))


def test_intrinsic_matrix_matches_alicevision_projection():
    """Real ExportImages intrinsic (biface): AliceVision's cam2ima maps the camera plane point (0.1, -0.05) to
    (5355.202700067083, 2165.7831709752545) (pyalicevision, AliceVision 2025, SfMData 1.2.14)."""
    intrinsic = {"intrinsicId": "485978989", "width": "8280", "height": "5520", "sensorWidth": "36.000000000000007",
                 "sensorHeight": "24.000000000000004", "type": "pinhole", "focalLength": "51.585677973111721",
                 "pixelRatio": "1", "principalPoint": ["28.732106685513827", "-0.98153233396078576"],
                 "distortionParams": "", "undistortionParams": "", "distortionType": "none"}
    K = lc.intrinsicMatrix(intrinsic, (1, 2, 14))
    pixel = K @ np.array([0.1, -0.05, 1.0])
    assert np.allclose(pixel[:2], [5355.202700067083, 2165.7831709752545], atol=1e-6)


@pytest.mark.parametrize("values", [
    {"type": "pinhole"},
    {"type": "pinhole", "distortionParams": []},
    {"type": "radial3", "distortionParams": ["0", "0", "0"]},  # no distortion left: pinhole
])
def test_check_pinhole_accepts(values):
    lc.checkPinhole(_intrinsic(**values))


@pytest.mark.parametrize("values", [
    {"type": "radial3", "distortionParams": ["0.1", "0", "0"]},
    {"type": "equidistant_r3", "distortionParams": ""},
    {"type": "pinhole", "distortionType": "radialk3", "distortionParams": ["0.01", "0", "0"]},
])
def test_check_pinhole_rejects(values):
    with pytest.raises(RuntimeError, match="ExportImages"):
        lc.checkPinhole(_intrinsic(**values))


# ---------------------------------------------------------------------------------------------- cameras

def _sfm(rotation=None, center=(1.0, -2.0, 0.5), **intrinsicValues):
    rotation = helpers.rotationZ(0.4) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) if rotation is None else rotation
    views = [{"viewId": v, "poseId": "7", "intrinsicId": "1", "path": "/x/{}.exr".format(v)} for v in ("7", "8", "9")]
    return {"version": ["1", "2", "14"], "views": views, "poses": [helpers.aliceVisionPose("7", rotation, center)],
            "intrinsics": [_intrinsic(**intrinsicValues)]}, rotation


def test_pose_camera_extrinsics():
    sfm, worldToCamera = _sfm()
    camera = lc.poseCamera(sfm, "7", sfm["views"], 1)
    # the JSON rotation is the world-to-camera rotation stored column-major: read row-major, it is camera-to-world
    assert np.allclose(camera.rotation, worldToCamera.T)
    assert np.allclose(camera.center, [1.0, -2.0, 0.5])
    world = np.array([[0.3, 0.2, 4.0], [-1.0, 2.0, 3.0]])
    inCamera = camera.worldToCamera(world)
    assert np.allclose(inCamera, (worldToCamera @ (world - camera.center).T).T)
    assert np.allclose(camera.cameraToWorld(inCamera), world)
    assert np.allclose(camera.directionsToWorld(np.array([[0.0, 0.0, 1.0]])), [worldToCamera[2]])
    assert (camera.width, camera.height) == (7200, 4800)


def test_pose_camera_projection_matches_ray_casting_convention():
    """A pixel (column u, row v) is the center of the image pixel [v, u]: its ray K^-1 (u, v, 1) projects back
    to (u, v) (AliceVision and LED-getter both put the center of pixel i at coordinate i)."""
    sfm, _ = _sfm()
    camera = lc.poseCamera(sfm, "7", sfm["views"], 1)
    pixels = np.array([[0, 0], [7199, 4799], [3610, 2397]], np.float64)
    rays = camera.pixelRays(pixels)
    assert np.allclose(rays[:, 2], 1.0)
    world = camera.cameraToWorld(rays * 3.0)
    assert np.allclose(camera.project(world), pixels)


@pytest.mark.parametrize("d", [2, 3])
def test_pose_camera_downscale(d):
    """The camera of the downscaled images is the one of the output SfMData (psCommon.scaleIntrinsics)."""
    sfm, _ = _sfm(pixelRatio="1")
    camera = lc.poseCamera(sfm, "7", sfm["views"], d)
    scaled = copy.deepcopy(sfm)
    ps.scaleIntrinsics(scaled, d)
    assert (camera.width, camera.height) == (7200 // d, 4800 // d)
    assert np.allclose(camera.K, lc.intrinsicMatrix(scaled["intrinsics"][0], (1, 2, 14)))
    assert np.isclose(camera.K[0, 2], (3610.5 + 0.5) / d - 0.5)
    assert sfm["intrinsics"][0]["width"] == "7200"  # input not modified


def test_pose_camera_errors():
    sfm, _ = _sfm(type="radial3", distortionParams=["0.1", "0", "0"])
    with pytest.raises(RuntimeError, match="ExportImages"):
        lc.poseCamera(sfm, "7", sfm["views"], 1)
    sfm, _ = _sfm()
    sfm["intrinsics"].append(dict(sfm["intrinsics"][0], intrinsicId="2"))
    sfm["views"][1]["intrinsicId"] = "2"
    with pytest.raises(RuntimeError, match="intrinsic"):
        lc.poseCamera(sfm, "7", sfm["views"], 1)
    sfm, _ = _sfm()
    with pytest.raises(RuntimeError, match="pose"):
        lc.poseCamera(sfm, "8", sfm["views"], 1)


# ---------------------------------------------------------------------------------------------- pixel grids

def test_grid_indices_and_pixels():
    rows, cols = lc.gridIndices(7, 10, 3)
    assert rows.tolist() == [0, 3, 6] and cols.tolist() == [0, 3, 6, 9]
    rows, cols = lc.gridIndices(7, 10, 3, offset=(1, 2))
    assert rows.tolist() == [1, 4] and cols.tolist() == [2, 5, 8]
    gridMask = np.array([[True, False, True], [False, True, False]])
    pixels = lc.gridPixels(rows, cols, gridMask)
    # (u = column, v = row), row-major order of the grid (the order of LED-getter's masked values)
    assert pixels.dtype == np.int32 and pixels.tolist() == [[2, 1], [8, 1], [5, 4]]


def test_split_offsets_cover_every_pixel_once():
    seen = np.zeros((7, 10), int)
    for offset in lc.splitOffsets(3):
        rows, cols = lc.gridIndices(7, 10, 3, offset)
        seen[np.ix_(rows, cols)] += 1
    assert (seen == 1).all() and len(lc.splitOffsets(3)) == 9


# ---------------------------------------------------------------------------------------------- images

def _views(tmp_path, fileType, nb=3):
    views = []
    for i in range(nb):
        rgb = np.zeros((helpers.HEIGHT, helpers.WIDTH, 3), np.float32)
        rgb[:, :, 0] = np.arange(helpers.WIDTH)[None, :] / 100.0 + i
        rgb[:, :, 1] = np.arange(helpers.HEIGHT)[:, None] / 100.0
        rgb[:, :, 2] = 0.5
        path = str(tmp_path / "{}.{}".format(i, fileType))
        helpers.writeImage(path, rgb if fileType == "exr" else rgb / 3.0)
        views.append({"viewId": str(i), "path": path})
    return views


COMMON = types.SimpleNamespace(
    inputs=ps.inputAttributes() + ps.advancedInputAttributes() + ps.settingsAttributes(),
    outputs=ps.outputAttributes())


def _params(tmp_path, **values):
    return ps.Params(helpers.FakeNode(COMMON, tmp_path, **values))


def test_sample_pixels_float_images(tmp_path):
    views = _views(tmp_path, "exr")
    pixels = np.array([[3, 2], [31, 23]], np.int32)  # (u, v)
    values = lc.samplePixels(views, _params(tmp_path), pixels, (helpers.WIDTH, helpers.HEIGHT))
    assert values.shape == (2, 3, 3) and values.dtype == np.float32  # LED-getter layout: pixels, channels, images
    assert np.allclose(values[0, :, 1], [0.03 + 1, 0.02, 0.5])
    assert np.allclose(values[1, :, 2], [0.31 + 2, 0.23, 0.5])


def test_sample_pixels_rejects_non_linear_images(tmp_path):
    views = _views(tmp_path, "png")
    with pytest.raises(lc.NonLinearInputError, match="exr"):
        lc.samplePixels(views, _params(tmp_path), np.array([[0, 0]], np.int32), (helpers.WIDTH, helpers.HEIGHT))


def test_sample_pixels_linearize_option(tmp_path):
    views = _views(tmp_path, "png")
    values = lc.samplePixels(views, _params(tmp_path, linearizeInput=True), np.array([[0, 0]], np.int32),
                             (helpers.WIDTH, helpers.HEIGHT))
    stored = np.round(0.5 / 3.0 * 255) / 255
    assert np.allclose(values[0, 2, :], ps.srgbToLinear(np.array([stored], np.float32))[0])


def test_sample_pixels_downscale_and_size_check(tmp_path):
    views = _views(tmp_path, "exr")
    params = _params(tmp_path, downscale=2)
    values = lc.samplePixels(views, params, np.array([[1, 1]], np.int32), (helpers.WIDTH // 2, helpers.HEIGHT // 2))
    assert np.allclose(values[0, :, 0], [0.025, 0.025, 0.5])  # average of the pixels [2, 3] x [2, 3]
    with pytest.raises(RuntimeError, match="size"):
        lc.samplePixels(views, params, np.array([[1, 1]], np.int32), (helpers.WIDTH, helpers.HEIGHT))


# ---------------------------------------------------------------------------------------------- normals, poses

def test_ledgetter_normals_to_opengl():
    normals = np.array([[0.0, 0.6, -0.8]], np.float32)  # LED-getter: OpenCV camera frame, facing the camera
    assert np.allclose(lc.ledgetterToOpenGL(normals), [[0.0, -0.6, 0.8]])


def test_lighting_views(tmp_path):
    views = [{"viewId": v, "path": "/x/{}.exr".format(v)} for v in ("7", "8", "9", "10")]
    assert [v["viewId"] for v in lc.lightingViews("7", views, _params(tmp_path))] == ["7", "8", "9", "10"]
    selected = lc.lightingViews("7", views, _params(tmp_path, excludePoseReferenceImage=True))
    assert [v["viewId"] for v in selected] == ["8", "9", "10"]
    selected = lc.lightingViews("7", views, _params(tmp_path, nbImages=2, imageSelection="first"))
    assert [v["viewId"] for v in selected] == ["7", "8"]
    with pytest.raises(RuntimeError, match="reference"):
        lc.lightingViews("7", views[:1], _params(tmp_path, excludePoseReferenceImage=True))


def test_pose_mask_resized(tmp_path):
    folder = tmp_path / "masks"
    folder.mkdir()
    mask = np.zeros((helpers.HEIGHT, helpers.WIDTH), np.uint8)
    mask[4:20, 6:26] = 255
    cv2.imwrite(str(folder / "7.png"), mask)
    views = [{"viewId": "7", "path": "/x/7.exr"}]
    params = _params(tmp_path, maskFolder=str(folder))
    poseMask, source = lc.poseMask("7", views, params, (helpers.WIDTH // 2, helpers.HEIGHT // 2), LOGGER)
    assert poseMask.shape == (helpers.HEIGHT // 2, helpers.WIDTH // 2) and poseMask.sum() == 8 * 10
    assert source.startswith("mask file")


def test_params_with_defaults_for_missing_attributes(tmp_path):
    """A node not exposing the output-only common attributes gets their psCommon default values."""
    inputs = [a for a in COMMON.inputs if a.name not in ("normalConvention", "keepLandmarks", "outputFormat")]
    node = helpers.FakeNode(types.SimpleNamespace(inputs=inputs, outputs=[]), tmp_path, nbImages=5)
    node.outputFolder = helpers.Value(str(tmp_path))
    params = lc.commonParams(node)
    assert params.nbImages == 5 and params.normalConvention == "opengl" and params.outputFormat == "png16"
    assert params.outputMaskFolder == str(tmp_path / "masks")


# ---------------------------------------------------------------------------------------------- light rig

def _frameViews(frameIds):
    return [{"viewId": str(10 + i), "frameId": str(f)} for i, f in enumerate(frameIds)]


def test_rig_light_keys_rank_follows_frame_numbers():
    ordered, keys = lc.rigLightKeys(_frameViews([1903, 1901, 1902]), "rank")
    assert [v["frameId"] for v in ordered] == ["1901", "1902", "1903"] and keys == [0, 1, 2]


def test_rig_light_keys_frame_id():
    ordered, keys = lc.rigLightKeys(_frameViews([51, 1, 6]), "frameId")
    assert keys == [1, 6, 51] and [v["viewId"] for v in ordered] == ["11", "12", "10"]


def test_rig_light_keys_errors():
    with pytest.raises(RuntimeError, match="no frame number"):
        lc.rigLightKeys([{"viewId": "1"}, {"viewId": "2", "frameId": "3"}], "rank")
    with pytest.raises(RuntimeError, match="share a frame number"):
        lc.rigLightKeys(_frameViews([4, 4]), "frameId")
    with pytest.raises(ValueError):
        lc.rigLightKeys(_frameViews([1]), "other")


def test_rig_groups_by_sequence():
    groups = lc.rigGroups({"a": [0, 1, 2], "b": [0, 1], "c": [0, 1, 2]})
    assert groups == [([0, 1, 2], ["a", "c"]), ([0, 1], ["b"])]


def test_stack_pose_data_order_and_separation():
    def data(gridMask, offset):
        n = int(gridMask.sum())
        return {"images": np.full((n, 3, 2), offset, np.float32), "pixels": np.full((n, 2), offset, np.int32),
                "points": np.full((n, 3), offset, np.float32), "normals": np.zeros((n, 3), np.float32),
                "objectsId": np.zeros(n, np.int32), "gridMask": gridMask}
    first = np.array([[True, False, True], [False, True, False]])
    second = np.array([[True, True], [False, True]])
    stacked, counts = lc.stackPoseData([data(first, 1), data(second, 2)])
    assert counts == [3, 3]
    gridMask = stacked["gridMask"]
    assert gridMask.shape == (2 + lc.RIG_SEPARATOR_ROWS + 2, 3)
    assert not gridMask[2:2 + lc.RIG_SEPARATOR_ROWS].any() and not gridMask[:, 2][-2:].any()
    # row-major order of the stacked grid = pose after pose
    assert stacked["pixels"][:, 0].tolist() == [1, 1, 1, 2, 2, 2] and gridMask.sum() == 6
    assert stacked["images"].shape == (6, 3, 2)


def test_split_pose_values():
    values = {"rho": np.arange(10).reshape(5, 2), "pixels": np.arange(10).reshape(5, 2),
              "light_locations": np.ones((4, 3)), "mu": np.array([0.5, 0.5, 0.5])}
    second = lc.splitPoseValues(values, [2, 3], 1)
    assert second["rho"].tolist() == [[4, 5], [6, 7], [8, 9]] and second["pixels"].shape == (3, 2)
    assert second["light_locations"] is values["light_locations"] and second["mu"] is values["mu"]
