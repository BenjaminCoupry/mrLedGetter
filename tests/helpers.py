"""Helpers of the LED-getter plugin tests (CPU only, no JAX / ledgetter import).

The node package (meshroom/ledGetter) uses relative imports: it is loaded here as the package
"ledGetterNodes". The LED-getter library calls are isolated in ledBackend.py, replaced in the tests by
FakeBackend (an analytic plane in front of the camera).
"""
import importlib
import importlib.util
import json
import logging
import os
import sys

import cv2
import numpy as np

PACKAGE = "ledGetterNodes"
FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshroom", "ledGetter")
LOGGER = logging.getLogger("test_ledGetter")


def loadModule(name):
    """Import meshroom/ledGetter/<name>.py as a submodule of the ledGetterNodes package."""
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(PACKAGE, os.path.join(FOLDER, "__init__.py"),
                                                      submodule_search_locations=[FOLDER])
        package = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = package
        spec.loader.exec_module(package)
    return importlib.import_module(PACKAGE + "." + name)


class Value:
    def __init__(self, value):
        self.value = value


class FakeNode:
    """Node with the attributes of a node description, set to their default values.

    Output paths are resolved in the cache folder; "{nodeCacheFolder}" expressions only.
    """

    def __init__(self, nodeDesc, cache, **values):
        for attribute in nodeDesc.inputs:
            setattr(self, attribute.name, Value(attribute.value))
        for attribute in nodeDesc.outputs:
            value = attribute.value
            if isinstance(value, str):
                value = value.replace("{nodeCacheFolder}", str(cache))
            setattr(self, attribute.name, Value(value))
        for name, value in values.items():
            getattr(self, name).value = value


class FakeChunk:
    def __init__(self, node):
        self.node = node
        self.logger = LOGGER


# ---------------------------------------------------------------------------------------------- scene

WIDTH, HEIGHT = 32, 24
FOCAL = 40.0
PLANE_DEPTH = 5.0
PLANE_RADIUS = 1.2  # the plane is a disc of this radius (world units) centered on the optical axis


def rotationZ(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def aliceVisionPose(poseId, rotationWorldToCamera, center):
    """SfMData pose as written by AliceVision: the world-to-camera rotation stored column-major."""
    return {"poseId": str(poseId), "pose": {"transform": {
        "rotation": [repr(float(v)) for v in np.asarray(rotationWorldToCamera).flatten(order="F")],
        "center": [repr(float(v)) for v in center]}, "locked": "false"}}


def pinholeIntrinsic(width=WIDTH, height=HEIGHT, focal=FOCAL, offset=(0.0, 0.0), pixelRatio=1.0):
    sensorWidth = 36.0
    return {"intrinsicId": "1", "width": str(width), "height": str(height), "sensorWidth": repr(sensorWidth),
            "sensorHeight": "24", "type": "pinhole", "distortionType": "none", "distortionParams": "",
            "undistortionParams": "", "focalLength": repr(focal * pixelRatio * sensorWidth / width),
            "pixelRatio": repr(pixelRatio), "principalPoint": [repr(offset[0]), repr(offset[1])]}


def lightIntensities(nbLights):
    return [0.2 + 0.1 * i for i in range(nbLights)]


def writeScene(folder, nbLights=4, poses=("100", "200"), fileType="exr", alphaMask=True):
    """Multi-lighting poses (one view per light, the first one with viewId == poseId) + one single view.

    Image values: linear intensity of the light, constant over the image (float EXR or 8-bit PNG); the alpha
    channel holds an object mask (a rectangle) when alphaMask is set.
    Returns the SfMData path.
    """
    folder = str(folder)
    os.makedirs(folder, exist_ok=True)
    views, sfmPoses = [], []
    alpha = np.zeros((HEIGHT, WIDTH), np.float32)
    alpha[4:20, 6:26] = 1.0
    for index, poseId in enumerate(poses):
        sfmPoses.append(aliceVisionPose(poseId, rotationZ(0.3 * index), [0.5 * index, 0.0, 0.0]))
        for light in range(nbLights):
            viewId = poseId if light == 0 else str(int(poseId) + light)
            rgb = np.full((HEIGHT, WIDTH, 3), lightIntensities(nbLights)[light], np.float32)
            rgb[:, :, 2] *= 0.5  # blue
            path = os.path.join(folder, "{}.{}".format(viewId, fileType))
            writeImage(path, rgb, alpha if alphaMask else None)
            views.append({"viewId": viewId, "poseId": poseId, "intrinsicId": "1", "path": path,
                          "width": str(WIDTH), "height": str(HEIGHT)})
    views.append({"viewId": "900", "poseId": "900", "intrinsicId": "1", "path": os.path.join(folder, "900.png"),
                  "width": str(WIDTH), "height": str(HEIGHT)})
    sfmPoses.append(aliceVisionPose("900", np.eye(3), [0.0, 0.0, -1.0]))
    sfm = {"version": ["1", "2", "14"], "views": views, "intrinsics": [pinholeIntrinsic()], "poses": sfmPoses,
           "structure": [{"landmarkId": "0", "X": ["0", "0", "5"], "observations": [{"observationId": "100"}]}]}
    path = os.path.join(folder, "input.sfm")
    with open(path, "w") as f:
        json.dump(sfm, f)
    return path


def writeImage(path, rgb, alpha=None):
    """Write an RGB(A) image: float32 EXR (raw values) or 8-bit PNG (values in [0, 1])."""
    image = rgb if alpha is None else np.concatenate([rgb, alpha[:, :, None]], axis=2)
    image = np.ascontiguousarray(image[:, :, [2, 1, 0] + ([3] if alpha is not None else [])])
    if path.endswith(".exr"):
        ok = cv2.imwrite(path, image.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    else:
        ok = cv2.imwrite(path, np.round(np.clip(image, 0, 1) * 255).astype(np.uint8))
    assert ok, path


# ---------------------------------------------------------------------------------------------- backend

class FakeBackend:
    """Stand-in for ledBackend: a disc of radius PLANE_RADIUS on the plane z = PLANE_DEPTH (camera frame),
    facing the camera; records the calls."""

    def __init__(self):
        self.calls = []

    def selectDevice(self, useGpu, logger):
        return "cpu"

    def loadMesh(self, meshPath, flipMesh, rotation, center):
        self.calls.append(("loadMesh", meshPath, flipMesh, np.array(rotation), np.array(center)))
        return {"meshPath": meshPath}

    def castRays(self, scene, K, pixels):
        pixels = np.asarray(pixels, np.float64)
        rays = np.linalg.solve(K, np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1).T).T
        points = rays * (PLANE_DEPTH / rays[:, 2:3])
        hit = np.linalg.norm(points[:, :2], axis=1) <= PLANE_RADIUS
        normals = np.tile([0.0, 0.0, -1.0], (len(pixels), 1))
        self.calls.append(("castRays", len(pixels)))
        return hit, points.astype(np.float32), normals.astype(np.float32), np.zeros(len(pixels), np.int32)

    def calibrateLights(self, scene, data, settings, exportFolder, lightNames, pose, logger):
        nbLights = data["images"].shape[2]
        self.calls.append(("calibrateLights", data, settings, exportFolder, list(lightNames)))
        os.makedirs(os.path.join(exportFolder, "LED"), exist_ok=True)
        with open(os.path.join(exportFolder, "LED", "light_plot.html"), "w") as f:
            f.write("<html></html>")
        locations = np.array([[np.cos(i), np.sin(i), -1.0] for i in range(nbLights)], np.float32)
        return {"model": "LED", "values": {
            "light_locations": locations,
            "light_power": np.arange(1, nbLights + 1, dtype=np.float32),
            "light_principal_direction": np.tile(np.array([0.0, 0.0, 1.0], np.float32), (nbLights, 1)),
            "mu": np.array([0.5, 0.6, 0.7], np.float32),
            "rho": np.full((len(data["pixels"]), 3), 0.25, np.float32),
            "pixels": data["pixels"], "points": data["points"], "normals": data["normals"],
        }}

    def solvePhotometricStereo(self, scene, data, lights, settings, logger):
        self.calls.append(("solvePhotometricStereo", data, lights, settings))
        nbPixels = len(data["pixels"])
        normals = np.tile(np.array([0.0, 0.6, -0.8], np.float32), (nbPixels, 1))  # OpenCV camera frame
        rho = np.tile(np.array([0.1, 0.2, 0.3], np.float32), (nbPixels, 1))
        return {"normals": normals, "rho": rho}
