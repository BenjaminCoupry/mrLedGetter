"""Lights calibrated by LedGetterLights, read by LedGetterPS (no JAX / LED-getter import).

Output folder of LedGetterLights:

    lights.json                     all the poses: {"format", "version", "poses": {poseId: <pose lights>}}
    <poseId>/lights.json            lights of the pose (documented below)
    <poseId>/calibration.npz        LED-getter values at the calibration pixels (albedo "rho", "pixels", "points",
                                    "normals") and light parameters (camera frame): initialization of LedGetterPS
    <poseId>/ledgetter/<stage>/     LED-getter exports of each calibration stage (directional, punctual, LED):
                                    light_plot.html, loss_plot.html, values, model.json, *.lp, ...

Pose lights (LED model of LED-getter, one light per lighting image):

    {
      "format": "ledgetter-lights", "version": "1.0", "poseId": "<poseId>", "model": "LED",
      "camera": {"K", "rotationCameraToWorld", "center", "width", "height", "downscale"},
      "anisotropy": [mu_R, mu_G, mu_B],
      "lights": [{"viewId", "imagePath", "power", "positionCamera", "positionWorld",
                  "principalDirectionCamera", "principalDirectionWorld"}, ...],
      ... (frames, units and model documented in the file itself)
    }
"""

import json
import os

import numpy as np

LIGHTS_FORMAT = "ledgetter-lights"
LIGHTS_FORMAT_VERSION = "1.0"
LIGHTS_FILE = "lights.json"
CALIBRATION_FILE = "calibration.npz"
LEDGETTER_FOLDER = "ledgetter"
LED_KEYS = ("light_locations", "light_power", "light_principal_direction", "mu")
INITIALIZATION_KEYS = ("rho", "pixels")

FRAMES = {
    "camera": "Camera frame of the pose (AliceVision / OpenCV): x right, y down, z forward (optical axis), origin at "
              "the camera center: X_camera = transpose(rotationCameraToWorld) (X_world - center).",
    "world": "World frame of the SfMData poses: X_world = rotationCameraToWorld X_camera + center (the AliceVision "
             "meshes are written in this frame with y and z negated).",
    "pixels": "(u, v) = (column, row), center of pixel i at coordinate i (AliceVision and LED-getter convention).",
}
MODEL = ("LED-getter LED model, per light l and color channel c, at a surface point x of normal n and albedo rho_c: "
         "w = (position_l - x) / |position_l - x|, "
         "irradiance_c = power_l / |position_l - x|^2 * max(0, -w . principalDirection_l)^anisotropy_c, "
         "image_c = rho_c * max(0, n . w) * irradiance_c (Lambertian).")
UNITS = ("Positions in SfMData world units (the units of the camera centers and of the proxy mesh); power in "
         "linear image intensity x world units^2 (relative to the albedo scale: the albedo is clamped to [0, 1]).")


def _vector(values):
    return [float(v) for v in np.asarray(values, np.float64).reshape(-1)]


def poseLights(poseId, camera, views, values, downscale):
    """Lights of a pose as a JSON dict, from the LED-getter values of the LED model (camera frame).

    Args:
        camera: ledCommon.Camera of the calibrated images.
        views: lighting views, in the order of the LED-getter lights.
        values: LED-getter values ("light_locations", "light_power", "light_principal_direction", "mu", ...).
    """
    missing = [key for key in LED_KEYS if key not in values]
    if missing:
        raise RuntimeError("Pose {}: LED model values missing: {}".format(poseId, ", ".join(missing)))
    locations = np.asarray(values["light_locations"], np.float64).reshape(-1, 3)
    if len(locations) != len(views):
        raise RuntimeError("Pose {}: {} calibrated lights for {} lighting images.".format(
            poseId, len(locations), len(views)))
    power = np.asarray(values["light_power"], np.float64).reshape(-1)
    directions = np.asarray(values["light_principal_direction"], np.float64).reshape(-1, 3)
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    positionsWorld = camera.cameraToWorld(locations)
    directionsWorld = camera.directionsToWorld(directions)
    lights = []
    for index, view in enumerate(views):
        lights.append({
            "viewId": str(view["viewId"]),
            "imagePath": view.get("path", ""),
            "power": float(power[index]),
            "positionCamera": _vector(locations[index]),
            "positionWorld": _vector(positionsWorld[index]),
            "principalDirectionCamera": _vector(directions[index]),
            "principalDirectionWorld": _vector(directionsWorld[index]),
        })
    return {
        "format": LIGHTS_FORMAT,
        "version": LIGHTS_FORMAT_VERSION,
        "poseId": str(poseId),
        "model": "LED",
        "modelDescription": MODEL,
        "frames": FRAMES,
        "units": UNITS,
        "camera": dict(camera.toDict(), downscale=int(downscale)),
        "anisotropy": _vector(values["mu"]),
        "lights": lights,
        "calibration": CALIBRATION_FILE,
        "ledgetterExports": LEDGETTER_FOLDER,
    }


def writePoseLights(folder, lights, values):
    """Write <folder>/<poseId>/lights.json and calibration.npz (the LED-getter values as float32/int32)."""
    poseFolder = os.path.join(folder, lights["poseId"])
    os.makedirs(poseFolder, exist_ok=True)
    with open(os.path.join(poseFolder, LIGHTS_FILE), "w") as f:
        json.dump(lights, f, indent=4)
    arrays = {key: np.asarray(value) for key, value in values.items()}
    np.savez(os.path.join(poseFolder, CALIBRATION_FILE), **arrays)


def writeSummary(path, posesLights):
    """Write the lights of all the poses in one JSON file."""
    with open(path, "w") as f:
        json.dump({"format": LIGHTS_FORMAT, "version": LIGHTS_FORMAT_VERSION, "frames": FRAMES, "model": MODEL,
                   "units": UNITS, "poses": posesLights}, f, indent=4)


def readPoseLights(folder, poseId):
    """Lights of a pose written by LedGetterLights (<folder>/<poseId>/lights.json)."""
    path = os.path.join(folder, str(poseId), LIGHTS_FILE)
    if not os.path.isfile(path):
        raise RuntimeError("No calibrated lights for pose {}: '{}' not found (output folder of LedGetterLights "
                           "expected).".format(poseId, path))
    with open(path) as f:
        lights = json.load(f)
    if lights.get("format") != LIGHTS_FORMAT or lights.get("model") != "LED":
        raise RuntimeError("'{}': unexpected lights format ({} / {}), '{}' with the LED model expected.".format(
            path, lights.get("format"), lights.get("model"), LIGHTS_FORMAT))
    return lights


def calibratedViewIds(lights):
    return {str(light["viewId"]) for light in lights["lights"]}


def ledValues(lights, viewIds):
    """LED-getter light values (camera frame, float32) of the lights of viewIds, in this order."""
    byView = {str(light["viewId"]): light for light in lights["lights"]}
    missing = [str(v) for v in viewIds if str(v) not in byView]
    if missing:
        raise RuntimeError("Pose {}: no calibrated light for the view(s) {}.".format(
            lights.get("poseId"), ", ".join(missing)))
    selected = [byView[str(v)] for v in viewIds]
    return {
        "light_locations": np.array([light["positionCamera"] for light in selected], np.float32).reshape(-1, 3),
        "light_power": np.array([light["power"] for light in selected], np.float32),
        "light_principal_direction": np.array([light["principalDirectionCamera"] for light in selected],
                                              np.float32).reshape(-1, 3),
        "mu": np.array(lights["anisotropy"], np.float32),
    }


def readInitialization(folder, lights, size, logger):
    """Albedo at the calibration pixels ("rho", "pixels"), to initialize the solve at size = (width, height).

    The pixels are rescaled when the solve resolution differs from the calibration one (downscale): pixel
    centers p' = (p + 0.5) * s - 0.5. Returns {} without calibration file (LED-getter then initializes the
    albedo from the images).
    """
    path = os.path.join(folder, str(lights["poseId"]), CALIBRATION_FILE)
    if not os.path.isfile(path):
        logger.warning("Pose {}: no '{}', the albedo is initialized from the images.".format(lights["poseId"], path))
        return {}
    with np.load(path) as archive:
        if not all(key in archive for key in INITIALIZATION_KEYS):
            return {}
        rho = np.asarray(archive["rho"], np.float32)
        pixels = np.asarray(archive["pixels"], np.float64)
    calibrationSize = (lights["camera"]["width"], lights["camera"]["height"])
    if tuple(calibrationSize) != tuple(size):
        scale = np.array([size[0] / float(calibrationSize[0]), size[1] / float(calibrationSize[1])])
        pixels = (pixels + 0.5) * scale - 0.5
    return {"rho": rho, "pixels": pixels.astype(np.float32)}


def checkSameCamera(lights, camera, logger, tolerance=1e-6):
    """Warn when the pose of the solve differs from the pose of the calibration (lights in the camera frame)."""
    rotation = np.array(lights["camera"]["rotationCameraToWorld"])
    center = np.array(lights["camera"]["center"])
    scale = max(1.0, float(np.linalg.norm(center)))
    if not (np.allclose(rotation, camera.rotation, atol=tolerance)
            and np.allclose(center, camera.center, atol=tolerance * scale)):
        logger.warning("Pose {}: the camera pose differs from the one of the light calibration: the lights are used "
                       "in the camera frame (rigid light / camera rig), check that both nodes use the same SfMData "
                       "(world units).".format(lights["poseId"]))
