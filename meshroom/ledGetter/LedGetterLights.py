__version__ = "1.0"
__author__ = "Benjamin Coupry"
__license__ = "MIT"

import os
import time
import traceback

import numpy as np

from meshroom.core import desc

from . import ledCommon
from . import lightsIO
from . import psCommon

METHOD_INFO = [
    ("method", "Self-calibrated Near-light Photometric Stereo using a Geometric Proxy (B. Coupry, F. Lauze, "
               "A. Laurent, J. Mélou, Y. Quéau, J.-D. Durou, SSVM 2025)"),
    ("methodLicense", "MIT"),
    ("methodRepository", "https://github.com/BenjaminCoupry/LED-getter"),
]
MIN_LIGHTS = 3
MIN_PIXELS = 50  # sanity floor: fewer pixels on the mesh means a misplaced mesh or mask


class LedGetterLights(desc.Node):
    """Self-calibration of the near lights of each multi-lighting pose with LED-getter."""

    category = "Photometric Stereo"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("inputSfm")
    __nodeInfo__ = METHOD_INFO

    documentation = """
Self-calibration of the lights of each multi-lighting pose with LED-getter (near-light photometric stereo using a
geometric proxy, B. Coupry et al., SSVM 2025). Given the 3D points and normals of a proxy mesh seen by the camera,
the lights are fitted to the images (Lambertian surface, Huber loss) with successive models, each initialized by
the previous one: directional, punctual, then LED (point source with a principal direction and an anisotropy).
This is the expensive step: its output folder is the 'Lights' input of LedGetterPS.

""" + ledCommon.LINEAR_INPUT_NOTE + """

**Inputs:** an SfMData where the lighting images of a pose share the same poseId (as created by CameraInit for
multi-lighting folders): undistorted pinhole images, i.e. ExportImages. The proxy mesh (Meshing / MeshFiltering)
must be in the SfMData world frame. The pose masks (mask folder or alpha channels, as in the other photometric
stereo nodes) are restricted to the pixels whose ray hits the mesh; the lights are calibrated on one pixel every
'Pixel Sampling Step' rows and columns.

**Outputs:** per pose, <poseId>/lights.json (position, power and principal direction of each light, in the camera
frame of the pose and in the SfMData world frame, anisotropy, camera), <poseId>/calibration.npz (albedo
initialization of LedGetterPS) and the LED-getter exports of each model (<poseId>/ledgetter/<model>/:
light_plot.html, loss_plot.html, values...); lights.json gathers all the poses.

**Frames:** camera frame x right, y down, z forward (AliceVision / OpenCV), world frame of the SfMData poses,
SfMData world units; pixel (u, v) = (column, row) with the center of pixel i at i (AliceVision and LED-getter).
"""

    inputs = psCommon.inputAttributes() + ledCommon.meshAttributes() + [
        desc.IntParam(
            name="samplingStep",
            label="Pixel Sampling Step",
            description="The lights are calibrated on one pixel every N rows and N columns of the pose mask (N^2 times "
                        "fewer pixels than the images, LED-getter '--step'). 1 uses every pixel: slower, more "
                        "memory.",
            value=10,
            range=(1, 30, 1),
        ),
        desc.IntParam(
            name="iterations",
            label="Iterations",
            description="Gradient descent iterations of each light model (directional, punctual, LED).",
            value=10000,
            range=(100, 30000, 100),
        ),
        desc.FloatParam(
            name="learningRate",
            label="Learning Rate",
            description="Learning rate of the Adam optimizer.",
            value=0.001,
            range=(0.0001, 0.1, 0.0001),
            advanced=True,
        ),
        desc.FloatParam(
            name="delta",
            label="Huber Loss Delta",
            description="Threshold of the Huber loss between the rendered and the observed intensities (linear "
                        "image values): residuals above it are penalized linearly (robustness to shadows, "
                        "highlights and outliers).",
            value=0.01,
            range=(0.001, 0.1, 0.001),
            advanced=True,
        ),
        desc.BoolParam(
            name="exportImages",
            label="Export Diagnostic Images",
            description="Also export the LED-getter images of each light (reference, simulated, residual, validity, "
                        "light maps): many files per light.",
            value=False,
            advanced=True,
        ),
    ] + ledCommon.selectAttributes(psCommon.advancedInputAttributes(), ("normalConvention", "keepLandmarks")) \
        + ledCommon.selectAttributes(psCommon.settingsAttributes(), ("outputFormat",))

    outputs = [
        desc.File(
            name="outputFolder",
            label="Lights Folder",
            description="Calibrated lights: <poseId>/lights.json, <poseId>/calibration.npz and the LED-getter "
                        "exports <poseId>/ledgetter/. Input of LedGetterPS.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="outputLights",
            label="Lights",
            description="Lights of all the poses (JSON, camera and world frames).",
            value="{nodeCacheFolder}/lights.json",
        ),
    ]

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            from . import ledBackend
            run(chunk, ledBackend)
        finally:
            chunk.logManager.end()


def ledgetterPose(camera):
    """Camera in the LED-getter pose format (exported with its results)."""
    return {"K": camera.K.tolist(), "R": camera.rotation.tolist(), "t": camera.center.tolist(),
            "width": camera.width, "height": camera.height}


def calibratePose(poseId, views, sfm, params, node, settings, backend, logger):
    """Calibrate the lights of one pose and write them; returns the pose lights (JSON dict)."""
    camera = ledCommon.poseCamera(sfm, poseId, views, params.downscale)
    selected = ledCommon.lightingViews(poseId, views, params)
    if len(selected) < MIN_LIGHTS:
        raise RuntimeError("{} lighting image(s), at least {} are needed".format(len(selected), MIN_LIGHTS))
    size = (camera.width, camera.height)
    mask, source = ledCommon.poseMask(poseId, views, params, size, logger)
    rows, cols = ledCommon.gridIndices(camera.height, camera.width, node.samplingStep.value)
    gridMask = mask[np.ix_(rows, cols)] if mask is not None else np.ones((len(rows), len(cols)), bool)
    pixels = ledCommon.gridPixels(rows, cols, gridMask)
    # the images are read first: non-linear inputs fail before the mesh is loaded
    images = ledCommon.samplePixels(selected, params, pixels, size)
    scene = backend.loadMesh(node.inputMesh.value, node.meshAliceVisionAxes.value, camera.rotation, camera.center)
    hit, points, normals, objectsId = backend.castRays(scene, camera.K, pixels)
    valid = hit & np.all(np.isfinite(images), axis=(1, 2))
    if valid.sum() < MIN_PIXELS:
        raise RuntimeError("only {} sampled pixel(s) of the pose mask see the proxy mesh ({} needed): check that the "
                           "mesh is in the SfMData world frame and the 'AliceVision Mesh Axes' option."
                           .format(int(valid.sum()), MIN_PIXELS))
    gridMask[gridMask] = valid
    logger.info("Pose {}: {} lighting images, {}x{}, mask: {}, {} calibration pixels (step {}, {} on the mesh)".format(
        poseId, len(selected), camera.width, camera.height, source, int(valid.sum()), node.samplingStep.value,
        int(hit.sum())))
    data = {"images": images[valid], "pixels": pixels[valid], "points": points[valid], "normals": normals[valid],
            "objectsId": objectsId[valid], "gridMask": gridMask}
    outputFolder = node.outputFolder.value
    exportFolder = os.path.join(outputFolder, poseId, lightsIO.LEDGETTER_FOLDER)
    result = backend.calibrateLights(scene, data, settings, exportFolder, [str(v["viewId"]) for v in selected],
                                     ledgetterPose(camera), logger)
    if result["model"] != "LED":
        raise RuntimeError("unexpected light model '{}' (LED expected)".format(result["model"]))
    lights = lightsIO.poseLights(poseId, camera, selected, result["values"], params.downscale)
    lightsIO.writePoseLights(outputFolder, lights, result["values"])
    return lights


def run(chunk, backend):
    """Calibrate the lights of every multi-lighting pose (backend: ledBackend or a stand-in)."""
    node, logger = chunk.node, chunk.logger
    params = ledCommon.commonParams(node)
    logger.info("LedGetterLights {}: {}".format(__version__, params.describe()))
    outputFolder = node.outputFolder.value
    os.makedirs(outputFolder, exist_ok=True)
    sfm = psCommon.loadSfm(params.inputSfm, outputFolder)
    sfm, poses = psCommon.filterPoses(sfm, params.minViewsPerPose, logger)
    logger.info("{} multi-lighting pose(s) to calibrate.".format(len(poses)))
    settings = {"device": backend.selectDevice(params.useGpu, logger), "iterations": int(node.iterations.value),
                "learningRate": float(node.learningRate.value), "delta": float(node.delta.value),
                "exportImages": bool(node.exportImages.value)}
    calibrated, failed = {}, []
    for index, (poseId, views) in enumerate(poses.items()):
        start = time.time()
        try:
            calibrated[poseId] = calibratePose(poseId, views, sfm, params, node, settings, backend, logger)
            logger.info("Pose {} ({}/{}): lights calibrated in {:.1f}s".format(
                poseId, index + 1, len(poses), time.time() - start))
        except Exception as exc:
            logger.error("Pose {}: failed: {}\n{}".format(poseId, exc, traceback.format_exc()))
            failed.append(poseId)
    lightsIO.writeSummary(node.outputLights.value, calibrated)
    logger.info("{} of {} pose(s) calibrated.".format(len(calibrated), len(poses)))
    psCommon.checkFailurePolicy(params.failurePolicy, len(calibrated), failed)
