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

MIN_PIXELS = 50  # sanity floor: fewer pixels on the mesh means a misplaced mesh or mask


def uidIgnoredWhen(value, attribute):
    """Attribute whose given value does not contribute to the node UID: an option added with its default value keeps
    the UID (and the computed caches) of the existing nodes. desc.BoolParam has no 'uidIgnoreValue' argument (only
    desc.StringParam has one): it is set on the description."""
    attribute._uidIgnoreValue = value
    return attribute


class LedGetterLights(desc.Node):
    """Self-calibration of the near lights of each multi-lighting pose with LED-getter."""

    category = "Photometric Stereo"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("inputSfm")
    __nodeInfo__ = ledCommon.METHOD_INFO

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

**Light rig:** when the lights are fixed to the camera (LED ring, dome or arm carrying the camera, the object moving
between poses), enable 'Light Rig': the poses lit by the same sequence of lights share one set of lights, calibrated
jointly on the pixels of all of them (one optimization, each shadow ray cast on the mesh of its pose). The light of
an image is identified by its frame number (the number in its filename): by rank in the pose (same shooting sequence)
or by the number itself (same numbering). The poses whose sequence of lights is shared by no other pose (e.g. a
different number of images) are calibrated alone. Each pose still gets its lights.json (camera and world frames).

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
        uidIgnoredWhen(False, desc.BoolParam(
            name="lightRig",
            label="Light Rig",
            description="The lights are fixed to the camera (the object moves between poses): the poses lit by the "
                        "same sequence of lights share one set of lights, calibrated jointly on all of them. The "
                        "poses whose sequence is shared by no other pose are calibrated alone.",
            value=False,
        )),
        desc.ChoiceParam(
            name="rigLightMatching",
            label="Rig Light Matching",
            description="How an image is matched to a light of the rig, from its frame number (the number in its "
                        "filename):\n"
                        " - rank: the n-th image of every pose, in frame number order (same shooting sequence in "
                        "every pose; poses with another number of images are calibrated alone).\n"
                        " - frameId: the same frame number in every pose (e.g. led_3-0051.nef, RTI_00012.NEF).",
            values=["rank", "frameId"],
            value="rank",
            exclusive=True,
            enabled=lambda node: node.lightRig.value,
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


def preparePose(poseId, selected, views, sfm, params, node, backend, logger):
    """Calibration data of one pose: its camera, the LED-getter data at the sampled pixels of the pose mask that see
    the mesh (ledBackend.calibrateLights) and the mesh scene in its camera frame."""
    camera = ledCommon.poseCamera(sfm, poseId, views, params.downscale)
    if len(selected) < ledCommon.MIN_LIGHTS:
        raise RuntimeError("{} lighting image(s), at least {} are needed".format(len(selected),
                                                                                 ledCommon.MIN_LIGHTS))
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
    return camera, data, scene


def calibrateResult(backend, scene, data, settings, exportFolder, lightNames, camera, logger):
    """LED-getter light values (camera frame) of a calibration."""
    result = backend.calibrateLights(scene, data, settings, exportFolder, lightNames, ledgetterPose(camera), logger)
    if result["model"] != "LED":
        raise RuntimeError("unexpected light model '{}' (LED expected)".format(result["model"]))
    return result["values"]


def calibratePose(poseId, views, sfm, params, node, settings, backend, logger):
    """Calibrate the lights of one pose and write them; returns the pose lights (JSON dict)."""
    selected = ledCommon.lightingViews(poseId, views, params)
    camera, data, scene = preparePose(poseId, selected, views, sfm, params, node, backend, logger)
    outputFolder = node.outputFolder.value
    exportFolder = os.path.join(outputFolder, poseId, lightsIO.LEDGETTER_FOLDER)
    values = calibrateResult(backend, scene, data, settings, exportFolder, [str(v["viewId"]) for v in selected],
                             camera, logger)
    lights = lightsIO.poseLights(poseId, camera, selected, values, params.downscale)
    lightsIO.writePoseLights(outputFolder, lights, values)
    return lights


def rigLighting(poses, params, matching):
    """Lighting views of each pose for a light rig: {poseId: (views ordered by light key, keys)}."""
    lighting = {}
    for poseId, views in poses.items():
        referenceId = (psCommon.poseReferenceViewId(poseId, views, params.poseReferencePattern)
                       if params.excludePoseReferenceImage else None)
        candidates = [v for v in views if str(v.get("viewId")) != referenceId]
        lighting[poseId] = ledCommon.rigLightKeys(candidates, matching)
    return lighting


def calibrateRig(poseIds, keys, lighting, poses, sfm, params, node, settings, backend, logger):
    """Calibrate one set of lights on the poses of a light rig and write the lights of each pose.

    Args:
        keys: light keys shared by the poses; lighting: rigLighting result.

    Returns:
        ({poseId: pose lights}, [poseIds that could not be prepared])
    """
    selectedKeys = psCommon.selectViews(keys, params.nbImages, params.imageSelection,
                                        psCommon.poseSeed(params.seed, poseIds[0]))
    prepared, failed = [], []
    for poseId in poseIds:
        ordered, poseKeys = lighting[poseId]
        byKey = dict(zip(poseKeys, ordered))
        selected = [byKey[key] for key in selectedKeys]
        try:
            prepared.append((poseId, selected) + preparePose(poseId, selected, poses[poseId], sfm, params, node,
                                                             backend, logger))
        except Exception as exc:
            logger.error("Pose {}: failed: {}\n{}".format(poseId, exc, traceback.format_exc()))
            failed.append(poseId)
    if not prepared:
        return {}, failed
    data, counts = ledCommon.stackPoseData([item[3] for item in prepared])
    scene = backend.mergeScenes([item[4] for item in prepared], counts)
    logger.info("Light rig: {} lights shared by {} pose(s) ({}), joint calibration on {} pixels.".format(
        len(selectedKeys), len(prepared), ", ".join(item[0] for item in prepared), sum(counts)))
    outputFolder = node.outputFolder.value
    exportFolder = os.path.join(outputFolder, "rig_" + prepared[0][0], lightsIO.LEDGETTER_FOLDER)
    values = calibrateResult(backend, scene, data, settings, exportFolder, [str(key) for key in selectedKeys],
                             prepared[0][2], logger)
    rig = {"poses": [item[0] for item in prepared], "lightKeys": [str(key) for key in selectedKeys],
           "matching": node.rigLightMatching.value, "ledgetterExports": os.path.relpath(exportFolder, outputFolder)}
    calibrated = {}
    for index, (poseId, selected, camera, _, _) in enumerate(prepared):
        poseValues = ledCommon.splitPoseValues(values, counts, index)
        lights = dict(lightsIO.poseLights(poseId, camera, selected, poseValues, params.downscale), rig=rig)
        lightsIO.writePoseLights(outputFolder, lights, poseValues)
        calibrated[poseId] = lights
    return calibrated, failed


def calibrateAll(poses, sfm, params, node, settings, backend, logger):
    """Calibrate the lights of every pose (alone, or jointly with the poses of its light rig).

    Returns:
        ({poseId: pose lights}, [failed poseIds])
    """
    if node.lightRig.value:
        lighting = rigLighting(poses, params, node.rigLightMatching.value)
        groups = ledCommon.rigGroups({poseId: keys for poseId, (_, keys) in lighting.items()})
    else:
        lighting, groups = None, [(None, [poseId]) for poseId in poses]
    calibrated, failed = {}, []
    for keys, poseIds in groups:
        start = time.time()
        if len(poseIds) > 1:
            rigCalibrated, rigFailed = calibrateRig(poseIds, keys, lighting, poses, sfm, params, node, settings,
                                                    backend, logger)
            calibrated.update(rigCalibrated)
            failed += rigFailed
            logger.info("Light rig of {} pose(s): lights calibrated in {:.1f}s".format(len(rigCalibrated),
                                                                                       time.time() - start))
            continue
        poseId = poseIds[0]
        if node.lightRig.value:
            logger.info("Pose {}: its sequence of {} lights is shared by no other pose, calibrated alone.".format(
                poseId, len(keys)))
        try:
            calibrated[poseId] = calibratePose(poseId, poses[poseId], sfm, params, node, settings, backend, logger)
            logger.info("Pose {} ({}/{}): lights calibrated in {:.1f}s".format(
                poseId, len(calibrated), len(poses), time.time() - start))
        except Exception as exc:
            logger.error("Pose {}: failed: {}\n{}".format(poseId, exc, traceback.format_exc()))
            failed.append(poseId)
    return calibrated, failed


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
    calibrated, failed = calibrateAll(poses, sfm, params, node, settings, backend, logger)
    lightsIO.writeSummary(node.outputLights.value, calibrated)
    logger.info("{} of {} pose(s) calibrated.".format(len(calibrated), len(poses)))
    psCommon.checkFailurePolicy(params.failurePolicy, len(calibrated), failed)
