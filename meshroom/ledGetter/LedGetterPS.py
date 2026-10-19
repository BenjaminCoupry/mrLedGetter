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

EXTRA_MAPS = ("albedo",)


class LedGetterPS(desc.Node):
    """Near-light photometric stereo (normals and albedo) with the lights calibrated by LedGetterLights."""

    category = "Photometric Stereo"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize("inputSfm")
    __nodeInfo__ = ledCommon.METHOD_INFO

    documentation = """
Estimate one normal map and one albedo map per multi-lighting pose with LED-getter (near-light photometric stereo
using a geometric proxy, B. Coupry et al., SSVM 2025), with the LED lights calibrated by LedGetterLights: per pixel,
the normal (initialized by the proxy mesh) and the albedo minimize a Huber loss between the Lambertian rendering
and the images.

""" + ledCommon.LINEAR_INPUT_NOTE + """

**Inputs:** the same SfMData (ExportImages: undistorted pinhole images) and proxy mesh as LedGetterLights, and its
output folder ('Lights'). Only the lighting images with a calibrated light are used. The pose mask (mask folder or
alpha channels, as in the other photometric stereo nodes) is intersected with the pixels whose ray hits the mesh.

**Outputs:** as the other photometric stereo nodes (LINOUniPS, UniMSPS, SDMUniPS): one normal map per pose
(<poseId>.png|exr, OpenGL camera frame by default), the pose masks (masks/<poseId>.png), the albedo maps
(albedo/<poseId>.png|exr, zero outside the mask; relative albedo, up to the scale of the light powers of the pose)
and one SfMData per map type (normalMaps.sfm, albedoMaps.sfm: one view per pose, intrinsics scaled by the
downscale factor), ready for RNb-NeuS2.
"""

    inputs = psCommon.inputAttributes() + ledCommon.meshAttributes() + [
        desc.File(
            name="inputLights",
            label="Lights",
            description="Output folder of LedGetterLights: <poseId>/lights.json (and calibration.npz, used to "
                        "initialize the albedo).",
            value="",
        ),
        desc.IntParam(
            name="iterations",
            label="Iterations",
            description="Gradient descent iterations of the per-pixel solve.",
            value=1000,
            range=(100, 30000, 100),
        ),
        desc.IntParam(
            name="splitFactor",
            label="Pixel Split",
            description="The pixels are solved in N x N interleaved subsets (one pixel every N rows and N columns, "
                        "LED-getter '--step' of the photometric stereo), one after the other, then merged at full "
                        "resolution. Higher values use less memory; the images are read again for each subset. "
                        "1 solves all the pixels at once.",
            value=2,
            range=(1, 8, 1),
        ),
        desc.IntParam(
            name="gpuChunks",
            label="GPU Chunks Per Subset",
            description="Each pixel subset is solved in this number of successive chunks on the GPU (LED-getter "
                        "'--ps_chunck_number'): increase it if the GPU runs out of memory.",
            value=100,
            range=(1, 1000, 1),
            advanced=True,
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
                        "image values): residuals above it are penalized linearly.",
            value=0.01,
            range=(0.001, 0.1, 0.001),
            advanced=True,
        ),
    ] + psCommon.advancedInputAttributes() + psCommon.settingsAttributes()

    outputs = psCommon.outputAttributes(extraMaps=EXTRA_MAPS)

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            from . import ledBackend
            run(chunk, ledBackend)
        finally:
            chunk.logManager.end()


def solveSubset(offset, camera, mask, selected, params, node, scene, lights, settings, backend, logger):
    """Solve the pixels of one interleaved subset; returns (pixels (n, 2) (u, v), normals, albedo) or None."""
    split = node.splitFactor.value
    rows, cols = ledCommon.gridIndices(camera.height, camera.width, split, offset)
    gridMask = mask[np.ix_(rows, cols)] if mask is not None else np.ones((len(rows), len(cols)), bool)
    pixels = ledCommon.gridPixels(rows, cols, gridMask)
    if not len(pixels):
        return None
    hit, points, normals, objectsId = backend.castRays(scene, camera.K, pixels)
    if not hit.any():
        return None
    # only the pixels seeing the mesh are read (memory)
    images = ledCommon.samplePixels(selected, params, pixels[hit], (camera.width, camera.height))
    valid = np.all(np.isfinite(images), axis=(1, 2))
    keep = np.flatnonzero(hit)[valid]
    if not len(keep):
        return None
    gridMask[gridMask] = np.isin(np.arange(len(pixels)), keep)
    data = {"images": images[valid], "pixels": pixels[keep], "points": points[keep], "normals": normals[keep],
            "objectsId": objectsId[keep], "gridMask": gridMask}
    logger.debug("Subset {}: {} pixels".format(offset, len(keep)))
    result = backend.solvePhotometricStereo(scene, data, lights, settings, logger)
    return pixels[keep], result["normals"], result["rho"]


def solvePose(poseId, views, sfm, params, node, settings, backend, logger):
    """Normals (OpenGL camera frame), albedo and support of one pose."""
    if psCommon.representativeView(poseId, views) is None:
        raise RuntimeError("no view with viewId == poseId to hold the pose in the output SfMData")
    lightsFolder = node.inputLights.value
    lights = lightsIO.readPoseLights(lightsFolder, poseId)
    camera = ledCommon.poseCamera(sfm, poseId, views, params.downscale)
    lightsIO.checkSameCamera(lights, camera, logger)
    calibrated = lightsIO.calibratedViewIds(lights)
    excluded = (psCommon.poseReferenceViewId(poseId, views, params.poseReferencePattern)
                if params.excludePoseReferenceImage else None)
    uncalibrated = [str(v["viewId"]) for v in views if str(v["viewId"]) not in calibrated | {excluded}]
    if uncalibrated:
        logger.warning("Pose {}: {} view(s) without calibrated light ignored: {}".format(
            poseId, len(uncalibrated), ", ".join(uncalibrated)))
    selected = ledCommon.lightingViews(poseId, views, params, calibrated)
    if len(selected) < ledCommon.MIN_LIGHTS:
        raise RuntimeError("{} lighting image(s) with a calibrated light, at least {} are needed".format(
            len(selected), ledCommon.MIN_LIGHTS))
    size = (camera.width, camera.height)
    lightValues = lightsIO.ledValues(lights, [v["viewId"] for v in selected])
    lightValues.update(lightsIO.readInitialization(lightsFolder, lights, size, logger))
    mask, source = ledCommon.poseMask(poseId, views, params, size, logger)
    logger.info("Pose {}: {} lighting images, {}x{}, mask: {}".format(
        poseId, len(selected), camera.width, camera.height, source))
    scene = backend.loadMesh(node.inputMesh.value, node.meshAliceVisionAxes.value, camera.rotation, camera.center)
    normalsCv = np.zeros((camera.height, camera.width, 3), np.float32)
    albedo = np.zeros((camera.height, camera.width, 3), np.float32)
    solved = np.zeros((camera.height, camera.width), bool)
    for offset in ledCommon.splitOffsets(node.splitFactor.value):
        result = solveSubset(offset, camera, mask, selected, params, node, scene, lightValues, settings, backend,
                             logger)
        if result is None:
            continue
        pixels, normals, rho = result
        normalsCv[pixels[:, 1], pixels[:, 0]] = normals
        albedo[pixels[:, 1], pixels[:, 0]] = rho
        solved[pixels[:, 1], pixels[:, 0]] = True
    if not solved.any():
        raise RuntimeError("no pixel of the pose mask sees the proxy mesh: check that the mesh is in the SfMData "
                           "world frame and the 'AliceVision Mesh Axes' option.")
    normal, support = psCommon.finalizeNormals(ledCommon.ledgetterToOpenGL(normalsCv), solved)
    return normal, albedo, support


def run(chunk, backend):
    """Photometric stereo on every multi-lighting pose (backend: ledBackend or a stand-in), with the outputs of
    psCommon.processPoses (the poses need the camera, the mesh and the lights: psCommon's predict(images, mask)
    does not provide them)."""
    node, logger = chunk.node, chunk.logger
    params = psCommon.Params(node)
    logger.info("LedGetterPS {} (photometric stereo common layer v{}): {}".format(
        __version__, psCommon.PS_COMMON_VERSION, params.describe()))
    outputFolder = params.outputFolder
    os.makedirs(outputFolder, exist_ok=True)
    sfm = psCommon.loadSfm(params.inputSfm, outputFolder)
    sfm, poses = psCommon.filterPoses(sfm, params.minViewsPerPose, logger)
    psCommon.saveSfm({k: v for k, v in sfm.items() if k != "structure"},
                     os.path.join(outputFolder, "photometricStereoViews.sfm"))
    logger.info("{} multi-lighting pose(s) to process.".format(len(poses)))
    settings = {"device": backend.selectDevice(params.useGpu, logger), "iterations": int(node.iterations.value),
                "learningRate": float(node.learningRate.value), "delta": float(node.delta.value),
                "chunks": int(node.gpuChunks.value)}
    extension = psCommon.mapExtension(node)
    mapPaths = {name: {} for name in ("normal",) + EXTRA_MAPS}
    processed, failed = [], []
    for index, (poseId, views) in enumerate(poses.items()):
        start = time.time()
        try:
            normal, albedo, support = solvePose(poseId, views, sfm, params, node, settings, backend, logger)
            normalPath = os.path.join(outputFolder, poseId + extension)
            psCommon.writeMap(normalPath, psCommon.toConvention(normal, params.normalConvention), params.outputFormat,
                              signed=True)
            psCommon.writeMask(os.path.join(params.outputMaskFolder, poseId + ".png"), support)
            albedoPath = os.path.join(outputFolder, "albedo", poseId + extension)
            psCommon.writeMap(albedoPath, albedo * support[:, :, None], params.outputFormat, signed=False)
            mapPaths["normal"][poseId] = normalPath
            mapPaths["albedo"][poseId] = albedoPath
            processed.append(poseId)
            logger.info("Pose {} ({}/{}): done in {:.1f}s ({:.1f}% of the image with normals)".format(
                poseId, index + 1, len(poses), time.time() - start, 100.0 * support.mean()))
        except Exception as exc:
            logger.error("Pose {}: failed: {}\n{}".format(poseId, exc, traceback.format_exc()))
            failed.append(poseId)

    sfmOutputs = {"normal": node.outputSfmDataNormal.value, "albedo": node.outputSfmDataAlbedo.value}
    for name, sfmPath in sfmOutputs.items():
        psCommon.saveSfm(psCommon.buildOutputSfm(sfm, mapPaths[name], params.downscale, params.keepLandmarks), sfmPath)
        logger.info("Saved {} ({} pose(s))".format(sfmPath, len(mapPaths[name])))
    logger.info("{} of {} pose(s) processed.".format(len(processed), len(poses)))
    psCommon.checkFailurePolicy(params.failurePolicy, len(processed), failed)
