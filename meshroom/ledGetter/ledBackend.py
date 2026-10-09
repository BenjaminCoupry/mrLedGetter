"""Calls of the LED-getter library (JAX), isolated from the rest of the nodes.

This is the only module using JAX and ledgetter. They are imported inside the functions: Meshroom loads every
module of the plugin to find the node descriptions, without the plugin environment; the computation runs in the
plugin virtual environment. The rest of the nodes only exchanges NumPy arrays with these functions (and an opaque
mesh scene), so that it is tested on CPU with a stand-in backend (tests/helpers.FakeBackend).

The functions reproduce LED-getter's own pipeline (estimate.py, ledgetter.pipeline.preprocessing) from arrays
instead of file paths: the images are loaded, linearized and masked by the nodes (psCommon / ledCommon), the mesh
is loaded and ray cast by LED-getter. LED-getter's camera frame is the AliceVision / OpenCV one (x right, y down,
z forward) and the ray of the pixel (u, v) is K^-1 (u, v, 1) (center of pixel i at i, as in AliceVision).

GPU memory: JAX defaults are kept (the GPU memory is preallocated on first use, 75% of it). Set
XLA_PYTHON_CLIENT_PREALLOCATE=false to allocate on demand, or XLA_PYTHON_CLIENT_MEM_FRACTION to change the
preallocated fraction (environment or meshroom/config.json).
"""

import os

CALIBRATION_STAGES = ("directional", "punctual", "LED")
CHANNELS = 3


def selectDevice(useGpu, logger):
    """JAX platform of the heavy computations: "gpu" if requested and available, "cpu" otherwise."""
    import jax
    logger.info("JAX {}, devices: {}; XLA_PYTHON_CLIENT_PREALLOCATE={}, XLA_PYTHON_CLIENT_MEM_FRACTION={}".format(
        jax.__version__, ", ".join(str(d) for d in jax.devices()),
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "<default>"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "<default>")))
    if useGpu:
        try:
            jax.devices("gpu")
            return "gpu"
        except RuntimeError:
            logger.warning("'Use GPU' is enabled but JAX has no GPU device: using the CPU.")
    return "cpu"


def loadMesh(meshPath, flipMesh, rotation, center):
    """Ray caster of the proxy mesh moved to the camera frame of a pose.

    Args:
        flipMesh: undo the AliceVision mesh convention (y and z negated with respect to the SfMData world frame).
        rotation: camera-to-world rotation of the pose; center: camera center (SfMData world frame).
    """
    import jax.numpy as jnp
    import ledgetter.image.camera as lgCamera
    import ledgetter.space.raycasting as raycasting
    import ledgetter.utils.loading as loading
    if not meshPath or not os.path.isfile(meshPath):
        raise RuntimeError("Proxy mesh not found: '{}'".format(meshPath))
    worldToCamera = lgCamera.get_rototranslation_matrix(jnp.asarray(rotation, jnp.float32),
                                                        jnp.asarray(center, jnp.float32), to_camera=True)
    mesh = loading.load_mesh(meshPath, transform=worldToCamera, flip_mesh=flipMesh)
    return raycasting.get_mesh_raycaster(mesh)


def castRays(scene, K, pixels):
    """Intersections of the rays of pixels (u, v) with the mesh (CPU, Open3D).

    Returns:
        (hit bool (n,), points (n, 3), normals (n, 3), object ids (n,)), in the camera frame.
    """
    import jax.numpy as jnp
    import numpy as np
    import ledgetter.image.camera as lgCamera
    geometry = lgCamera.get_geometry(scene, jnp.asarray(K, jnp.float32))
    hit, normals, points, objectsId = geometry(jnp.asarray(pixels, jnp.int32))
    return (np.asarray(hit, bool), np.asarray(points, np.float32), np.asarray(normals, np.float32),
            np.asarray(objectsId, np.int32))


def _ledgetterInputs(data):
    """LED-getter values, images, grid mask, shapes and scene scale (as ledgetter.pipeline.preprocessing)."""
    import jax.numpy as jnp
    images = jnp.asarray(data["images"], jnp.float32)
    nbPixels, nbImages = images.shape[0], images.shape[2]
    points = jnp.asarray(data["points"], jnp.float32)
    values = {
        "points": points,
        "normals": jnp.asarray(data["normals"], jnp.float32),
        "pixels": jnp.asarray(data["pixels"], jnp.int32),
        "objects_id": jnp.asarray(data["objectsId"], jnp.int32),
        "filters": jnp.tile(jnp.arange(CHANNELS, dtype=jnp.int32), (nbPixels, 1)),
    }
    scale = jnp.quantile(jnp.linalg.norm(points - jnp.mean(points, axis=0), axis=-1), 0.95)
    shapes = (nbPixels, nbImages, CHANNELS, CHANNELS)
    return values, images, jnp.asarray(data["gridMask"], bool), shapes, scale


def calibrateLights(scene, data, settings, exportFolder, lightNames, pose, logger):
    """Self-calibration of the lights of a pose: directional, then punctual, then LED model, each stage starting
    from the previous one (LED-getter estimate.py with --pattern directional punctual LED).

    Args:
        scene: loadMesh result (cast shadows).
        data: "images" (n, 3, m) linear values of the m lighting images at the n pixels, "pixels" (n, 2) (u, v),
            "points", "normals" (n, 3) and "objectsId" (n,) from castRays, "gridMask": bool grid of the sampled
            pixels (the n pixels are its True nodes in row-major order, used by the validity morphology).
        settings: "device" ("gpu" / "cpu"), "iterations" (per stage), "learningRate", "delta" (Huber loss),
            "exportImages" (also export the per-light images and light maps).
        exportFolder: LED-getter exports of each stage (<exportFolder>/<stage>/), None to skip them.
        lightNames: names of the m lights (viewIds); pose: camera dict exported with the results.

    Returns:
        {"model": "LED", "values": LED-getter values as NumPy arrays (camera frame)}.
    """
    import jax
    import numpy as np
    import optax
    import ledgetter.pipeline.light_estimation as lightEstimation
    import ledgetter.pipeline.outputs as outputs
    import ledgetter.utils.logs as logs
    iterations = int(settings["iterations"])
    skip = None if settings.get("exportImages") else ["images", "lightmaps"]
    with jax.default_device(jax.devices(settings["device"])[0]):
        values, images, gridMask, shapes, scale = _ledgetterInputs(data)
        output = logs.get_tqdm_output(max(1, iterations // 10))
        optimizer = optax.adam(settings["learningRate"])
        lightDict = {"model": None, "light_values": {}, "light": None, "losses": []}
        for stage in CALIBRATION_STAGES:
            logger.info("Light calibration, {} model: {} iterations on {} pixels x {} images".format(
                stage, iterations, shapes[0], shapes[1]))
            lightDict, validity = lightEstimation.estimate_light(iterations, stage, values, images, gridMask, scene,
                                                                 shapes, output, optimizer, scale, lightDict,
                                                                 delta=settings["delta"])
            logger.info("  final loss {:.4e}, {:.1f}% valid observations".format(
                float(np.asarray(lightDict["losses"][-1][1])[-1]), 100.0 * float(np.mean(np.asarray(validity)))))
            if exportFolder:
                outputs.export_results(os.path.join(exportFolder, stage), shapes, validity, lightDict, gridMask,
                                       images, list(lightNames), skip=skip, pose=pose)
    return {"model": lightDict["model"]["light"],
            "values": {key: np.asarray(value) for key, value in lightDict["light_values"].items()}}


def solvePhotometricStereo(scene, data, lights, settings, logger):
    """Normals and albedo of pixels with calibrated LED lights (LED-getter estimate.py --pattern PS).

    The data are kept on the CPU and solved by chunks on settings["device"] (as LED-getter with --backend cpu
    --ps_backend gpu).

    Args:
        data: as calibrateLights, for the pixels to solve.
        lights: LED-getter light values in the camera frame ("light_locations", "light_power",
            "light_principal_direction", "mu") of the m images, plus optionally "rho" and "pixels" from the
            calibration (albedo initialization, interpolated at the nearest pixel).
        settings: "device", "iterations", "learningRate", "delta", "chunks" (number of chunks of the solve).

    Returns:
        {"normals": (n, 3) camera frame (x right, y down, z forward), "rho": (n, 3) albedo}.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import ledgetter.models.models as models
    import ledgetter.pipeline.ps_estimation as psEstimation
    import ledgetter.utils.functions as functions
    import ledgetter.utils.logs as logs
    with jax.default_device(jax.devices("cpu")[0]):
        values, images, gridMask, shapes, scale = _ledgetterInputs(data)
        lightValues = {key: jnp.asarray(value) for key, value in lights.items()}
        ledLight = models.get_light("LED")
        # same light function as ledgetter.utils.loading.load_light for values loaded from a file
        light = functions.filter_args(jax.jit(lambda points, pixels, shapes: ledLight(
            **(lightValues | {"points": points, "pixels": pixels, "shapes": shapes}))))
        lightDict = {"model": {"light": "LED"}, "light_values": lightValues, "light": light, "losses": []}
        logger.info("Photometric stereo: {} iterations on {} pixels x {} images, {} chunks".format(
            settings["iterations"], shapes[0], shapes[1], settings["chunks"]))
        lightDict, _ = psEstimation.estimate_ps(int(settings["iterations"]), values, images, gridMask, scene, shapes,
                                                logs.get_tqdm_output(1), optax.adam(settings["learningRate"]), scale,
                                                lightDict, delta=settings["delta"],
                                                chunck_number=int(settings["chunks"]), return_ps_only=True,
                                                ps_backend=settings["device"])
    solved = lightDict["light_values"]
    return {"normals": np.asarray(solved["normals"], np.float32), "rho": np.asarray(solved["rho"], np.float32)}
