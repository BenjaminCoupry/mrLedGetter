"""Shared code of the LED-getter nodes (LedGetterLights, LedGetterPS), without any JAX / LED-getter import.

On top of psCommon (the common photometric stereo layer shared by the PS plugins), it provides:

- the node attributes specific to LED-getter,
- the pinhole camera of a pose, read from the SfMData (intrinsics, extrinsics, frame conversions),
- the sampling of linear image values at given pixels, in the LED-getter layout,
- the pixel grids of the light calibration (subsampling) and of the photometric stereo solve (split).

Conventions (shared by AliceVision and LED-getter, no conversion needed):

- Camera frame: x right, y down, z forward (optical axis), origin at the camera center, in SfMData world units:
  X_camera = R_worldToCamera (X_world - center). LED-getter casts the ray K^-1 (u, v, 1) through the pixel (u, v)
  in this frame.
- Pixels: (u, v) = (column, row). The center of the image pixel [v, u] is at the coordinate (u, v): AliceVision
  puts the center of pixel i at i (principal point = image center + offset), and LED-getter samples the pixel
  floor(u), floor(v) for the ray through (u, v).
"""

import copy
import os

import numpy as np

from meshroom.core import desc

from . import psCommon


# --------------------------------------------------------------------------------------------------------
# Node attributes
# --------------------------------------------------------------------------------------------------------

LINEAR_INPUT_NOTE = (
    "**Linear input images are mandatory**: the near-light model is physical (intensity proportional to the "
    "irradiance). Use the float images of ExportImages (outputFileType=exr: linear half float). 8/16-bit images "
    "are rejected unless 'Linearize Input' is enabled: they are then converted from sRGB to linear values, an "
    "approximation that ignores any tone curve of the camera or of the raw development."
)


def meshAttributes():
    """Proxy mesh inputs shared by the LED-getter nodes."""
    return [
        desc.File(
            name="inputMesh",
            label="Proxy Mesh",
            description="Proxy mesh of the object (.obj or .ply), in the SfMData world frame, e.g. the output of "
                        "Meshing or MeshFiltering. Its rendering in each pose gives the 3D points and normals used "
                        "to calibrate the lights; the pose masks are restricted to the pixels whose ray hits it.",
            value="",
        ),
        desc.BoolParam(
            name="meshAliceVisionAxes",
            label="AliceVision Mesh Axes",
            description="The mesh is written with the AliceVision mesh convention (Meshing, MeshFiltering, "
                        "Texturing: y and z negated with respect to the SfMData world frame): the flip is undone "
                        "when loading it. Disable it for a mesh already in the SfMData world frame.",
            value=True,
            advanced=True,
        ),
    ]


def selectAttributes(attributes, excluded):
    """Attributes whose name is not in excluded."""
    return [attribute for attribute in attributes if attribute.name not in excluded]


class _Value:
    def __init__(self, value):
        self.value = value


class _NodeWithDefaults:
    """Node proxy giving the psCommon default value to the common attributes a node does not expose."""

    def __init__(self, node):
        self._node = node
        attributes = psCommon.inputAttributes() + psCommon.advancedInputAttributes() + psCommon.settingsAttributes()
        self._defaults = {attribute.name: attribute.value for attribute in attributes}

    def __getattr__(self, name):
        attribute = getattr(self._node, name, None)
        if attribute is not None:
            return attribute
        if name == "outputMaskFolder":
            return _Value(os.path.join(self._node.outputFolder.value, "masks"))
        if name in self._defaults:
            return _Value(self._defaults[name])
        raise AttributeError(name)


def commonParams(node):
    """psCommon.Params of a node, with the default value of the common attributes it does not expose."""
    return psCommon.Params(_NodeWithDefaults(node))


# --------------------------------------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------------------------------------

def _floats(value):
    """Float list from an SfMData JSON value (list, number or string; '' or None: empty list)."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [float(str(v)) for v in value]
    return [float(str(value))]


def sfmVersion(sfm):
    """SfMData version as a tuple (current version when absent)."""
    return tuple(int(str(v)) for v in sfm.get("version", ["1", "2", "14"]))


def checkPinhole(intrinsic):
    """Raise unless the intrinsic is a pinhole camera without distortion (e.g. after ExportImages)."""
    intrinsicType = str(intrinsic.get("type", "pinhole"))
    distortion = _floats(intrinsic.get("distortionParams")) + _floats(intrinsic.get("undistortionParams"))
    if intrinsicType.startswith(("equidistant", "equirectangular")) or any(v != 0.0 for v in distortion):
        raise RuntimeError("Intrinsic {}: '{}' camera with distortion parameters {}: a pinhole camera without "
                           "distortion is expected. Use the undistorted images and SfMData of ExportImages."
                           .format(intrinsic.get("intrinsicId"), intrinsicType, distortion))


def intrinsicMatrix(intrinsic, version):
    """Calibration matrix K of a pinhole intrinsic of the SfMData JSON format (AliceVision sfmDataIO/jsonIO.cpp).

    Both focals derive from the width and the sensor width (the sensor height is not used); the meaning of
    "focalLength" and "pixelRatio" depends on the SfMData version. The principal point is stored as an offset
    from the image center (version >= 1.2.1).
    """
    version = tuple(version)
    if version < (1, 2, 2):
        raise RuntimeError("SfMData version {} is not supported (focal length in pixels): convert the SfMData "
                           "with a recent AliceVision (e.g. ExportImages, ConvertSfMFormat).".format(version))
    width, height = float(str(intrinsic["width"])), float(str(intrinsic["height"]))
    millimetersToPixels = width / float(str(intrinsic.get("sensorWidth", 36.0)))
    focal = float(str(intrinsic["focalLength"])) * millimetersToPixels
    ratio = float(str(intrinsic.get("pixelRatio", 1.0)))
    if version < (1, 2, 5):  # "pixelRatio" stored the focal ratio fx / fy
        fx, fy = focal, focal / ratio
    elif version < (1, 2, 11):  # focal length in mm along x
        fx, fy = focal, focal * ratio
    else:  # focal length in mm along y, pixel aspect ratio fy / fx
        fx, fy = focal / ratio, focal
    offset = _floats(intrinsic.get("principalPoint")) or [0.0, 0.0]
    return np.array([[fx, 0.0, width / 2.0 + offset[0]],
                     [0.0, fy, height / 2.0 + offset[1]],
                     [0.0, 0.0, 1.0]])


class Camera:
    """Pinhole camera of a pose.

    Attributes:
        K: 3 x 3 calibration matrix (pixels, center of pixel i at i).
        rotation: 3 x 3 camera-to-world rotation (columns: camera axes in the world frame).
        center: camera center in the world frame.
        width, height: image size (pixels).
    """

    def __init__(self, K, rotation, center, width, height):
        self.K = np.asarray(K, np.float64)
        self.rotation = np.asarray(rotation, np.float64)
        self.center = np.asarray(center, np.float64)
        self.width, self.height = int(width), int(height)

    def worldToCamera(self, points):
        return (np.asarray(points, np.float64) - self.center) @ self.rotation

    def cameraToWorld(self, points):
        return np.asarray(points, np.float64) @ self.rotation.T + self.center

    def directionsToWorld(self, directions):
        return np.asarray(directions, np.float64) @ self.rotation.T

    def pixelRays(self, pixels):
        """Rays (camera frame, z = 1) through pixels (u, v)."""
        pixels = np.asarray(pixels, np.float64)
        homogeneous = np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1)
        return homogeneous @ np.linalg.inv(self.K).T

    def project(self, points):
        """Pixels (u, v) of world points."""
        projected = self.worldToCamera(points) @ self.K.T
        return projected[:, :2] / projected[:, 2:]

    def toDict(self):
        return {"K": self.K.tolist(), "rotationCameraToWorld": self.rotation.tolist(),
                "center": self.center.tolist(), "width": self.width, "height": self.height}


def poseCamera(sfm, poseId, views, downscale):
    """Camera of a pose, for the images downscaled by an integer factor (intrinsics scaled as in the output
    SfMData, see psCommon.scaleIntrinsics).

    The SfMData JSON stores the world-to-camera rotation of a pose column-major (Eigen order): read row-major,
    it is the camera-to-world rotation.
    """
    intrinsicIds = sorted({str(view.get("intrinsicId")) for view in views})
    if len(intrinsicIds) != 1:
        raise RuntimeError("Pose {}: the views use {} intrinsics ({}), a single intrinsic per pose is expected."
                           .format(poseId, len(intrinsicIds), ", ".join(intrinsicIds)))
    intrinsics = {str(i.get("intrinsicId")): i for i in sfm.get("intrinsics", [])}
    if intrinsicIds[0] not in intrinsics:
        raise RuntimeError("Pose {}: intrinsic {} not found in the SfMData.".format(poseId, intrinsicIds[0]))
    intrinsic = intrinsics[intrinsicIds[0]]
    checkPinhole(intrinsic)
    scaled = {"intrinsics": [copy.deepcopy(intrinsic)]}
    psCommon.scaleIntrinsics(scaled, downscale)
    intrinsic = scaled["intrinsics"][0]
    poses = {str(p.get("poseId")): p for p in sfm.get("poses", [])}
    if str(poseId) not in poses:
        raise RuntimeError("Pose {}: no pose in the SfMData (not localized).".format(poseId))
    transform = poses[str(poseId)]["pose"]["transform"]
    rotation = np.array(_floats(transform["rotation"])).reshape(3, 3)
    return Camera(intrinsicMatrix(intrinsic, sfmVersion(sfm)), rotation, _floats(transform["center"]),
                  int(float(str(intrinsic["width"]))), int(float(str(intrinsic["height"]))))


# --------------------------------------------------------------------------------------------------------
# Views, masks
# --------------------------------------------------------------------------------------------------------

def lightingViews(poseId, views, params, calibrated=None):
    """Lighting images of a pose: all the views (without the pose reference image if excluded), restricted to
    the viewIds of calibrated (if given), then selected by psCommon.selectViews."""
    lighting = list(views)
    if params.excludePoseReferenceImage:
        lighting = [v for v in lighting if str(v.get("viewId")) != str(poseId)]
        if not lighting:
            raise RuntimeError("no lighting image once the pose reference image is excluded")
    if calibrated is not None:
        lighting = [v for v in lighting if str(v.get("viewId")) in calibrated]
    return psCommon.selectViews(lighting, params.nbImages, params.imageSelection,
                                psCommon.poseSeed(params.seed, poseId))


def poseMask(poseId, views, params, size, logger):
    """psCommon pose mask resized to size = (width, height): (bool mask or None, description of the source)."""
    if params.maskPoseReferenceOnly:
        reference = psCommon.representativeView(poseId, views)
        if reference is None:
            raise RuntimeError("no pose reference image (viewId == poseId) to compute the mask from")
        mask, source = psCommon.computePoseMask(poseId, [reference], params, logger)
        source += " (pose reference image only)"
    else:
        mask, source = psCommon.computePoseMask(poseId, views, params, logger)
    if mask is not None:
        mask = psCommon.resizeMask(mask, tuple(size))
        if not mask.any():
            raise RuntimeError("empty pose mask ({})".format(source))
    return mask, source


# --------------------------------------------------------------------------------------------------------
# Pixel grids
# --------------------------------------------------------------------------------------------------------

def gridIndices(height, width, step, offset=(0, 0)):
    """Rows and columns of the pixel grid with the given step and (row, column) offset."""
    return np.arange(offset[0], height, step), np.arange(offset[1], width, step)


def splitOffsets(split):
    """(row, column) offsets of the split x split interleaved grids covering an image (LED-getter order)."""
    return [(row, column) for row in range(split) for column in range(split)]


def gridPixels(rows, cols, gridMask):
    """Pixels (u, v) of the selected grid nodes, in the row-major order of the grid (LED-getter's order)."""
    gridRows, gridCols = np.nonzero(gridMask)
    return np.stack([cols[gridCols], rows[gridRows]], axis=1).astype(np.int32)


# --------------------------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------------------------

class NonLinearInputError(RuntimeError):
    """An 8/16-bit (non linear) image while 'Linearize Input' is disabled."""


def readLinearImage(path, params):
    """Linear RGB float32 image, downscaled: float images as is, 8/16-bit images converted from sRGB if
    'Linearize Input' is enabled (NonLinearInputError otherwise)."""
    rgb, _, isFloat = psCommon.readImage(path)
    if not isFloat:
        if not params.linearizeInput:
            raise NonLinearInputError(
                "'{}' is an 8/16-bit image: LED-getter needs linear images (physical near-light model). Export "
                "float images with ExportImages (outputFileType=exr, linear half float), or enable 'Linearize "
                "Input' to convert sRGB images to linear values (approximation).".format(path))
        rgb = psCommon.srgbToLinear(rgb)
    return psCommon.downscaleImage(rgb, params.downscale)


def samplePixels(views, params, pixels, size):
    """Linear values of the views at pixels (u, v), in the LED-getter layout (n_pixels, 3 channels, n_images).

    Each image is read, sampled and released in turn: the memory holds one image and the sampled values.
    size = (width, height) expected after downscale (the size of the SfMData intrinsics).
    """
    pixels = np.asarray(pixels)
    values = np.empty((len(pixels), 3, len(views)), np.float32)
    for index, view in enumerate(views):
        rgb = readLinearImage(view["path"], params)
        if (rgb.shape[1], rgb.shape[0]) != tuple(size):
            raise RuntimeError("'{}': image size {}x{} (after downscale) instead of {}x{} (SfMData intrinsics)."
                               .format(view["path"], rgb.shape[1], rgb.shape[0], size[0], size[1]))
        values[:, :, index] = rgb[pixels[:, 1], pixels[:, 0]]
    return values


# --------------------------------------------------------------------------------------------------------
# Normals
# --------------------------------------------------------------------------------------------------------

def ledgetterToOpenGL(normals):
    """LED-getter normals (camera frame x right, y down, z forward) to the OpenGL camera frame of psCommon
    (x right, y up, z towards the camera)."""
    return np.asarray(normals, np.float32) * np.array([1.0, -1.0, -1.0], np.float32)
