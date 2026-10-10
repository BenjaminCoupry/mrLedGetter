# mrLedGetter

Meshroom plugin for [LED-getter](https://github.com/BenjaminCoupry/LED-getter): self-calibrated **near-light**
photometric stereo using a geometric proxy (B. Coupry et al., SSVM 2025).

The lights of each multi-lighting pose (point sources with a principal direction and an anisotropy, e.g. LEDs) are
calibrated from the images and a proxy mesh of the object, then the normals and the albedo are estimated per pixel
with the calibrated lights. The plugin provides two nodes, harmonized with the other photometric stereo plugins
(mrLINOUniPS, mrUniMSPS, mrSDMUniPS, through the shared `psCommon.py` layer):

| Node | Step | Output |
|------|------|--------|
| **LedGetterLights** | Light self-calibration per pose (expensive: directional, punctual, then LED model) | `lights.json` per pose (camera and world frames) |
| **LedGetterPS** | Normals and albedo per pose with the calibrated lights | normal maps, masks, albedo maps, `normalMaps.sfm`, `albedoMaps.sfm` (RNb-NeuS2 inputs) |

> [!IMPORTANT]
> **Linear input images are mandatory.** The near-light model is physical: the image intensity must be
> proportional to the irradiance. Use the float images of **ExportImages with `outputFileType=exr`** (linear half
> float, the default). 8/16-bit images (PNG, JPG, TIFF) are **rejected** unless `Linearize Input` is enabled; they
> are then converted from sRGB to linear values, an approximation that ignores any tone curve of the camera or of
> the raw development.

---

## Requirements

- **Meshroom** 2025+ (its bundled **Python 3.11**: JAX 0.9 needs Python >= 3.11)
- **CUDA 12** and an NVIDIA GPU (the nodes fall back to the CPU, much slower)

## Installation

Meshroom uses the virtual environment named **`venv`** at the plugin root to run the nodes.

```bash
cd /path/to/your/plugins
git clone https://github.com/BenjaminCoupry/mrLedGetter.git
cd mrLedGetter

# Python 3.11, e.g. the Python bundled with Meshroom
/path/to/Meshroom/python/bin/python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install .

export MESHROOM_PLUGINS_PATH=/path/to/your/plugins/mrLedGetter:$MESHROOM_PLUGINS_PATH
```

`pip install .` installs the dependencies declared in `pyproject.toml` (Meshroom still discovers the nodes from
`MESHROOM_PLUGINS_PATH`, not from site-packages):

- `ledgetter`, pinned to a commit of [LED-getter](https://github.com/BenjaminCoupry/LED-getter) (with its own
  dependencies: optax, open3d, plotly, sphericart-jax 1.0.8, ...),
- `jax[cuda12]==0.9.0.1`: sphericart-jax 1.0.8 requires `jax>=0.6.0,<0.9.1`, 0.9.0.1 is the latest release in that
  range (LED-getter itself requires plain `jax`, CPU only),
- `numpy`, `opencv-python-headless` 4.x (image and mask I/O of `psCommon.py`; the OpenCV 5 wheels cannot read EXR), `optax`, `psutil`.

To work on a local LED-getter clone: `venv/bin/pip install -e /path/to/LED-getter` after the install above.

Launch Meshroom: the nodes appear under **Photometric Stereo**.

---

## Pipeline

```
CameraInit -> ... -> StructureFromMotion -> SfMFilter* -> ExportImages (outputFileType=exr) --+--> LedGetterLights
                                         \-> ... -> Meshing -> MeshFiltering ------------------+         |
                                                                                               |    outputFolder
                                                                                               |         v
                                                                                               +--> LedGetterPS --> RNbNeuS2
```

| From | To |
|------|----|
| `ExportImages.outputSfMData` | `LedGetterLights.inputSfm` and `LedGetterPS.inputSfm` (the same SfMData) |
| `MeshFiltering.outputMesh` (or `Meshing.outputMesh`) | `LedGetterLights.inputMesh` and `LedGetterPS.inputMesh` |
| `LedGetterLights.outputFolder` | `LedGetterPS.inputLights` |
| `LedGetterPS.outputSfmDataNormal` | `RNbNeuS2.inputNormalSfm` |
| `LedGetterPS.outputSfmDataAlbedo` | `RNbNeuS2.inputAlbedoSfm` |
| `LedGetterPS.outputMaskFolder` | `RNbNeuS2.inputMaskFolder` |

\* The all-light / ambient reference images that are not lit by a single light (e.g. `all_leds-*`, `LIGHT.NEF`,
`*_ambient_all_leds`) must be removed before ExportImages (SfMFilter); `Exclude Pose Reference Image` removes the
reference view of each pose (viewId == poseId) from the lighting images while keeping it for the mask and the
output SfMData.

Masks: as in the other photometric stereo nodes, from `Mask Folder` (`<poseId>.png` or `<viewId>.png`) or from the
alpha channel of the images (e.g. ExportImages with the masks of an ImageSegmentation node). The pose masks are
also restricted to the pixels whose ray hits the proxy mesh.

---

## LedGetterLights

Calibrates the lights of each multi-lighting pose: given the 3D points and normals of the proxy mesh seen by the
camera, LED-getter fits successive light models to the images (Lambertian surface, Huber loss, Adam), each one
initialized by the previous: directional, punctual, then LED.

### Inputs

| Parameter | Label | Default | Description |
|-----------|-------|---------|-------------|
| `inputSfm` | SfMData | | Multi-lighting views grouped by poseId: undistorted pinhole images of ExportImages (non pinhole or distorted intrinsics are rejected) |
| `maskFolder` | Mask Folder | | Masks `<poseId>.png` or `<viewId>.png`; alpha channels otherwise |
| `downscale` | Downscale Factor | 1 | Integer downscale of the images (intrinsics scaled accordingly) |
| `nbImages` | Number Of Images | -1 | Maximum number of lighting images per pose (-1: all) |
| `inputMesh` | Proxy Mesh | | `.obj` / `.ply` mesh in the SfMData world frame (Meshing / MeshFiltering) |
| `samplingStep` | Pixel Sampling Step | 10 | Calibration on one pixel every N rows and columns of the mask (formerly `decimation`, LED-getter `--step`) |
| `lightRig` | Light Rig | false | Lights fixed to the camera: the poses lit by the same sequence of lights share one set of lights, calibrated jointly (see [Light rig](#light-rig)) |
| `rigLightMatching` | Rig Light Matching | rank | How an image is matched to a light of the rig: `rank` (n-th image of each pose in frame number order) or `frameId` (same frame number in every pose) |
| `iterations` | Iterations | 10000 | Iterations of each light model |
| `useGpu` | Use GPU | true | JAX on the GPU (CPU if unavailable) |

Advanced: `meshAliceVisionAxes` (true: undo the y/z flip of the AliceVision mesh export; disable for a mesh already
in the SfMData world frame), `learningRate` (0.001), `delta` (Huber loss threshold in linear intensity, 0.01, the
LED-getter default), `exportImages` (false: per-light diagnostic images of LED-getter), and the common
`minViewsPerPose`, `excludePoseReferenceImage`, `imageSelection`, `seed`, `linearizeInput`, `maskThreshold`,
`maskVoteThreshold`, `maskPoseReferenceOnly`, `maskRemoveBorderComponents`, `maskUseGlobalFile`, `failurePolicy`,
`verboseLevel`.

### Light rig

By default the lights of each pose are calibrated alone (LED-getter calibrates one pose at a time). When the lights
are fixed to the camera (LED ring or dome carrying the camera, lighting arm; the object moves between poses), the
same light has the same position, direction and power in the camera frame of every pose. With `lightRig`, the poses
lit by the same sequence of lights are calibrated **jointly**: their calibration pixels (camera-frame points and
normals of the proxy mesh, image values) are concatenated into one LED-getter problem with one set of lights, the
albedo staying free per pixel; the cast shadows of each pixel are tested on the mesh of its own pose. Each pose still
gets its `<poseId>/lights.json` (the same camera-frame lights, the world-frame values of its pose) with a `rig` entry
(poses, light keys, matching); the LED-getter exports of the joint calibration are in `rig_<first poseId>/ledgetter/`.

The light of an image is identified by its frame number (AliceVision `frameId`, the number in the image filename):

- `rank`: the n-th lighting image of every pose (frame number order) is lit by the same light, i.e. the shooting
  sequence is the same in every pose (e.g. `DSC_1903.NEF`... `DSC_1937.NEF` then `DSC_1940.NEF`...). The pose
  reference image is not counted when `excludePoseReferenceImage` is enabled.
- `frameId`: the same frame number in every pose (e.g. `led_3-photo-...-0051.nef`, `RTI_00012.NEF`).

Poses whose sequence of lights is shared by no other pose (another number of images, other frame numbers) are
calibrated alone, as without the option. `nbImages` selects the same lights in every pose of a rig.

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | `<poseId>/lights.json`, `<poseId>/calibration.npz`, `<poseId>/ledgetter/<model>/` (or `rig_<poseId>/ledgetter/<model>/` for a light rig) (LED-getter exports: `misc/light_plot.html`, `losses/loss_plot.html`, `values/`, `model.json`, `light/*.lp`, ...) |
| `outputLights` | `lights.json`: the lights of all the poses |

`<poseId>/lights.json` (format `ledgetter-lights` 1.0, frames and model also described in the file):

```json
{
  "format": "ledgetter-lights", "version": "1.0", "poseId": "1992575257", "model": "LED",
  "camera": {"K": [[...]], "rotationCameraToWorld": [[...]], "center": [...], "width": 8280, "height": 5520, "downscale": 1},
  "anisotropy": [0.8, 0.9, 1.1],
  "lights": [
    {"viewId": "11348718", "imagePath": ".../11348718.exr", "power": 2.4,
     "positionCamera": [x, y, z], "positionWorld": [X, Y, Z],
     "principalDirectionCamera": [dx, dy, dz], "principalDirectionWorld": [DX, DY, DZ]}
  ]
}
```

Model (per light, color channel c, surface point x with normal n and albedo rho_c):
`w = (position - x) / |position - x|`,
`irradiance_c = power / |position - x|^2 * max(0, -w . principalDirection)^anisotropy_c`,
`image_c = rho_c * max(0, n . w) * irradiance_c`.
Positions are in SfMData world units; the power is relative to the albedo scale (the albedo is clamped to [0, 1]).

---

## LedGetterPS

Estimates the normal and the albedo of each pixel with the calibrated lights (only the lighting images with a
calibrated light are used); the normals are initialized by the proxy mesh and the albedo by the calibration.

### Inputs

Same inputs as the other photometric stereo nodes (`inputSfm`, `maskFolder`, `downscale`, `nbImages`,
`outputFormat` (`png16` / `exr`), `useGpu` and the common advanced inputs, including `normalConvention` and
`keepLandmarks`), plus:

| Parameter | Label | Default | Description |
|-----------|-------|---------|-------------|
| `inputMesh` | Proxy Mesh | | Same mesh as LedGetterLights (`meshAliceVisionAxes`, advanced, as above) |
| `inputLights` | Lights | | Output folder of LedGetterLights |
| `iterations` | Iterations | 1000 | Iterations of the per-pixel solve |
| `splitFactor` | Pixel Split | 2 | Pixels solved in N x N interleaved subsets, then merged at full resolution (formerly `decimation` of SolvePS): less memory, the images are read again per subset |
| `gpuChunks` | GPU Chunks Per Subset | 100 | Advanced: successive GPU chunks per subset (LED-getter `--ps_chunck_number`) |
| `learningRate`, `delta` | | 0.001, 0.01 | Advanced: Adam learning rate, Huber loss threshold |

### Outputs

As the other photometric stereo nodes, ready for RNb-NeuS2:

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Normal maps `<poseId>.png` (16-bit, `(n + 1) / 2`) or `.exr`, OpenGL camera frame by default |
| `outputSfmDataNormal` | `normalMaps.sfm`: one view per pose (viewId == poseId), intrinsics scaled by `downscale` |
| `outputMaskFolder` | Pose masks `masks/<poseId>.png`: pose mask, on the mesh, with a normal |
| `outputSfmDataAlbedo` | `albedoMaps.sfm`, albedo maps `albedo/<poseId>.png|exr` (zero outside the mask; relative albedo, up to the light power scale of the pose) |

---

## Conventions

- **Cameras**: K, R, center are read from the SfMData (`focalLength`, `sensorWidth`, `pixelRatio` with the
  semantics of the SfMData version, principal point = image center + offset). The JSON pose `rotation` is the
  world-to-camera rotation stored column-major: read row-major, it is the camera-to-world rotation that LED-getter
  expects. Pinhole intrinsics without distortion are required (ExportImages output).
- **Camera frame**: x right, y down, z forward (AliceVision / OpenCV, LED-getter's frame); lights are estimated in
  the camera frame of each pose, in SfMData world units, and also exported in the SfMData world frame.
- **Pixel centers**: the center of pixel `i` is at coordinate `i` in AliceVision and in LED-getter (ray
  `K^-1 (u, v, 1)` for the pixel of column u, row v): no half-pixel offset. Downscaling keeps it exact
  (`psCommon.scaledPrincipalPoint`).
- **Mesh**: AliceVision writes meshes with y and z negated with respect to the SfMData world frame; LED-getter
  undoes it when `meshAliceVisionAxes` is enabled (default).
- **Normals**: LED-getter's normals (camera frame above) are converted to the psCommon convention (OpenGL camera
  frame: x right, y up, z towards the camera) or kept in the OpenCV frame with `normalConvention=opencv`.

## GPU memory

JAX defaults are kept: on first use, JAX preallocates 75% of the GPU memory of the node process. To allocate on
demand instead, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` to preallocate
more), in the environment or in `meshroom/config.json`:

```json
[
    {"key": "XLA_PYTHON_CLIENT_PREALLOCATE", "type": "string", "value": "false"}
]
```

Memory knobs: `samplingStep` (LedGetterLights), `splitFactor` and `gpuChunks` (LedGetterPS), `downscale`, `nbImages`.

---

## Development

CPU unit tests (no JAX needed: the LED-getter calls are isolated in `ledBackend.py`, replaced by a stand-in):

```bash
PYTHONPATH=/path/to/Meshroom venv/bin/python -m pytest tests
```

```
mrLedGetter/
├── meshroom/
│   ├── config.json
│   └── ledGetter/
│       ├── LedGetterLights.py   # node: light calibration
│       ├── LedGetterPS.py       # node: normals and albedo
│       ├── ledCommon.py         # cameras, frames, linear image sampling, pixel grids (no JAX)
│       ├── lightsIO.py          # lights.json / calibration.npz between the nodes (no JAX)
│       ├── ledBackend.py        # the only module calling JAX / LED-getter
│       └── psCommon.py          # common photometric stereo layer (identical in mrLINOUniPS, mrUniMSPS, mrSDMUniPS)
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Publications

- **Self-calibrated Near-light Photometric Stereo using a Geometric Proxy**  
  B. Coupry, F. Lauze, A. Laurent, J. Mélou, Y. Quéau, J.-D. Durou (2025)  
  In *International Conference on Scale Space and Variational Methods in Computer Vision (SSVM)*, pp. 364–376.  
  [PDF](https://hal.science/hal-05004867/document)

- **Assessing the quality of 3D reconstruction in the absence of ground truth: application to a multimodal archaeological dataset**  
  B. Coupry, B. Brument, A. Laurent, J. Mélou, Y. Quéau, J.-D. Durou (2025)  
  In *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, pp. 3457–3466.  
  [PDF](https://hal.science/hal-04942610v1/file/WACV_25_bis.pdf)

## License

[MIT](LICENSE) © Benjamin Coupry
