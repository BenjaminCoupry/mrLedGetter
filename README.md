# LED-getter
This repository implements a 3D reconstruction technique leveraging near-light photometric stereo
## Installation

LED-getter is used as a [Meshroom](https://alicevision.org/#meshroom) plugin. It requires two repositories: [LED-getter](https://github.com/BenjaminCoupry/LED-getter) (the Python library) and [mrLedgetter](https://github.com/BenjaminCoupry/mrLedgetter) (the Meshroom plugin). The installation below uses the Python interpreter bundled with Meshroom.

1. **Clone both repositories** in the same folder:

   ```bash
   mkdir mrLedGetter-workspace && cd mrLedGetter-workspace
   git clone https://github.com/BenjaminCoupry/mrLedgetter.git
   git clone https://github.com/BenjaminCoupry/LED-getter.git
   ```

2. **Declare the plugin path** (add it to your shell profile to make it permanent):

   ```bash
   export MESHROOM_PLUGINS_PATH=$(pwd)/mrLedgetter:$MESHROOM_PLUGINS_PATH
   ```

3. **Create and activate a virtual environment** with Meshroom's Python:

   ```bash
   /path/to/Meshroom-2026.10.05/python/bin/python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install the dependencies** (JAX with CUDA 12 support is required):

   ```bash
   pip install -U "jax[cuda12]" matplotlib plotly tqdm optax pyaml networkx \
       sphericart sphericart-jax==1.0.8 scipy numpy pillow pandas ImageIO rawpy \
       open3d OpenEXR exifread palettable cvxpy flatbuffers kaleido \
       scikit-image dm-haiku miniball numpngw
   ```

5. **Install LED-getter** in editable mode:

   ```bash
   pip install -e LED-getter/.
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
