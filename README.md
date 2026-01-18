### 1. Installation

The code was tested on AutoDL using an NVIDIA RTX 4090 GPU. All experiments were conducted with Python 3.12, PyTorch 2.8.0, and CUDA 12.8, selected from AutoDL’s pre-configured container environments. Compatibility with other versions is expected but has not been explicitly tested.

1. Open the `Diffusion-Illusions` folder.

2. Install the remaining requirements (PyTorch 2.8.0 and CUDA 12.8 are pre-installed on AutoDL and therefore excluded from the requirements file).

```
pip install --upgrade -r requirements.txt
pip install rp --upgrade
```

3. Download the model (this may take some time due to the large file size).

```
pip install modelscope
```

```
python -c "from modelscope import snapshot_download; snapshot_download('AI-ModelScope/stable-diffusion-v1-4', local_dir='./weights/sd-v1-4')"
```

### 2. Code Reproduction

The files `parker_puzzle_colab.ipynb`, `rotation_overlays_for_colab.ipynb`, `flippy_illusions_for_colab.ipynb`, `hidden_characters_for_colab.ipynb`, and `twisting_squares_colab.ipynb` are implementations from the original paper. We only made minimal modifications, such as updating file paths, to ensure compatibility with our execution environment.

You can run each code cell in order to reproduce the results.

（When running `.ipynb` files on AutoDL to generate images, if the execution is interrupted midway, the Python process may not terminate properly. In such cases, GPU memory can remain occupied until the process is manually killed.

If multiple `.ipynb` files are executed sequentially without cleaning up these residual processes, GPU memory usage may accumulate and eventually exhaust the available GPU resources. Therefore, it is recommended to manually terminate the corresponding Python processes (e.g., via `nvidia-smi` and `kill -9 <PID>`) when interrupting a running notebook.）

### 3. Motion Blur Illusions and Multi-Angle Rotation Illusions

open these two files:

- **`motion_blur.ipynb`**: Implements **Motion Blur Illusions**.
- **`rotation_overlays_for_more_angles.ipynb`**: Implements **Multi-Angle Rotation Illusions**.

Then run each code cell in order to reproduce the results.

### 4. Component-Aware Diffusion Illusions

- **`fruit_face.ipynb`**: Implements **Component-Aware Diffusion Illusions**.

  - The files **`face_parser.py`**, **`model.py`**, and **`resnet.py`** are auxiliary modules used for facial component segmentation in this task and **do not need to be run separately**.

- - **`image_masked.py`** and **`image_inversion.py`**: Implement **Image Prompt Supporting**.  
    When the input image has a white or uniform background, the masked version (**`image_masked.py`**) should be used to remove background influence by applying an explicit foreground mask.
  - The file **`new_stable_diffusion.py`** in the `source/` directory is an auxiliary module that enables image-based inputs and **does not need to be run separately**.

  You can open **`fruit_face.ipynb`** and run each code cell in order to reproduce the result.

### 5. Image Prompt Supporting

When using the Image Prompt Supporting code, if the input image has a white background, the masked version can be used to eliminate background interference. For example, with the default text prompt, you can run:

```
python image_masked.py --target-image "images/lion.jpg" --structure-strength 0.83 --text-strength 1.0
```

to generate an image in which the lion is integrated into the background room.

Alternatively, you can run:

```
python image_version.py --target-image "images/monalisa.jpg" --structure-strength 0.8 --text-strength 1.0
```

to generate an image in which the Mona Lisa is integrated into an oil painting.

### 6. High–Low Frequency Separation

You can run:

```
python run_illusion_input.py
```

to generate image with High-low frequency separation. (According to our tests, this module is highly stochastic. Even with the same parameter settings, the generated results may vary and may not always be optimal.)
