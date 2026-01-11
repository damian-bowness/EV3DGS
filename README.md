<div align="center">

# EV3DGS: Extreme Views 3DGS Filter

**3DGS Filter for Novel View Synthesis from Out-of-Distribution Camera Poses**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://damian-bowness.github.io/EV3DGS/)
[![Paper](https://img.shields.io/badge/arXiv-2510.20027-b31b1b.svg)](https://arxiv.org/abs/2510.20027)

*20th International Symposium on Visual Computing (ISVC) 2025*

[Damian Bowness](https://damian-bowness.github.io), [Charalambos Poullis](https://scholar.google.com/citations?user=IhcdDa8AAAAJ&hl=en)  
Concordia University


</div>

<div align="center">
  <img src="readme_assets/drjohnson_split.gif" width="640" alt="Comparison GIF">
  <br>
  <table width="640">
    <tr>
      <td width="100"></td>
      <td align="left"><b>3DGS</b></td>
      <td align="right"><b>EV3DGS (ours)</b></td>
      <td width="80"></td>
    </tr>
  </table>
</div>


## Requirements

### Dependencies

- **Python** >= 3.8
- **PyTorch (CUDA build)**: CUDA-enabled PyTorch matching your GPU/driver (e.g., +cu118)
- **[Nerfstudio](https://docs.nerf.studio/)**: used to train, load, and run Splatfacto models
- **[gsplat](https://github.com/nerfstudio-project/gsplat)**: Splatfacto’s CUDA rasterization backend (required for training)

> **Tested setup:** This project has been tested with **Python 3.8**, **PyTorch 2.1.2+cu118**, **Nerfstudio 1.1.0**, and **gsplat 0.1.12**. Other versions of this software stack may work, but haven’t been validated.


### Hardware

- NVIDIA GPU with CUDA support

> **Tested setup:** This project has been tested on an **NVIDIA GeForce RTX 2080 Ti** with **CUDA 11.8**. Other GPUs/CUDA versions may work, but haven’t been validated.

## Installation

### 1. Set Up Nerfstudio Environment (for CUDA 11.8)

Install nerfstudio following the [official installation guide](https://docs.nerf.studio/quickstart/installation.html):

```bash
# Create conda environment
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install nerfstudio
pip install nerfstudio
```

### 2. Install gsplat

gsplat is required for training splatfacto models:

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```
> **Tip:** If you're using **nerfstudio==1.1.5**, install the matching gsplat version:
>
> `pip install "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0"`

### 3. Install EV3DGS

Clone EV3DGS into your `nerfstudio-main/` directory (repo root), then install:

```bash
git clone https://github.com/damian-bowness/EV3DGS.git
cd EV3DGS

# Install the package
pip install -e . --no-build-isolation
```

This will compile the CUDA extensions for the EV3DGS rasterizer.

## Usage

EV3DGS works with **trained nerfstudio splatfacto models**. You must first preprocess your data and train a model using nerfstudio before using EV3DGS for viewing or rendering.

### Preprocess Data (Prerequisite)
```bash
# Preprocess data with nerfstudio
ns-process-data video  --data <path-to-video>        --output-dir <path-to-output> 
# OR
ns-process-data images --data <path-to-image-folder> --output-dir <path-to-output>
```

### Training a Splatfacto Model (Prerequisite)

```bash
# Train a splatfacto model with nerfstudio
ns-train splatfacto --data <path-to-your-data>
```

### Interactive EV3DGS Viewer

Launch the EV3DGS viewer with a trained model:

```bash
ev3dgs --load-config <path-to-config.yml>
```

### Rendering Camera Paths

Render a video along a predefined camera path:

```bash
ev3dgs-render camera-path \
    --load-config <path-to-config.yml> \
    --camera-path-filename <path-to-camera-path.json> \
    --output-path renders/output.mp4 \
    --ev3dgs.two-pass True \
    --ev3dgs.xg-thresh 0.01 \
    --ev3dgs.cc-thresh 0.5
```

> **Tip:** You can also preview and plot camera paths interactively in the EV3DGS viewer.  
> Open the **Viewer → Render tab/panel**, load or create a camera path there, and visualize it before rendering to video.

### Rendering Dataset Views

Render all views from the training/test dataset:

```bash
ev3dgs-render dataset \
    --load-config <path-to-config.yml> \
    --output-path renders/dataset_output/ \
    --split {train,test,train+test} # choose which split(s) to render \
    --ev3dgs.two-pass True \
    --ev3dgs.xg-thresh 0.01 \
    --ev3dgs.cc-thresh 0.5
```

---

## Parameters

### EV3DGS Filtering Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `two_pass` | Enable two-pass filtering | Boolean | `True` |
| `xg_thresh` | Gradient sensitivity threshold. Count Gaussians with intermediate gradients above this threshold. | 0.0–1.0 | 0.01 |
| `cc_thresh` | Contribution ratio threshold. Gaussians with flagged/total ratio above this are filtered. | 0.0–1.0 | 0.5 |

---

## Project Structure

```
nerfstudio-main
└──EV3DGS/
   ├── cuda_ev3dgs/           # CUDA rasterizer implementation
   │   ├── forward.cu         # Forward pass rendering kernels
   │   ├── rasterizer_impl.cu # Core rasterization logic
   │   └── *.h                # Header files
   ├── ev3dgs/
   │   ├── scripts/
   │   │   ├── run_ev3dgs.py      # Interactive viewer entry point
   │   │   ├── render_ev3dgs.py   # Rendering script
   │   │   └── ev3dgs_viewer.py   # Viewer implementation
   │   └── utils/
   │       ├── utils.py               # Core rendering utilities
   │       └── ev3dgs_render_panel.py # Render UI components
   ├── setup.py               # Package installation
   ├── pyproject.toml         # Build configuration
   ├── ext.cpp                # PyTorch extension bindings
   ├── rasterize_points.cu    # CUDA-Python interface
   └── rasterize_points.h     # Interface header
```

## Citation

If you use EV3DGS in your research, please cite:

```bibtex
@article{EV3DGS25,
  title   = {Extreme Views: 3DGS Filter for Novel View Synthesis from Out-of-Distribution Camera Poses},
  author  = {Bowness, Damian and Poullis, Charalambos},
  journal = {arXiv},
  year    = {2025},
}
```

## Acknowledgments

We’re grateful to the authors and maintainers of the following projects, which made this work possible::
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Nerfstudio](https://docs.nerf.studio/)
- [gsplat](https://docs.gsplat.studio/main/)
- [Gaussian Opacity Fields](https://niujinshuchong.github.io/gaussian-opacity-fields/)

## License

This software is free for non-commercial, research and evaluation use.

For commercial licensing inquiries, please contact the authors.
