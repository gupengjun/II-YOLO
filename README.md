# II-YOLO

Official implementation of **II-YOLO**, a reflection-robust detector for small objects in water-surface scenes.

This repository accompanies the manuscript:

> **Reflection-Robust Cross-Scale Feature Learning for Water-Surface Small-Object Detection**  
> Hao Wang, Pengjun Gu, Shuguang Zhang, and Xianyu Ge

II-YOLO is developed on top of the Ultralytics YOLO framework. It addresses the degradation of weak small-object features caused by specular reflections, low texture, and repeated downsampling.

## Main Components

- **Illumination-Invariant Feature Module (IIFM):** implicitly identifies reflection-sensitive feature responses and enhances texture and boundary cues weakened by specular highlights.
- **Dynamic Tensor-Product Attention (DTPA):** models second-order channel interactions using an input-adaptive tensor-product formulation to strengthen weak-object representations.
- **Cross-scale Alignment Feature Pyramid (CAFP):** aligns shallow spatial details with deep semantic features and introduces a high-resolution detection branch for small-object localization.

## Repository Structure

```text
II-YOLO/
|-- II-yolo.yaml                  # II-YOLO model configuration
|-- train.py                      # training entry point
|-- val.py                        # validation and efficiency evaluation
|-- README.md
`-- ultralytics/
    |-- cfg/                      # model and dataset configurations
    |-- data/                     # data loading and augmentation code
    |-- engine/                   # training and inference engine
    |-- models/                   # detection models
    |-- nn/
    |   |-- Addmodules/           # IIFM and tensor-product attention modules
    |   `-- extra_modules/        # additional feature-fusion components
    `-- utils/
```

Important implementation locations include:

- IIFM: `ultralytics/nn/Addmodules/IIFM.py`
- DTPA/C2TPA: `ultralytics/nn/Addmodules/`
- Cross-scale fusion (`Zoom_cat`): `ultralytics/nn/extra_modules/block.py`
- II-YOLO configuration: `II-yolo.yaml`

## Experimental Environment

The experiments reported in the manuscript were conducted with:

- Python 3.10.16
- Ultralytics 8.3.9-based codebase
- CentOS 7
- NVIDIA RTX 3080 GPU with 10 GB memory
- Input resolution: 640 x 640

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/gupengjun/II-YOLO.git
cd II-YOLO

conda create -n ii-yolo python=3.10.16 -y
conda activate ii-yolo
```

Install PyTorch using the command appropriate for your CUDA version, then install the required Python packages:

```bash
pip install ultralytics==8.3.9
pip install prettytable numpy
```

Run all commands from the repository root so that Python imports the modified local `ultralytics` package.

## Dataset Preparation

Organize a YOLO-format dataset as follows:

```text
dataset/
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
|-- labels/
|   |-- train/
|   |-- val/
|   `-- test/
`-- data.yaml
```

An example `data.yaml` is:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: floating-waste
```

The two datasets used in this project are currently available from the following Google Drive mirrors:

| Dataset | Download |
|---|---|
| floating-waste-I-enhanced | [Download](https://orca-tech.cn/datasets/FloW/FloW-Img) |
| flow-img | [Download](https://github.com/wangruichen01/FloatingWaste-I) |

After downloading, extract the archives and update the `path` field in the corresponding dataset YAML file. These Google Drive URLs are convenient download mirrors rather than immutable archival identifiers. Permanent dataset DOI link(s) will be added after the datasets and the exact code release used in the study have been archived in a DOI-issuing repository such as Zenodo.

Only data that can be legally redistributed should be included in the public archive.

## Training

The default settings used in the manuscript are 640-pixel inputs.

Alternatively, update the model and dataset paths in `train.py`, then run:

```bash
python train.py
```

## Validation

Evaluate a trained checkpoint using:


```bash
python val.py
```

