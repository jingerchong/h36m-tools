# H3.6M Preprocessing and Visualization Tools

This repository provides tools to **preprocess and visualize the Human3.6M (H3.6M) dataset**. It includes scripts to convert raw `D3_Angles` files downloaded from the official H3.6M website into a variety of SO(3) representations, including:

* **Quaternions**
* **Expmap / axis-angle**
* **Euler / Tait-Bryan angles**
* **Rotation matrices**
* [**6D rotation matrices**](https://arxiv.org/abs/1812.07035)

These tools originally aimed to reproduce the preprocessed expmap dataset commonly referenced in motion prediction research, whose [original download link](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) is no longer available.

## Features

* **SO(3) representation conversion** from raw D3_Angles to multiple rotation formats
* **Forward Kinematics (FK)** to convert rotation-based representations to 3D joint positions
* **3D visualization and animation** of skeleton sequences for validation
* **Static dimension filtering** and normalization statistics computation
* **Configuration files** for joint mappings, skeleton topology, and dataset metadata
* Simple PyTorch-based implementation using [Roma](https://naver.github.io/roma/) for rotation conversions

## Installation

Install the required dependencies via pip:

```bash
# For CPU or Colab
pip install -r requirements.txt

# For GPU-enabled environments (default is GPU)
pip install -r requirements-gpu.txt
```
For saving animations to mp4, additionally install [`ffmpeg`](https://www.ffmpeg.org/).

## Downloading the H3.6M Dataset

To use these tools, you must first download the H3.6M dataset:

1. Visit the official H3.6M website: [http://vision.imar.ro/human3.6m/](http://vision.imar.ro/human3.6m/)
2. Create an account and request dataset access
3. Download the **D3_Angles** files for all actions and subjects:
   * Navigate to **Training Data → By Subject → Poses → D3_Angles**
   * Download and extract each `.tgz` archive into `data/raw`

### Expected Directory Structure

```
data/raw/
├── S1/
│   └── MyPoseFeatures/
│       └── D3_Angles/
│           ├── Directions 1.cdf
│           ├── Directions 2.cdf
│           ├── Discussion 1.cdf
│           └── ...
├── S5/
│   └── MyPoseFeatures/
│       └── D3_Angles/
│           └── ...
└── ...
```

**Note:** The D3_Angles files contain **Tait-Bryan angles** in the ZXY order, defined relative to each joint's parent. The root position (first 3 values) and root rotation (next 3 values) are often discarded for human motion prediction tasks.

## Preprocessing

Generate a preprocessed dataset for a specific SO(3) representation:

```bash
python -m scripts.preprocess -r "expmap" -i "data/raw" -o "data/protocol1"
```

This will:
* Apply a default downsampling from the original 50 fps to 25 fps
* Remove static dimensions
* Convert to the desired SO(3) representation
* Save processed `.pt` tensors
* Compute global mean and standard deviation for normalization

**Note:** The train/test split protocol is defined in `metadata.py`

## Comparison with Expmap Zip Dataset

If you have access to the original expmap zip file, you can verify that our preprocessing script generates equivalent outputs:

```bash
python -m scripts.compare_expmap -r "data/expmap_zip" -p "data/protocol1"
```
This compares all non-static dimensions for equivalent sequences (same action and subject). Note that in some cases, the sequence order may differ (e.g., "S1 Eating 1" in our processed dataset might correspond to "S1 Eating 2" in the original zip).

## Visualization

The repository provides tools to visualize skeleton sequences in multiple formats. Example outputs can be found in the `outputs/` folder.

Plot a series of frames:
```bash
python -m scripts.plot_sequence -i "data/protocol1/expmap/train/S1_walking_1.pt" -s 100 
```

Create MP4 animations of entire sequences:
```bash
python -m scripts.animate_sequence -i "data/protocol1/quat/train/S1_walking_1.pt" 
```
**Note:** This script infers the representation from the parent directory of the data.

## Data Inspection

Quickly inspect raw or processed files to view tensor properties and sample frames:
```bash
python -m scripts.inspect_files -i "data/expmap_zip/S1/directions_1.txt" "data/protocol1/expmap/train/S1_directions_1.pt"
```

**Supported formats:** `.cdf` (raw H3.6M), `.pt` (processed tensors), `.txt` (expmap zip format)

This displays tensor shape, dtype, min/max values, and the first `n` frames for each file.


## Integration with Models

Install as Python package to use modules in other projects
```bash
pip install .
```

Load processed H3.6M data for training or evaluation:

```python
from h36m_tools.load import load_processed

train, test, mean, std = load_processed("h36m-tools/data/protocol1", rep="quat", include_stats=True)
```

Compute standard human motion prediction metrics:

```python
from h36m_tools.metrics import mae_l2, mpjpe

print(mae_l2(y_pred, y_gt, "quat"))
print(mpjpe(y_pred, y_gt, "quat"))
```

## Citation

If you use this repository in your research or projects, please cite it as follows:

```bibtex
@misc{h36m_tools,
  author = {Jinger Chong},
  title = {H3.6M Tools},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jingerchong/h36m_tools}
}
```

## Development Status

This repository is **under active development** and has been shared publicly in the spirit of open research. New features and improvements are being added and some functionality may contain bugs or change in future releases. Contributions, bug reports, and questions are welcome via GitHub.