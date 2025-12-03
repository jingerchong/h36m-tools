# H3.6M Preprocessing and Visualization Tools

This repository provides tools to **preprocess and visualize the Human3.6M (H3.6M) dataset**. It includes scripts to convert raw `D3_Angles` files downloaded from the official H3.6M website into a variety of SO(3) representations, including:

* **Quaternions**
* **Expmap / axis-angle**
* **Euler angles**
* **Rotation matrices**
* [**6D rotation matrix**](https://arxiv.org/abs/1812.07035)

These scripts aim to reproduce the preprocessed expmap dataset which is commonly referenced in motion prediction research but whose [original download link](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) is no longer available.

Additionally, this repository includes:
* Conversion from rotation-based representations (e.g., quaternions) to **3D joint positions via Forward Kinematics (FK)**
* 3D visualization and animation of skeleton sequences to validate preprocessing and predictions
* Filtering and adding of **static dimensions**
* Computation of **global normalization statistics** (mean and standard deviation)
* Configuration files describing joint mappings, skeleton topology, and other metadata to help new users get started with H3.6M

The implementation uses **PyTorch**, **NumPy**, and **Matplotlib**, with rotation conversions based on [Roma](https://naver.github.io/roma/). The design goal is simplicity, making it easy to integrate into other research projects.

---

## Installation

Install the required dependencies via pip:

```bash
# For CPU or Colab
pip install -r requirements.txt

# For GPU-enabled environments
pip install -r requirements-gpu.txt
```

---

## Downloading the H3.6M Dataset

To use these tools, you must first download the H3.6M dataset:

1. Visit the official H3.6M website: [http://vision.imar.ro/human3.6m/](http://vision.imar.ro/human3.6m/)
2. Create a login and request access to the dataset.
3. Download the **D3_Angles** files for all actions and subjects.
4. 
   * Navigate to **Training Data → By Subject → Poses → D3_Angles**
   * Download and extract each `.tgz` archive into `data/raw`

The expected directory structure is:

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
```

**Note:** The D3_Angles files contain **Cardan/Euler angles** in the ZXY order, defined relative to each joint's parent. The root position (first 3 values) and root rotation (next 3 values) are often discarded for motion prediction training.

---

## Preprocessing

While this repository does **not directly provide XYZ position preprocessing**, the FK functionality can be adapted to compute world-space joint positions, ignoring the root. What we do provide is preprocessing from D3_angles to:

* Removing static dimensions
* Converting to the desired SO(3) representation
* Saving processed `.pt` tensors
* Computing global mean and standard deviation for normalization

You can generate a preprocessed dataset for a specific target representation by calling 
```bash
python -m scripts.preprocess \ -rep "expmap"
    -i "data/raw" \
    -o "data/processed"
```
---



**Compare expmap preprocessing outputs**
If you happen to have access to the zip file, you can confirm that our preprocessing generates the same expmap representation by unzipping it in data/ezpmap and running this script, which compares all non static dims for equivalent sequences (same action and subject). We note that in certain cases, the order seems to be swapped. That is, sometimes S1 Eating 1 in our processed dataset = S1 Eating 2 in the zip.
```bash
python -m scripts.compare_expmap -r "data/raw" -p "data/expmap_zip"
```

Confirms equivalence between raw D3_Angles and preprocessed expmap tensors in the zip


## Visualization

You can inspect raw or processed files with 3D visualization:

```bash
python -m scripts.inspect_files \
    -i "data/expmap_zip/S1/directions_1.txt" \
       "data/processed/expmap/train/S1_directions_1.pt"
```

This will show a 3D skeleton animation for each sequence.

---

## Using in Modeling

Load processed H3.6M data for training or evaluation:

```python
from h36m_tools.load import load_processed

train, test, mean, std = load_processed("data/processed", rep="quat")
```

Compute standard motion metrics:

```python
from h36m_tools.metrics import mae_l2, mpjpe

print(mae_l2(y_pred, y_gt, "quat"))
print(mpjpe(y_pred, y_gt, "quat"))
```

---

## Inspect multiple files

```bash
python -m scripts.inspect_files \
    -i "data/expmap_zip/S1/directions_1.txt" "data/processed/expmap/train/S1_directions_1.pt"
```

Loads a batch of files. easy way to look into some of the raw data, zip data, processed data. works with CDF, PT , and txt In the zip.
