from pathlib import Path
import logging
from typing import List, Union, Dict, Tuple
from tqdm import tqdm
import torch

from h36m_tools.files import read_files
from h36m_tools.metadata import DOWNSAMPLE_FACTOR
from h36m_tools.representations import get_rep_dir


logger = logging.getLogger(__name__)


def _standardize_action(raw_action: str) -> str:
    """Normalize / standardize Human3.6M action names to consistent canonical forms."""
    action = "".join(c for c in raw_action.lower() if c.isalpha())
    if "walk" in action and "walking" not in action:
        action = action.replace("walk", "walking")
    elif "photo" in action and "takingphoto" not in action:
        action = "takingphoto"
    return action


def load_raw(root_dir: Union[str, Path] = Path("data/raw"),
             downsample: int = DOWNSAMPLE_FACTOR
             ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Load raw H3.6M D3_Angles CDF data, optionally filtering by subjects/actions,
    reshape to [T, J, 3], downsample, and return a nested dictionary:
        data[subject][action] -> list of tensors

    Args:
        root_dir (Path | str): Root directory of raw H3.6M dataset.
        subjects (list[str], optional): Subjects to include (e.g., ['S1', 'S5']).
        actions (list[str], optional): Actions to include (e.g., ['Walking', 'Eating']).
        downsample (int): Factor to downsample temporal frames.

    Returns:
        dict: Nested dictionary of tensors keyed by subject -> action -> list of tensors
    """
    input_files = list(Path(root_dir).rglob("**/MyPoseFeatures/D3_Angles/*.cdf"))
    if not input_files:
        raise FileNotFoundError("No D3_Angles CDF files found")

    logger.debug(f"Found {len(input_files)} D3_Angles CDF files to process")
    all_angles = read_files(input_files)
    data: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    for file, angles in tqdm(zip(input_files, all_angles), desc="Processing angles", total=len(input_files)):
        subject = file.parts[-4].upper()
        action = _standardize_action(file.stem)

        angles = angles.reshape(angles.shape[0], -1, 3)[:, 1:]
        if downsample and downsample > 1:
            angles = angles[::downsample]

        if subject not in data:
            data[subject] = {}
        if action not in data[subject]:
            data[subject][action] = []
        data[subject][action].append(angles)

    return data


def load_processed(root_dir: Union[str, Path] = Path("h36m-tools/data/processed"),
                   rep: str = "expmap",
                   convention: str = "ZXY",
                   degrees: bool = False
                   ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Load preprocessed H3.6M data (train/test splits + normalization stats).

    Directory structure assumed:
        root_dir/rep=<rep>_conv=<convention>_deg=<degrees>/
            train/*.pt
            test/*.pt
            mean.pt
            std.pt

    Args:
        root_dir (Path): Base processed directory (e.g. h36m-tools/data/processed)
        rep (str): Representation type ("expmap", "quat", "rot6", ...)
        convention (str): Euler convention if used during preprocessing
        degrees (bool): Whether Euler was saved in degrees

    Returns:
        train (list[Tensor])
        test  (list[Tensor])
        mean  (Tensor)
        std   (Tensor)
    """
    rep_dir = get_rep_dir(Path(root_dir), rep=rep, convention=convention, degrees=degrees)
    if not rep_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {rep_dir}")

    mean = read_files(rep_dir / "mean.pt")
    std = read_files(rep_dir / "std.pt")

    train_dir = rep_dir / "train"
    test_dir = rep_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Missing train/test folders inside {rep_dir}")
    train = read_files(sorted(train_dir.glob("*.pt")))
    test = read_files(sorted(test_dir.glob("*.pt")))

    return train, test, mean, std