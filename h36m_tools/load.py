from pathlib import Path
import logging
from typing import List, Union, Dict
from tqdm import tqdm
import torch

from h36m_tools.files import read_files  
from h36m_tools.metadata import DOWNSAMPLE_FACTOR


logger = logging.getLogger(__name__)


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
        dict: Nested dictionary of tensors keyed by subject → action → list of tensors
    """
    input_files = list(Path(root_dir).rglob("**/MyPoseFeatures/D3_Angles/*.cdf"))
    if not input_files:
        raise FileNotFoundError("No D3_Angles CDF files found")

    logger.info(f"Found {len(input_files)} D3_Angles CDF files to process")
    all_angles = read_files(input_files)
    data: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    for file, angles in tqdm(zip(input_files, all_angles), desc="Processing angles", total=len(input_files)):
        subject = file.parts[-4].upper()
        action = "".join(c for c in file.stem if c.isalpha()).lower()

        T, total_dims = angles.shape
        J = total_dims // 3
        angles = angles.reshape(T, J, 3)[:, 1:]

        if downsample and downsample > 1:
            angles = angles[::downsample]

        if subject not in data:
            data[subject] = {}
        if action not in data[subject]:
            data[subject][action] = []
        data[subject][action].append(angles)

    return data