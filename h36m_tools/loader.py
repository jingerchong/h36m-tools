from pathlib import Path
import logging
from typing import List, Optional, Union, Dict
from tqdm import tqdm
import torch

from h36m_tools.files import read_files  


def _clean_action_name(filename: str) -> str:
    """Keep only alphabetic characters, lowercase. Removes spaces, digits, punctuation, etc."""
    return "".join(c for c in filename if c.isalpha()).lower()


def load_raw(root_dir: Union[str, Path] = Path("data/raw"),
             subjects: Optional[List[str]] = None,
             actions: Optional[List[str]] = None,
             downsample: int = 2
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

    if subjects is not None:
        subjects_upper = [s.upper() for s in subjects]
        input_files = [f for f in input_files if f.parts[-4].upper() in subjects_upper]
        logging.debug(f"Filtered to subjects: {subjects_upper}, remaining files: {len(input_files)}")

    if actions is not None:
        actions_clean = [_clean_action_name(a) for a in actions]
        input_files = [f for f in input_files if _clean_action_name(f.stem) in actions_clean]
        logging.debug(f"Filtered to actions: {actions_clean}, remaining files: {len(input_files)}")

    logging.info(f"Found {len(input_files)} D3_Angles CDF files to process")

    all_angles = read_files(input_files)

    data: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    for file, angles in tqdm(zip(input_files, all_angles), desc="Processing angles", total=len(input_files)):
        subject = file.parts[-4].upper()
        action = _clean_action_name(file.stem)

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