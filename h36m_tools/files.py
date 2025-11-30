from pathlib import Path
import cdflib
import torch
import logging
from typing import List, Union
from tqdm import tqdm
import numpy as np

from h36m_tools.metadata import DEVICE


def read_files(inputs: List[Union[str, Path]]) -> List[torch.Tensor]:
    """
    Read multiple files and convert them into PyTorch tensors.
    Supported formats:
        - .pt   : saved PyTorch tensor
        - .cdf  : raw H36M CDF file (from original dataset)
        - .txt  : expmap txt file (from Martinez preprocessing zip)

    Args:
        inputs (list[str | Path]): List of paths to files to load.

    Returns:
        list[torch.Tensor]: List of tensors loaded from each input file, in original order.
    """
    outputs = []

    for file in tqdm(inputs, desc="Reading files"):
        suffix = Path(file).suffix.lower()

        try:
            if suffix == ".pt":
                outputs.append(torch.load(file, map_location=DEVICE))
            elif suffix == ".cdf":
                cdf = cdflib.CDF(str(file))
                outputs.append(torch.tensor(cdf.varget("Pose"), device=DEVICE).squeeze(0))
            elif suffix == ".txt":
                pose_np = np.loadtxt(file, delimiter=",", dtype=np.float32)
                outputs.append(torch.tensor(pose_np, device=DEVICE))
            else:
                logging.error(f"Skipped unsupported file type: {file}")
                continue
            logging.debug(f"Loaded {file} → tensor shape: {outputs[-1].shape}")

        except Exception as e:
            logging.error(f"Failed to read {file}: {e}")
            continue

    return outputs


def save_tensor(path: Union[str, Path], tensor: torch.Tensor):
    """
    Save a single PyTorch tensor to a .pt file.

    Args:
        path (str | Path): Output file path to save the tensor.
        tensor (torch.Tensor): Tensor to save.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
        logging.debug(f"Saved tensor shape: {tensor.shape} → {path}")
    except Exception as e:
        logging.error(f"Failed to save tensor to {path}: {e}")