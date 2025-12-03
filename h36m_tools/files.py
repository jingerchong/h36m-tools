from pathlib import Path
import cdflib
import torch
import logging
from typing import List, Union
from tqdm import tqdm
import numpy as np

from h36m_tools.metadata import DEVICE


logger = logging.getLogger(__name__)


def read_files(inputs: Union[str, Path, List[Union[str, Path]]]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Read one or many files and convert them into PyTorch tensors.
    Supported formats:
        - .pt   : saved PyTorch tensor
        - .cdf  : raw H36M CDF file (from original dataset)
        - .txt  : expmap txt file (from Martinez preprocessing zip)

    Args:
        inputs (str | Path | list[str | Path]): A single file path or a list of file paths to load.

    Returns:
        torch.Tensor | list[torch.Tensor]:
            - If a single input file is provided → returns one tensor.
            - If multiple files are provided → returns a list of tensors,
            in the same order as the input list.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    elif not isinstance(inputs, (list, tuple)):
        raise TypeError(f"read_files expected str, Path, or list: got {type(inputs)}")
    
    outputs = []

    for file in tqdm(inputs, desc="Reading files", disable=len(inputs) < 20):
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
                logger.error(f"Skipped unsupported file type: {file}")
                continue
            logger.debug(f"Loaded {file} -> tensor shape: {outputs[-1].shape}")

        except Exception as e:
            logger.error(f"Failed to read {file}: {e}")
            continue
    
    if len(outputs) == 1:
        return outputs[0]
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
        logger.debug(f"Saved tensor shape: {tensor.shape} -> {path}")
    except Exception as e:
        logger.error(f"Failed to save tensor to {path}: {e}")
