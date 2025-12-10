from pathlib import Path
import cdflib
import torch
import logging
from typing import List, Union, Any, Union
from tqdm import tqdm
import numpy as np


logger = logging.getLogger(__name__)

DATA_DEVICE = "cpu"  # Default CPU, discouraged can also be changed to GPU 
# DATA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def read_files(inputs: Union[str, Path, List[Union[str, Path]]], device: torch.device = DATA_DEVICE) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Read one or many files and convert them into PyTorch tensors.
    Supported formats:
        - .pt/.pth : saved PyTorch tensor or model state dict
        - .cdf     : raw H36M CDF file (from original dataset)
        - .txt     : expmap txt file (from Martinez preprocessing zip)

    Args:
        inputs (str | Path | list[str | Path]): A single file path or a list of file paths to load.

    Returns:
        torch.Tensor | dict | list[torch.Tensor | dict]:
            - If a single input file is provided -> returns one tensor or dict.
            - If multiple files are provided -> returns a list, in the same order as the input list.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    elif not isinstance(inputs, (list, tuple)):
        raise TypeError(f"read_files expected str, Path, or list: got {type(inputs)}")
    
    outputs = []

    for file in tqdm(inputs, desc="Reading files", disable=len(inputs) < 20):
        suffix = Path(file).suffix.lower()

        try:
            if suffix in (".pt", ".pth"):
                outputs.append(torch.load(file, map_location=device))
            elif suffix == ".cdf":
                cdf = cdflib.CDF(str(file))
                outputs.append(torch.tensor(cdf.varget("Pose"), device=device).squeeze(0))
            elif suffix == ".txt":
                pose_np = np.loadtxt(file, delimiter=",", dtype=np.float32)
                outputs.append(torch.tensor(pose_np, device=device))
            else:
                logger.error(f"Skipped unsupported file type: {file}")
                continue
            logger.debug(f"Loaded {file} -> type: {type(outputs[-1])}, shape: {getattr(outputs[-1], 'shape', 'N/A')}")

        except Exception as e:
            logger.error(f"Failed to read {file}: {e}")
            continue
    
    if len(outputs) == 1:
        return outputs[0]
    return outputs


def save_object(path: Union[str, Path], obj: Any):
    """
    Save a PyTorch-serializable object (.pt or .pth file).

    Supports:
        - torch.Tensor
        - dict
        - list
        - anything torch.save can handle
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
        if isinstance(obj, torch.Tensor):
            logger.debug(f"Saved tensor with shape {obj.shape} -> {path}")
    except Exception as e:
        logger.error(f"Failed to save object to {path}: {e}")
