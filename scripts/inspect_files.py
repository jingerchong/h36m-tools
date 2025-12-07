import argparse
import torch
import logging
from pathlib import Path

from h36m_tools.utils import setup_logger
from h36m_tools.files import read_files


logger = logging.getLogger(__name__)


def inspect_files(paths: list[Path], n_frames: int = 1):
    """
    Load and inspect one or multiple tensor files (.pt, .txt, or .cdf), logging 
    basic information such as shape, dtype, device, min/max values, and the first few frames.

    Args:
        paths (list[Path]): List of file paths to load. Supported formats:
            - .pt   : PyTorch tensor file
            - .txt  : Text file with numeric pose/rotation data
            - .cdf  : H3.6M raw CDF file
        n_frames (int, optional): Number of initial frames to display for each tensor. 
            Defaults to 1. If the tensor has fewer frames than `n_frames`, all frames are shown.

    Returns:
        None. Logs information about each tensor to the configured logger.
    """
    tensors = read_files(paths)

    for path, tensor in zip(paths, tensors):
        logger.info("")
        logger.info(f"Loaded tensor from: {path}")
        logger.info(f"Shape: {tensor.shape}")
        logger.info(f"Dtype: {tensor.dtype}")
        logger.info(f"Device (loaded): {tensor.device}")

        try:
            logger.info(f"Min:  {tensor.min().item():.4f}")
            logger.info(f"Max:  {tensor.max().item():.4f}")
        except Exception:
            logger.warning("Could not compute statistics (tensor may be non-numeric).")

        if tensor.ndim >= 1:
            n = min(n_frames, tensor.shape[0])
            logger.info(f"First {n} frame(s):\n{tensor[:n]}")
        else:
            logger.info(f"Tensor value:\n{tensor}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Tensor inspection debugger")
    parser.add_argument("-i", "--input", type=Path, nargs="+", required=True, help="One or more .cdf, .txt, or .pt files")
    parser.add_argument("-n", "--n_frames", type=int, default=1, help="Number of initial frames to show")

    args = parser.parse_args()
    setup_logger()

    torch.set_printoptions(precision=4, sci_mode=False)
    inspect_files(args.input, args.n_frames)
