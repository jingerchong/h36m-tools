import torch
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def debug_tensor(tensor: torch.Tensor, n_frames: int = 1):
    """
    Log basic info about a tensor, including shape and first (few) frames.

    Args:
        tensor (torch.Tensor): Tensor to inspect.
        n_frames (int): Number of frames to log from the start (default: 1).
    """
    n_frames = min(n_frames, tensor.shape[0])
    logging.info(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
    logging.info(f"First {n_frames} frame(s):\n{tensor[:n_frames].cpu()}")
