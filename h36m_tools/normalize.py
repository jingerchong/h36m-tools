import torch
import logging
from typing import Tuple, List


logger = logging.getLogger(__name__)


def compute_stats(data_list: List[torch.Tensor], eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute global mean and std over a list of tensors.

    Args:
        data_list: List of tensors, each [T_i, J, D]
        eps: Minimum std threshold (for numerical stability)

    Returns:
        mean: [J, D] tensor
        std: [J, D] tensor (clamped to be >= eps)
    """
    all_frames = torch.cat(data_list, dim=0)  # [total_frames, J, D]
    mean = all_frames.mean(dim=0)  # [J, D]
    std = all_frames.std(dim=0, unbiased=False)  # [J, D]
    
    small_std_mask = std < eps
    if small_std_mask.any():
        n_small = small_std_mask.sum().item()
        min_std = std[small_std_mask].min().item()
        logger.warning(f"{n_small} dimension(s) have std < {eps:.1e}. Smallest std: {min_std:.2e}.")

    logger.debug(f"compute_stats: {len(data_list)} sequences, "
                  f"total {all_frames.shape[0]} frames -> mean/std shape {mean.shape}")
    return mean, std


def normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize data using z-score normalization.

    Args:
        data: [..., J, D] tensor with arbitrary batch dimensions
        mean: [J, D] normalization mean
        std: [J, D] normalization std

    Returns:
        Normalized data: [..., J, D]
    """
    normalized = (data - mean) / std
    logger.debug(f"normalize: {data.shape} -> {normalized.shape}")
    return normalized


def unnormalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Reverse z-score normalization.

    Args:
        data: [..., J, D] normalized tensor with arbitrary batch dimensions
        mean: [J, D] normalization mean
        std: [J, D] normalization std

    Returns:
        Unnormalized data: [..., J, D]
    """
    unnormalized = data * std + mean
    logger.debug(f"unnormalize: {data.shape} -> {unnormalized.shape}")
    return unnormalized