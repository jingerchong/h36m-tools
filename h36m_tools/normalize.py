import torch
import logging


def compute_stats(data_list):
    """
    Compute global mean and std over a list of tensors [T, J, D].

    Args:
        data_list: list of tensors, each [T_i, J, D]

    Returns:
        mean: [J, D] tensor
        std: [J, D] tensor
    """
    all_frames = torch.cat([d.reshape(-1, *d.shape[1:]) for d in data_list], dim=0)  # [total_frames, J, D]
    mean = all_frames.mean(dim=0)
    std = all_frames.std(dim=0, unbiased=False)
    logging.debug(f"compute_stats: concatenated shape {all_frames.shape}, mean shape {mean.shape}, std shape {std.shape}")
    return mean, std


def _expand_for_batch(stat: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """Expand [J, D] to broadcast over leading batch dims of data [..., J, D]."""
    return stat.view((1,) * (data.ndim - 2) + stat.shape)


def normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor using provided mean/std. Supports arbitrary leading batch dims.

    Args:
        data: [..., J, D] tensor
        mean: [J, D]
        std: [J, D]

    Returns:
        normalized data: [..., J, D]
    """
    norm_data = (data - _expand_for_batch(mean, data)) / _expand_for_batch(std, data)
    logging.debug(f"normalize: input {data.shape} → output {norm_data.shape}")
    return norm_data


def unnormalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Reverse normalization. Supports arbitrary leading batch dims.

    Args:
        data: [..., J, D] normalized tensor
        mean: [J, D]
        std: [J, D]

    Returns:
        unnormalized data: [..., J, D]
    """
    unnorm_data = data * _expand_for_batch(std, data) + _expand_for_batch(mean, data)
    logging.debug(f"unnormalize: input {data.shape} → output {unnorm_data.shape}")
    return unnorm_data