import torch
import logging
from typing import Iterable, Union


def create_dim_mask(selected_dims: Iterable[int], total_dims: int) -> torch.BoolTensor:
    """
    Create a boolean mask of length total_dims where selected dims = True.

    Args:
        selected_dims: List of integers representing the selected dimensions to be marked as True in the mask.
        total_dims: Total number of dimensions to create a mask for.

    Returns:
        Tensor of shape [total_dims] with True for selected dims, False otherwise.
    """
    if not all(isinstance(dim, int) for dim in selected_dims):
        raise ValueError(f"All elements of selected_dims must be integers.")
    if any(dim < 0 or dim >= total_dims for dim in selected_dims):
        raise ValueError(f"All elements of selected_dims must be in the range [0, {total_dims - 1}]")

    mask = torch.zeros(total_dims, dtype=torch.bool)
    mask[selected_dims] = True
    logging.debug(f"create_dim_mask: Created mask for selected dims {selected_dims} → mask shape {mask.shape}")
    return mask


def remove_dims(seq: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """
    Remove dims where mask == True.

    Args:
        seq: Input tensor with shape [..., D]
        mask: Boolean mask of shape [D] where True values indicate dimensions to be removed.

    Returns:
        Tensor with shape [..., D_remaining] where dimensions marked by mask are removed.
    """
    assert seq.shape[-1] == mask.numel(), "Mask length must match the last dimension of the sequence."
    
    out = seq[..., ~mask] 
    logging.debug(f"remove_dims: Removed dims with mask {mask.sum().item()} True values, "
                  f"input shape {seq.shape} → output shape {out.shape}")
    return out


def add_dims(seq: torch.Tensor,
             mask: torch.BoolTensor,
             fill_values: Union[float, torch.Tensor] = 0.0) -> torch.Tensor:
    """
    Add dims back into the sequence at positions where mask == True.

    Args:
        seq: Tensor with shape [..., D_kept] (kept dimensions).
        mask: Boolean mask with shape [D_total] specifying where dims should be added.
        fill_values: Scalar or tensor to fill the added dimensions. Default is 0.0.

    Returns:
        Tensor with shape [..., D_total] after adding back the missing dimensions.
    """
    assert seq.shape[-1] == mask.numel(), "Mask length must match the last dimension of the sequence."

    D_total = mask.numel()
    out = torch.empty(seq.shape[:-1] + (D_total,), dtype=seq.dtype, device=seq.device)

    out[..., ~mask] = seq

    if isinstance(fill_values, torch.Tensor):
        assert fill_values.shape == out[..., mask].shape, \
            f"Shape of fill_values tensor {fill_values.shape} must match shape of the masked output {out[..., mask].shape}."
        fill = fill_values
    else:
        fill = torch.full(out[..., mask].shape, fill_values, dtype=seq.dtype, device=seq.device)
    out[..., mask] = fill

    logging.debug(f"add_dims: input shape {seq.shape}, mask shape {mask.shape}, output shape {out.shape}, "
                  f"filled {mask.sum().item()} dims with {fill_values}")
    return out