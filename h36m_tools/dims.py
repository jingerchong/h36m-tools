import torch
from typing import Iterable, Union
import logging


logger = logging.getLogger(__name__)


def create_dim_mask(selected_dims: Iterable[int], total_dims: int) -> torch.BoolTensor:
    """
    Create a boolean mask of length total_dims where selected dims = True.
    """
    mask = torch.zeros(total_dims, dtype=torch.bool)
    mask[list(selected_dims)] = True
    logger.debug(f"create_dim_mask: selected_dims={list(selected_dims)}, total_dims={total_dims}, mask={mask}")
    return mask


def remove_dims(tensor: torch.Tensor,
                mask: torch.BoolTensor,
                axis: int = -1) -> torch.Tensor:
    """
    Remove dimensions from a tensor at positions specified by a boolean mask along a given axis.

    Args:
        tensor: Input tensor.
        mask: Boolean mask of length equal to tensor.shape[axis]. True dims are removed.
        axis: Axis along which to remove dimensions. Default: -1.

    Returns:
        Tensor with specified dims removed.
    """
    assert tensor.shape[axis] == mask.numel(), "Mask length must match tensor size along specified dim"
    idx = (~mask).nonzero(as_tuple=True)[0]
    out = torch.index_select(tensor, axis, idx)
    
    logger.debug(f"remove_dims: axis={axis}, tensor_shape={tensor.shape}, mask={mask}, "
                 f"kept_indices={idx.tolist()}, output_shape={out.shape}")
    return out


def add_dims(tensor: torch.Tensor,
             mask: torch.BoolTensor,
             fill_values: Union[float, torch.Tensor] = 0.0,
             axis: int = -1) -> torch.Tensor:
    """
    Add dimensions into a tensor at positions specified by a boolean mask along a given axis.

    Args:
        tensor: Tensor with shape [..., D_kept, ...].
        mask: Boolean mask of length total_dims. True dims are added, False dims come from tensor.
        fill_values: Scalar or tensor to fill added dimensions.
        axis: Axis along which to add dimensions. Default: -1.

    Returns:
        Tensor with added dimensions.
    """
    total_dims = mask.numel()
    kept_count = (~mask).sum().item()
    
    if axis < 0:
        axis = tensor.ndim + axis
    
    assert tensor.shape[axis] == kept_count, \
        f"Tensor has {tensor.shape[axis]} dims along axis {axis}, but mask indicates {kept_count} kept dims"
    
    out_shape = list(tensor.shape)
    out_shape[axis] = total_dims
    out = torch.empty(*out_shape, dtype=tensor.dtype, device=tensor.device)

    kept_idx = (~mask).nonzero(as_tuple=True)[0]
    out.index_copy_(axis, kept_idx, tensor)

    add_idx = mask.nonzero(as_tuple=True)[0]
    if isinstance(fill_values, torch.Tensor):
        expected_shape = list(out_shape)
        expected_shape[axis] = add_idx.numel()
        assert fill_values.shape == tuple(expected_shape), \
            f"fill_values shape {fill_values.shape} does not match expected {tuple(expected_shape)}"
        out.index_copy_(axis, add_idx, fill_values)
    else:
        out.index_fill_(axis, add_idx, fill_values)

    logger.debug(f"add_dims: axis={axis}, tensor_shape={tensor.shape}, mask={mask}, "
                 f"added_indices={add_idx.tolist()}, fill_values={fill_values}, output_shape={out.shape}")
    return out