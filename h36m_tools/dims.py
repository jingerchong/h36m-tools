import torch
from typing import Union, Iterable, List, Sequence, Tuple
import logging


logger = logging.getLogger(__name__)


def _create_mask(selected_dims: Iterable[int], total_dims: int, device: torch.device) -> torch.BoolTensor:
    """Create a boolean mask of length total_dims where selected_dims are True."""
    mask = torch.zeros(total_dims, dtype=torch.bool, device=device)
    mask[list(selected_dims)] = True
    return mask


def remove_dims(tensor: torch.Tensor,
                selected_dims: Iterable[int],
                total_dims: int,
                axis: int = -1) -> torch.Tensor:
    """
    Remove dimensions from a tensor at positions specified by selected_dims along a given axis.

    Args:
        tensor: Input tensor.
        selected_dims: Indices of dimensions to remove.
        total_dims: Total number of dimensions along the axis.
        axis: Axis along which to remove dimensions. Default: -1.

    Returns:
        Tensor with specified dims removed.
    """
    mask = _create_mask(selected_dims, total_dims, tensor.device)
    idx = (~mask).nonzero(as_tuple=True)[0]
    out = torch.index_select(tensor, axis, idx)
    
    logger.debug(f"remove_dims: selected_dims={list(selected_dims)}, axis={axis}, "
                 f"tensor_shape={tensor.shape}, output_shape={out.shape}")
    return out


def add_dims(tensor: torch.Tensor,
             selected_dims: Iterable[int],
             total_dims: int,
             fill_values: Union[float, torch.Tensor] = 0.0,
             axis: int = -1) -> torch.Tensor:
    """
    Add dimensions into a tensor at positions specified by selected_dims along a given axis.

    Args:
        tensor: Tensor with shape [..., D_kept, ...].
        selected_dims: Indices of dimensions to add.
        total_dims: Total number of dimensions along the axis after addition.
        fill_values: Scalar or tensor to fill added dimensions.
        axis: Axis along which to add dimensions. Default: -1.

    Returns:
        Tensor with added dimensions.
    """
    mask = _create_mask(selected_dims, total_dims, tensor.device)
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

    logger.debug(f"add_dims: selected_dims={list(selected_dims)}, axis={axis}, "
                 f"tensor_shape={tensor.shape}, output_shape={out.shape}")
    return out


def remap_index_group(total_items: int,
                      removed_indices: Iterable[int],
                      group: List[Sequence[int]],
                      ) -> List[Tuple[int, ...]]:
    """
    Remap a list of index sequences after removing certain indices.

    Args:
        total_items: total number of items before removal
        removed_indices: indices to remove
        group: list of sequences (e.g., chains or any grouped indices)

    Returns:
        remapped_group: list of sequences remapped according to the new indices
    """
    removed_set = set(removed_indices)
    mapping = {i: (j := None) if i in removed_set else (j := sum(1 for k in range(i) if k not in removed_set))
               for i in range(total_items)}
    
    remapped_group = [
        tuple(mapping[i] for i in seq if mapping[i] is not None)
        for seq in group
        if any(mapping[i] is not None for i in seq)
    ]

    logger.debug(f"remap_index_group: original_group={group} -> remapped_group={remapped_group}")
    return remapped_group


def compute_dynamic_dims(std: torch.Tensor, threshold: float = 1e-4) -> List[List[int]]:
    """
    Compute dynamic dimensions per item based on a variance threshold.

    Args:
        std: Tensor of shape [N, D] representing per-dim standard deviations.
        threshold: Minimum std value to consider a dimension dynamic.

    Returns:
        List of length N; each element is a list of dynamic dim indices for that item.
    """
    N, D = std.shape
    dynamic_dims: List[List[int]] = []

    for item_idx in range(N):
        item_dynamic_dims = [dim_idx for dim_idx in range(D) if std[item_idx, dim_idx] > threshold]
        dynamic_dims.append(item_dynamic_dims)

    logger.debug(f"compute_dynamic_dims: dynamic_dims_lengths={[len(v) for v in dynamic_dims]}")
    return dynamic_dims