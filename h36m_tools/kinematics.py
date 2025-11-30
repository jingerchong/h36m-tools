import torch
import logging
import roma
from typing import List

from h36m_tools.rotations import to_quat
from h36m_tools.metadata import PARENTS, OFFSETS


def _fk_quat(quat: torch.Tensor,
             parents: List[int] = PARENTS,
             offsets: torch.Tensor = OFFSETS,
             ignore_root: bool = True) -> torch.Tensor:
    """
    Forward kinematics using quaternion rotations, supports arbitrary batch dims.

    Args: 
        quat: [..., J, 4] quaternions in XYZW format (Roma convention)
        parents: list[int] length J 
        offsets: [J, 3] tensor 
        ignore_root: zero root rotation 
 
    Returns: 
        Tensor [..., J, 3] joint positions 
    """
    quat = roma.quat_normalize(quat)  # [..., J, 4]
    pos = torch.zeros_like(quat[..., :3])  # [..., J, 3]
 
    if ignore_root: 
        quat[..., 0, :] = roma.quat_identity(size=quat.shape[:-2], dtype=quat.dtype, device=quat.device)  # [..., 4]
 
    for i in range(1, quat.shape[-2]): 
        parent = parents[i] 
        if parent == -1: 
            continue 
        # Accumulate global rotation from parent
        quat[..., i, :] = roma.quat_product(quat[..., parent, :], quat[..., i, :])  # [..., 4]
        # Rotate bone offset by parent rotation and add to parent position
        pos[..., i, :] = pos[..., parent, :] + roma.quat_action(quat[..., parent, :], offsets[i])  # [..., 3]
 
    return pos


def fk(rot: torch.Tensor,
       rep: str = "quat",
       parents: List[int] = PARENTS,
       offsets: torch.Tensor = OFFSETS,
       ignore_root: bool = True,
       **kwargs) -> torch.Tensor:
    """
    Unified forward kinematics entry point.
    Accepts any rotation representation, converts to quaternion, and runs FK.

    Args:
        rot: rotation tensor [..., J, ?] depending on representation
        rep: input representation ("quat", "expmap", "euler", "rot6", "rot9")
        parents: list[int] length J
        offsets: [J, 3] joint offsets
        ignore_root: if True, zero root rotation 
        **kwargs: passed to to_quat() 

    Returns:
        Joint positions [..., J, 3]
    """
    quat = to_quat(rot, rep=rep, **kwargs)
    logging.debug(f"fk() converting {rep} → quat, resulting shape {quat.shape}")
    return _fk_quat(quat, parents=parents, offsets=offsets, ignore_root=ignore_root)
