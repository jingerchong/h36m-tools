import torch
import logging
import roma
from typing import List

from h36m_tools.rotations import to_quat, identity_rotation
from h36m_tools.dims import add_dims
from h36m_tools.metadata import PARENTS, OFFSETS, STATIC_JOINTS, SITE_JOINTS, NUM_JOINTS, TOTAL_JOINTS


logger = logging.getLogger(__name__)


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
        quat[..., 0, :] = roma.identity_quat(size=quat.shape[:-2], dtype=quat.dtype, device=quat.device)  # [..., 4]
 
    for i in range(1, quat.shape[-2]): 
        parent = parents[i] 
        if parent == -1: 
            continue 
        # Accumulate global rotation from parent
        quat[..., i, :] = roma.quat_product(quat[..., parent, :], quat[..., i, :])  # [..., 4]
        # Rotate bone offset by parent rotation and add to parent position
        offset = offsets[i].expand(*quat[..., parent, :].shape[:-1], 3)
        pos[..., i, :] = pos[..., parent, :] + roma.quat_action(quat[..., parent, :], offset)  # [..., 3]
 
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
    logger.debug(f"fk() converting {rep} -> quat, resulting shape {quat.shape}")
    return _fk_quat(quat, parents=parents, offsets=offsets, ignore_root=ignore_root)


def fill_static_and_site_joints(rot: torch.Tensor, rep: str, **kwargs) -> torch.Tensor:
    """
    Fill static and site joints with identity rotations for FK compatibility.
    Uses globals: STATIC_JOINTS, SITE_JOINTS, NUM_JOINTS, TOTAL_JOINTS.
    """
    orig_shape = rot.shape
    fill = identity_rotation(rep, (rot.shape[0], len(STATIC_JOINTS)), **kwargs)
    rot = add_dims(rot, STATIC_JOINTS, NUM_JOINTS, fill, axis=-2)
    fill = identity_rotation(rep, (rot.shape[0], len(SITE_JOINTS)), **kwargs)
    rot = add_dims(rot, SITE_JOINTS, TOTAL_JOINTS, fill, axis=-2)
    logger.debug(f"fill_static_and_site_joints: tensor shape {orig_shape} -> {rot.shape}")
    return rot