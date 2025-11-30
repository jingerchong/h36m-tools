import torch
import logging 

from h36m_tools.rotations import to_quat, quat_to
from h36m_tools.kinematics import fk
from h36m_tools.metadata import PARENTS, OFFSETS


def mae_l2(y_pred: torch.Tensor,
           y_gt: torch.Tensor,
           rep: str = "quat",
           ignore_root: bool = True,
           reduce_all: bool = False,
           **kwargs) -> torch.Tensor:
    """
    Compute the mean angular L2 error between predicted and target rotations.

    The inputs are converted to Euler angles (ZYX) before computing
    the angular difference. Supports ignoring the root joint.

    Args:
        y_pred: Predicted rotations, shape [..., J, D].
        y_gt: Ground-truth rotations, shape [..., J, D].
        rep: Representation of input rotations ("quat", "euler", "expmap", "rot6", "rot9").
        ignore_root: If True, exclude the root joint from error computation.
        reduce_all: If True, returns scalar; otherwise preserves leading dimensions.
        **kwargs: Forwarded to conversion functions for additional options (e.g., convention, degrees).

    Returns:
        Tensor of shape [...] if reduce_all=False, otherwise scalar.
    """
    quat_pred = to_quat(y_pred, rep=rep, **kwargs)
    quat_gt = to_quat(y_gt, rep=rep, **kwargs)
    
    euler_pred = quat_to(quat_pred, rep="euler", convention="ZYX", **kwargs)
    euler_gt = quat_to(quat_gt, rep="euler", convention="ZYX", **kwargs)

    diff = torch.remainder(euler_gt - euler_pred + torch.pi, 2 * torch.pi) - torch.pi  # [..., J]
    if ignore_root:
        diff[..., 0, :] = 0.0  

    error = diff.norm(dim=-1).mean(dim=-1)  # [...]
    if reduce_all:
        error = error.mean()  # scalar
    
    logging.debug(f"MAE L2 result shape: {error.shape}, reduce_all={reduce_all}")
    return error


def mpjpe(y_pred: torch.Tensor,
          y_gt: torch.Tensor,
          rep: str = "quat",
          ignore_root: bool = True,
          reduce_all: bool = False,
          **kwargs) -> torch.Tensor:
    """
    Compute Mean Per-Joint Position Error (MPJPE) between predicted and ground-truth poses.
    Converts inputs to 3D joint positions using differentiable FK.

    Args:
        y_pred: Predicted rotations/poses, shape [..., J, D].
        y_gt: Ground-truth rotations/poses, shape [..., J, D].
        rep: Rotation representation ("quat", "euler", "expmap", "rot6", "rot9").
        ignore_root: If True, zero out root rotation before FK.
        reduce_all: If True, return scalar; otherwise preserve leading dimensions.
        **kwargs: Forwarded to `fk` (e.g., convention, degrees).

    Returns:
        Tensor of shape [...] if reduce_all=False, otherwise scalar.
    """
    pos_pred = fk(y_pred, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)
    pos_gt = fk(y_gt, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)

    diff = (pos_gt - pos_pred).norm(dim=-1)    # [..., J]

    error = diff.mean(dim=-1)                  # [...]
    if reduce_all:
        error = error.mean()

    logging.debug(f"MPJPE result shape: {error.shape}, reduce_all={reduce_all}")
    return error

