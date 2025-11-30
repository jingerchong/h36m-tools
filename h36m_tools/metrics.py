import torch
import logging 

from h36m_tools.rotations import to_euler
from h36m_tools.kinematics import fk
from h36m_tools.metadata import PARENTS, OFFSETS


def mae_l2(y_pred: torch.Tensor,
           y: torch.Tensor,
           rep: str = "quat",
           ignore_root: bool = True,
           keep_time_dim: bool = True,
           **kwargs) -> torch.Tensor:
    """
    Compute the mean angular L2 error between predicted and target rotations.

    The inputs are converted to Euler angles (ZYX) before computing
    the angular difference. Supports ignoring the root joint.

    Args:
        y_pred: Predicted rotations, shape [..., J, ?].
        y: Ground-truth rotations, shape [..., J, ?].
        rep: Representation of input rotations ("quat", "euler", "expmap", "rot6", "rot9").
        ignore_root: If True, exclude the root joint from error computation.
        keep_time_dim: If True, returns per-time-step errors; otherwise returns a scalar mean.
        **kwargs: Forwarded to `to_euler` for additional options (e.g., src_order).

    Returns:
        Tensor of shape [T] if keep_time_dim=True, otherwise scalar.
    """
    euler_pred = to_euler(y_pred, rep=rep, order="zyx", **kwargs)
    euler = to_euler(y, rep=rep, order="zyx", **kwargs)

    diff = torch.remainder(euler - euler_pred + torch.pi, 2 * torch.pi) - torch.pi
    if ignore_root:
        diff[..., :3] = 0.0

    error = torch.mean(diff.norm(dim=-1), dim=0)  # L2 norm over joints, mean over batch
    error = error if keep_time_dim else error.mean()
    logging.debug(f"MAE L2 result shape: {error.shape}, keep_time_dim={keep_time_dim}")
    return error


def mpjpe(y_pred: torch.Tensor,
          y: torch.Tensor,
          rep: str = "quat",
          ignore_root: bool = True,
          keep_time_dim: bool = True,
          **kwargs) -> torch.Tensor:
    """
    Compute Mean Per-Joint Position Error (MPJPE) between predicted and target poses.
    Inputs are converted to 3D joint positions using forward kinematics.

    Args:
        y_pred: Predicted rotations/poses, shape [..., J, ?].
        y: Ground-truth rotations/poses, shape [..., J, ?].
        rep: Representation of input rotations ("quat", "euler", "expmap", "rot6", "rot9").
        ignore_root: If True, zero out the root joint in FK before computing error.
        keep_time_dim: If True, returns per-time-step errors; otherwise returns a scalar mean.
        **kwargs: Forwarded to `fk` for additional options.

    Returns:
        Tensor of shape [T] if keep_time_dim=True, otherwise scalar.
    """
    pos_pred = fk(y_pred, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)
    pos = fk(y, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)

    error = (pos - pos_pred).norm(dim=-1).mean(dim=[0, 2])  # norm over XYZ, mean over batch and joints
    error = error if keep_time_dim else error.mean()
    logging.debug(f"MPJPE result shape: {error.shape}, keep_time_dim={keep_time_dim}")
    return error