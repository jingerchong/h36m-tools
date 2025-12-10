import torch
import logging 
from torch.distributions import MultivariateNormal
from scipy.stats import gaussian_kde
import numpy as np
from tqdm import trange

from h36m_tools.rotations import to_quat, quat_to
from h36m_tools.kinematics import fk
from h36m_tools.metadata import PARENTS, OFFSETS


logger = logging.getLogger(__name__)


def mae_l2(y_pred: torch.Tensor,
           y_gt: torch.Tensor,
           rep: str = "quat",
           ignore_root: bool = False,
           time_dim: int = 1,
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
        time_dim: Axis corresponding to the time dimension.
        **kwargs: Forwarded to conversion functions for additional options (e.g., convention, degrees).

    Returns:
        Tensor of shape [..., T], mean L2 error per time step.
    """
    quat_pred = to_quat(y_pred, rep=rep, **kwargs)
    quat_gt = to_quat(y_gt, rep=rep, **kwargs)
    
    euler_pred = quat_to(quat_pred, rep="euler", convention="ZYX")
    euler_gt = quat_to(quat_gt, rep="euler", convention="ZYX")

    diff = torch.remainder(euler_gt - euler_pred + torch.pi, 2 * torch.pi) - torch.pi  # [..., J]
    if ignore_root:
        diff[..., 0, :] = 0.0  

    error = diff.view(*diff.shape[:-2], -1).norm(dim=-1)  # [...], L2 over J*D
    reduce_dims = [i for i in range(error.dim()) if i != time_dim]
    error = error.mean(dim=reduce_dims)  # [T]
    
    logger.debug(f"MAE L2 result shape: {error.shape}, time_dim={time_dim}")
    return error


def mpjpe(y_pred: torch.Tensor,
          y_gt: torch.Tensor,
          rep: str = "quat",
          ignore_root: bool = False,
          time_dim: int = 1,
          **kwargs) -> torch.Tensor:
    """
    Compute Mean Per-Joint Position Error (MPJPE) between predicted and ground-truth poses.
    Converts inputs to 3D joint positions using differentiable FK.

    Args:
        y_pred: Predicted rotations/poses, shape [..., J, D].
        y_gt: Ground-truth rotations/poses, shape [..., J, D].
        rep: Rotation representation ("quat", "euler", "expmap", "rot6", "rot9").
        ignore_root: If True, zero out root rotation before FK.
        time_dim: Axis corresponding to the time dimension.
        **kwargs: Forwarded to `fk` (e.g., convention, degrees).

    Returns:
        Tensor of shape [..., T], mean per joint position error per time step.
    """
    pos_pred = fk(y_pred, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)
    pos_gt = fk(y_gt, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)

    diff = (pos_gt - pos_pred).norm(dim=-1)    # [..., J]

    reduce_dims = [i for i in range(diff.dim()) if i != time_dim]
    error = diff.mean(dim=reduce_dims)  # [T]

    logger.debug(f"MPJPE result shape: {error.shape}, time_dim={time_dim}")
    return error


def nll_gaussian(y_pred_samples: torch.Tensor,
                 y_gt: torch.Tensor,
                 rep: str = "quat",
                 ignore_root: bool = False,
                 time_dim: int = 1,
                 min_std: float = 1e-6,
                 jitter: float = 1e-4,
                 **kwargs) -> torch.Tensor:
    """
    Compute Gaussian Negative Log-Likelihood (NLL) per time step.

    Args:
        y_pred_samples: Predicted samples, shape [S, ..., J, D]
                        (S stochastic samples per joint). Sample axis is first dimension.
        y_gt: Ground-truth positions, shape [..., J, D]
        rep: Rotation representation ("quat", "euler", "expmap", "rot6", "rot9").
        ignore_root: If True, zero out root rotation before FK.
        time_dim: Axis of time in y_gt
        min_std: Minimum standard deviation threshold for valid dimensions.
        **kwargs: Forwarded to `fk` (e.g., convention, degrees).

    Returns:
        Tensor of shape [T], NLL per time step.
    """
    S = y_pred_samples.shape[0]
    if S < 1000:
        logger.warning(f"Number of stochastic samples is small: {S} < {1000}. NLL estimates may be unreliable or very high.")

    pos_pred = fk(y_pred_samples, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)
    pos_gt = fk(y_gt, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)

    if time_dim != 1:
        pos_pred = pos_pred.transpose(time_dim, 2)  # move time to axis 2
        pos_gt   = pos_gt.transpose(time_dim, 1)    # same for ground truth

    *batch_dims, T, J, D = pos_gt.shape
    B = int(torch.tensor(batch_dims).prod().item())
    pos_gt_flat   = pos_gt.reshape(B, T, J, D)          # [B, T, J, D]
    pos_pred_flat = pos_pred.reshape(pos_pred.shape[0], B, T, J, D)  # [S, B, T, J, D]

    mu   = pos_pred_flat.mean(dim=0)        # [B, T, J, D]
    stds = pos_pred_flat.std(dim=0)         # [B, T, J, D]
    valid_mask = (stds >= min_std).all(-1)  # [B, T, J]
    logger.debug(f"Mean std: {stds.mean():.6f}, Fraction degenerate: {(~valid_mask).float().mean():.4f}")

    diffs = pos_pred_flat - mu.unsqueeze(0)  # [S, B, T, J, D]
    # cov[b,t,j] = (diffs^T @ diffs) / (S-1)
    cov = torch.einsum("sbtjd,sbtje->btjde", diffs, diffs) / (S - 1)  # [B, T, J, D, D]
    cov = cov + jitter * torch.eye(D, device=cov.device)

    dist = MultivariateNormal(loc=mu, covariance_matrix=cov)  # batch MVN
    logp = dist.log_prob(pos_gt_flat)  # [B, T, J]
    logp = logp.masked_fill(~valid_mask, float("nan"))
    nll_per_time = -logp.nanmean(dim=(0, 2))  # [T]

    logger.debug(f"Gaussian NLL result shape: {nll_per_time.shape}, time_dim={time_dim}")
    return nll_per_time
