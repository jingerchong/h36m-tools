import torch
import logging 
from scipy.stats import gaussian_kde
import numpy as np

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


def nll(y_pred_samples: torch.Tensor,
        y_gt: torch.Tensor,
        rep: str = "quat",
        ignore_root: bool = False,
        time_dim: int = 1,
        **kwargs) -> torch.Tensor:
    """
    Compute KDE-based Negative Log-Likelihood (NLL) per time step.

    Args:
        y_pred_samples: Predicted samples, shape [S, ..., J, D]
                        (S stochastic samples per joint). Sample axis is `sample_dim`.
        y_gt: Ground-truth positions, shape [..., J, D]
        rep: Rotation representation ("quat", "euler", "expmap", "rot6", "rot9").
        ignore_root: If True, zero out root rotation before FK.
        time_dim: Axis of time in y_gt
        **kwargs: Forwarded to `fk` (e.g., convention, degrees).

    Returns:
        Tensor of shape [T] if keep_time_dim=True, else scalar.
    """
    pos_pred = fk(y_pred_samples, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)
    pos_gt = fk(y_gt, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)

    pos_pred = pos_pred.transpose(time_dim + 1, -2)  # +1 because first dim = samples
    pos_gt = pos_gt.transpose(time_dim, -2)

    B = int(torch.prod(torch.tensor(pos_gt.shape[:-2])))
    T, J, D = pos_gt.shape[-3:]
    S = pos_pred.shape[0]
    pos_gt_flat = pos_gt.reshape(B, T, J, D)
    pos_pred_flat = pos_pred.reshape(S, B, T, J, D)

    nll_tensor = torch.zeros((B, T, J), dtype=torch.float32, device=y_gt.device)
    y_gt_np = pos_gt_flat.cpu().numpy()
    y_pred_np = pos_pred_flat.cpu().numpy()

    for b in range(B):
        for t in range(T):
            for j in range(J):
                try:
                    kde = gaussian_kde(y_pred_np[:, b, t, j, :].T)
                    log_prob = kde.logpdf(y_gt_np[b, t, j, :])
                    nll_tensor[b, t, j] = torch.tensor(-log_prob, device=y_gt.device)
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.warning(f"KDE failed for batch {b}, time {t}, joint {j}: {e}")
                    nll_tensor[b, t, j] = torch.nan  

    nll_per_time = torch.nanmean(torch.nanmean(nll_tensor, dim=-1), dim=0)  # [T]

    logger.debug(f"NLL result shape: {nll_per_time.shape}, time_dim={time_dim}")
    return nll_per_time  