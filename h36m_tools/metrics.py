import torch
import logging 

from h36m_tools.rotations import to_quat, quat_to
from h36m_tools.kinematics import fk
from h36m_tools.metadata import PARENTS, OFFSETS, TOTAL_JOINTS, DOWNSAMPLE_FACTOR


logger = logging.getLogger(__name__)


def mae_l2(y_pred: torch.Tensor,
           y_gt: torch.Tensor,
           rep: str = "quat",
           ignore_root: bool = True,
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
        **kwargs: Forwarded to conversion functions for additional options (e.g., convention, degrees).

    Returns:
        Tensor of shape [..., T], mean L2 error per time step.
    """
    quat_pred = to_quat(y_pred, rep=rep, **kwargs)
    quat_gt = to_quat(y_gt, rep=rep, **kwargs)
    
    euler_pred = quat_to(quat_pred, rep="euler", convention="ZYX")
    euler_gt = quat_to(quat_gt, rep="euler", convention="ZYX")

    diff = torch.remainder(euler_gt - euler_pred + torch.pi, 2 * torch.pi) - torch.pi  # [..., J, D]

    if ignore_root:
        diff[..., 0, :] = 0.0  

    error = diff.view(*diff.shape[:-2], -1).norm(dim=-1)  # [...], L2 over J*D
    error = error.mean(dim=-2)  # [T] or [n_samples, T]
    
    logger.debug(f"MAE L2 result shape: {error.shape}")
    return error


def to_pos(y_pred: torch.Tensor,
           y_gt: torch.Tensor = None,
           rep: str = "quat", 
           ignore_root: bool = False,
           **kwargs):
    """
    Convert rotation representations to 3D joint positions using FK.
    
    Args:
        y_pred: Predicted rotations, shape [..., J, D] or [n_samples, ..., J, D].
        y_gt: Ground-truth rotations, same shape as y_pred (optional).
        rep: Rotation representation ("quat", "euler", "rot6", "rot9", "pos").
        ignore_root: If True, zero out root rotation before FK.
        **kwargs: Extra arguments for FK (e.g., convention, degrees).

    Returns:
        pos_pred: Tensor of positions in meters.
        pos_gt: Tensor of ground-truth positions in meters (or None if y_gt is None).
    """
    if rep == "pos":
        pos_pred = y_pred
        pos_gt = y_gt
    else:
        pos_pred = fk(y_pred, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)
        pos_gt = fk(y_gt, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs) if y_gt is not None else None

    # Convert from mm to meters
    pos_pred = pos_pred / 1000.0
    if pos_gt is not None:
        pos_gt = pos_gt / 1000.0

    return pos_pred, pos_gt


def mpjpe(pos_pred: torch.Tensor, pos_gt: torch.Tensor) -> torch.Tensor:
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
    diff = (pos_gt - pos_pred).norm(dim=-1)    # [..., J]

    # Evaluation protocol as defined in
    # https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/test.py
    # https://github.com/TUM-AAS/motron-cvpr22/blob/master/notebooks/RES%20H3.6M%20Evaluation%20Comparison%20HistRepItself.ipynb
    joint_to_ignore = [16, 20, 23, 24, 28, 31]
    joint_equal = [13, 19, 22, 13, 27, 30]
    diff[..., joint_to_ignore] = diff[..., joint_equal]

    error = diff.mean(dim=(-1, -3))*1000.0  # [T] or [n_samples, T], expresesd in mm

    logger.debug(f"MPJPE result shape: {error.shape}")
    return error


# Ignoring certain joints in generative eval according to
# https://github.com/TUM-AAS/motron-cvpr22/blob/master/notebooks/RES%20Eval%20NLL.ipynb
IGNORED_JOINTS = {0, 4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
KEPT_JOINTS = [x for x in range(TOTAL_JOINTS) if x not in IGNORED_JOINTS]


def nll_kde(y_pred_gen: torch.Tensor,
            y_gt_gen: torch.Tensor,
            rep: str = "quat",
            ignore_root: bool = False,
            **kwargs) -> torch.Tensor:
    """
    Fully vectorized batch-local KDE negative log-likelihood (NLL) in position space.

    Args:
        y_pred_gen: Generator yielding [n_samples, B, T, J, D] per batch.
        y_gt_gen: Generator yielding [B, T, J, D] per batch.
        rep: Rotation representation ("quat", "euler", "expmap", "rot6", "rot9").
        ignore_root: If True, zero out root rotation before FK.
        **kwargs: Forwarded to `fk()` (e.g., convention, degrees).

    Returns:
        Tensor of shape [T], mean NLL over joints and batches per time step.
    """
    nll_list = []

    for y_pred_batch, y_gt_batch in zip(y_pred_gen, y_gt_gen):
        device = y_gt_batch.device

        pos_pred, pos_gt = to_pos(y_pred_batch, y_gt_batch, rep, ignore_root, **kwargs)  # [n_samples, B, T, J, D], [B, T, J, D]
        n_samples, B, T, J, D = pos_pred.shape

        # Compute mean and covariance
        mean = pos_pred.mean(dim=0)  # [B, T, J, D]
        diff = pos_pred - mean.unsqueeze(0)  # [n_samples, B, T, J, D]
        cov = torch.einsum("sbtjd,sbtje->btjde", diff, diff) / n_samples  # [B, T, J, D, D]

        # Scott's rule for bandwidth
        cov = cov * (n_samples ** (-2.0 / (D + 4)))

        # Regularize covariance
        eps = 1e-14
        eye = torch.eye(D, device=device).view(1, 1, 1, D, D)
        L = torch.linalg.cholesky(cov + eps * eye)  # [B, T, J, D, D]

        # Compute log-likelihood
        diff_gt = pos_gt.unsqueeze(0) - pos_pred  # [n_samples, B, T, J, D]
        x = torch.linalg.solve_triangular(L.unsqueeze(0), diff_gt.unsqueeze(-1), upper=False)
        quad = (x**2).sum(dim=(-2, -1))  # [n_samples, B, T, J]

        logdet = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # [B, T, J]
        log_k = -0.5 * (quad + logdet.unsqueeze(0) + D * torch.log(torch.tensor(2 * torch.pi, device=device)))

        # Log-sum-exp over samples
        kde_ll = torch.logsumexp(log_k, dim=0) - torch.log(torch.tensor(float(n_samples), device=device))
        nll_list.append(-kde_ll)  # [B, T, J]

    nll_all = torch.cat(nll_list, dim=0)  # [N, T, J]
    nll_all = nll_all[..., KEPT_JOINTS]

    threshold = 20.0
    clamped = nll_all.clip(max=threshold)
    
    logger.warning(f"{(nll_all > threshold).sum().item()}/{nll_all.numel()} elements of NLL were clamped to {threshold}")
    error = clamped.sum(dim=2).mean(dim=0)  # [T]

    logger.debug(f"NLL-KDE result shape: {error.shape}")
    return error


def _upsample_trajectory(y_down: torch.Tensor, factor: int = DOWNSAMPLE_FACTOR) -> torch.Tensor:
    """
    Upsample a downsampled trajectory by a given factor.
    """
    n_samples, B, T_down, J, D = y_down.shape
    T_full = int(T_down * factor)
    y_flat = y_down.permute(0, 1, 3, 4, 2).reshape(n_samples * B * J * D, T_down, 1)  # [N, T, 1]
    y_up_flat = torch.nn.functional.interpolate(y_flat, size=T_full, mode='linear', align_corners=True)
    y_upsampled = y_up_flat.reshape(n_samples, B, J, D, T_full).permute(0, 1, 4, 2, 3)  # [S, B, T_full, J, D]
    return y_upsampled


def apd(pos_pred: torch.Tensor) -> float:
    """
    Compute the Average Pairwise Distance (APD) between multiple predicted sequences.

    Args:
        pos_pred: Tensor of shape [n_samples, B, T, J, D] representing predicted trajectories.

    Returns:
        float: Mean APD across batch, averaged over samples per batch element.
    """
    n_samples, B = pos_pred.shape[0], pos_pred.shape[1]
    if n_samples == 1:
        return 0.0
    
    pos_pred = _upsample_trajectory(pos_pred[..., KEPT_JOINTS, :])
    apd_total = 0.0
    for b in range(B):
        traj_flat = pos_pred[:, b].reshape(n_samples, -1)  # [S, F]
        dist = torch.pdist(traj_flat)  # [S*(S-1)/2]
        apd_total += dist.mean()
    return (apd_total / B).item()


def ade(pos_pred: torch.Tensor, pos_gt: torch.Tensor) -> float:
    """
    Compute Average Displacement Error (ADE) between predicted and ground-truth trajectories.

    Args:
        pos_pred: Tensor of shape [n_samples, B, T, J, D] predicted trajectories.
        pos_gt: Tensor of shape [B, T, J, D] ground-truth trajectories.

    Returns:
        float: Minimum ADE over all predicted samples (best-of-K) averaged over batch.
    """
    pos_pred = _upsample_trajectory(pos_pred[..., KEPT_JOINTS, :])
    pos_gt = _upsample_trajectory(pos_gt[None, ..., KEPT_JOINTS, :])[0]
    pos_pred_flat = pos_pred.flatten(start_dim=-2)  # [n_samples, B, T, J*D]
    pos_gt_flat = pos_gt.flatten(start_dim=-2)      # [B, T, J*D]

    diff = pos_pred_flat - pos_gt_flat              # [n_samples, B, T, J*D]
    dist = diff.norm(dim=-1)                        # [n_samples, B, T]
    # Mean over time dimension
    ade_per_sample = dist.mean(dim=-1)               # [n_samples, B]
    # Take min over samples, then average over batch
    ade_min = ade_per_sample.min(dim=0).values.mean().item()

    logger.debug(f"compute_ade: n_samples={pos_pred.shape[0]}, ade_min={ade_min:.6f}")
    return ade_min


def fde(pos_pred: torch.Tensor, pos_gt: torch.Tensor) -> float:
    """
    Compute Final Displacement Error (FDE) between predicted and ground-truth trajectories.

    Args:
        pos_pred: Tensor of shape [n_samples, B, T, J, D] predicted trajectories.
        pos_gt: Tensor of shape [B, T, J, D] ground-truth trajectories.

    Returns:
        float: Minimum FDE over all predicted samples at the final timestep, averaged over batch.
    """
    pos_pred = pos_pred[..., KEPT_JOINTS, :]
    pos_gt = pos_gt[..., KEPT_JOINTS, :]
    pos_pred_flat = pos_pred.flatten(start_dim=-2)  # [n_samples, B, T, J*D]
    pos_gt_flat = pos_gt.flatten(start_dim=-2)      # [B, T, J*D]

    diff = pos_pred_flat[:, :, -1, :] - pos_gt_flat[:, -1, :]  # [n_samples, B, J*D]
    dist = diff.norm(dim=-1)                                   # [n_samples, B]
    # Take min over samples, then average over batch
    fde_min = dist.min(dim=0).values.mean().item()

    logger.debug(f"compute_fde: n_samples={pos_pred.shape[0]}, fde_min={fde_min:.6f}")
    return fde_min