import torch
import logging 

from h36m_tools.rotations import to_quat, quat_to
from h36m_tools.kinematics import fk
from h36m_tools.metadata import PARENTS, OFFSETS, STATIC_JOINTS, STATIC_PARENTS, SITE_JOINTS, SITE_PARENTS, NUM_JOINTS, TOTAL_JOINTS
from h36m_tools.dims import add_dims


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
        time_dim: Axis corresponding to the time dimension.
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

    time_dim = diff.ndim - 3
    error = diff.view(*diff.shape[:-2], -1).norm(dim=-1)  # [...], L2 over J*D
    reduce_dims = [i for i in range(error.dim()) if i != time_dim]
    error = error.mean(dim=reduce_dims)  # [T]
    
    logger.debug(f"MAE L2 result shape: {error.shape}")
    return error


def _to_pos(y_pred: torch.Tensor,
            y_gt: torch.Tensor,
            rep: str = "quat", 
            ignore_root: bool = False,
            meters: bool = False,
            **kwargs):
    if rep == "pos":
        pos_pred, pos_gt = y_pred, y_gt
    else:
        pos_pred = fk(y_pred, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)
        pos_gt = fk(y_gt, rep=rep, parents=PARENTS, offsets=OFFSETS, ignore_root=ignore_root, **kwargs)
    if meters:
        pos_pred = pos_pred / 1000.0
        pos_gt   = pos_gt / 1000.0
    return pos_pred, pos_gt


def mpjpe(y_pred: torch.Tensor,
          y_gt: torch.Tensor,
          rep: str = "quat",
          ignore_root: bool = False,
          return_pos: bool = False,
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
    pos_pred, pos_gt = _to_pos(y_pred, y_gt, rep, ignore_root, **kwargs)
    diff = (pos_gt - pos_pred).norm(dim=-1)    # [..., J]
    if ignore_root:
        diff[..., 0] = 0.0  

    # Evaluation protocol as defined in
    # https://github.com/dulucas/siMLPe/blob/main/exps/baseline_h36m/test.py
    # https://github.com/TUM-AAS/motron-cvpr22/blob/master/notebooks/RES%20H3.6M%20Evaluation%20Comparison%20HistRepItself.ipynb
    joint_to_ignore = [16, 20, 23, 24, 28, 31]
    joint_equal = [13, 19, 22, 13, 27, 30]
    diff[..., joint_to_ignore] = diff[..., joint_equal]

    reduce_dims = [i for i in range(diff.dim()) if i != 1]
    error = diff.mean(dim=reduce_dims)  # [T]

    logger.debug(f"MPJPE result shape: {error.shape}")
    return error, pos_pred, pos_gt if return_pos else error


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

        pos_pred, pos_gt = _to_pos(y_pred_batch, y_gt_batch, rep, ignore_root, meters=True, **kwargs)  # [n_samples, B, T, J, D], [B, T, J, D]
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



# def compute_diversity(pred, *args):
#     if pred.shape[0] == 1:
#         return 0.0
#     dist = pdist(pred.reshape(pred.shape[0], -1))
#     diversity = dist.mean().item()
#     return diversity


# def compute_ade(pred, gt, *args):
#     diff = pred - gt
#     dist = np.linalg.norm(diff, axis=2).mean(axis=1)
#     return dist.min()


# def compute_fde(pred, gt, *args):
#     diff = pred - gt
#     dist = np.linalg.norm(diff, axis=2)[:, -1]
#     return dist.min()


# def compute_mmade(pred, gt, gt_multi):
#     gt_dist = []
#     for gt_multi_i in gt_multi:
#         dist = compute_ade(pred, gt_multi_i)
#         gt_dist.append(dist)
#     gt_dist = np.array(gt_dist).mean()
#     return gt_dist


# def compute_mmfde(pred, gt, gt_multi):
#     gt_dist = []
#     for gt_multi_i in gt_multi:
#         dist = compute_fde(pred, gt_multi_i)
#         gt_dist.append(dist)
#     gt_dist = np.array(gt_dist).mean()
#     return gt_dist
