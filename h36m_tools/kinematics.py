import torch
import logging
from kornia.geometry.quaternion import Quaternion

from h36m_tools.rotations import to_quat


def _fk_quat(quat: torch.Tensor,
            parents,
            offsets,
            ignore_root=True) -> torch.Tensor:
    """
    Forward kinematics using quaternion rotations, supports arbitrary batch dims.

    Args:
        quat: [..., J, 4] quaternions (w, x, y, z)
        parents: list[int] length J
        offsets: [J, 3] tensor
        ignore_root: zero root rotation

    Returns:
        Tensor [..., J, 3] joint positions
    """
    orig_shape = quat.shape[:-2]
    J = quat.shape[-2]
    q = quat.reshape(-1, J, 4)                 # [B, J, 4], B = product of batch dims
    B = q.shape[0]

    pos = torch.zeros(B, J, 3, device=q.device, dtype=q.dtype)

    q = q / q.norm(dim=-1, keepdim=True)

    if ignore_root:
        q[:, 0, :] = 0.0
        q[:, 0, 0] = 1.0

    for i in range(1, J):
        parent = parents[i]
        if parent == -1:
            continue

        q_i = Quaternion(q[:, i])         # [B, 4]
        q_p = Quaternion(q[:, parent])    # [B, 4]
        q[:, i] = (q_p * q_i).data

        offset_quat = Quaternion(torch.cat([torch.zeros(B, 1, device=q.device), 
                                            offsets[i].unsqueeze(0).expand(B, -1)], dim=-1))  # [B, 4]
        pos[:, i] = pos[:, parent] + (q_p * offset_quat * q_p.conj()).data[..., 1:]  # [B, 3]

    return pos.view(*orig_shape, J, 3)


def fk(rot: torch.Tensor,
       rep: str = "quat",
       parents=None,
       offsets=None,
       ignore_root=True,
       **kwargs) -> torch.Tensor:
    """
    Unified forward kinematics entry point.
    Accepts any rotation representation, converts to quaternion, and runs FK.

    Args:
        rot: rotation tensor [..., J, ?] depending on representation
        rep: input representation ("quat", "expmap", "euler", "rot6", "rot9")
        parents: list[int] length J
        offsets: [J, 3] joint offsets
        ignore_root: if True, zero root rotation (keep only translation)
        **kwargs: passed to to_quaternion()

    Returns:
        Joint positions [..., J, 3]
    """
    quat = rot if rep == "quat" else to_quat(rot, rep=rep, **kwargs)
    logging.debug(f"fk() converting {rep} → quat, resulting shape {quat.shape}")
    return _fk_quat(quat, parents=parents, offsets=offsets, ignore_root=ignore_root)
