import torch
import logging
import kornia.geometry.conversions as Kconv
from kornia.geometry.conversions import rotation_6d_to_matrix


def quat_to(quat: torch.Tensor, rep: str = "expmap", **kwargs) -> torch.Tensor:
    """
    Convert quaternion [..., 4] to another rotation representation.
    
    Args:
        quat: [..., 4] quaternions [w, x, y, z]
        rep: "expmap" | "euler" | "rot6" | "rot9"
        **kwargs: extra args (e.g., Euler order)
    
    Returns:
        Tensor of shape [..., ?] depending on target.
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Last dimension must be 4 (quaternion), got {quat.shape[-1]}")

    orig_shape = quat.shape[:-1]
    quat_flat = torch.nan_to_num(quat.reshape(-1, 4))
    quat_flat = quat_flat / quat_flat.norm(dim=-1, keepdim=True)

    if rep == "expmap":
        out_flat = Kconv.quaternion_to_axis_angle(quat_flat)
    elif rep == "euler":
        order = kwargs.get("order", "zyx")
        out_flat = Kconv.quaternion_to_euler_angles(quat_flat, convention=order)
    elif rep == "rot9":
        rotm = Kconv.quaternion_to_rotation_matrix(quat_flat)  # [N, 3, 3]
        out_flat = rotm.reshape(rotm.shape[0], 9)               # [N, 9]
    elif rep == "rot6":
        rotm = Kconv.quaternion_to_rotation_matrix(quat_flat)  # [N, 3, 3]
        out_flat = rotm[:, :, :2].reshape(rotm.shape[0], 6)    # [N, 6]
    else:
        raise ValueError(f"Unknown target representation: '{rep}'")

    out = out_flat.view(*orig_shape, out_flat.shape[-1])
    logging.debug(f"quat_to('{rep}') input {quat.shape} → output {out.shape}")
    return out


def to_quat(rot: torch.Tensor, rep: str = "expmap", **kwargs) -> torch.Tensor:
    """
    Convert other rotation representations to quaternion [..., 4].
    
    Args:
        rot: Tensor of shape [..., ?] depending on rep
        rep: "expmap" | "euler" | "rot9" | "rot6"
        **kwargs: extra args (e.g., Euler order)
    
    Returns:
        Tensor [..., 4] quaternions [w, x, y, z]
    """
    orig_shape = rot.shape[:-1]
    rot_flat = rot.reshape(-1, rot.shape[-1])
    
    if rep == "expmap":
        quat_flat = Kconv.axis_angle_to_quaternion(rot_flat)
    elif rep == "euler":
        order = kwargs.get("order", "zyx")
        quat_flat = Kconv.euler_angles_to_quaternion(rot_flat, convention=order)
    elif rep == "rot9":
        rotm = rot_flat.view(-1, 3, 3)
        quat_flat = Kconv.rotation_matrix_to_quaternion(rotm)
    elif rep == "rot6":
        # reconstruct full rotation matrix from 6D representation
        rotm = rotation_6d_to_matrix(rot_flat)
        quat_flat = Kconv.rotation_matrix_to_quaternion(rotm)
    else:
        raise ValueError(f"Unknown source representation: {rep}")
    
    quat_flat = quat_flat / quat_flat.norm(dim=-1, keepdim=True)
    quat = quat_flat.view(*orig_shape, 4)
    logging.debug(f"to_quat('{rep}') input {rot.shape} → output {quat.shape}")
    return quat