import torch
import logging
import roma


logger = logging.getLogger(__name__)


def quat_to(quat: torch.Tensor, rep: str, **kwargs) -> torch.Tensor:
    """
    Convert quaternion to any supported rotation representation.
    
    Args:
        quat: [..., 4] quaternions in XYZW format (Roma convention)
        rep: Target representation - "quat" | "expmap" | "euler" | "rot6" | "rot9"
        **kwargs: Extra arguments for Euler representation:
            convention (str): Euler angle convention, case sensitive. Default: "ZXY"
            degrees (bool): If True, return angles in degrees instead of radians. Default: False
    
    Returns:
        Tensor of shape [..., D] where D depends on representation:
            - quat: [..., 4] in XYZW format
            - expmap: [..., 3] (axis-angle/rotation vector in radians)
            - euler: [..., 3] (Euler angles in radians or degrees)
            - rot6: [..., 6] (first two columns of rotation matrix, flattened)
            - rot9: [..., 9] (flattened 3x3 rotation matrix)
    """
    quat_norm = roma.quat_normalize(quat)
    
    if rep == "quat":
        out = quat_norm
    elif rep == "expmap":
        out = roma.unitquat_to_rotvec(quat_norm)  # [..., 3]
    elif rep == "euler":
        convention = kwargs.get("convention", "ZXY")
        degrees = kwargs.get("degrees", False)
        out = roma.unitquat_to_euler(quat_norm, convention=convention, degrees=degrees)  # [..., 3]
    elif rep == "rot9":
        rotm = roma.unitquat_to_rotmat(quat_norm)  # [..., 3, 3]
        out = rotm.reshape(*rotm.shape[:-2], 9)  # [..., 9]
    elif rep == "rot6":
        rotm = roma.unitquat_to_rotmat(quat_norm)  # [..., 3, 3]
        out = rotm[..., :, :2].reshape(*rotm.shape[:-2], 6)  # [..., 6]
    else:
        raise ValueError(f"Unknown target representation: '{rep}'")

    logger.debug(f"quat_to('{rep}'): input {quat.shape} → output {out.shape}")
    return out


def to_quat(rot: torch.Tensor, rep: str, **kwargs) -> torch.Tensor:
    """
    Convert any supported rotation representation to quaternion.
    
    Args:
        rot: Tensor of shape [..., D] where D depends on representation:
            - quat: [..., 4] in XYZW format
            - expmap: [..., 3] (axis-angle/rotation vector in radians)
            - euler: [..., 3] (Euler angles in radians or degrees)
            - rot6: [..., 6] (first two columns of rotation matrix, flattened)
            - rot9: [..., 9] (flattened 3x3 rotation matrix)
        rep: Source representation - "quat" | "expmap" | "euler" | "rot6" | "rot9"
        **kwargs: Extra arguments for Euler representation:
            convention (str): Euler angle convention, case sensitive. Default: "ZXY"
            degrees (bool): If True, interpret input angles as degrees instead of radians. Default: False

    Returns:
        Tensor [..., 4] with quaternions in XYZW format (Roma convention)
    """
    if rep == "quat":
        quat = rot
    elif rep == "expmap":
        quat = roma.rotvec_to_unitquat(rot)
    elif rep == "euler":
        convention = kwargs.get("convention", "ZXY")
        degrees = kwargs.get("degrees", False)
        quat = roma.euler_to_unitquat(convention, rot, degrees=degrees)
    elif rep == "rot9":
        rotm = rot.reshape(*rot.shape[:-1], 3, 3)  # [..., 3, 3]
        quat = roma.rotmat_to_unitquat(rotm)
    elif rep == "rot6":
        rot_reshaped = rot.reshape(*rot.shape[:-1], 3, 2)  # [..., 3, 2]        
        rotm = roma.special_gramschmidt(rot_reshaped)  # [..., 3, 3]
        quat = roma.rotmat_to_unitquat(rotm)
    else:
        raise ValueError(f"Unknown source representation: '{rep}'")

    logger.debug(f"to_quat('{rep}'): input {rot.shape} → output {quat.shape}")
    return quat
