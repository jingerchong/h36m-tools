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
        out = roma.unitquat_to_euler(convention, quat_norm, degrees=degrees)  # [..., 3]
    elif rep == "rot9":
        rotm = roma.unitquat_to_rotmat(quat_norm)  # [..., 3, 3]
        out = rotm.reshape(*rotm.shape[:-2], 9)  # [..., 9]
    elif rep == "rot6":
        rotm = roma.unitquat_to_rotmat(quat_norm)  # [..., 3, 3]
        out = rotm[..., :, :2].reshape(*rotm.shape[:-2], 6)  # [..., 6]
    else:
        raise ValueError(f"Unknown target representation: '{rep}'")

    logger.debug(f"quat_to('{rep}'): input {quat.shape} -> output {out.shape}")
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

    logger.debug(f"to_quat('{rep}'): input {rot.shape} -> output {quat.shape}")
    return quat


def mean_rotation(rot: torch.Tensor, rep: str, axis: int = -1, **kwargs) -> torch.Tensor:
    """
    Compute the mean rotation over a given axis using roma.special_procrustes
    (chordal L2 rotation averaging).

    Args:
        rot: rotation in any representation [..., D]
        rep: "quat" | "expmap" | "euler" | "rot6" | "rot9"
        axis: which axis to average over (default = -1)
        **kwargs: passed to Euler conversions

    Returns:
        Rotation in SAME representation as input, with dimension reduced on `axis`.
    """
    quat = to_quat(rot, rep, **kwargs)        # [..., 4]
    R = roma.unitquat_to_rotmat(quat)         # [..., 3, 3]

    R = R.movedim(axis, 0)                    # [N, ..., 3, 3]
    N = R.shape[0]
    M = R.sum(dim=0) / N                      # [..., 3, 3]
    R_mean = roma.special_procrustes(M)       # [..., 3, 3]

    quat_mean = roma.rotmat_to_unitquat(R_mean)
    out = quat_to(quat_mean, rep, **kwargs)

    logger.debug(f"mean_rotation('{rep}'): input {rot.shape} -> output {out.shape}")
    return out


def delta_rotation(target: torch.Tensor, anchor: torch.Tensor, rep: str, **kwargs) -> torch.Tensor:
    """
    Compute relative (delta) rotation: R_delta = R_anchor^{-1} ∘ R_target

    Args:
        target: [..., D] rotation at time t
        anchor: [..., D] reference rotation
        rep: rotation representation ("quat" | "expmap" | "euler" | "rot6" | "rot9")
        **kwargs: passed to conversion functions

    Returns:
        delta rotation in SAME representation as input [..., D]
    """
    q_t = to_quat(target, rep, **kwargs)   # [..., 4]
    q_a = to_quat(anchor, rep, **kwargs)   # [..., 4]

    # Relative rotation: q_delta = q_a^{-1} ⊗ q_t
    q_delta = roma.quat_product(roma.quat_conjugation(q_a), q_t)
    out = quat_to(q_delta, rep, **kwargs)

    logger.debug(f"delta_rotation('{rep}'): target {target.shape}, anchor {anchor.shape} -> output {out.shape}")
    return out


def add_rotation(delta: torch.Tensor, anchor: torch.Tensor, rep: str, **kwargs) -> torch.Tensor:
    """
    Apply a relative (delta) rotation to an anchor rotation.

    Computes:
        R_out = R_delta ∘ R_anchor

    This is the inverse of `delta_rotation`, where:
        R_delta = R_anchor^{-1} ∘ R_target

    Args:
        delta: Relative rotation in given representation.
            Shape: [..., D]
        anchor: Anchor (base) rotation in same representation.
            Shape: [..., D] (broadcastable to rot_delta)
        rep: Rotation representation of inputs and output.
            One of: {"quat", "expmap", "euler", "rot6", "rot9"}
        **kwargs: Passed to conversion utilities (e.g. Euler convention, degrees).

    Returns:
        Absolute rotation in the same representation.
        Shape: [..., D]
    """
    q_delta = to_quat(delta, rep, **kwargs)   # [..., 4]
    q_anchor = to_quat(anchor, rep, **kwargs) # [..., 4]

    # Compose rotations: q_out = q_delta ⊗ q_anchor
    q_out = roma.quat_product(q_delta, q_anchor)

    out = quat_to(q_out, rep, **kwargs)

    logger.debug(f"add_rotation('{rep}'): delta {delta.shape}, anchor {anchor.shape} -> out {out.shape}")
    return out
