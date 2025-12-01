import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.text import Text

from h36m_tools.metadata import JOINT_NAMES, RIGHT_LEFT_JOINTS_IDX, PARENTS, DOWNSAMPLED_FPS
from h36m_tools.kinematics import fk


RotData = Union[np.ndarray, torch.Tensor]
logger = logging.getLogger(__name__)


def _get_right_joints(right_left_joints_idx: List[Tuple[int, int]] = RIGHT_LEFT_JOINTS_IDX) -> List[int]:
    """Return right-side joint indices from right-left pairs."""
    return [r for r, _ in right_left_joints_idx]


def _to_numpy_pos(data: RotData, rep: str, **kwargs: Any) -> np.ndarray:
    """Convert rot/pos → numpy positions [T, J, 3]."""
    data_t = torch.as_tensor(data, dtype=torch.float32)
    pos = fk(data_t, rep=rep, **kwargs)
    arr = pos.detach().cpu().numpy()
    arr = arr.copy()
    arr[..., [1, 2]] = arr[..., [2, 1]]
    return arr


def _compute_bounds(sequences: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """ Compute global bounding box center and radius for all sequences."""
    all_pos = np.concatenate(sequences, axis=0)
    mins, maxs = all_pos.reshape(-1, 3).min(0), all_pos.reshape(-1, 3).max(0)
    center = (mins + maxs) / 2
    radius = float(np.max(maxs - mins) / 2)
    return center, radius


def _setup_axes(fig: Optional[plt.Figure] = None, 
                elev: float = 20, 
                azim: float = 60, 
                center: Optional[np.ndarray] = None, 
                radius: Optional[float] = None, 
                title: str = "") -> Tuple[plt.Figure, plt.Axes]:
    """Create or configure 3D axes with fixed bounds."""
    if fig is None:
        fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    if center is not None and radius is not None:
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=18)
    return fig, ax


def _draw_skeleton_lines(ax: plt.Axes, 
                        frame: np.ndarray, 
                        parents: List[int], 
                        right_joints: List[int], 
                        alpha: float = 1.0, 
                        line_objs: Optional[List[Optional[Line2D]]] = None) -> List[Optional[plt.Line2D]]:
    """Draw or update skeleton lines connecting joints."""
    J = frame.shape[0]
    if line_objs is None:
        line_objs = [None] * J
        
    for j in range(J):
        p = parents[j]
        if p == -1: 
            continue
        xs = [frame[j, 0], frame[p, 0]]
        ys = [frame[j, 1], frame[p, 1]]
        zs = [frame[j, 2], frame[p, 2]]
        color = "red" if j in right_joints else "gray"
        if line_objs[j] is None:
            line_objs[j] = ax.plot(xs, ys, zs, lw=2, c=color, alpha=alpha)[0]
        else:
            line_objs[j].set_data(xs, ys)
            line_objs[j].set_3d_properties(zs, zdir="z")
            line_objs[j].set_color(color)
            line_objs[j].set_alpha(alpha)

    return line_objs


def _update_joint_labels(ax: plt.Axes, 
                        frame: np.ndarray, 
                        joint_names: List[str], 
                        text_objs: Optional[List[Text]] = None, 
                        radius: Optional[float] = None) -> List[Text]:
    """Draw or update text labels for joints."""
    J = frame.shape[0]
    offset = (radius * 0.05) if radius is not None else 30
    if text_objs is None:
        text_objs = [ax.text(frame[j, 0], frame[j, 1], frame[j, 2] + offset,
                     joint_names[j], fontsize=8, ha="center", va="center") for j in range(J)]
        return text_objs
    for j in range(J):
        x, y, z = frame[j]
        text_objs[j].set_position((x, y))
        text_objs[j].set_3d_properties(z + offset, zdir="z")
    return text_objs


def plot_frames(sequences: Union[RotData, List[RotData]],
                rep: str = "quat",
                title: str = "",
                parents: List[int] = PARENTS,
                right_left_joints_idx=RIGHT_LEFT_JOINTS_IDX,
                plot_joint_names: bool = False,
                show: bool = True,
                save_path: Optional[Path] = None,
                **rep_kwargs
                ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more skeleton sequences with fixed global scaling.
    
    Multiple sequences are overlaid with varying transparency to show progression
    or comparison. The plot bounds are computed globally to ensure all sequences
    are visible and comparable.

    Args:
        sequences: Single array [T, J, 3] or list of arrays, each [T, J, 3]
        title: Figure title
        parents: Skeleton parent indices (default from metadata)
        right_left_joints_idx: List of (right, left) joint tuples for coloring
        plot_joint_names: Whether to render joint name labels
        show: Whether to display the figure (set False if only saving)
        save_path: Optional path to save figure as image file

    Returns:
        Tuple of (figure, axes) objects
    """
    if isinstance(sequences, (np.ndarray, torch.Tensor)):
        sequences = [sequences]
    
    seq_pos = [_to_numpy_pos(s, rep=rep, **rep_kwargs) for s in sequences]
    right_joints = _get_right_joints(right_left_joints_idx)
    center, radius = _compute_bounds(seq_pos)
    fig, ax = _setup_axes(center=center, radius=radius, title=title)

    for i, seq in enumerate(seq_pos):
        alpha = 1.0 if len(seq_pos) == 1 else (0.3 + 0.7 * (i / (len(seq_pos) - 1)))
        for frame in seq:
            _draw_skeleton_lines(ax, frame, parents, right_joints, alpha)
        if plot_joint_names:
            _update_joint_labels(ax, seq[-1], JOINT_NAMES, radius=radius)

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.debug(f"plot_frames: saved figure to {save_path}")
    elif show:
        plt.show()
        logger.debug(f"plot_frames: displayed figure for {len(sequences)} sequence(s)")
    else:
        plt.close(fig)

    return fig, ax


def animate_frames(pred: RotData,
                   rep: str = "quat",
                   gt: Optional[RotData] = None,
                   parents: List[int] = PARENTS,
                   right_left_joints_idx=RIGHT_LEFT_JOINTS_IDX,
                   fps: int = DOWNSAMPLED_FPS,
                   title: str = "",
                   plot_joint_names: bool = False,
                   **rep_kwargs
                   ) -> FuncAnimation:
    """
    Animate predicted and optional ground truth skeleton sequences.
    
    Creates a matplotlib animation showing skeleton motion over time. If ground truth
    is provided, it's overlaid with 50% transparency for comparison.
    
    IMPORTANT: Keep a reference to the returned animation object to prevent garbage
    collection, which would stop the animation.

    Args:
        pred: Predicted poses [T, J, 3]
        gt: Ground truth poses [T, J, 3], optional for comparison
        parents: Skeleton parent indices (default from metadata)
        right_left_joints_idx: (right, left) joint index pairs for coloring
        fps: Animation frames per second
        title: Figure title
        plot_joint_names: Whether to render joint name labels

    Returns:
        FuncAnimation object (must keep reference to prevent garbage collection)
    """
    pred_pos = _to_numpy_pos(pred, rep=rep, **rep_kwargs)
    gt_pos = _to_numpy_pos(gt, rep=rep, **rep_kwargs) if gt is not None else None

    sequences = [pred_pos] + ([gt_pos] if gt_pos is not None else [])
    right_joints = _get_right_joints(right_left_joints_idx)

    center, radius = _compute_bounds(sequences)
    fig, ax = _setup_axes(center=center, radius=radius, title=title)

    J = pred_pos.shape[1] 
    pred_lines = [None] * J
    gt_lines = [None] * J if gt_pos is not None else None
    
    text_pred = None
    text_gt = None

    def update(t: int):
        nonlocal text_pred, text_gt
        
        _draw_skeleton_lines(ax, pred_pos[t], parents, right_joints, line_objs=pred_lines)  
        if plot_joint_names:
            text_pred = _update_joint_labels(ax, pred_pos[t], JOINT_NAMES, text_pred, radius=radius)
        
        if gt_pos is not None:  # BUG FIX: Check gt_pos instead of gt
            _draw_skeleton_lines(ax, gt_pos[t], parents, right_joints, alpha=0.5, line_objs=gt_lines)  
            if plot_joint_names:
                text_gt = _update_joint_labels(ax, gt_pos[t], JOINT_NAMES, text_gt, radius=radius)

        objs = pred_lines + (gt_lines if gt_lines else [])
        if plot_joint_names:
            objs += text_pred or []
            if text_gt:
                objs += text_gt
        return objs

    anim = FuncAnimation(fig, update, frames=pred_pos.shape[0], interval=1000 / fps, blit=False) 
    plt.close(fig)  
    logger.debug(f"animate_frames: created animation for {pred_pos.shape[0]} frames at {fps} fps")  
    return anim