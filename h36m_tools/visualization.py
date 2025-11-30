import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from h36m_tools.metadata import JOINT_NAMES, RIGHT_LEFT_JOINTS_IDX, PARENTS, DOWNSAMPLED_FPS

# Type alias for cleaner type hints
PoseData = Union[np.ndarray, torch.Tensor]


def _get_right_joints(right_left_joints_idx: List[Tuple[int, int]] = RIGHT_LEFT_JOINTS_IDX) -> List[int]:
    """Return right-side joint indices from right-left pairs."""
    return [r for r, _ in right_left_joints_idx]


def _prepare_poses(poses: Optional[PoseData]) -> Optional[np.ndarray]:
    """Convert tensor to numpy and swap Y/Z axes for plotting."""
    if poses is None:
        return None
    if torch.is_tensor(poses):
        poses = poses.detach().cpu().numpy()
    poses = poses.copy()
    # Swap Y and Z axes for standard 3D plotting convention
    poses[..., [1, 2]] = poses[..., [2, 1]]
    return poses


def _compute_bounds(sequences: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """ Compute global bounding box center and radius for all sequences."""
    all_pos = np.concatenate(sequences, axis=0)
    mins, maxs = all_pos.reshape(-1, 3).min(0), all_pos.reshape(-1, 3).max(0)
    center = (mins + maxs) / 2
    radius = float(np.max(maxs - mins) / 2)
    return center, radius


def _setup_axes(fig=None, elev=20, azim=60, center=None, radius=None, title=""):
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


def _draw_skeleton_lines(ax, frame, parents, right_joints, alpha=1.0, line_objs=None):
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
        color = "red" if j in right_joints else "black"
        
        if line_objs[j] is None:
            line_objs[j] = ax.plot(xs, ys, zs, lw=2, c=color, alpha=alpha)[0]
        else:
            line_objs[j].set_data(xs, ys)
            line_objs[j].set_3d_properties(zs, zdir="z")
            line_objs[j].set_color(color)
            line_objs[j].set_alpha(alpha)
            
    return line_objs


def _update_joint_labels(ax, frame, joint_names, text_objs=None, radius=None):
    """Draw or update text labels for joints."""
    J = frame.shape[0]
    offset = (radius * 0.05) if radius is not None else 30
    
    if text_objs is None:
        text_objs = [ax.text(frame[j, 0], frame[j, 1], frame[j, 2] + offset,
                     joint_names[j], fontsize=8, ha="center", va="center")for j in range(J)]
        return text_objs
        
    for j in range(J):
        x, y, z = frame[j]
        text_objs[j].set_position((x, y))
        text_objs[j].set_3d_properties(z + offset, zdir="z")

    return text_objs


def plot_frames(sequences: Union[PoseData, List[PoseData]],
                title: str = "",
                parents: List[int] = PARENTS,
                right_left_joints_idx: List[Tuple[int, int]] = RIGHT_LEFT_JOINTS_IDX,
                plot_joint_names: bool = False,
                show: bool = True,
                save_path: Optional[Path] = None,
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
    
    sequences = [_prepare_poses(seq) for seq in sequences]
    right_joints = _get_right_joints(right_left_joints_idx)
    center, radius = _compute_bounds(sequences)
    fig, ax = _setup_axes(center=center, radius=radius, title=title)

    for seq_idx, seq in enumerate(sequences):
        if len(sequences) == 1:
            alpha = 1.0
        else:
            alpha = 0.3 + 0.7 * (seq_idx / (len(sequences) - 1))

        for frame in seq:
            _draw_skeleton_lines(ax, frame, parents, right_joints, alpha)
            
        if plot_joint_names:
            _update_joint_labels(ax, seq[-1], JOINT_NAMES, radius=radius)

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logging.debug(f"plot_frames: saved figure to {save_path}")
    elif show:
        plt.show()
        logging.debug(f"plot_frames: displayed figure for {len(sequences)} sequence(s)")
    else:
        plt.close(fig)

    return fig, ax


def animate_frames(pred: PoseData,
                   gt: Optional[PoseData] = None,
                   parents: List[int] = PARENTS,
                   right_left_joints_idx: List[Tuple[int, int]] = RIGHT_LEFT_JOINTS_IDX,
                   fps: int = DOWNSAMPLED_FPS,
                   title: str = "",
                   plot_joint_names: bool = False,
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
    pred = _prepare_poses(pred)
    gt = _prepare_poses(gt)
    right_joints = _get_right_joints(right_left_joints_idx)
    
    sequences = [pred] + ([gt] if gt is not None else [])
    center, radius = _compute_bounds(sequences)
    fig, ax = _setup_axes(center=center, radius=radius, title=title)

    J = pred.shape[1]
    pred_lines = [None] * J
    gt_lines = [None] * J if gt is not None else None
    text_pred: Optional[List[plt.Text]] = None
    text_gt: Optional[List[plt.Text]] = None

    def update(t: int):
        nonlocal text_pred, text_gt
        
        _draw_skeleton_lines(ax, pred[t], parents, right_joints, line_objs=pred_lines)
        if plot_joint_names:
            text_pred = _update_joint_labels(ax, pred[t], JOINT_NAMES, text_pred, radius=radius)
        
        if gt is not None:
            _draw_skeleton_lines(ax, gt[t], parents, right_joints, alpha=0.5, line_objs=gt_lines)
            if plot_joint_names:
                text_gt = _update_joint_labels(ax, gt[t], JOINT_NAMES, text_gt, radius=radius)
        
        objs = pred_lines + (gt_lines if gt_lines else [])
        if plot_joint_names:
            objs += text_pred or []
            if text_gt:
                objs += text_gt
        return objs

    anim = FuncAnimation(fig, update, frames=pred.shape[0], interval=1000 / fps, blit=False)
    plt.close(fig)  
    logging.debug(f"animate_frames: created animation for {pred.shape[0]} frames at {fps} fps")
    return anim