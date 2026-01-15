import argparse
from pathlib import Path
import logging
from tqdm import tqdm

from h36m_tools.metadata import STATIC_JOINTS, SITE_JOINTS, NUM_JOINTS, TOTAL_JOINTS, RAW_FPS, DOWNSAMPLE_FACTOR
from h36m_tools.files import read_files
from h36m_tools.rotations import identity_rotation
from h36m_tools.dims import add_dims
from h36m_tools.visualize import animate_frames
from h36m_tools.logging import setup_logger
from h36m_tools.representations import parse_rep_dir


logger = logging.getLogger(__name__)


def animate_sequence(input_file: Path,
                     output_file: Path = None,
                     n_frames: int = None,
                     dpi: int = 150,
                     downsample: int = DOWNSAMPLE_FACTOR,
                     show_joint_names: bool = False):
    """
    Load a processed H3.6M rotation-representation tensor and generate an MP4 skeleton animation.

    Args:
        input_file (Path): Path to the processed tensor file.
        output_file (Path, optional): Output .mp4 path. If None, defaults to `<input_file>.mp4`.
        downsample (int): Temporal downsample factor.
        show_joint_names (bool): Whether to draw joint labels.

    Returns:
        None. Saves animation to disk.
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading file: {input_file}")
    rot = read_files(input_file)   # [T, D_rot]
    if n_frames is not None:
        logger.info(f"Animating first {n_frames} of {rot.shape[0]} total frames")
        rot = rot[:n_frames]

    rep_dir = input_file.parent.parent.name
    rep, convention, degrees = parse_rep_dir(rep_dir)
    logger.info(f"Parsed representation: rep={rep}, convention='{convention}', degrees={degrees}")

    fill = identity_rotation(rep, (rot.shape[0], len(STATIC_JOINTS)), convention=convention, degrees=degrees)
    rot = add_dims(rot, STATIC_JOINTS, NUM_JOINTS, fill, -2)
    fill = identity_rotation(rep, (rot.shape[0], len(SITE_JOINTS)), convention=convention, degrees=degrees)
    rot = add_dims(rot, SITE_JOINTS, TOTAL_JOINTS, fill, -2)

    fps = RAW_FPS // downsample
    default_name = f"{rep_dir}_{input_file.stem}"
    anim = animate_frames(pred=rot, rep=rep, fps=fps, title=default_name, show_joint_names=show_joint_names,
                          convention=convention, degrees=degrees)
    logger.info("Animation created")

    output_path = Path("outputs") / f"{default_name}.mp4" if output_file is None else output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(rot), desc="Saving animation", unit="fr") as pbar:
        anim.save(str(output_path), writer='ffmpeg', dpi=dpi, fps=fps, 
                  extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'], 
                  progress_callback=lambda i, n: pbar.update(1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Animate a processed H3.6M sequence")
    parser.add_argument("-i", "--input", dest="input_file", type=Path, required=True, help="Path to processed rotation tensor")
    parser.add_argument("-o", "--output", dest="output_file", type=Path, default=None, help="Output .mp4 file path (default: outputs/<rep>_<input>.mp4)")
    parser.add_argument("-d", "--downsample", type=int, default=DOWNSAMPLE_FACTOR, help="Downsample factor")
    parser.add_argument("-n", "--n_frames", type=int, default=None, help="Number of initial frames to animate (default: all frames)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output video (default: 150)")
    parser.add_argument("--label", action="store_true", help="Show joint names in the animation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logger(debug=args.debug)

    animate_sequence(args.input_file, args.output_file, args.n_frames, args.dpi, args.downsample, args.label)
