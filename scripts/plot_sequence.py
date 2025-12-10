import argparse
from pathlib import Path
import logging

from h36m_tools.metadata import STATIC_JOINTS, NUM_JOINTS
from h36m_tools.files import read_files
from h36m_tools.dims import add_dims
from h36m_tools.visualize import plot_frames
from h36m_tools.logging import setup_logger
from h36m_tools.representations import parse_rep_dir


logger = logging.getLogger(__name__)


def plot_sequence(input_file: Path,
                  output_file: Path = None,
                  start_frame: int = 0,
                  n_frames: int = 10,
                  show_joint_names: bool = False,
                  dpi: int = 150):
    """
    Plot skeleton frames from a processed H3.6M tensor.

    Args:
        input_file (Path): Path to processed tensor file.
        output_file (Path, optional): Path to save plot image.
        start_frame (int): Index of first frame to plot.
        n_frames (int): Number of frames to plot (default: 1).
        show_joint_names (bool): Whether to label joints.
        dpi (int): DPI for saved image.

    Returns:
        None. Saves a figure to disk.
    """
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading file: {input_file}")
    rot = read_files(input_file)  # [T, D_rot]

    if start_frame >= rot.shape[0]:
        logger.error(f"start_frame {start_frame} exceeds sequence length {rot.shape[0]}")
        return
    end_frame = min(start_frame + n_frames, rot.shape[0])
    rot = rot[start_frame:end_frame]
    logger.info(f"Plotting frames {start_frame} to {end_frame - 1} (total {len(rot)} frames)")
    data = add_dims(rot, STATIC_JOINTS, NUM_JOINTS, 0.0, -2)

    rep_dir = input_file.parent.parent.name
    rep, convention, degrees = parse_rep_dir(rep_dir)
    logger.info(f"Parsed representation: rep={rep}, convention='{convention}', degrees={degrees}")

    default_name = f"{rep_dir}_{input_file.stem}_f{start_frame}"
    fig = plot_frames(data, rep=rep, title=default_name, show_joint_names=show_joint_names,
                      convention=convention if convention else None,
                      degrees=degrees if degrees is not None else None)

    
    output_file = Path("outputs") / f"{default_name}.png" if output_file is None else output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_file, dpi=dpi)
    logger.info(f"Saved plot to {output_file}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Plot H3.6M skeleton frames")
    parser.add_argument("-i", "--input", dest="input_file", type=Path, required=True, help="Path to processed rotation tensor")
    parser.add_argument("-o", "--output", dest="output_file", type=Path, default=None, help="Output image file path (default: outputs/<rep>_<input>_<start_frame>.png)")
    parser.add_argument("-s", "--start", dest="start_frame", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("-n", "--n_frames", type=int, default=10, help="Number of frames to plot (default: 10)")
    parser.add_argument("--label", action="store_true", help="Show joint names")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved image")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logger(debug=args.debug)

    plot_sequence(args.input_file, args.output_file, args.start_frame, args.n_frames, args.label, args.dpi)
