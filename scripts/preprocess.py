import argparse
from pathlib import Path
import logging
import shutil

from h36m_tools.rotations import to_quat, quat_to
from h36m_tools.files import save_tensor
from h36m_tools.metadata import PROTOCOL, STATIC_JOINTS, NUM_JOINTS, DOWNSAMPLE_FACTOR
from h36m_tools.load import load_raw
from h36m_tools.normalize import compute_stats
from h36m_tools.dims import remove_dims
from h36m_tools.utils import setup_logger, get_rep_dir


logger = logging.getLogger(__name__)


def preprocess(raw_dir: Path, 
               rep_dir: Path, 
               rep: str = "expmap",
               downsample: int = DOWNSAMPLE_FACTOR, 
               **rep_kwargs):
    """
    Preprocess H3.6M raw CDF data, converting raw joint rotations to the 
    specified representation and computing normalization statistics.

    This function performs the following steps:
        1. Loads raw data from `raw_dir` (optionally downsampling).
        2. Removes static joints from the data.
        3. Converts joint rotations from Euler angles to the target representation (`rep`).
        4. Saves each processed sequence as a `.pt` file in the proper `rep_dir` subfolder
           (train/test split according to protocol).
        5. Computes and saves mean and standard deviation tensors for training data.

    Args:
        raw_dir (Path): Directory containing raw H3.6M data.
        rep_dir (Path): Directory where processed data will be saved. Subfolders for 
            each representation and train/test split will be created automatically.
        rep (str, optional): Target rotation representation. Supported values:
            - "expmap" : rotation vectors / axis-angle
            - "quat"   : quaternions
            - "euler"  : Euler angles
            - "rot6"   : 6D rotation representation
            - "rot9"   : 9D rotation matrix flattened
            Defaults to "expmap".
        downsample (int, optional): Factor by which to downsample the temporal frames. 
            Defaults to `DOWNSAMPLE_FACTOR`.
        **rep_kwargs: Extra keyword arguments passed to `quat_to` for Euler angles:
            - convention (str): Euler angle convention (e.g., "ZYX"). Default: "ZXY".
            - degrees (bool): If True, return Euler angles in degrees. Default: False.

    Returns:
        None. Saves processed tensors to disk, including:
            - Individual sequences in `.pt` files per subject/action/split.
            - Normalization statistics (`mean.pt` and `std.pt`) for training data.
    """
    logging.info(f"Loading raw data from {raw_dir}...")
    data = load_raw(raw_dir, downsample=downsample) 
    rep_dir.mkdir(parents=True, exist_ok=True)

    train_tensors = []

    for subject, actions in data.items():
        split = "train" if subject in PROTOCOL["train"] else "test"
        split_dir = rep_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for action, sequences in actions.items():
            for idx, tensor in enumerate(sequences):
                tensor = remove_dims(tensor, STATIC_JOINTS, NUM_JOINTS, -2)    

                quat = to_quat(tensor, rep="euler", convention="ZXY", degrees=True)
                data_rep = quat_to(quat, rep, **rep_kwargs)

                filename = f"{subject}_{action}_{idx+1}.pt"
                save_tensor(split_dir / filename, data_rep)

                if subject in PROTOCOL["train"]:
                    train_tensors.append(data_rep)

    logging.info("Computing normalization statistics...")
    mean, std = compute_stats(train_tensors)
    save_tensor(rep_dir / "mean.pt", mean)
    save_tensor(rep_dir / "std.pt", std)
    
    logging.info(f"Preprocessing finished. Processed data saved in {rep_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="H3.6M preprocessing script")
    parser.add_argument("-i", "--input", type=Path, default=Path("data/raw"), help="Raw H3.6M data directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("-r", "--rep", type=str, default="expmap", help="Target rotation representation")
    parser.add_argument("-d", "--downsample", type=int, default=DOWNSAMPLE_FACTOR, help="Downsample factor")
    parser.add_argument("--convention", type=str, default="ZYX", help="Target Euler angle convention")
    parser.add_argument("--degrees", action="store_true", help="If set, target Euler angles returned as degrees")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    rep_dir = get_rep_dir(args.output, rep=args.rep, convention=args.convention, degrees=args.degrees)
    if rep_dir.exists():
        logging.info(f"Removing existing directory: {rep_dir}")
        shutil.rmtree(rep_dir) 
    
    setup_logger(rep_dir, debug=args.debug)

    preprocess(args.input, rep_dir, rep=args.rep, downsample=args.downsample,
               convention=args.convention, degrees=args.degrees)