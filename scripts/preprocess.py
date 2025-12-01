import argparse
from pathlib import Path
import logging

from h36m_tools.rotations import to_quat, quat_to
from h36m_tools.files import save_tensor
from h36m_tools.metadata import PROTOCOL, STATIC_JOINTS, DOWNSAMPLE_FACTOR
from h36m_tools.load import load_raw
from h36m_tools.normalize import compute_stats
from h36m_tools.dims import create_dim_mask, remove_dims


def preprocess(raw_dir: Path = Path("data/raw"), 
               processed_dir: Path = Path("data/processed"), 
               rep: str = "expmap",
               downsample: int = DOWNSAMPLE_FACTOR, 
               **rep_kwargs):

    logging.info(f"Loading raw data from {raw_dir}...")
    data = load_raw(raw_dir, downsample=downsample)  
    
    if rep == "euler":
        rep_folder = f"{rep}_{rep_kwargs.get('convention','ZXY')}_{'deg' if rep_kwargs.get('degrees',False) else 'rad'}"
    else:
        rep_folder = rep
    rep_dir = processed_dir / rep_folder
    rep_dir.mkdir(parents=True, exist_ok=True)

    train_tensors = []

    for subject, actions in data.items():
        split = "train" if subject in PROTOCOL["train"] else "test"
        split_dir = rep_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for action, sequences in actions.items():
            for idx, tensor in enumerate(sequences):
                static_mask = create_dim_mask(STATIC_JOINTS, tensor.shape[-2])
                tensor = remove_dims(tensor, static_mask, axis=-2)    

                quat = to_quat(tensor, rep="euler", convention="ZXY", degrees=True)
                data_rep = quat_to(quat, rep, **rep_kwargs)

                filename = f"{subject}_{action}_{idx+1}.pt"
                save_tensor(split_dir / filename, data_rep)

                if subject in PROTOCOL["train"]:
                    train_tensors.append(data_rep)

    mean, std = compute_stats(train_tensors)
    save_tensor(rep_dir / "mean.pt", mean)
    save_tensor(rep_dir / "std.pt", std)
    logging.info(f"Preprocessing finished. Processed data saved in {rep_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="H3.6M preprocessing script")
    parser.add_argument("-i", "--input", type=str, default=Path("data/raw"), help="Raw H3.6M data directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("-r", "--rep", type=str, default="expmap", help="Target rotation representation")
    parser.add_argument("-d", "--downsample", type=int, default=DOWNSAMPLE_FACTOR, help="Downsample factor")
    parser.add_argument("--convention", type=str, default="ZYX", help="Target Euler angle convention")
    parser.add_argument("--degrees", action="store_true", help="If set, target Euler angles returned as degrees")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    preprocess(args.input, args.output, rep=args.rep, downsample=args.downsample,
               convention=args.convention, degrees=args.degrees)