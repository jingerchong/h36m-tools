import argparse
from pathlib import Path
import logging
from tqdm import tqdm

from h36m_tools.files import read_files
from h36m_tools.dims import create_dim_mask, remove_dims
from h36m_tools.metadata import STATIC_JOINTS, NUM_JOINTS, DOWNSAMPLE_FACTOR
from h36m_tools.utils import setup_logger, compare_tensors


logger = logging.getLogger(__name__)

SITE_JOINTS = (0, 6, 11, 16, 22, 24, 30, 32)


def compare_expmap(ref_dir: Path, 
                   processed_dir: Path, 
                   downsample: int = DOWNSAMPLE_FACTOR
                   ) -> bool:
    """
    Compare processed expmap tensors against reference files in ref_dir.

    Args:
        ref_dir: Path to reference expmap files (.txt, .cdf, or .pt)
        processed_dir: Path to processed expmap .pt files
        downsample: Optional temporal downsample factor

    Returns:
        bool: True if all tensors match, False if any mismatches found.
    """
    processed_expmap_dir = processed_dir / "expmap"
    if not processed_expmap_dir.exists():
        logger.warning("Expmap folder not found. Please run preprocess script first.")
        return False

    processed_files = [f for f in processed_expmap_dir.rglob("*.pt") if f.parent.name in ("train", "test")]
    if not processed_files:
        logger.warning("No processed expmap files found.")
        return False

    logger.info(f"Found {len(processed_files)} processed expmap files to compare")

    site_mask  = create_dim_mask(SITE_JOINTS, NUM_JOINTS + len(SITE_JOINTS))
    static_mask = create_dim_mask(STATIC_JOINTS, NUM_JOINTS)

    all_processed = read_files(processed_files)
    has_mismatch = False

    for file, processed_tensor in tqdm(zip(processed_files, all_processed), desc="Comparing expmap", total=len(processed_files)):
        subject, action, _ = file.stem.split('_')
        match_found = False 

        for ref_idx in (1, 2):
            ref_file = ref_dir / subject / f"{action}_{ref_idx}.txt"

            if not ref_file.exists():
                logger.debug(f"Reference missing: {ref_file}")
                continue
            try:
                ref_tensor = read_files([ref_file])[0]
            except Exception as e:
                logger.error(f"Failed to read reference tensor {ref_file}: {e}")
                continue

            ref_tensor = ref_tensor.reshape(ref_tensor.shape[0], -1, 3)
            if downsample > 1:
                ref_tensor = ref_tensor[::downsample]
            ref_tensor = remove_dims(ref_tensor, site_mask, axis=-2)
            ref_tensor = remove_dims(ref_tensor, static_mask, axis=-2)

            if compare_tensors(processed_tensor, -ref_tensor, name=f"{file} vs {ref_file}"):
                match_found = True
                tqdm.write(f"Match found {file} <=> {ref_file}")
                break  

        if not match_found:
            logger.error(f"No matching reference found for: {file}")
            has_mismatch = True

    if not has_mismatch:
        logger.info("All tensors match!")
    return not has_mismatch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compare processed H3.6M data in expmap to reference files from zip")
    parser.add_argument("--ref_dir", type=Path, default=Path("data/expmap_zip"), help="Reference expmap folder extracted from zip")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"), help="Processed data directory")
    parser.add_argument("--downsample", type=int, default=DOWNSAMPLE_FACTOR, help="Downsample factor")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logger(debug=args.debug)

    compare_expmap(args.ref_dir, args.processed_dir, args.downsample)