import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import torch

from h36m_tools.files import read_files
from h36m_tools.dims import remove_dims
from h36m_tools.metadata import STATIC_JOINTS, SITE_JOINTS, NUM_JOINTS, TOTAL_JOINTS, DOWNSAMPLE_FACTOR
from h36m_tools.logging import setup_logger


logger = logging.getLogger(__name__)


def _compare_tensors(processed: torch.Tensor, reference: torch.Tensor, name: str = "", atol: float = 1e-4) -> bool:
    """Compare two tensors and log detailed statistics if they differ."""
    reference = reference.to(processed.device)

    if processed.shape != reference.shape:
        logger.debug(f"Shape mismatch: {name}")
        logger.debug(f"Shape processed: {processed.shape}, reference: {reference.shape}")
        return False

    if torch.allclose(processed, reference, atol=atol):
        return True

    logger.debug(f"Tensors do not match: {name}")
    diff = processed - reference
    n_diff = torch.sum(diff != 0).item()
    max_diff = torch.max(diff.abs()).item()
    mean_diff = torch.mean(diff.abs()).item()
    logger.debug(f"Number of differing elements: {n_diff}")
    logger.debug(f"Max absolute difference: {max_diff:.4f}")
    logger.debug(f"Mean absolute difference: {mean_diff:.4f}")

    return False


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
                ref_tensor = read_files(ref_file)
            except Exception as e:
                logger.error(f"Failed to read reference tensor {ref_file}: {e}")
                continue

            ref_tensor = ref_tensor.reshape(ref_tensor.shape[0], -1, 3)[:, 1:]
            if downsample > 1:
                ref_tensor = ref_tensor[::downsample]
            ref_tensor = remove_dims(ref_tensor, SITE_JOINTS, TOTAL_JOINTS, -2)
            ref_tensor = remove_dims(ref_tensor, STATIC_JOINTS, NUM_JOINTS, -2)

            if _compare_tensors(processed_tensor, -ref_tensor, name=f"{file} vs {ref_file}"):
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
    parser.add_argument("-r", "--ref_dir", type=Path, default=Path("data/expmap_zip"), help="Reference expmap folder extracted from zip")
    parser.add_argument("-p", "--processed_dir", type=Path, default=Path("data/protocol1"), help="Processed data directory")
    parser.add_argument("-d", "--downsample", type=int, default=DOWNSAMPLE_FACTOR, help="Downsample factor")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logger(debug=args.debug)

    compare_expmap(args.ref_dir, args.processed_dir, args.downsample)