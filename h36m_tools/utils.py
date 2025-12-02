import sys
import logging
from pathlib import Path
import torch
from typing import Tuple, Optional, Union


logger = logging.getLogger(__name__)


def setup_logger(output_dir: Path | None = None, debug: bool = False) -> logging.Logger:
    """
    Configure application logging for console-only or console + file output.

    Args:
        output_dir: Directory where the log file will be stored.
            - If None: only console logging is enabled.
            - If a Path: a log file is created inside this directory with the
              name `<script_name>.log`. File always logs at DEBUG level.
        debug: If True, console shows DEBUG messages. Otherwise INFO.

    Returns:
        The configured root logger. Depending on arguments:
            - Console handler is always added (level depends on `debug`).
            - File handler is added only when `output_dir` is provided (always DEBUG).

    Notes:
        - Handlers are not duplicated; calling this function multiple times is safe.
        - Log file directory is created automatically if it does not exist.
        - File logs include timestamps; console logs do not.
        - File handler always captures DEBUG+ messages for debugging purposes.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    console_level = logging.DEBUG if debug else logging.INFO
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    if output_dir is not None:
        script_name = Path(sys.argv[0]).stem 
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"{script_name}.log"

        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG) 
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logger.addHandler(fh)

    return logger


def get_rep_dir(processed_dir: Path, rep: str, convention: str | None = None, degrees: bool = False) -> Path:
    """
    Return the directory path for a given rotation representation,
    creating it if it does not exist. Handles special naming for Euler angles.

    Args:
        processed_dir: Base directory for processed data.
        rep: Rotation representation name, e.g., "expmap", "quat", "euler".
        convention: Euler convention (only used if rep=="euler"), e.g., "ZXY".
        degrees: If True and rep=="euler", append "deg"; else "rad".

    Returns:
        Path object pointing to the directory.
        The directory is created if it does not exist.
    """
    if rep.lower() == "euler":
        conv = convention if convention is not None else "ZXY"
        rep_folder = f"{rep}_{conv}_{'deg' if degrees else 'rad'}"
    else:
        rep_folder = rep
    rep_dir = processed_dir / rep_folder
    rep_dir.mkdir(parents=True, exist_ok=True)
    return rep_dir


def parse_rep_dir(dir_path: Union[str, Path]) -> Tuple[str, Optional[str], Optional[bool]]:
    """
    Parse representation (rep), convention, and degrees flag
    from a directory name produced by get_rep_dir().
    Works if a full path is provided.

    Args:
        dir_path: str or Path pointing to a rep folder or full path.
        
    Returns:
        (rep, convention, degrees)
    """
    dir_name = Path(dir_path).name
    parts = dir_name.split("_")

    if parts[0] != "euler":
        return parts[0], None, None
    if len(parts) == 3:
        _, conv, unit = parts
        degrees = unit.lower() == "deg"
        return "euler", conv, degrees
    return "euler", "ZXY", False



def compare_tensors(processed: torch.Tensor, reference: torch.Tensor, name: str = "", atol: float = 1e-4) -> bool:
    """
    Compare two tensors and log detailed statistics if they differ.

    Args:
        processed: The processed tensor.
        reference: The reference tensor.
        name: Optional identifier to include in log messages.
        atol: Absolute tolerance for comparison (default 1e-4).

    Returns:
        True if tensors are equal within tolerance, False if they differ.
    """
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


def standardize_action(raw_action: str) -> str:
    """
    Normalize / standardize Human3.6M action names to consistent canonical forms.

    This function:
      - Strips non-alphabetic characters.
      - Converts to lowercase.
      - Normalizes known variants (e.g., 'walk' → 'walking').
      - Applies rule-based replacements.

    Args:
        raw_action (str): The raw action name extracted from a filename.

    Returns:
        str: A cleaned, standardized action string.
    """
    action = "".join(c for c in raw_action.lower() if c.isalpha())

    if "walk" in action and "walking" not in action:
        action = action.replace("walk", "walking")

    elif "photo" in action and "takingphoto" not in action:
        action = "takingphoto"

    return action