from pathlib import Path
from typing import Union, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


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