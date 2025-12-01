import sys
import logging
from pathlib import Path


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

    if logger.handlers:
        return logger

    console_level = logging.DEBUG if debug else logging.INFO
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    if output_dir is not None:
        script_name = Path(sys.argv[0]).stem 
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"{script_name}.log"

        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)  # File always gets DEBUG
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
