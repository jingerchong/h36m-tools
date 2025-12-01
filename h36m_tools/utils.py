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