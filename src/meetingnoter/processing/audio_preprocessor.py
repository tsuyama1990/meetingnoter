import logging
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def preprocess_audio(
    input_filepath: str, ffmpeg_path: str, mode: Literal["none", "loudnorm", "compressor"] = "none"
) -> str:
    """
    Applies optional audio preprocessing using FFmpeg.
    Returns the path to the original or processed audio file.
    """
    if mode == "none":
        logger.info("No audio preprocessing applied.")
        return input_filepath

    input_path = Path(input_filepath)
    output_path = input_path.with_name(f"temp_preprocessed_{input_path.name}")

    cmd = [ffmpeg_path, "-y", "-i", str(input_path.resolve())]

    if mode == "loudnorm":
        logger.info("Applying 'loudnorm' audio preprocessing.")
        cmd.extend(["-af", "loudnorm"])
    elif mode == "compressor":
        logger.info("Applying 'compressor' audio preprocessing.")
        cmd.extend(["-af", "acompressor"])
    else:
        logger.warning("Unknown preprocessing mode '%s'. Proceeding without preprocessing.", mode)
        return input_filepath

    cmd.extend([str(output_path.resolve())])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
        logger.info("Audio preprocessing completed successfully. Saved to: %s", output_path)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        logger.exception("FFmpeg preprocessing failed. Output: %s", e.stderr)
        msg = f"Audio preprocessing failed: {e.stderr}"
        raise RuntimeError(msg) from e
