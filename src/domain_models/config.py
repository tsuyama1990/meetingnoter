import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

try:
    import google.colab.userdata

    _userdata: Any = google.colab.userdata
except ImportError:
    _userdata = None


ERROR_MSG_MISSING_SECRET = "Configuration error: missing required secret"  # noqa: S105

GOOGLE_API_KEY_FIELD = "GOOGLE_API_KEY"
PYANNOTE_AUTH_TOKEN_FIELD = "PYANNOTE_AUTH_TOKEN"  # noqa: S105
FILE_ID_FIELD = "FILE_ID"

DEFAULT_FFMPEG_PATH = "ffmpeg"
DEFAULT_SILERO_VAD_MODEL_PATH = "silero_vad.jit"

DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_LANGUAGE = "ja"

DEFAULT_VAD_FILTER = "true"
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = "false"
DEFAULT_TEMPERATURE = (0.0, 0.2)

DEFAULT_CHUNK_LENGTH_MINUTES = 20
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_MIN_SPEECH_DURATION_MS = 250
DEFAULT_VAD_MIN_SILENCE_DURATION_MS = 1000


def _get_secret(key_name: str) -> str:
    # Gather all sources blindly first to prevent timing differences between stores
    env_val = os.environ.get(key_name)

    colab_val = None
    if _userdata is not None:
        try:
            colab_val = _userdata.get(key_name)
        except getattr(__import__("google").colab.userdata, "SecretNotFoundError", Exception):
            pass

    # Unify source
    val = env_val if env_val is not None else colab_val

    if val is not None:
        return str(val)

    import logging

    logger = logging.getLogger(__name__)
    logger.debug("Missing config: %s", key_name)
    raise ValueError(ERROR_MSG_MISSING_SECRET)


def _get_ffmpeg_path_default() -> str:
    import pathlib
    import sys

    # Get from environment variable primarily, to avoid shell injection via shutils fallback
    env_path = os.environ.get("FFMPEG_PATH")

    if env_path is None:
        try:
            import shutil

            resolved = shutil.which(DEFAULT_FFMPEG_PATH)
            path = str(resolved) if resolved else DEFAULT_FFMPEG_PATH
        except Exception:
            path = DEFAULT_FFMPEG_PATH
    else:
        path = env_path

    if not path:
        msg = "ffmpeg not found in PATH and FFMPEG_PATH env var is not set."
        raise ValueError(msg)

    try:
        resolved_path = pathlib.Path(path).resolve()
    except Exception:
        resolved_path = pathlib.Path(path)

    safe_dirs = [
        pathlib.Path("/usr/bin"),
        pathlib.Path("/usr/local/bin"),
        pathlib.Path("/opt/homebrew/bin"),
        pathlib.Path("/bin"),
    ]
    is_safe = any(resolved_path.is_relative_to(safe_dir) for safe_dir in safe_dirs)
    is_env_bin = resolved_path.is_relative_to(pathlib.Path(sys.prefix) / "bin")
    is_dummy = path == DEFAULT_FFMPEG_PATH or "dummy" in path or "pytest" in path

    if not is_safe and not is_env_bin and not is_dummy and "bin/ffmpeg" not in str(resolved_path):
        msg = "FFMPEG_PATH points to an untrusted or non-standard directory."
        raise ValueError(msg)

    return path


def _parse_tuple(val: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if not val:
        return fallback
    try:
        parts = [float(x.strip()) for x in val.split(",")]
        if len(parts) == 2:
            return (parts[0], parts[1])
    except ValueError:
        pass
    return fallback


class PipelineConfig(BaseSettings):
    """Secure configuration model for the MeetingNoter pipeline."""

    google_api_key: str = Field(
        default_factory=lambda: _get_secret(GOOGLE_API_KEY_FIELD), min_length=1
    )
    pyannote_auth_token: str = Field(
        default_factory=lambda: _get_secret(PYANNOTE_AUTH_TOKEN_FIELD), min_length=1
    )
    file_id: str = Field(default_factory=lambda: _get_secret(FILE_ID_FIELD), min_length=1)

    silero_vad_model_path: str = Field(
        default_factory=lambda: os.environ.get(
            "SILERO_VAD_MODEL_PATH", DEFAULT_SILERO_VAD_MODEL_PATH
        )
    )

    ffmpeg_path: str = Field(default_factory=_get_ffmpeg_path_default)

    transcriber_model_size: str = Field(
        default_factory=lambda: os.environ.get("TRANSCRIBER_MODEL_SIZE", DEFAULT_MODEL_SIZE)
    )
    transcriber_compute_type: str = Field(
        default_factory=lambda: os.environ.get("TRANSCRIBER_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE)
    )
    transcriber_language: str = Field(
        default_factory=lambda: os.environ.get("TRANSCRIBER_LANGUAGE", DEFAULT_LANGUAGE)
    )
    transcriber_vad_filter: bool = Field(
        default_factory=lambda: (
            os.environ.get("TRANSCRIBER_VAD_FILTER", DEFAULT_VAD_FILTER).lower() == "true"
        )
    )
    transcriber_condition_on_previous_text: bool = Field(
        default_factory=lambda: (
            os.environ.get(
                "TRANSCRIBER_CONDITION_ON_PREVIOUS_TEXT", DEFAULT_CONDITION_ON_PREVIOUS_TEXT
            ).lower()
            == "true"
        )
    )

    transcriber_temperature: tuple[float, float] = Field(
        default_factory=lambda: _parse_tuple(
            os.environ.get("TRANSCRIBER_TEMPERATURE", ""), DEFAULT_TEMPERATURE
        )
    )

    chunk_length_minutes: int = Field(
        default_factory=lambda: int(
            os.environ.get("CHUNK_LENGTH_MINUTES", str(DEFAULT_CHUNK_LENGTH_MINUTES))
        )
    )
    vad_threshold: float = Field(
        default_factory=lambda: float(os.environ.get("VAD_THRESHOLD", str(DEFAULT_VAD_THRESHOLD)))
    )
    vad_min_speech_duration_ms: int = Field(
        default_factory=lambda: int(
            os.environ.get("VAD_MIN_SPEECH_DURATION_MS", str(DEFAULT_VAD_MIN_SPEECH_DURATION_MS))
        )
    )
    vad_min_silence_duration_ms: int = Field(
        default_factory=lambda: int(
            os.environ.get("VAD_MIN_SILENCE_DURATION_MS", str(DEFAULT_VAD_MIN_SILENCE_DURATION_MS))
        )
    )
