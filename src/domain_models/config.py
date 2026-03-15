import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

try:
    import google.colab.userdata

    _userdata: Any = google.colab.userdata
except ImportError:
    _userdata = None


ERROR_MSG_MISSING_SECRET = "Missing required configuration secret"  # noqa: S105
GOOGLE_API_KEY_DESCRIPTION = "API key for Google Drive access"
PYANNOTE_AUTH_TOKEN_DESCRIPTION = "HuggingFace token for Pyannote Diarization"  # noqa: S105
FILE_ID_DESCRIPTION = "The Google Drive file ID to process"

SILERO_VAD_MODEL_PATH_DEFAULT = "silero_vad.jit"
SILERO_VAD_MODEL_PATH_DESCRIPTION = "Local path to verified Silero VAD .jit model"

FFMPEG_PATH_DESCRIPTION = "Path to the ffmpeg executable"

TRANSCRIBER_MODEL_SIZE_DEFAULT = "large-v3"
TRANSCRIBER_MODEL_SIZE_DESCRIPTION = "Faster Whisper model size"

TRANSCRIBER_COMPUTE_TYPE_DEFAULT = "int8"
TRANSCRIBER_COMPUTE_TYPE_DESCRIPTION = "Faster Whisper compute type for optimization"

TRANSCRIBER_LANGUAGE_DEFAULT = "ja"
TRANSCRIBER_LANGUAGE_DESCRIPTION = "Target language for transcription"

TRANSCRIBER_VAD_FILTER_DEFAULT = True
TRANSCRIBER_VAD_FILTER_DESCRIPTION = "Enable VAD filtering in whisper"

TRANSCRIBER_CONDITION_ON_PREVIOUS_TEXT_DEFAULT = False
TRANSCRIBER_CONDITION_ON_PREVIOUS_TEXT_DESCRIPTION = "Disable hallucination loops"

TRANSCRIBER_TEMPERATURE_DEFAULT = (0.0, 0.2)
TRANSCRIBER_TEMPERATURE_DESCRIPTION = "Decoding temperature"


def _get_ffmpeg_path_default() -> str:
    return str(__import__("shutil").which("ffmpeg") or "ffmpeg")


def _get_secret(key_name: str) -> str:
    val = os.environ.get(key_name)
    if val is not None:
        return str(val)
    if _userdata is not None:
        try:
            val = _userdata.get(key_name)
            if val is not None:
                return str(val)
        except getattr(__import__("google").colab.userdata, "SecretNotFoundError", Exception):
            pass

    import logging

    logger = logging.getLogger(__name__)
    logger.error("Missing required configuration secret: %s", key_name)

    msg = f"{ERROR_MSG_MISSING_SECRET}: {key_name}"
    raise ValueError(msg)


class PipelineConfig(BaseSettings):
    """Secure configuration model for the MeetingNoter pipeline."""

    # Required Secrets fetched dynamically to prevent hardcoding or exposure
    google_api_key: str = Field(
        default_factory=lambda: _get_secret("GOOGLE_API_KEY"),
        description=GOOGLE_API_KEY_DESCRIPTION,
        min_length=1,
    )
    pyannote_auth_token: str = Field(
        default_factory=lambda: _get_secret("PYANNOTE_AUTH_TOKEN"),
        description=PYANNOTE_AUTH_TOKEN_DESCRIPTION,
        min_length=1,
    )
    file_id: str = Field(
        default_factory=lambda: _get_secret("FILE_ID"),
        description=FILE_ID_DESCRIPTION,
        min_length=1,
    )

    # Optional / Default Configuration
    silero_vad_model_path: str = Field(
        default=SILERO_VAD_MODEL_PATH_DEFAULT, description=SILERO_VAD_MODEL_PATH_DESCRIPTION
    )

    # Chunker Configuration
    ffmpeg_path: str = Field(
        default_factory=_get_ffmpeg_path_default,
        description=FFMPEG_PATH_DESCRIPTION,
    )

    # Transcriber Configuration
    transcriber_model_size: str = Field(
        default=TRANSCRIBER_MODEL_SIZE_DEFAULT, description=TRANSCRIBER_MODEL_SIZE_DESCRIPTION
    )
    transcriber_compute_type: str = Field(
        default=TRANSCRIBER_COMPUTE_TYPE_DEFAULT, description=TRANSCRIBER_COMPUTE_TYPE_DESCRIPTION
    )
    transcriber_language: str = Field(
        default=TRANSCRIBER_LANGUAGE_DEFAULT, description=TRANSCRIBER_LANGUAGE_DESCRIPTION
    )
    transcriber_vad_filter: bool = Field(
        default=TRANSCRIBER_VAD_FILTER_DEFAULT, description=TRANSCRIBER_VAD_FILTER_DESCRIPTION
    )
    transcriber_condition_on_previous_text: bool = Field(
        default=TRANSCRIBER_CONDITION_ON_PREVIOUS_TEXT_DEFAULT,
        description=TRANSCRIBER_CONDITION_ON_PREVIOUS_TEXT_DESCRIPTION,
    )
    transcriber_temperature: tuple[float, float] = Field(
        default=TRANSCRIBER_TEMPERATURE_DEFAULT, description=TRANSCRIBER_TEMPERATURE_DESCRIPTION
    )

    # Architectural Module Injection Paths
    drive_client_module_path: str = Field(
        default="meetingnoter.ingestion.drive_client",
        description="Path to Drive Client implementation.",
    )
    chunker_module_path: str = Field(
        default="meetingnoter.processing.chunker",
        description="Path to Audio Splitter implementation.",
    )
    vad_module_path: str = Field(
        default="meetingnoter.processing.vad", description="Path to VAD implementation."
    )
    transcriber_module_path: str = Field(
        default="meetingnoter.processing.transcriber",
        description="Path to Transcriber implementation.",
    )
    diarizer_module_path: str = Field(
        default="meetingnoter.processing.diarizer", description="Path to Diarizer implementation."
    )

    # Process Hyperparameters
    chunk_length_minutes: int = Field(
        default=20, description="Audio segment split length to prevent Pyannote OOM."
    )
    vad_threshold: float = Field(
        default=0.5, description="Probability threshold for speech detection."
    )
    vad_min_speech_duration_ms: int = Field(
        default=250, description="Minimum length of a speech segment to recognize."
    )
    vad_min_silence_duration_ms: int = Field(
        default=1000, description="Minimum length of silence to split phrases."
    )
