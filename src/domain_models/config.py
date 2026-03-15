import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

try:
    import google.colab.userdata

    _userdata: Any = google.colab.userdata
except ImportError:
    _userdata = None


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
    logger.error("Missing config: %s", key_name)
    msg = f"Missing config: {key_name}"
    raise ValueError(msg)


def _get_ffmpeg_path_default() -> str:
    return os.environ.get("FFMPEG_PATH", str(__import__("shutil").which("ffmpeg") or "ffmpeg"))


class PipelineConfig(BaseSettings):
    """Secure configuration model for the MeetingNoter pipeline."""

    google_api_key: str = Field(default_factory=lambda: _get_secret("GOOGLE_API_KEY"), min_length=1)
    pyannote_auth_token: str = Field(
        default_factory=lambda: _get_secret("PYANNOTE_AUTH_TOKEN"), min_length=1
    )
    file_id: str = Field(default_factory=lambda: _get_secret("FILE_ID"), min_length=1)

    silero_vad_model_path: str = Field(
        default_factory=lambda: os.environ.get("SILERO_VAD_MODEL_PATH", "silero_vad.jit")
    )

    ffmpeg_path: str = Field(default_factory=_get_ffmpeg_path_default)

    transcriber_model_size: str = Field(
        default_factory=lambda: os.environ.get("TRANSCRIBER_MODEL_SIZE", "large-v3")
    )
    transcriber_compute_type: str = Field(
        default_factory=lambda: os.environ.get("TRANSCRIBER_COMPUTE_TYPE", "int8")
    )
    transcriber_language: str = Field(
        default_factory=lambda: os.environ.get("TRANSCRIBER_LANGUAGE", "ja")
    )
    transcriber_vad_filter: bool = Field(
        default_factory=lambda: os.environ.get("TRANSCRIBER_VAD_FILTER", "true").lower() == "true"
    )
    transcriber_condition_on_previous_text: bool = Field(
        default_factory=lambda: (
            os.environ.get("TRANSCRIBER_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
        )
    )

    transcriber_temperature: tuple[float, float] = Field(default_factory=lambda: (0.0, 0.2))

    aggregator_module_path: str = Field(
        default_factory=lambda: os.environ.get(
            "AGGREGATOR_MODULE_PATH", "meetingnoter.processing.aggregator"
        )
    )
    drive_client_module_path: str = Field(
        default_factory=lambda: os.environ.get(
            "DRIVE_CLIENT_MODULE_PATH", "meetingnoter.ingestion.drive_client"
        )
    )
    chunker_module_path: str = Field(
        default_factory=lambda: os.environ.get(
            "CHUNKER_MODULE_PATH", "meetingnoter.processing.chunker"
        )
    )
    vad_module_path: str = Field(
        default_factory=lambda: os.environ.get("VAD_MODULE_PATH", "meetingnoter.processing.vad")
    )
    transcriber_module_path: str = Field(
        default_factory=lambda: os.environ.get(
            "TRANSCRIBER_MODULE_PATH", "meetingnoter.processing.transcriber"
        )
    )
    diarizer_module_path: str = Field(
        default_factory=lambda: os.environ.get(
            "DIARIZER_MODULE_PATH", "meetingnoter.processing.diarizer"
        )
    )

    chunk_length_minutes: int = Field(
        default_factory=lambda: int(os.environ.get("CHUNK_LENGTH_MINUTES", "20"))
    )
    vad_threshold: float = Field(
        default_factory=lambda: float(os.environ.get("VAD_THRESHOLD", "0.5"))
    )
    vad_min_speech_duration_ms: int = Field(
        default_factory=lambda: int(os.environ.get("VAD_MIN_SPEECH_DURATION_MS", "250"))
    )
    vad_min_silence_duration_ms: int = Field(
        default_factory=lambda: int(os.environ.get("VAD_MIN_SILENCE_DURATION_MS", "1000"))
    )
