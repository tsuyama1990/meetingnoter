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
    msg = f"Missing required configuration secret: {key_name}"
    raise ValueError(msg)


class PipelineConfig(BaseSettings):
    """Secure configuration model for the MeetingNoter pipeline."""

    # Required Secrets fetched dynamically to prevent hardcoding or exposure
    google_api_key: str = Field(
        default_factory=lambda: _get_secret("GOOGLE_API_KEY"),
        description="API key for Google Drive access",
        min_length=1,
    )
    pyannote_auth_token: str = Field(
        default_factory=lambda: _get_secret("PYANNOTE_AUTH_TOKEN"),
        description="HuggingFace token for Pyannote Diarization",
        min_length=1,
    )
    file_id: str = Field(
        default_factory=lambda: _get_secret("FILE_ID"),
        description="The Google Drive file ID to process",
        min_length=1,
    )

    # Optional / Default Configuration
    silero_vad_model_path: str = Field(
        default="silero_vad.jit", description="Local path to verified Silero VAD .jit model"
    )

    # Chunker Configuration
    ffmpeg_path: str = Field(
        default_factory=lambda: __import__("shutil").which("ffmpeg") or "ffmpeg",
        description="Path to the ffmpeg executable",
    )

    # Transcriber Configuration
    transcriber_model_size: str = Field(default="large-v3", description="Faster Whisper model size")
    transcriber_compute_type: str = Field(
        default="int8", description="Faster Whisper compute type for optimization"
    )
    transcriber_language: str = Field(default="ja", description="Target language for transcription")
    transcriber_vad_filter: bool = Field(
        default=True, description="Enable VAD filtering in whisper"
    )
    transcriber_condition_on_previous_text: bool = Field(
        default=False, description="Disable hallucination loops"
    )
    transcriber_temperature: tuple[float, float] = Field(
        default=(0.0, 0.2), description="Decoding temperature"
    )
