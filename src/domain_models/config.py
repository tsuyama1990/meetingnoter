import os
from typing import Any, cast

from pydantic import Field
from pydantic_settings import BaseSettings

try:
    from google.colab import userdata
except ImportError:
    userdata: Any = None  # type: ignore[no-redef]


class PipelineConfig(BaseSettings):
    """Secure configuration model for the MeetingNoter pipeline."""

    # Required Secrets fetched dynamically to prevent hardcoding or exposure
    google_api_key: str = Field(
        default_factory=lambda: cast(
            str,
            os.environ.get("GOOGLE_API_KEY")
            or (userdata.get("GOOGLE_API_KEY") if userdata else None),
        ),
        description="API key for Google Drive access",
        min_length=1,
    )
    pyannote_auth_token: str = Field(
        default_factory=lambda: cast(
            str,
            os.environ.get("PYANNOTE_AUTH_TOKEN")
            or (userdata.get("PYANNOTE_AUTH_TOKEN") if userdata else None),
        ),
        description="HuggingFace token for Pyannote Diarization",
        min_length=1,
    )
    file_id: str = Field(
        default_factory=lambda: cast(
            str,
            os.environ.get("FILE_ID")
            or (userdata.get("FILE_ID") if userdata else None),
        ),
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
