import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    """Secure configuration model for the MeetingNoter pipeline."""

    # Required Secrets explicitly passed to prevent default_factory exposure
    google_api_key: Any = Field(
        ...,
        description="API key for Google Drive access",
    )
    pyannote_auth_token: Any = Field(
        ...,
        description="HuggingFace token for Pyannote Diarization",
    )

    @field_validator("google_api_key", "pyannote_auth_token", mode="before")
    @classmethod
    def cast_to_credential_manager(cls, v: Any) -> Any:
        from domain_models.credentials import CredentialManager

        if isinstance(v, CredentialManager):
            return v
        return CredentialManager(str(v))

    file_id: str = Field(
        ...,
        description="The Google Drive file ID to process",
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


def resolve_secrets() -> dict[str, str]:
    import os

    try:
        from google.colab import userdata
    except ImportError:
        userdata = None

    secrets = {}
    for key in ["GOOGLE_API_KEY", "PYANNOTE_AUTH_TOKEN", "FILE_ID"]:
        val = os.environ.get(key)
        if not val and userdata is not None:
            import contextlib

            with contextlib.suppress(Exception):
                val = userdata.get(key)

        if val:
            secrets[key.lower()] = val

    return secrets


class VADConfig(BaseModel):
    """Secure configuration model for Voice Activity Detection."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    fallback_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence for fallback behavior."
    )
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(
        default=1000, ge=500, le=1000, description="Strict Japanese nuances."
    )
    model_path: str = Field(..., min_length=1)
    model_hash: str | None = Field(default=None, description="SHA256 checksum for model integrity.")
    frame_duration: float = Field(default=0.032, gt=0.0)

    # Architectural and Security Limits
    max_model_size_bytes: int = Field(default=100 * 1024 * 1024)  # 100MB
    max_audio_size_bytes: int = Field(default=1024 * 1024 * 1024)  # 1GB
    max_audio_duration_seconds: int = Field(default=3600)  # 1 Hour
    target_sample_rate: int = Field(default=16000)

    @field_validator("model_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        path = Path(v).resolve()

        # Must exist and be a file
        if not path.is_file():
            msg = f"Model path {path} must be an existing file."
            raise ValueError(msg)

        # Must have valid extension
        if path.suffix.lower() not in [".jit", ".pt"]:
            msg = f"Model path {path} has invalid extension."
            raise ValueError(msg)

        allowed_dirs = [Path.cwd(), Path(__import__("tempfile").gettempdir())]
        for d in allowed_dirs:
            if path.is_relative_to(d):
                return str(path)
        msg = f"Path {path} is not within allowed directories: {allowed_dirs}"
        raise ValueError(msg)

    @field_validator("model_hash")
    @classmethod
    def validate_hash(cls, v: str | None) -> str | None:
        if v is not None and not re.match(r"^[a-fA-F0-9]{64}$", v):
            msg = "Model hash must be a valid SHA256 hex digest."
            raise ValueError(msg)
        return v
