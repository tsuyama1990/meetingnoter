import os

from pydantic import Field
from pydantic_settings import BaseSettings


def get_secret(key: str) -> str:
    """
    Dynamically retrieves a secret via environment variables, or
    falls back securely to Google Colab's userdata API if available.
    """
    # 1. Check environment variables first
    value = os.environ.get(key)
    if value:
        return value

    # 2. Securely check Colab userdata dynamically at runtime
    try:
        from google.colab import userdata

        colab_value = userdata.get(key)
        if colab_value:
            return str(colab_value)
    except ImportError:
        pass  # Not running in Colab

    # 3. Fail gracefully if required secret is missing
    msg = f"Missing required configuration secret: {key}"
    raise RuntimeError(msg)


class PipelineConfig(BaseSettings):
    """Secure configuration model for the MeetingNoter pipeline."""

    # Required Secrets fetched dynamically to prevent hardcoding or exposure
    google_api_key: str = Field(
        default_factory=lambda: get_secret("GOOGLE_API_KEY"),
        description="API key for Google Drive access",
    )
    pyannote_auth_token: str = Field(
        default_factory=lambda: get_secret("PYANNOTE_AUTH_TOKEN"),
        description="HuggingFace token for Pyannote Diarization",
    )
    file_id: str = Field(
        default_factory=lambda: get_secret("FILE_ID"),
        description="The Google Drive file ID to process",
    )

    # Optional / Default Configuration
    silero_vad_model_path: str = Field(
        default="silero_vad.jit", description="Local path to verified Silero VAD .jit model"
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
