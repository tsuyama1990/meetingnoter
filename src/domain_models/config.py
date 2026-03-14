from pydantic import Field
from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    """Secure configuration model for the MeetingNoter pipeline."""

    # Required Secrets
    google_api_key: str = Field(..., description="API key for Google Drive access")
    pyannote_auth_token: str = Field(..., description="HuggingFace token for Pyannote Diarization")
    file_id: str = Field(..., description="The Google Drive file ID to process")

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
