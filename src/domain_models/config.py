from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Secure configuration model for the MeetingNoter pipeline."""
    google_api_key: str = Field(..., description="API key for Google Drive access")
    pyannote_auth_token: str = Field(..., description="HuggingFace token for Pyannote Diarization")
    silero_vad_model_path: str | None = Field(default=None, description="Local path to Silero VAD .jit model")
    file_id: str = Field(..., description="The Google Drive file ID to process")
