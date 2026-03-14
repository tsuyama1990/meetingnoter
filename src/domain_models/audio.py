from pydantic import BaseModel, ConfigDict, Field, model_validator


class AudioSource(BaseModel):
    """Represents a raw audio file to be processed."""

    model_config = ConfigDict(extra="forbid")

    filepath: str = Field(..., min_length=1, description="Path to the audio file.")
    duration_seconds: float = Field(..., gt=0, description="Duration of the audio in seconds.")


class AudioChunk(BaseModel):
    """Represents a discrete chunk of an audio file."""

    model_config = ConfigDict(extra="forbid")

    chunk_filepath: str = Field(..., min_length=1, description="Path to the chunk file.")
    start_time: float = Field(..., ge=0, description="Start time in the original audio.")
    end_time: float = Field(..., gt=0, description="End time in the original audio.")
    chunk_index: int = Field(..., ge=0, description="Index of this chunk.")

    @model_validator(mode="after")
    def check_time_ordering(self) -> "AudioChunk":
        if self.start_time >= self.end_time:
            msg = "start_time must be strictly less than end_time."
            raise ValueError(msg)
        return self


class SpeechSegment(BaseModel):
    """Represents a segment of audio that has been classified as containing speech."""

    model_config = ConfigDict(extra="forbid")

    start_time: float = Field(..., ge=0, description="Start time of speech.")
    end_time: float = Field(..., gt=0, description="End time of speech.")

    @model_validator(mode="after")
    def check_time_ordering(self) -> "SpeechSegment":
        if self.start_time >= self.end_time:
            msg = "start_time must be strictly less than end_time."
            raise ValueError(msg)
        return self
