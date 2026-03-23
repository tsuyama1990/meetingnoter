from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptionSegment(BaseModel):
    """Represents a segment of text transcribed from audio."""

    model_config = ConfigDict(extra="forbid")

    start_time: float = Field(..., ge=0, description="Start time of the transcribed text.")
    end_time: float = Field(..., gt=0, description="End time of the transcribed text.")
    text: str = Field(..., description="The transcribed text.")
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0."
    )
    uncertain: bool | None = Field(
        default=None, description="True if confidence is below threshold, else None."
    )

    @model_validator(mode="after")
    def check_time_ordering(self) -> "TranscriptionSegment":
        if self.start_time >= self.end_time:
            msg = "start_time must be strictly less than end_time."
            raise ValueError(msg)
        return self


class SpeakerLabel(BaseModel):
    """Represents a speaker identified in an audio segment."""

    model_config = ConfigDict(extra="forbid")

    start_time: float = Field(..., ge=0, description="Start time of the speaker's speech.")
    end_time: float = Field(..., gt=0, description="End time of the speaker's speech.")
    speaker_id: str = Field(..., min_length=1, description="Unique identifier for the speaker.")

    @model_validator(mode="after")
    def check_time_ordering(self) -> "SpeakerLabel":
        if self.start_time >= self.end_time:
            msg = "start_time must be strictly less than end_time."
            raise ValueError(msg)
        return self


class DiarizedSegment(BaseModel):
    """Represents a transcribed segment assigned to a specific speaker."""

    model_config = ConfigDict(extra="forbid")

    start_time: float = Field(..., ge=0, description="Start time of the segment.")
    end_time: float = Field(..., gt=0, description="End time of the segment.")
    speaker_id: str = Field(..., min_length=1, description="Identifier of the speaker.")
    text: str = Field(..., description="The spoken text.")
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0."
    )
    uncertain: bool | None = Field(
        default=None, description="True if confidence is below threshold, else None."
    )

    @model_validator(mode="after")
    def check_time_ordering(self) -> "DiarizedSegment":
        if self.start_time >= self.end_time:
            msg = "start_time must be strictly less than end_time."
            raise ValueError(msg)
        return self


class DiarizedTranscript(BaseModel):
    """Represents the complete output of the diarization and transcription process."""

    model_config = ConfigDict(extra="forbid")

    segments: list[DiarizedSegment] = Field(
        default_factory=list, description="Ordered list of diarized segments."
    )
