from typing import Protocol

from .audio import AudioChunk, AudioSource, SpeechSegment
from .transcription import DiarizedSegment, SpeakerLabel, TranscriptionSegment


class StorageClient(Protocol):
    def download(self, file_id: str) -> AudioSource:
        """Downloads an audio file and returns an AudioSource."""
        ...


class AudioSplitter(Protocol):
    def split(self, source: AudioSource) -> list[AudioChunk]:
        """Splits an audio source into smaller chunks."""
        ...


class SpeechDetector(Protocol):
    def detect_speech(self, chunk: AudioChunk) -> list[SpeechSegment]:
        """Detects speech segments within an audio chunk."""
        ...


class Transcriber(Protocol):
    def transcribe(
        self, chunk: AudioChunk, speech_segments: list[SpeechSegment]
    ) -> list[TranscriptionSegment]:
        """Transcribes the given speech segments of an audio chunk."""
        ...


class Diarizer(Protocol):
    def diarize(self, chunk: AudioChunk) -> list[SpeakerLabel]:
        """Identifies speaker labels with timestamps for an audio chunk."""
        ...

class Aggregator(Protocol):
    def merge(
        self,
        chunk: AudioChunk,
        transcriptions: list[TranscriptionSegment],
        speaker_labels: list[SpeakerLabel],
    ) -> list[DiarizedSegment]:
        """Merges transcriptions and speaker labels into diarized segments, applying chunk offset."""
        ...
