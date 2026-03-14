import pytest
from pydantic import ValidationError

from domain_models import (
    AudioChunk,
    AudioSource,
    AudioSplitter,
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerLabel,
    SpeechSegment,
    StorageClient,
    TranscriptionSegment,
)


def test_audio_source_valid() -> None:
    source = AudioSource(filepath="/mock_dir/audio.wav", duration_seconds=60.0)
    assert source.filepath == "/mock_dir/audio.wav"
    assert source.duration_seconds == 60.0


def test_audio_source_invalid_duration() -> None:
    with pytest.raises(ValidationError):
        AudioSource(filepath="/mock_dir/audio.wav", duration_seconds=0)


def test_audio_chunk_valid() -> None:
    chunk = AudioChunk(
        chunk_filepath="/mock_dir/chunk1.wav", start_time=0.0, end_time=30.0, chunk_index=0
    )
    assert chunk.start_time == 0.0
    assert chunk.end_time == 30.0


def test_audio_chunk_invalid_ordering() -> None:
    with pytest.raises(ValidationError, match="start_time must be strictly less than end_time"):
        AudioChunk(
            chunk_filepath="/mock_dir/chunk1.wav", start_time=30.0, end_time=10.0, chunk_index=0
        )


def test_speech_segment_valid() -> None:
    seg = SpeechSegment(start_time=1.5, end_time=5.0)
    assert seg.start_time == 1.5


def test_speech_segment_invalid_ordering() -> None:
    with pytest.raises(ValidationError):
        SpeechSegment(start_time=5.0, end_time=1.0)


def test_transcription_segment_valid() -> None:
    seg = TranscriptionSegment(start_time=1.5, end_time=5.0, text="Hello world")
    assert seg.text == "Hello world"


def test_transcription_segment_invalid_ordering() -> None:
    with pytest.raises(ValidationError):
        TranscriptionSegment(start_time=5.0, end_time=1.0, text="Invalid")


def test_speaker_label_valid() -> None:
    label = SpeakerLabel(start_time=1.0, end_time=10.0, speaker_id="SPEAKER_00")
    assert label.speaker_id == "SPEAKER_00"


def test_speaker_label_invalid_ordering() -> None:
    with pytest.raises(ValidationError):
        SpeakerLabel(start_time=10.0, end_time=1.0, speaker_id="SPEAKER_00")


def test_diarized_segment_valid() -> None:
    seg = DiarizedSegment(start_time=1.5, end_time=5.0, speaker_id="SPEAKER_00", text="Hi")
    assert seg.speaker_id == "SPEAKER_00"


def test_diarized_segment_invalid_ordering() -> None:
    with pytest.raises(ValidationError):
        DiarizedSegment(start_time=5.0, end_time=1.0, speaker_id="SPEAKER_00", text="Hi")


def test_diarized_transcript_valid() -> None:
    seg = DiarizedSegment(start_time=1.5, end_time=5.0, speaker_id="SPEAKER_00", text="Hi")
    transcript = DiarizedTranscript(segments=[seg])
    assert len(transcript.segments) == 1


# Mock Interfaces for validation checks
class MockStorageClient:
    def download(self, file_id: str) -> AudioSource:
        return AudioSource(filepath=f"/mock_dir/{file_id}.wav", duration_seconds=60.0)


class MockAudioSplitter:
    def split(self, source: AudioSource) -> list[AudioChunk]:
        return [
            AudioChunk(
                chunk_filepath="/mock_dir/chunk.wav",
                start_time=0.0,
                end_time=source.duration_seconds,
                chunk_index=0,
            )
        ]


def test_mock_storage_client_implements_protocol() -> None:
    client: StorageClient = MockStorageClient()
    source = client.download("test_id")
    assert isinstance(source, AudioSource)


def test_mock_audio_splitter_implements_protocol() -> None:
    splitter: AudioSplitter = MockAudioSplitter()
    chunks = splitter.split(AudioSource(filepath="/mock_dir/audio.wav", duration_seconds=60.0))
    assert len(chunks) == 1
    assert isinstance(chunks[0], AudioChunk)
