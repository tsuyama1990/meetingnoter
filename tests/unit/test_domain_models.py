import tempfile
from unittest.mock import MagicMock, patch

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
from meetingnoter.ingestion.drive_client import GoogleDriveClient


def test_audio_source_valid() -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        source = AudioSource(filepath=tf.name, duration_seconds=60.0)
        assert source.filepath == tf.name
        assert source.duration_seconds == 60.0


def test_audio_source_invalid_duration() -> None:
    with (
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf,
        pytest.raises(ValidationError),
    ):
        AudioSource(filepath=tf.name, duration_seconds=0)


def test_audio_chunk_valid() -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        chunk = AudioChunk(chunk_filepath=tf.name, start_time=0.0, end_time=30.0, chunk_index=0)
        assert chunk.start_time == 0.0
        assert chunk.end_time == 30.0


def test_audio_chunk_invalid_ordering() -> None:
    with (
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf,
        pytest.raises(ValidationError, match="start_time must be strictly less than end_time"),
    ):
        AudioChunk(chunk_filepath=tf.name, start_time=30.0, end_time=10.0, chunk_index=0)


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


def test_mock_storage_client_implements_protocol() -> None:
    client: StorageClient = MagicMock(spec=StorageClient)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        mock_source = AudioSource(filepath=tf.name, duration_seconds=60.0)

    # We must explicitly set the type of the mock return to please mypy
    client.download = MagicMock(return_value=mock_source)  # type: ignore[method-assign]

    source = client.download("test_id")
    assert isinstance(source, AudioSource)
    assert source.duration_seconds == 60.0


def test_mock_audio_splitter_implements_protocol() -> None:
    splitter: AudioSplitter = MagicMock(spec=AudioSplitter)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_chunk:
        mock_chunk = AudioChunk(
            chunk_filepath=tf_chunk.name, start_time=0.0, end_time=60.0, chunk_index=0
        )

    splitter.split = MagicMock(return_value=[mock_chunk])  # type: ignore[method-assign]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_source:
        chunks = splitter.split(AudioSource(filepath=tf_source.name, duration_seconds=60.0))

    assert len(chunks) == 1
    assert isinstance(chunks[0], AudioChunk)
    assert chunks[0].start_time == 0.0


@patch("urllib.request.urlretrieve")
@patch("wave.open")
def test_google_drive_client_success(
    mock_wave_open: MagicMock, mock_urlretrieve: MagicMock
) -> None:
    mock_wave = MagicMock()
    mock_wave.getnframes.return_value = 44100
    mock_wave.getframerate.return_value = 44100
    mock_wave_open.return_value.__enter__.return_value = mock_wave

    client = GoogleDriveClient(api_key="test_token")
    source = client.download("test_id")
    assert isinstance(source, AudioSource)
    assert source.duration_seconds == 1.0


@patch("urllib.request.urlretrieve")
def test_google_drive_client_failure(mock_urlretrieve: MagicMock) -> None:
    import urllib.error

    mock_urlretrieve.side_effect = urllib.error.URLError("Network error")

    client = GoogleDriveClient(api_key="test_token")
    with pytest.raises(RuntimeError, match="Network error"):
        client.download("test_id")


@patch("urllib.request.urlretrieve")
def test_google_drive_client_unexpected_error(mock_urlretrieve: MagicMock) -> None:
    mock_urlretrieve.side_effect = Exception("Unexpected error")

    client = GoogleDriveClient(api_key="test_token")
    with pytest.raises(RuntimeError, match="Unexpected error during download"):
        client.download("test_id")
