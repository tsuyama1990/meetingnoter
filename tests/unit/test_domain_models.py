import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from domain_models import (
    AudioChunk,
    AudioSource,
    AudioSplitter,
    DiarizedSegment,
    DiarizedTranscript,
    PipelineConfig,
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


class DummyStorageClientForMock(StorageClient):
    def download(self, file_id: str) -> AudioSource:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            return AudioSource(filepath=tf.name, duration_seconds=60.0)


class DummyAudioSplitterForMock(AudioSplitter):
    def split(self, source: AudioSource) -> list[AudioChunk]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_chunk:
            return [
                AudioChunk(
                    chunk_filepath=tf_chunk.name, start_time=0.0, end_time=60.0, chunk_index=0
                )
            ]


def test_dummy_storage_client_implements_protocol() -> None:
    client = DummyStorageClientForMock()
    source = client.download("test_id")
    assert isinstance(source, AudioSource)
    assert source.duration_seconds == 60.0


def test_dummy_audio_splitter_implements_protocol() -> None:
    splitter = DummyAudioSplitterForMock()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_source:
        chunks = splitter.split(AudioSource(filepath=tf_source.name, duration_seconds=60.0))

    assert len(chunks) == 1
    assert isinstance(chunks[0], AudioChunk)
    assert chunks[0].start_time == 0.0


@patch("tempfile.mkstemp")
@patch("os.fdopen")
@patch("wave.open")
def test_google_drive_client_success(
    mock_wave_open: MagicMock,
    mock_fdopen: MagicMock,
    mock_mkstemp: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Set up safe test configuration instead of hardcoding API keys
    monkeypatch.setenv("GOOGLE_API_KEY", "env_dummy_key_123")
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "env_dummy_token_123")
    monkeypatch.setenv("FILE_ID", "env_dummy_file_123")
    config = PipelineConfig()

    mock_http = MagicMock(spec=requests.Session)
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_http.get.return_value = mock_response

    mock_mkstemp.return_value = (1, "/tmp/dummy.wav")

    mock_wave = MagicMock()
    mock_wave.getnframes.return_value = 44100
    mock_wave.getframerate.return_value = 44100
    mock_wave_open.return_value.__enter__.return_value = mock_wave

    client = GoogleDriveClient(config=config, http_client=mock_http)
    source = client.download("test_id")
    assert isinstance(source, AudioSource)
    assert source.duration_seconds == 1.0


def test_google_drive_client_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "env_dummy_key_123")
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "env_dummy_token_123")
    monkeypatch.setenv("FILE_ID", "env_dummy_file_123")
    config = PipelineConfig()

    mock_http = MagicMock(spec=requests.Session)
    mock_http.get.side_effect = requests.exceptions.RequestException("Network error")

    client = GoogleDriveClient(config=config, http_client=mock_http)
    with pytest.raises(RuntimeError, match="Network or Request Error during download"):
        client.download("test_id")


def test_google_drive_client_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "env_dummy_key_123")
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "env_dummy_token_123")
    monkeypatch.setenv("FILE_ID", "env_dummy_file_123")
    config = PipelineConfig()

    mock_http = MagicMock(spec=requests.Session)
    mock_http.get.side_effect = requests.exceptions.HTTPError("403 Forbidden")

    client = GoogleDriveClient(config=config, http_client=mock_http)
    with pytest.raises(RuntimeError, match="HTTP Error during download"):
        client.download("test_id")


def test_google_drive_client_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "env_dummy_key_123")
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "env_dummy_token_123")
    monkeypatch.setenv("FILE_ID", "env_dummy_file_123")
    config = PipelineConfig()

    mock_http = MagicMock(spec=requests.Session)
    mock_http.get.side_effect = Exception("Unexpected error")

    client = GoogleDriveClient(config=config, http_client=mock_http)
    with pytest.raises(RuntimeError, match="Unexpected error while parsing audio"):
        client.download("test_id")

def test_transcript_merger_temporal_offset() -> None:
    from domain_models import AudioChunk, SpeakerLabel, TranscriptionSegment
    from meetingnoter import TranscriptMerger

    merger = TranscriptMerger()
    chunk = AudioChunk(chunk_filepath="dummy.wav", start_time=1200.0, end_time=2400.0, chunk_index=1)

    transcriptions = [
        TranscriptionSegment(start_time=10.0, end_time=20.0, text="Hello offset")
    ]
    labels = [
        SpeakerLabel(start_time=10.0, end_time=20.0, speaker_id="SPEAKER_01")
    ]

    result = merger.merge(chunk, transcriptions, labels)
    assert len(result) == 1
    assert result[0].start_time == 1210.0
    assert result[0].end_time == 1220.0
    assert result[0].text == "Hello offset"
    assert result[0].speaker_id == "SPEAKER_01"


def test_transcript_merger_overlap() -> None:
    from domain_models import AudioChunk, SpeakerLabel, TranscriptionSegment
    from meetingnoter import TranscriptMerger

    merger = TranscriptMerger()
    chunk = AudioChunk(chunk_filepath="dummy.wav", start_time=0.0, end_time=60.0, chunk_index=0)

    transcriptions = [
        TranscriptionSegment(start_time=5.0, end_time=15.0, text="I overlap more with speaker 2")
    ]
    labels = [
        SpeakerLabel(start_time=0.0, end_time=6.0, speaker_id="SPEAKER_01"), # 1s overlap
        SpeakerLabel(start_time=6.0, end_time=20.0, speaker_id="SPEAKER_02"), # 9s overlap
    ]

    result = merger.merge(chunk, transcriptions, labels)
    assert len(result) == 1
    assert result[0].speaker_id == "SPEAKER_02"
    assert result[0].start_time == 5.0
    assert result[0].end_time == 15.0
