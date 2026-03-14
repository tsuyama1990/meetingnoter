import tempfile

from domain_models import (
    AudioChunk,
    AudioSource,
    AudioSplitter,
    Diarizer,
    SpeakerLabel,
    SpeechDetector,
    SpeechSegment,
    StorageClient,
    Transcriber,
    TranscriptionSegment,
)


class DummyStorageClient:
    def download(self, file_id: str) -> AudioSource:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            return AudioSource(filepath=tf.name, duration_seconds=60.0)


class DummyAudioSplitter:
    def split(self, source: AudioSource) -> list[AudioChunk]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf0, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf1:
            return [
                AudioChunk(chunk_filepath=tf0.name, start_time=0.0, end_time=30.0, chunk_index=0),
                AudioChunk(chunk_filepath=tf1.name, start_time=30.0, end_time=60.0, chunk_index=1)
            ]


class DummySpeechDetector:
    def detect_speech(self, chunk: AudioChunk) -> list[SpeechSegment]:
        return [SpeechSegment(start_time=chunk.start_time, end_time=chunk.start_time + 10.0)]


class DummyTranscriber:
    def transcribe(self, chunk: AudioChunk, speech_segments: list[SpeechSegment]) -> list[TranscriptionSegment]:
        return [TranscriptionSegment(start_time=seg.start_time, end_time=seg.end_time, text=f"Text {seg.start_time}") for seg in speech_segments]


class DummyDiarizer:
    def diarize(self, chunk: AudioChunk) -> list[SpeakerLabel]:
        return [SpeakerLabel(start_time=chunk.start_time, end_time=chunk.start_time + 10.0, speaker_id="SPEAKER_00")]



class FailingDummyStorageClient:
    """A clean test double that fails on download."""
    def download(self, file_id: str) -> AudioSource:
        msg = "Network Error"
        raise RuntimeError(msg)


def test_pipeline_integration_failure() -> None:
    # Test handling of download failures using proper test double instead of monkey-patching
    storage: StorageClient = FailingDummyStorageClient()

    import pytest
    with pytest.raises(RuntimeError, match="Network Error"):
        storage.download("test_id")

    # Ensure run_pipeline raises the error to the caller
    from main import run_pipeline
    with pytest.raises(RuntimeError, match="Network Error"):
        run_pipeline(
            storage=storage,
            splitter=DummyAudioSplitter(),
            detector=DummySpeechDetector(),
            transcriber=DummyTranscriber(),
            diarizer=DummyDiarizer(),
            file_id="test_id"
        )


def test_pipeline_integration() -> None:
    # Use the main.py run_pipeline orchestration logic to actually test the integration SUT
    from main import run_pipeline

    storage: StorageClient = DummyStorageClient()
    splitter: AudioSplitter = DummyAudioSplitter()
    detector: SpeechDetector = DummySpeechDetector()
    transcriber: Transcriber = DummyTranscriber()
    diarizer: Diarizer = DummyDiarizer()

    transcript = run_pipeline(
        storage=storage,
        splitter=splitter,
        detector=detector,
        transcriber=transcriber,
        diarizer=diarizer,
        file_id="test_id"
    )

    assert len(transcript.segments) == 2
    assert transcript.segments[0].speaker_id == "SPEAKER_00"
    assert transcript.segments[0].text == "Text 0.0"
