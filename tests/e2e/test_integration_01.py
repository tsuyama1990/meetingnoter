import tempfile

from domain_models import (
    AudioChunk,
    AudioSource,
    AudioSplitter,
    DiarizedTranscript,
    Diarizer,
    SpeakerLabel,
    SpeechDetector,
    SpeechSegment,
    StorageClient,
    Transcriber,
    TranscriptionSegment,
)


class SyntheticDatasetStorageClient(StorageClient):
    def download(self, file_id: str) -> AudioSource:
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            with wave.open(tf.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                # Create 1 second of synthetic silence
                w.writeframes(b"\x00" * 16000 * 2)
            return AudioSource(filepath=tf.name, duration_seconds=1.0)


class SyntheticDatasetSpeechDetector(SpeechDetector):
    def detect_speech(self, chunk: AudioChunk) -> list[SpeechSegment]:
        return [SpeechSegment(start_time=chunk.start_time, end_time=chunk.start_time + 0.5)]


class SyntheticDatasetTranscriber(Transcriber):
    def transcribe(
        self, chunk: AudioChunk, speech_segments: list[SpeechSegment]
    ) -> list[TranscriptionSegment]:
        return [
            TranscriptionSegment(
                start_time=seg.start_time, end_time=seg.end_time, text=f"Synthetic Text {seg.start_time}"
            )
            for seg in speech_segments
        ]


class SyntheticDatasetDiarizer(Diarizer):
    def diarize(self, chunk: AudioChunk) -> list[SpeakerLabel]:
        return [
            SpeakerLabel(
                start_time=chunk.start_time,
                end_time=chunk.start_time + 0.5,
                speaker_id="SPEAKER_00",
            )
        ]


class FailingSyntheticStorageClient(StorageClient):
    """A clean synthetic double that simulates a network failure on download."""

    def download(self, file_id: str) -> AudioSource:
        msg = "Network Error"
        raise RuntimeError(msg)


def test_pipeline_integration_failure() -> None:
    # Test handling of download failures using proper test double instead of monkey-patching
    storage: StorageClient = FailingSyntheticStorageClient()
    import shutil

    import pytest

    from meetingnoter.processing.chunker import FFmpegChunker

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")

    # Ensure run_pipeline raises the error to the caller
    from main import run_pipeline

    with pytest.raises(RuntimeError, match="Network Error"):
        run_pipeline(
            storage=storage,
            splitter=FFmpegChunker(chunk_length_minutes=1),
            detector=SyntheticDatasetSpeechDetector(),
            transcriber=SyntheticDatasetTranscriber(),
            diarizer=SyntheticDatasetDiarizer(),
            file_id="test_id",
        )


def test_pipeline_integration() -> None:
    # Use the main.py run_pipeline orchestration logic to actually test the integration SUT
    import shutil

    import pytest

    from main import run_pipeline
    from meetingnoter.processing.chunker import FFmpegChunker

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")

    storage: StorageClient = SyntheticDatasetStorageClient()
    splitter: AudioSplitter = FFmpegChunker(chunk_length_minutes=1)
    detector: SpeechDetector = SyntheticDatasetSpeechDetector()
    transcriber: Transcriber = SyntheticDatasetTranscriber()
    diarizer: Diarizer = SyntheticDatasetDiarizer()

    transcript: DiarizedTranscript = run_pipeline(
        storage=storage,
        splitter=splitter,
        detector=detector,
        transcriber=transcriber,
        diarizer=diarizer,
        file_id="test_id",
    )

    assert len(transcript.segments) == 1
    assert transcript.segments[0].speaker_id == "SPEAKER_00"
    assert transcript.segments[0].text == "Synthetic Text 0.0"

def test_ffmpeg_chunker_integration() -> None:
    import shutil
    if not shutil.which("ffmpeg"):
        import pytest
        pytest.skip("ffmpeg not installed")

    import wave
    from pathlib import Path

    from meetingnoter.processing.chunker import FFmpegChunker

    # Create a valid minimal wave file
    with (
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf,
        wave.open(tf.name, "wb") as w,
    ):
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        # Create 1 second of silence
        w.writeframes(b"\x00" * 16000 * 2)

    try:
        source: AudioSource = AudioSource(filepath=tf.name, duration_seconds=1.0)
        chunker: FFmpegChunker = FFmpegChunker(chunk_length_minutes=1) # 1 minute chunks

        chunks: list[AudioChunk] = chunker.split(source)

        assert len(chunks) == 1
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 1.0
        assert Path(chunks[0].chunk_filepath).exists()
        assert Path(chunks[0].chunk_filepath).stat().st_size > 0
    finally:
        Path(tf.name).unlink(missing_ok=True)
