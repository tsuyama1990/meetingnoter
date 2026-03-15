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
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=f"Synthetic Text {seg.start_time}",
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


class SyntheticDatasetAudioSplitter(AudioSplitter):
    def split(self, source: AudioSource) -> list[AudioChunk]:
        return [
            AudioChunk(
                chunk_filepath=source.filepath,
                start_time=0.0,
                end_time=source.duration_seconds,
                chunk_index=0,
            )
        ]


def create_failing_storage_client() -> StorageClient:
    return FailingSyntheticStorageClient()


def create_storage_client() -> StorageClient:
    return SyntheticDatasetStorageClient()


def create_audio_splitter() -> AudioSplitter:
    return SyntheticDatasetAudioSplitter()


def create_speech_detector() -> SpeechDetector:
    return SyntheticDatasetSpeechDetector()


def create_transcriber() -> Transcriber:
    return SyntheticDatasetTranscriber()


def create_diarizer() -> Diarizer:
    return SyntheticDatasetDiarizer()


def test_pipeline_integration_failure() -> None:
    import pytest

    from main import run_pipeline

    storage: StorageClient = create_failing_storage_client()

    with pytest.raises(RuntimeError, match="Network Error"):
        run_pipeline(
            storage=storage,
            splitter=create_audio_splitter(),
            detector=create_speech_detector(),
            transcriber=create_transcriber(),
            diarizer=create_diarizer(),
            file_id="test_id",
        )


def test_pipeline_integration() -> None:
    # Use the main.py run_pipeline orchestration logic to actually test the integration SUT

    from main import run_pipeline

    storage: StorageClient = create_storage_client()
    splitter: AudioSplitter = create_audio_splitter()
    detector: SpeechDetector = create_speech_detector()

    transcriber: Transcriber = create_transcriber()
    diarizer: Diarizer = create_diarizer()

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
    import wave
    from pathlib import Path
    from typing import Any
    from unittest.mock import patch

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
        import os

        chunk_length_str = os.environ.get("TEST_CHUNK_LENGTH_MINUTES", "1")
        chunk_length = int(chunk_length_str)

        source: AudioSource = AudioSource(filepath=tf.name, duration_seconds=1.0)
        chunker: FFmpegChunker = FFmpegChunker(chunk_length_minutes=chunk_length)

        with patch("subprocess.run") as mock_run:
            # We mock the subprocess.run call to ffmpeg, but create the output file artificially
            def side_effect(*args: list[Any], **kwargs: dict[str, Any]) -> None:
                # The output file is args[-1]
                cmd = args[0]
                output_file = Path(cmd[-1])
                import shutil

                shutil.copy(tf.name, output_file)

            mock_run.side_effect = side_effect
            chunks: list[AudioChunk] = chunker.split(source)

        assert len(chunks) == 1
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 1.0
        assert Path(chunks[0].chunk_filepath).exists()
        assert Path(chunks[0].chunk_filepath).stat().st_size > 0
    finally:
        Path(tf.name).unlink(missing_ok=True)


def test_pipeline_diarization_integration() -> None:
    # Test integration between transcription and diarization specifically
    from main import run_pipeline

    class MultiSpeakerDiarizer(Diarizer):
        def diarize(self, chunk: AudioChunk) -> list[SpeakerLabel]:
            return [
                SpeakerLabel(
                    start_time=chunk.start_time,
                    end_time=chunk.start_time + 0.5,
                    speaker_id="SPEAKER_00",
                ),
                SpeakerLabel(
                    start_time=chunk.start_time + 0.5,
                    end_time=chunk.start_time + 1.0,
                    speaker_id="SPEAKER_01",
                ),
            ]

    class MultiSpeechTranscriber(Transcriber):
        def transcribe(
            self, chunk: AudioChunk, speech_segments: list[SpeechSegment]
        ) -> list[TranscriptionSegment]:
            return [
                TranscriptionSegment(
                    start_time=chunk.start_time,
                    end_time=chunk.start_time + 0.5,
                    text="Hello",
                ),
                TranscriptionSegment(
                    start_time=chunk.start_time + 0.5,
                    end_time=chunk.start_time + 1.0,
                    text="World",
                ),
            ]

    storage = create_storage_client()
    splitter = create_audio_splitter()
    detector = create_speech_detector()
    transcriber = MultiSpeechTranscriber()
    diarizer = MultiSpeakerDiarizer()

    transcript: DiarizedTranscript = run_pipeline(
        storage=storage,
        splitter=splitter,
        detector=detector,
        transcriber=transcriber,
        diarizer=diarizer,
        file_id="test_id",
    )

    assert len(transcript.segments) == 2
    assert transcript.segments[0].speaker_id == "SPEAKER_00"
    assert transcript.segments[0].text == "Hello"
    assert transcript.segments[1].speaker_id == "SPEAKER_01"
    assert transcript.segments[1].text == "World"
