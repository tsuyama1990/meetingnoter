from collections.abc import Callable
from typing import Any

import marimo

try:
    import google.colab
except ImportError:
    google: Any = None  # type: ignore[no-redef]

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def cell_imports() -> tuple[Any]:
    import marimo as mo

    return (mo,)


@app.cell
def cell_markdown(mo: Any) -> Any:
    return mo.md(
        r"""
        # CYCLE01 User Acceptance Testing (UAT)

        This interactive notebook validates the Pydantic schemas created in Cycle 01 for the Voice Analysis Pipeline.
        """
    )


@app.cell
def cell_tests(
    mo: Any,
) -> tuple[
    Callable[[], Any],
    Callable[[], Any],
    type,
    type,
    type,
    type,
    type,
    type,
    type,
    type,
    type,
    type,
    type,
    type,
    type,
    Any,
]:
    from pydantic import ValidationError

    from domain_models import (
        AudioChunk,
        AudioSource,
        AudioSplitter,
        DiarizedSegment,
        DiarizedTranscript,
        Diarizer,
        SpeakerLabel,
        SpeechDetector,
        SpeechSegment,
        StorageClient,
        Transcriber,
        TranscriptionSegment,
    )

    def test_happy_path() -> Any:
        try:
            source = AudioSource(filepath="sample.m4a", duration_seconds=120.0)
            chunk = AudioChunk(
                chunk_filepath="sample_chunk_1.wav", start_time=0.0, end_time=60.0, chunk_index=0
            )
            speech = SpeechSegment(start_time=1.0, end_time=15.0)
            transcription = TranscriptionSegment(
                start_time=1.0, end_time=15.0, text="Thank you for joining this interview."
            )
            speaker = SpeakerLabel(start_time=1.0, end_time=15.0, speaker_id="SPEAKER_00")
            diarized = DiarizedSegment(
                start_time=1.0,
                end_time=15.0,
                speaker_id="SPEAKER_00",
                text="Thank you for joining this interview.",
            )
            transcript = DiarizedTranscript(segments=[diarized])

            # verify we use variables to prevent F841
            assert source.duration_seconds > 0
            assert chunk.end_time > 0
            assert speech.end_time > 0
            assert transcription.start_time >= 0
            assert speaker.speaker_id == "SPEAKER_00"

            return mo.md(f"**Happy Path Passed!**\n\nGenerated Transcript: {transcript}")
        except Exception as e:
            return mo.md(f"**Happy Path Failed!** Error: {e}")

    def test_error_handling() -> Any:
        try:
            # Trigger error: start_time > end_time
            AudioChunk(
                chunk_filepath="sample_chunk_1.wav", start_time=60.0, end_time=10.0, chunk_index=0
            )
            return mo.md("**Error Handling Failed:** Exception was not triggered!")
        except ValidationError as e:
            return mo.md(
                f"**Error Handling Passed!** Caught validation error gracefully:\n```\n{e}\n```"
            )

    tests_output = mo.vstack([test_happy_path(), test_error_handling()])
    return (
        test_happy_path,
        test_error_handling,
        ValidationError,
        AudioSource,
        AudioChunk,
        SpeechSegment,
        TranscriptionSegment,
        SpeakerLabel,
        DiarizedSegment,
        DiarizedTranscript,
        StorageClient,
        AudioSplitter,
        SpeechDetector,
        Transcriber,
        Diarizer,
        tests_output,
    )


@app.cell
def cell_markdown_c03(mo: Any) -> Any:
    return mo.md(
        r"""
        # CYCLE03 User Acceptance Testing (UAT)

        This section validates the Audio Preprocessing (Chunking) component natively on synthetic datasets, providing a visual demonstration of the chunking mechanism without mocking abstractions.
        """
    )


@app.cell
def cell_tests_c03(mo: Any) -> tuple[Callable[[], Any], Any]:
    def test_c03_ffmpeg_chunker() -> Any:
        import shutil
        import tempfile
        import wave
        from pathlib import Path

        from domain_models import AudioChunk, AudioSource
        from meetingnoter.processing.chunker import FFmpegChunker

        if not shutil.which("ffmpeg"):
            return mo.md("**Cycle 03 UAT Skipped:** FFmpeg is not installed on this system.")

        with (
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf,
            wave.open(tf.name, "wb") as w,
        ):
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            # Create 3 seconds of synthetic silence
            w.writeframes(b"\x00" * 16000 * 2 * 3)

        try:
            source = AudioSource(filepath=tf.name, duration_seconds=3.0)
            # Create a chunker that splits every 1 minute (should only produce 1 chunk for 3s audio)
            chunker = FFmpegChunker(chunk_length_minutes=1)
            chunks: list[AudioChunk] = chunker.split(source)

            output_msg = (
                f"**Cycle 03 Chunker Passed!**\n\nGenerated {len(chunks)} chunks successfully.\n\n"
            )
            for chunk in chunks:
                output_msg += (
                    f"- Chunk {chunk.chunk_index}: {chunk.start_time}s to {chunk.end_time}s\n"
                )

            return mo.md(output_msg)
        except Exception as e:
            return mo.md(f"**Cycle 03 Chunker Failed:** {e}")
        finally:
            Path(tf.name).unlink(missing_ok=True)

    c03_tests_output = mo.vstack([test_c03_ffmpeg_chunker()])
    return (
        test_c03_ffmpeg_chunker,
        c03_tests_output,
    )


if __name__ == "__main__":
    app.run()


@app.cell
def cell_tests_c05_1() -> tuple[object, ...]:
    import marimo as mo

    return (mo,)


@app.cell
def cell_tests_c05_2(mo: object) -> tuple[object, ...]:
    import typing as _typing

    mo_typed = _typing.cast(_typing.Any, mo)
    mo_typed.md(
        """
        # CYCLE05 User Acceptance Testing (UAT)

        This notebook serves as the interactive tutorial and UAT suite for the `Advanced Transcription Engine` implemented in Cycle 05.

        The UAT scenarios are derived directly from `dev_documents/system_prompts/CYCLE05/UAT.md`.
        """
    )
    return ()


@app.cell
def cell_tests_c05_3() -> tuple[object, ...]:
    import tempfile
    import wave
    from pathlib import Path
    from unittest.mock import MagicMock as _MagicMock
    from unittest.mock import patch as _patch

    from domain_models import AudioChunk as _AudioChunk
    from domain_models import SpeechSegment as _SpeechSegment
    from meetingnoter.processing.transcriber import (
        FasterWhisperTranscriber as _FasterWhisperTranscriber,
    )

    # 1. Setup a dummy wav file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        with wave.open(tf.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            # Create 1 second of synthetic silence
            w.writeframes(b"\x00" * 16000 * 2)
        chunk_01_name = tf.name

    chunk_01 = _AudioChunk(
        chunk_filepath=chunk_01_name, start_time=10.0, end_time=20.0, chunk_index=0
    )

    speech_segments_01 = [_SpeechSegment(start_time=10.0, end_time=15.0)]

    # 2. Mock FasterWhisper model so we can run this without actually downloading models
    # This is strictly for the UAT notebook happy path test without GPU
    mock_model_instance = _MagicMock()

    # Mock transcript segments
    mock_segment1 = _MagicMock()
    mock_segment1.start = 0.5
    mock_segment1.end = 2.0
    mock_segment1.text = "Hello "

    mock_model_instance.transcribe.return_value = ([mock_segment1], None)

    transcriber_01 = _FasterWhisperTranscriber(language="ja")

    # We must patch inspect.signature to return a signature matching what we need to test
    mock_sig = _MagicMock()
    mock_sig.parameters = {
        "audio": _MagicMock(),
        "language": _MagicMock(),
        "vad_filter": _MagicMock(),
        "condition_on_previous_text": _MagicMock(),
        "temperature": _MagicMock(),
        "compression_ratio_threshold": _MagicMock(),
        "log_prob_threshold": _MagicMock(),
        "no_speech_threshold": _MagicMock(),
    }

    with (
        _patch("faster_whisper.WhisperModel", return_value=mock_model_instance),
        _patch("inspect.signature", return_value=mock_sig),
    ):
        results_01 = transcriber_01.transcribe(chunk_01, speech_segments_01)

    Path(chunk_01_name).unlink()

    return chunk_01, speech_segments_01, transcriber_01, results_01


@app.cell
def cell_tests_c05_4() -> tuple[object, ...]:
    from unittest.mock import patch as _patch_err

    from domain_models import AudioChunk as _AudioChunk_err
    from meetingnoter.processing.transcriber import (
        FasterWhisperTranscriber as _FasterWhisperTranscriber_err,
    )

    transcriber_err = _FasterWhisperTranscriber_err(language="ja")

    chunk_err = _AudioChunk_err(
        chunk_filepath="/path/to/nonexistent.wav", start_time=0.0, end_time=10.0, chunk_index=0
    )

    import contextlib

    with contextlib.suppress(FileNotFoundError), _patch_err.object(transcriber_err, "_load_model"):
        # Avoid loading the real model if we can't find the file
        transcriber_err.transcribe(chunk_err, [])

    return chunk_err, transcriber_err
