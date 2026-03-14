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
def cell_markdown_c02(mo: Any) -> Any:
    return mo.md(
        r"""
        # CYCLE02 User Acceptance Testing (UAT)

        This section validates the Secure Data Ingestion component logic through test doubles to prevent actual API calls and secret exposure.
        """
    )


@app.cell
def cell_tests_c02(mo: Any) -> tuple[Callable[[], Any], Any]:
    def test_c02_error_handling() -> Any:
        from unittest.mock import MagicMock, patch

        import requests

        from domain_models import PipelineConfig
        from meetingnoter.ingestion.drive_client import GoogleDriveClient

        try:
            # Safely configure a fake environment without hardcoding into os.environ globally
            with patch("os.environ.get", return_value="dummy_env_var_value_for_uat"):
                config = PipelineConfig()

            # Inject a mock HTTP Client
            mock_http = MagicMock(spec=requests.Session)
            mock_http.get.side_effect = requests.exceptions.HTTPError("403 Forbidden")

            client = GoogleDriveClient(config=config, http_client=mock_http)

            # This should fail naturally due to the mock raising HTTPError
            client.download("fake_file_id_for_testing_12345")

            return mo.md("**Cycle 02 Error Handling Failed:** Exception was not triggered!")
        except RuntimeError as e:
            return mo.md(
                f"**Cycle 02 Error Handling Passed!** Caught runtime error gracefully from simulated API failure:\n```\n{e}\n```"
            )

    c02_tests_output = mo.vstack([test_c02_error_handling()])
    return (
        test_c02_error_handling,
        c02_tests_output,
    )


if __name__ == "__main__":
    app.run()

@app.cell
def cell_markdown_c03(mo: Any) -> Any:
    return mo.md(
        r"""
        # CYCLE03 User Acceptance Testing (UAT)

        This section validates the Audio Preprocessing (Chunking) component. It tests the behavior when attempting to create malformed audio chunks, ensuring that precise timestamp offset validation prevents state errors before heavy compute begins.
        """
    )


@app.cell
def cell_tests_c03(mo: Any) -> tuple[Callable[[], Any], Any]:
    def test_c03_error_handling() -> Any:
        from pydantic import ValidationError

        from domain_models import AudioChunk

        try:
            # Inject malformed data: start_time >= end_time
            _ = AudioChunk(
                chunk_filepath="malformed_chunk.wav",
                start_time=120.0,
                end_time=60.0, # Error: end time is before start time
                chunk_index=1
            )
            return mo.md("**Cycle 03 Error Handling Failed:** Exception was not triggered!")
        except ValidationError as e:
            return mo.md(
                f"**Cycle 03 Error Handling Passed!** Caught validation error gracefully:\n```\n{e}\n```"
            )

    c03_tests_output = mo.vstack([test_c03_error_handling()])
    return (
        test_c03_error_handling,
        c03_tests_output,
    )
