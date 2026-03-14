import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def cell_imports() -> tuple:  # type: ignore[type-arg]
    import marimo as mo

    return (mo,)


@app.cell
def cell_markdown(mo: object) -> object:
    return mo.md(  # type: ignore[attr-defined]
        r"""
        # CYCLE01 User Acceptance Testing (UAT)

        This interactive notebook validates the Pydantic schemas created in Cycle 01 for the Voice Analysis Pipeline.
        """
    )


@app.cell
def cell_tests(mo: object) -> tuple:  # type: ignore[type-arg]
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

    def test_happy_path() -> object:
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

            return mo.md(f"**Happy Path Passed!**\n\nGenerated Transcript: {transcript}")  # type: ignore[attr-defined]
        except Exception as e:
            return mo.md(f"**Happy Path Failed!** Error: {e}")  # type: ignore[attr-defined]

    def test_error_handling() -> object:
        try:
            # Trigger error: start_time > end_time
            AudioChunk(
                chunk_filepath="sample_chunk_1.wav", start_time=60.0, end_time=10.0, chunk_index=0
            )
            return mo.md("**Error Handling Failed:** Exception was not triggered!")  # type: ignore[attr-defined]
        except ValidationError as e:
            return mo.md(  # type: ignore[attr-defined]
                f"**Error Handling Passed!** Caught validation error gracefully:\n```\n{e}\n```"
            )

    tests_output = mo.vstack([test_happy_path(), test_error_handling()])  # type: ignore[attr-defined]
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
def cell_markdown_c02(mo: object) -> object:
    return mo.md(  # type: ignore[attr-defined]
        r"""
        # CYCLE02 User Acceptance Testing (UAT)

        This section validates the Secure Data Ingestion component logic through test doubles to prevent actual API calls and secret exposure.
        """
    )


@app.cell
def cell_tests_c02(mo: object) -> tuple:  # type: ignore[type-arg]
    def test_c02_error_handling() -> object:
        from unittest.mock import MagicMock

        import requests

        from domain_models import PipelineConfig
        from meetingnoter.ingestion.drive_client import GoogleDriveClient

        try:
            # Safely configure a fake environment
            import os

            os.environ["GOOGLE_API_KEY"] = "mock_api_key_for_testing_12345"
            os.environ["PYANNOTE_AUTH_TOKEN"] = "mock_pyannote"
            os.environ["FILE_ID"] = "mock_file"

            config = PipelineConfig()  # type: ignore[call-arg]

            # Inject a mock HTTP Client
            mock_http = MagicMock(spec=requests.Session)
            mock_http.get.side_effect = requests.exceptions.HTTPError("403 Forbidden")

            client = GoogleDriveClient(config=config, http_client=mock_http)

            # This should fail naturally due to the mock raising HTTPError
            client.download("fake_file_id_for_testing_12345")

            return mo.md("**Cycle 02 Error Handling Failed:** Exception was not triggered!")  # type: ignore[attr-defined]
        except RuntimeError as e:
            return mo.md(  # type: ignore[attr-defined]
                f"**Cycle 02 Error Handling Passed!** Caught runtime error gracefully from simulated API failure:\n```\n{e}\n```"
            )

    c02_tests_output = mo.vstack([test_c02_error_handling()])  # type: ignore[attr-defined]
    return (
        test_c02_error_handling,
        c02_tests_output,
    )


if __name__ == "__main__":
    app.run()
