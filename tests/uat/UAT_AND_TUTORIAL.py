import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def cell_imports() -> tuple: # type: ignore[type-arg]
    import marimo as mo

    return (mo,)


@app.cell
def cell_markdown(mo: object) -> None:
    mo.md( # type: ignore[attr-defined]
        r"""
        # CYCLE01 User Acceptance Testing (UAT)

        This interactive notebook validates the Pydantic schemas created in Cycle 01 for the Voice Analysis Pipeline.
        """
    )


@app.cell
def cell_tests(mo: object) -> tuple: # type: ignore[type-arg]
    from pydantic import ValidationError

    from domain_models import (
        AudioChunk,
        AudioSource,
        DiarizedSegment,
        DiarizedTranscript,
        SpeakerLabel,
        SpeechSegment,
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

            return mo.md(f"**Happy Path Passed!**\n\nGenerated Transcript: {transcript}") # type: ignore[attr-defined]
        except Exception as e:
            return mo.md(f"**Happy Path Failed!** Error: {e}") # type: ignore[attr-defined]

    def test_error_handling() -> object:
        try:
            # Trigger error: start_time > end_time
            AudioChunk(
                chunk_filepath="sample_chunk_1.wav", start_time=60.0, end_time=10.0, chunk_index=0
            )
            return mo.md("**Error Handling Failed:** Exception was not triggered!") # type: ignore[attr-defined]
        except ValidationError as e:
            return mo.md( # type: ignore[attr-defined]
                f"**Error Handling Passed!** Caught validation error gracefully:\n```\n{e}\n```"
            )

    mo.vstack([test_happy_path(), test_error_handling()]) # type: ignore[attr-defined]
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
    )


if __name__ == "__main__":
    app.run()
