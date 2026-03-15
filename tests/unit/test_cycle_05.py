import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from domain_models import AudioChunk, SpeechSegment, TranscriptionSegment
from meetingnoter.processing.transcriber import FasterWhisperTranscriber


@patch("faster_whisper.WhisperModel")
def test_transcriber_initialization(mock_whisper_model: MagicMock) -> None:
    transcriber = FasterWhisperTranscriber(
        model_size="tiny",
        compute_type="int8",
        language="ja",
        vad_filter=True,
        condition_on_previous_text=False,
    )

    # Model should not be loaded on init
    assert transcriber.model is None
    mock_whisper_model.assert_not_called()

    # Check parameters
    assert transcriber.model_size == "tiny"
    assert transcriber.compute_type == "int8"
    assert transcriber.language == "ja"
    assert transcriber.vad_filter is True
    assert transcriber.condition_on_previous_text is False


@patch("faster_whisper.WhisperModel")
@patch("pathlib.Path.exists", return_value=True)
def test_transcriber_load_and_transcribe(
    mock_exists: MagicMock, mock_whisper_model: MagicMock, tmp_path: Path
) -> None:
    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance

    # Mock transcript segments
    mock_segment1 = MagicMock()
    mock_segment1.start = 0.5
    mock_segment1.end = 2.0
    mock_segment1.text = "Hello "

    mock_segment2 = MagicMock()
    mock_segment2.start = 2.5
    mock_segment2.end = 4.0
    mock_segment2.text = "World"

    mock_model_instance.transcribe.return_value = ([mock_segment1, mock_segment2], None)

    # We must patch inspect.signature to return a signature matching what we need to test
    # (specifically the presence of compression_ratio_threshold)
    import inspect

    mock_sig = MagicMock()
    mock_sig.parameters = {
        "audio": MagicMock(),
        "language": MagicMock(),
        "vad_filter": MagicMock(),
        "condition_on_previous_text": MagicMock(),
        "temperature": MagicMock(),
        "compression_ratio_threshold": MagicMock(),
        "log_prob_threshold": MagicMock(),
        "no_speech_threshold": MagicMock(),
    }

    with patch("inspect.signature", return_value=mock_sig):
        transcriber = FasterWhisperTranscriber(language="ja")

        chunk = AudioChunk(
            chunk_filepath=str(tmp_path / "test.wav"), start_time=10.0, end_time=20.0, chunk_index=0
        )

        speech_segments = [SpeechSegment(start_time=10.0, end_time=15.0)]

        results = transcriber.transcribe(chunk, speech_segments)

        # Check that model was loaded
        assert transcriber.model is not None
        mock_whisper_model.assert_called_once()

        # Check that transcribe was called with correct parameters
        call_kwargs = mock_model_instance.transcribe.call_args.kwargs
        assert call_kwargs["language"] == "ja"
        assert call_kwargs["compression_ratio_threshold"] is None
        assert call_kwargs["log_prob_threshold"] is None
        assert call_kwargs["no_speech_threshold"] is None
        assert call_kwargs["condition_on_previous_text"] is False

        # Check results mapping to global timestamps
        assert len(results) == 2
        assert results[0].start_time == 10.5
        assert results[0].end_time == 12.0
        assert results[0].text == "Hello"

        assert results[1].start_time == 12.5
        assert results[1].end_time == 14.0
        assert results[1].text == "World"


def test_transcriber_file_not_found() -> None:
    transcriber = FasterWhisperTranscriber()

    chunk = AudioChunk(
        chunk_filepath="/path/to/nonexistent.wav", start_time=0.0, end_time=10.0, chunk_index=0
    )

    with pytest.raises(FileNotFoundError, match="Audio chunk file not found"):
        # load model needs to be patched out or we'll get an error trying to load it
        with patch.object(transcriber, "_load_model"):
            transcriber.transcribe(chunk, [])


@patch("faster_whisper.WhisperModel")
@patch("pathlib.Path.exists", return_value=True)
def test_transcriber_inference_error(
    mock_exists: MagicMock, mock_whisper_model: MagicMock, tmp_path: Path
) -> None:
    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.side_effect = Exception("Inference failed")
    mock_whisper_model.return_value = mock_model_instance

    import inspect

    mock_sig = MagicMock()
    mock_sig.parameters = {"audio": MagicMock()}

    with patch("inspect.signature", return_value=mock_sig):
        transcriber = FasterWhisperTranscriber()

        chunk = AudioChunk(
            chunk_filepath=str(tmp_path / "test.wav"), start_time=0.0, end_time=10.0, chunk_index=0
        )

        with pytest.raises(
            RuntimeError, match="Faster whisper transcription failed: Inference failed"
        ):
            transcriber.transcribe(chunk, [])

@patch("faster_whisper.WhisperModel")
@patch("pathlib.Path.exists", return_value=True)
def test_transcriber_load_model_not_found(mock_exists: MagicMock, mock_whisper_model: MagicMock, tmp_path: Path) -> None:
    mock_whisper_model.side_effect = Exception("Model initialization failed")

    transcriber = FasterWhisperTranscriber()

    chunk = AudioChunk(
        chunk_filepath=str(tmp_path / "test.wav"),
        start_time=0.0,
        end_time=10.0,
        chunk_index=0
    )

    with pytest.raises(RuntimeError, match="Failed to load Faster Whisper model"):
        transcriber.transcribe(chunk, [])


@patch("faster_whisper.WhisperModel")
@patch("pathlib.Path.exists", return_value=True)
def test_transcriber_model_none_after_load(mock_exists: MagicMock, mock_whisper_model: MagicMock, tmp_path: Path) -> None:
    # Simulate a scenario where model is still None after _load_model
    # This might happen if the try/except block caught something but didn't raise, or just to cover the `if self.model:` branch
    transcriber = FasterWhisperTranscriber()

    chunk = AudioChunk(
        chunk_filepath=str(tmp_path / "test.wav"),
        start_time=0.0,
        end_time=10.0,
        chunk_index=0
    )

    # Force _load_model to do nothing, so model remains None
    with patch.object(transcriber, "_load_model"):
        with pytest.raises(RuntimeError, match="Faster Whisper model was not properly loaded"):
            transcriber.transcribe(chunk, [])

def test_transcriber_import_error() -> None:
    # Test handling of missing faster-whisper dependency
    transcriber = FasterWhisperTranscriber()

    import sys

    # Hide the faster_whisper module to trigger ImportError
    with patch.dict(sys.modules, {"faster_whisper": None}):
        with pytest.raises(ImportError, match="Required library 'faster-whisper' or 'torch' is not installed."):
            transcriber._load_model()
