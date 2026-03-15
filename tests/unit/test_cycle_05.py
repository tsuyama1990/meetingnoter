from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain_models import AudioChunk, PipelineConfig, SpeechSegment
from meetingnoter.processing.transcriber import FasterWhisperTranscriber


@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_initialization(mock_whisper_model: MagicMock) -> None:

    with patch("domain_models.config._get_secret", return_value="dummy"):
        config = PipelineConfig(transcriber_model_size="tiny")
        transcriber = FasterWhisperTranscriber(config)

    # Model should not be loaded on init
    assert transcriber.model is None
    mock_whisper_model.assert_not_called()

    # Check parameters
    assert transcriber.config.transcriber_model_size == "tiny"


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_load_and_transcribe(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
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

    with (
        patch("pathlib.Path.is_relative_to", return_value=True),
        patch("domain_models.config._get_secret", return_value="dummy"),
    ):
        config = PipelineConfig(transcriber_language="ja")
        transcriber = FasterWhisperTranscriber(config)

    test_file = tmp_path / "test.wav"
    test_file.touch()
    chunk = AudioChunk(chunk_filepath=str(test_file), start_time=10.0, end_time=20.0, chunk_index=0)

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

    with patch("domain_models.config._get_secret", return_value="dummy"):
        config = PipelineConfig()
        transcriber = FasterWhisperTranscriber(config)

    chunk = AudioChunk(
        chunk_filepath="/path/to/nonexistent.wav", start_time=0.0, end_time=10.0, chunk_index=0
    )

    with pytest.raises(FileNotFoundError, match="Audio chunk file not found"):
        transcriber.transcribe(chunk, [])


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_inference_error(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.side_effect = Exception("Inference failed")
    mock_whisper_model.return_value = mock_model_instance

    with patch("pathlib.Path.is_relative_to", return_value=True):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file), start_time=0.0, end_time=10.0, chunk_index=0
        )

        with pytest.raises(
            RuntimeError, match="Faster whisper transcription failed: Inference failed"
        ):
            transcriber.transcribe(chunk, [])


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_load_model_not_found(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    mock_whisper_model.side_effect = Exception("Model initialization failed")

    with patch("pathlib.Path.is_relative_to", return_value=True):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file), start_time=0.0, end_time=10.0, chunk_index=0
        )

        with pytest.raises(RuntimeError, match="Failed to load Faster Whisper model"):
            transcriber.transcribe(chunk, [])


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_model_none_after_load(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    with patch("pathlib.Path.is_relative_to", return_value=True):
        # Simulate a scenario where model is still None after _load_model
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file), start_time=0.0, end_time=10.0, chunk_index=0
        )

        # Force _load_model to do nothing, so model remains None
        with (
            patch.object(transcriber, "_load_model"),
            pytest.raises(RuntimeError, match="Faster Whisper model was not properly loaded"),
        ):
            transcriber.transcribe(chunk, [])


def test_transcriber_import_error() -> None:

    import importlib
    import sys

    # Hide the faster_whisper module to trigger ImportError
    import meetingnoter.processing.transcriber

    with (
        patch.dict(sys.modules, {"faster_whisper": None}),
        pytest.raises(
            ImportError, match="Required library 'faster-whisper' or 'torch' is not installed:"
        ),
    ):
        importlib.reload(meetingnoter.processing.transcriber)


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.empty_cache")
def test_transcriber_garbage_collection(
    mock_empty_cache: MagicMock,
    mock_cuda: MagicMock,
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    mock_model_instance = MagicMock()
    mock_whisper_model.return_value = mock_model_instance
    mock_model_instance.transcribe.return_value = ([], None)

    with patch("pathlib.Path.is_relative_to", return_value=True):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file),
            start_time=0.0,
            end_time=10.0,
            chunk_index=0,
        )

        transcriber.transcribe(chunk, [])

        mock_empty_cache.assert_called_once()


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_cuda_oom_error_during_load(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    mock_whisper_model.side_effect = RuntimeError("CUDA out of memory")

    with patch("pathlib.Path.is_relative_to", return_value=True):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file), start_time=0.0, end_time=10.0, chunk_index=0
        )

        with pytest.raises(
            RuntimeError, match="CUDA Out of Memory when trying to load Faster Whisper model."
        ):
            transcriber.transcribe(chunk, [])


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_cuda_oom_error_during_inference(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.side_effect = RuntimeError("CUDA out of memory")
    mock_whisper_model.return_value = mock_model_instance

    with patch("pathlib.Path.is_relative_to", return_value=True):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file), start_time=0.0, end_time=10.0, chunk_index=0
        )

        with pytest.raises(RuntimeError, match="CUDA Out of Memory during transcription."):
            transcriber.transcribe(chunk, [])


def test_transcriber_cleanup_resources_no_torch() -> None:

    import sys

    with patch.dict(sys.modules, {"torch": None}):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)
        # Should catch ImportError and pass
        transcriber._cleanup_resources()


def test_transcriber_invalid_path_relative() -> None:

    with patch("domain_models.config._get_secret", return_value="dummy"):
        config = PipelineConfig()
        transcriber = FasterWhisperTranscriber(config)

    with (
        patch("pathlib.Path.is_file", return_value=True),
        patch("pathlib.Path.is_relative_to", return_value=False),
        pytest.raises(ValueError, match="is not within allowed directories"),
    ):
        transcriber._validate_audio_file(Path("/etc/passwd"))


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_general_error_during_inference(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.side_effect = Exception("Some other error")
    mock_whisper_model.return_value = mock_model_instance

    with patch("pathlib.Path.is_relative_to", return_value=True):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file), start_time=0.0, end_time=10.0, chunk_index=0
        )

        with pytest.raises(
            RuntimeError, match="Faster whisper transcription failed: Some other error"
        ):
            transcriber.transcribe(chunk, [])


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_general_runtime_error_during_load(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    mock_whisper_model.side_effect = RuntimeError("Other runtime load error")

    with patch("pathlib.Path.is_relative_to", return_value=True):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file), start_time=0.0, end_time=10.0, chunk_index=0
        )

        with pytest.raises(
            RuntimeError, match="Failed to load Faster Whisper model: Other runtime load error"
        ):
            transcriber.transcribe(chunk, [])


@patch("pathlib.Path.is_file", return_value=True)
@patch("meetingnoter.processing.transcriber.WhisperModel")
def test_transcriber_general_runtime_error_during_inference(
    mock_whisper_model: MagicMock,
    mock_is_file: MagicMock,
    tmp_path: Path,
) -> None:

    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.side_effect = RuntimeError("Other runtime inference error")
    mock_whisper_model.return_value = mock_model_instance

    with patch("pathlib.Path.is_relative_to", return_value=True):
        with patch("domain_models.config._get_secret", return_value="dummy"):
            config = PipelineConfig()
            transcriber = FasterWhisperTranscriber(config)

        test_file = tmp_path / "test.wav"
        test_file.touch()
        chunk = AudioChunk(
            chunk_filepath=str(test_file), start_time=0.0, end_time=10.0, chunk_index=0
        )

        with pytest.raises(
            RuntimeError, match="Faster whisper transcription failed: Other runtime inference error"
        ):
            transcriber.transcribe(chunk, [])


def test_transcriber_cleanup_resources_torch_cuda_empty_cache() -> None:

    with patch("domain_models.config._get_secret", return_value="dummy"):
        config = PipelineConfig()
        transcriber = FasterWhisperTranscriber(config)

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache") as mock_empty_cache,
    ):
        transcriber._cleanup_resources()
        mock_empty_cache.assert_called_once()
