from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain_models import (
    AudioChunk,
    AudioSource,
    AudioSplitter,
    Diarizer,
    PipelineConfig,
    SpeechDetector,
    SpeechSegment,
    StorageClient,
    Transcriber,
)
from main import _process_single_chunk, run_pipeline
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
        patch("pathlib.Path.is_symlink", return_value=True),
        pytest.raises(ValueError, match="is a symlink, which is not permitted."),
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


# ---------------------------------------------------------
# CYCLE 07 Tests for Pipeline Orchestration
# ---------------------------------------------------------


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


def test_pipeline_orchestration_cleanup_on_download_fail(tmp_path: Path) -> None:
    # Arrange
    storage: StorageClient = FailingSyntheticStorageClient()
    splitter: AudioSplitter = SyntheticDatasetAudioSplitter()
    detector = MagicMock(spec=SpeechDetector)
    transcriber = MagicMock(spec=Transcriber)
    diarizer = MagicMock(spec=Diarizer)

    # Act & Assert
    with pytest.raises(RuntimeError, match="Network Error"):
        run_pipeline(
            storage=storage,
            splitter=splitter,
            detector=detector,
            transcriber=transcriber,
            diarizer=diarizer,
            file_id="test_id",
        )
    # The source file was never created, so we don't assert deletion,
    # but we ensure the code safely passed the finally block without error.


def test_pipeline_orchestration_cleanup_on_chunking_fail(tmp_path: Path) -> None:
    # Arrange
    class FakeStorage(StorageClient):
        def download(self, file_id: str) -> AudioSource:
            f = tmp_path / "source.wav"
            f.touch()
            return AudioSource(filepath=str(f), duration_seconds=10.0)

    class FailingSplitter(AudioSplitter):
        def split(self, source: AudioSource) -> list[AudioChunk]:
            msg = "Invalid audio format"
            raise ValueError(msg)

    storage = FakeStorage()
    splitter = FailingSplitter()
    detector = MagicMock(spec=SpeechDetector)
    transcriber = MagicMock(spec=Transcriber)
    diarizer = MagicMock(spec=Diarizer)

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid audio format"):
        run_pipeline(
            storage=storage,
            splitter=splitter,
            detector=detector,
            transcriber=transcriber,
            diarizer=diarizer,
            file_id="test_id",
        )
    # Ensure source was cleaned up in the finally block
    assert not (tmp_path / "source.wav").exists()


def test_pipeline_orchestration_cleanup_on_processing_fail(tmp_path: Path) -> None:
    # Arrange
    class FakeStorage(StorageClient):
        def download(self, file_id: str) -> AudioSource:
            f = tmp_path / "source.wav"
            f.touch()
            return AudioSource(filepath=str(f), duration_seconds=10.0)

    class FakeSplitter(AudioSplitter):
        def split(self, source: AudioSource) -> list[AudioChunk]:
            f1 = tmp_path / "chunk1.wav"
            f1.touch()
            f2 = tmp_path / "chunk2.wav"
            f2.touch()
            return [
                AudioChunk(chunk_filepath=str(f1), start_time=0.0, end_time=5.0, chunk_index=0),
                AudioChunk(chunk_filepath=str(f2), start_time=5.0, end_time=10.0, chunk_index=1),
            ]

    storage = FakeStorage()
    splitter = FakeSplitter()
    detector = MagicMock(spec=SpeechDetector)
    # Force detector to fail on the second chunk
    detector.detect_speech.side_effect = [[], RuntimeError("CUDA out of memory")]

    transcriber = MagicMock(spec=Transcriber)
    transcriber.transcribe.return_value = []

    diarizer = MagicMock(spec=Diarizer)
    diarizer.diarize.return_value = []

    # Act & Assert
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run_pipeline(
            storage=storage,
            splitter=splitter,
            detector=detector,
            transcriber=transcriber,
            diarizer=diarizer,
            file_id="test_id",
        )
    # Ensure all temp files were cleaned up
    assert not (tmp_path / "source.wav").exists()
    assert not (tmp_path / "chunk1.wav").exists()
    assert not (tmp_path / "chunk2.wav").exists()


def test_process_single_chunk_network_error(tmp_path: Path) -> None:
    chunk = AudioChunk(
        chunk_filepath=str(tmp_path / "test.wav"), start_time=0.0, end_time=10.0, chunk_index=0
    )
    detector = MagicMock(spec=SpeechDetector)
    detector.detect_speech.side_effect = RuntimeError("API down")
    transcriber = MagicMock(spec=Transcriber)
    diarizer = MagicMock(spec=Diarizer)

    with pytest.raises(RuntimeError, match="API down"):
        _process_single_chunk(chunk, detector, transcriber, diarizer)


def test_process_single_chunk_validation_error(tmp_path: Path) -> None:
    chunk = AudioChunk(
        chunk_filepath=str(tmp_path / "test.wav"), start_time=0.0, end_time=10.0, chunk_index=0
    )
    detector = MagicMock(spec=SpeechDetector)
    detector.detect_speech.side_effect = ValueError("Invalid inputs")
    transcriber = MagicMock(spec=Transcriber)
    diarizer = MagicMock(spec=Diarizer)

    with pytest.raises(ValueError, match="Invalid inputs"):
        _process_single_chunk(chunk, detector, transcriber, diarizer)


def test_process_single_chunk_unexpected_error(tmp_path: Path) -> None:
    chunk = AudioChunk(
        chunk_filepath=str(tmp_path / "test.wav"), start_time=0.0, end_time=10.0, chunk_index=0
    )
    detector = MagicMock(spec=SpeechDetector)
    detector.detect_speech.side_effect = TypeError("Something totally weird")
    transcriber = MagicMock(spec=Transcriber)
    diarizer = MagicMock(spec=Diarizer)

    with pytest.raises(
        RuntimeError,
        match="Unexpected failure in pipeline processing chunk 0: Something totally weird",
    ):
        _process_single_chunk(chunk, detector, transcriber, diarizer)
