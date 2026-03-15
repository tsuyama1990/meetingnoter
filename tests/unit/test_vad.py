from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from domain_models import AudioChunk
from meetingnoter.processing.vad import SileroVADDetector


@patch("pathlib.Path.is_relative_to", return_value=True)
def test_silero_vad_init_validation(mock_is_rel: MagicMock) -> None:
    vad = SileroVADDetector(min_silence_duration_ms=500, model_path="dummy.jit")
    assert vad.config.min_silence_duration_ms == 500

    vad = SileroVADDetector(min_silence_duration_ms=1000, model_path="dummy.jit")
    assert vad.config.min_silence_duration_ms == 1000

    with pytest.raises(ValueError, match="min_silence_duration_ms"):
        SileroVADDetector(min_silence_duration_ms=2000, model_path="dummy.jit")

    with pytest.raises(ValueError, match="min_silence_duration_ms"):
        SileroVADDetector(min_silence_duration_ms=499, model_path="dummy.jit")


@patch("pathlib.Path.is_relative_to", return_value=True)
@patch("pathlib.Path.stat")
@patch("pathlib.Path.exists", return_value=True)
@patch("torch.jit.load")
@patch("torchaudio.load")
def test_silero_vad_detect_speech(
    mock_ta_load: MagicMock,
    mock_jit_load: MagicMock,
    mock_exists: MagicMock,
    mock_stat: MagicMock,
    mock_is_rel: MagicMock,
    tmp_path: Path,
) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    # Mock file size to be valid
    mock_stat_obj = MagicMock()
    mock_stat_obj.st_size = 1024
    mock_stat.return_value = mock_stat_obj

    chunk = AudioChunk(
        chunk_filepath=str(audio_file), start_time=10.0, end_time=20.0, chunk_index=0
    )

    mock_wav = MagicMock()
    mock_wav.shape = [1, 16000]
    mock_wav.mean.return_value = mock_wav
    mock_ta_load.return_value = (mock_wav, 16000)

    mock_model = MagicMock()
    mock_jit_load.return_value = mock_model

    probs = torch.zeros(21)
    probs[5:21] = 0.9

    mock_out = MagicMock()
    mock_out.squeeze.return_value = probs
    mock_model.return_value = mock_out

    # Disable hash checking for this test
    with patch.object(SileroVADDetector, "_verify_model_integrity"):
        vad = SileroVADDetector(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            model_path="dummy.jit",
            frame_duration=0.032,
        )

        segments = vad.detect_speech(chunk)

        assert len(segments) == 1
        assert segments[0].start_time == 10.0 + 0.16
        assert segments[0].end_time == 10.0 + 0.672


@patch("pathlib.Path.is_relative_to", return_value=False)
def test_silero_vad_invalid_path(mock_is_rel: MagicMock) -> None:
    with pytest.raises(ValueError, match="is not within allowed directories"):
        SileroVADDetector(min_silence_duration_ms=500, model_path="/etc/passwd")


def test_silero_vad_invalid_audio_size(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    chunk = AudioChunk(
        chunk_filepath=str(audio_file), start_time=10.0, end_time=20.0, chunk_index=0
    )

    vad = SileroVADDetector(model_path="dummy.jit")
    vad.model = MagicMock()
    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = 2 * 1024 * 1024 * 1024  # 2GB
        mock_stat.return_value = mock_stat_obj

        with patch.object(vad, "_load_model"), pytest.raises(ValueError, match="exceeds maximum allowed size"):
                vad.detect_speech(chunk)


def test_silero_vad_invalid_audio_format(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    chunk = AudioChunk(
        chunk_filepath=str(audio_file), start_time=10.0, end_time=20.0, chunk_index=0
    )

    vad = SileroVADDetector(model_path="dummy.jit")
    vad.model = MagicMock()
    with patch.object(vad, "_load_model"), pytest.raises(ValueError, match="Audio format must be .wav"):
            vad.detect_speech(chunk)


def test_silero_vad_hash_mismatch(tmp_path: Path) -> None:
    model_file = tmp_path / "dummy.jit"
    model_file.write_bytes(b"dummy model data")

    vad = SileroVADDetector(model_path=str(model_file), model_hash="invalidhash")
    with pytest.raises(RuntimeError, match="Model checksum mismatch!"):
        vad._load_model()
