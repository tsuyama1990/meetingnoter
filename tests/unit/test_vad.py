from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from domain_models import AudioChunk
from meetingnoter.processing.vad import SileroVADDetector


def test_silero_vad_init_validation(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy.jit"
    model_path.touch()

    vad = SileroVADDetector(min_silence_duration_ms=500, model_path=str(model_path))
    assert vad.config.min_silence_duration_ms == 500

    vad = SileroVADDetector(min_silence_duration_ms=1000, model_path=str(model_path))
    assert vad.config.min_silence_duration_ms == 1000

    with pytest.raises(ValueError, match="min_silence_duration_ms"):
        SileroVADDetector(min_silence_duration_ms=2000, model_path=str(model_path))

    with pytest.raises(ValueError, match="min_silence_duration_ms"):
        SileroVADDetector(min_silence_duration_ms=499, model_path=str(model_path))


@patch("pathlib.Path.exists", return_value=True)
@patch("torch.jit.load")
@patch("torchaudio.load")
def test_silero_vad_detect_speech(
    mock_ta_load: MagicMock,
    mock_jit_load: MagicMock,
    mock_exists: MagicMock,
    tmp_path: Path,
) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    model_path = tmp_path / "dummy.jit"
    model_path.touch()

    chunk = AudioChunk(
        chunk_filepath=str(audio_file), start_time=10.0, end_time=20.0, chunk_index=0
    )

    mock_wav = torch.zeros(1, 16000)
    mock_ta_load.return_value = (mock_wav, 16000)

    mock_model = MagicMock()
    mock_jit_load.return_value = mock_model

    probs = torch.zeros(21)
    probs[5:21] = 0.9

    mock_out = MagicMock()
    mock_out.squeeze.return_value = probs
    mock_model.return_value = mock_out

    # Disable hash checking for this test
    with patch.object(SileroVADDetector, "_verify_and_load_model"):
        vad = SileroVADDetector(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            model_path=str(model_path),
            frame_duration=0.032,
        )
        vad.model = mock_model

        # Mock validate_audio_file to pass without actually parsing WAV header
        with patch.object(vad, "_validate_audio_file"):
            segments = vad.detect_speech(chunk)

        assert len(segments) == 1
        assert segments[0].start_time == 10.0 + 0.16
        assert segments[0].end_time == 10.0 + 0.672


def test_silero_vad_invalid_path(tmp_path: Path) -> None:
    model_path = Path("/etc/passwd")
    with pytest.raises(ValueError, match="has invalid extension"):
        SileroVADDetector(min_silence_duration_ms=500, model_path=str(model_path))

    # Fake an existing path check for testing directory containment violation
    with patch("pathlib.Path.is_file", return_value=True):
        model_path2 = Path("/etc/passwd.jit")
        with pytest.raises(ValueError, match="is not within allowed directories"):
            SileroVADDetector(min_silence_duration_ms=500, model_path=str(model_path2))


def test_silero_vad_invalid_audio_size(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    chunk = AudioChunk(
        chunk_filepath=str(audio_file), start_time=10.0, end_time=20.0, chunk_index=0
    )

    model_path = tmp_path / "dummy.jit"
    model_path.touch()
    vad = SileroVADDetector(model_path=str(model_path))
    vad.model = MagicMock()
    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = 2 * 1024 * 1024 * 1024  # 2GB
        mock_stat.return_value = mock_stat_obj

        with (
            patch.object(vad, "_verify_and_load_model"),
            pytest.raises(ValueError, match="exceeds maximum allowed size"),
        ):
            vad.detect_speech(chunk)


def test_silero_vad_invalid_audio_format(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.mp3"
    audio_file.touch()

    chunk = AudioChunk(
        chunk_filepath=str(audio_file), start_time=10.0, end_time=20.0, chunk_index=0
    )

    model_path = tmp_path / "dummy.jit"
    model_path.touch()
    vad = SileroVADDetector(model_path=str(model_path))
    vad.model = MagicMock()
    with (
        patch.object(vad, "_verify_and_load_model"),
        pytest.raises(ValueError, match="Audio format must be .wav"),
    ):
        vad.detect_speech(chunk)


def test_silero_vad_hash_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy.jit"
    model_path.write_bytes(b"dummy model data")

    vad = SileroVADDetector(
        model_path=str(model_path),
        model_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    with pytest.raises(RuntimeError, match="Model checksum mismatch!"):
        vad._verify_and_load_model()
