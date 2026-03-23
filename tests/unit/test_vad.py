from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from domain_models import AudioChunk
from meetingnoter.processing.vad import SileroVADDetector


def test_silero_vad_init_validation(tmp_path: Path) -> None:
    vad = SileroVADDetector(min_silence_duration_ms=500)
    assert vad.config.min_silence_duration_ms == 500

    vad = SileroVADDetector(min_silence_duration_ms=1000)
    assert vad.config.min_silence_duration_ms == 1000

    with pytest.raises(ValueError, match="min_silence_duration_ms"):
        SileroVADDetector(min_silence_duration_ms=2000)

    with pytest.raises(ValueError, match="min_silence_duration_ms"):
        SileroVADDetector(min_silence_duration_ms=499)


@patch("pathlib.Path.exists", return_value=True)
@patch("torch.hub.load")
@patch("torchaudio.load")
def test_silero_vad_detect_speech(
    mock_ta_load: MagicMock,
    mock_hub_load: MagicMock,
    mock_exists: MagicMock,
    tmp_path: Path,
) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    chunk = AudioChunk(
        chunk_filepath=str(audio_file), start_time=10.0, end_time=20.0, chunk_index=0
    )

    # 1 second of audio
    mock_wav = torch.zeros(1, 16000)
    mock_ta_load.return_value = (mock_wav, 16000)

    class MockModel:
        def __init__(self) -> None:
            self.call_count = 0

        def reset_states(self) -> None:
            pass

        def __call__(self, x: torch.Tensor, sr: int | None = None) -> torch.Tensor:
            if x.shape[-1] != 512:
                msg = f"Input tensor shape must be (..., 512), got {x.shape}"
                raise ValueError(msg)
            if sr != 16000:
                msg = "Missing or incorrect 'sr' argument. Expected 16000."
                raise ValueError(msg)
            self.call_count += 1

            # 16000 / 512 = 31.25 -> 32 calls
            # Let's say we output speech around call 5 to 21
            if 5 <= self.call_count <= 21:
                return torch.tensor(0.9)
            return torch.tensor(0.1)

    mock_model = MockModel()
    mock_hub_load.return_value = (mock_model, None)

    with patch.object(SileroVADDetector, "_verify_and_load_model"):
        vad = SileroVADDetector(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            frame_duration=0.032,
        )
        vad.model = mock_model

        # Mock validate_audio_file to pass without actually parsing WAV header
        with patch.object(vad, "_validate_audio_file"):
            segments = vad.detect_speech(chunk)

        # 16000 / 512 = 32 chunks
        assert mock_model.call_count == 32
        assert len(segments) == 1
        # It's an approximation of start time:
        assert segments[0].start_time == 10.0 + 4 * 0.032
        # End is idx 21 * 0.032
        assert segments[0].end_time == 10.0 + 21 * 0.032


@patch("pathlib.Path.exists", return_value=True)
@patch("torch.hub.load")
@patch("torchaudio.load")
def test_silero_vad_detect_speech_long(
    mock_ta_load: MagicMock,
    mock_hub_load: MagicMock,
    mock_exists: MagicMock,
    tmp_path: Path,
) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    chunk = AudioChunk(chunk_filepath=str(audio_file), start_time=0.0, end_time=3.0, chunk_index=0)

    # 3 seconds of audio
    mock_wav = torch.zeros(1, 16000 * 3)
    mock_ta_load.return_value = (mock_wav, 16000)

    class MockModelLong:
        def __init__(self) -> None:
            self.call_count = 0

        def reset_states(self) -> None:
            pass

        def __call__(self, x: torch.Tensor, sr: int | None = None) -> torch.Tensor:
            if x.shape[-1] != 512:
                msg = f"Input tensor shape must be (..., 512), got {x.shape}"
                raise ValueError(msg)
            if sr != 16000:
                msg = "Missing or incorrect 'sr' argument. Expected 16000."
                raise ValueError(msg)
            self.call_count += 1
            return torch.tensor(0.1)

    mock_model = MockModelLong()
    mock_hub_load.return_value = (mock_model, None)

    with patch.object(SileroVADDetector, "_verify_and_load_model"):
        vad = SileroVADDetector(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            frame_duration=0.032,
        )
        vad.model = mock_model

        # Mock validate_audio_file to pass without actually parsing WAV header
        with patch.object(vad, "_validate_audio_file"):
            segments = vad.detect_speech(chunk)

        # 48000 / 512 = 93.75 -> 94 calls
        assert mock_model.call_count == 94
        assert len(segments) == 0


def test_silero_vad_invalid_audio_size(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.touch()

    chunk = AudioChunk(
        chunk_filepath=str(audio_file), start_time=10.0, end_time=20.0, chunk_index=0
    )

    vad = SileroVADDetector()
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

    vad = SileroVADDetector()
    vad.model = MagicMock()
    with (
        patch.object(vad, "_verify_and_load_model"),
        pytest.raises(ValueError, match="Audio format must be .wav"),
    ):
        vad.detect_speech(chunk)
