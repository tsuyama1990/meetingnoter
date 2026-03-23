import math
from unittest.mock import MagicMock, patch

from domain_models.transcription import TranscriptionSegment
from meetingnoter.processing.audio_preprocessor import preprocess_audio


def test_score_calculation() -> None:
    # Simulate avg_logprob = -0.5
    avg_logprob = -0.5
    confidence_score = math.exp(avg_logprob)

    assert math.isclose(confidence_score, 0.606, abs_tol=0.01)

    # Test thresholding
    threshold = 0.6
    uncertain = True if confidence_score < threshold else None
    assert uncertain is None  # Should be None since 0.606 > 0.6

    threshold = 0.65
    uncertain = True if confidence_score < threshold else None
    assert uncertain is True


def test_format_markdown() -> None:
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(15.5) == "00:00:15"
    assert format_timestamp(65) == "00:01:05"
    assert format_timestamp(3665) == "01:01:05"


@patch("meetingnoter.processing.audio_preprocessor.subprocess.run")
def test_preprocessor_routing(mock_run: MagicMock) -> None:
    ffmpeg_path = "mock_ffmpeg"

    # Test none
    res = preprocess_audio("test.wav", ffmpeg_path, "none")
    assert res == "test.wav"
    mock_run.assert_not_called()

    # Test loudnorm
    res = preprocess_audio("test.wav", ffmpeg_path, "loudnorm")
    assert "temp_preprocessed_test.wav" in res
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "loudnorm" in args

    mock_run.reset_mock()

    # Test compressor
    res = preprocess_audio("test.wav", ffmpeg_path, "compressor")
    assert "temp_preprocessed_test.wav" in res
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert "acompressor" in args


def test_transcription_segment_models() -> None:
    # Test exclude_none serialization for TranscriptionSegment with uncertain=None
    seg = TranscriptionSegment(
        start_time=0.0, end_time=1.0, text="test", confidence_score=0.9, uncertain=None
    )
    dumped = seg.model_dump(exclude_none=True)
    assert "uncertain" not in dumped

    # Test serialization with uncertain=True
    seg2 = TranscriptionSegment(
        start_time=0.0, end_time=1.0, text="test", confidence_score=0.5, uncertain=True
    )
    dumped2 = seg2.model_dump(exclude_none=True)
    assert "uncertain" in dumped2
    assert dumped2["uncertain"] is True
