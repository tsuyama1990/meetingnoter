import tempfile
from unittest.mock import MagicMock, patch

import pytest

from domain_models import AudioSource


@patch("subprocess.run")
def test_ffmpeg_chunker_single_chunk(mock_run: MagicMock) -> None:
    from meetingnoter.processing.chunker import FFmpegChunker

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        pass

    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = 1024
        # Return a mode integer mimicking a regular file so is_file() passes
        mock_stat_obj.st_mode = 33188
        mock_stat.return_value = mock_stat_obj

        chunker = FFmpegChunker(chunk_length_minutes=20)
        source = AudioSource(filepath=tf.name, duration_seconds=600.0) # 10 mins
        chunks = chunker.split(source)

        assert len(chunks) == 1
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 600.0
        assert chunks[0].chunk_index == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "-c" in args
        assert "copy" in args

@patch("subprocess.run")
def test_ffmpeg_chunker_multiple_chunks(mock_run: MagicMock) -> None:
    from meetingnoter.processing.chunker import FFmpegChunker

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        pass

    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = 1024
        mock_stat_obj.st_mode = 33188
        mock_stat.return_value = mock_stat_obj

        chunker = FFmpegChunker(chunk_length_minutes=20)
        source = AudioSource(filepath=tf.name, duration_seconds=3000.0) # 50 mins
        chunks = chunker.split(source)

        assert len(chunks) == 3
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 1200.0
        assert chunks[1].start_time == 1200.0
        assert chunks[1].end_time == 2400.0
        assert chunks[2].start_time == 2400.0
        assert chunks[2].end_time == 3000.0
        assert mock_run.call_count == 3

@patch("subprocess.run")
def test_ffmpeg_chunker_missing_ffmpeg(mock_run: MagicMock) -> None:

    from meetingnoter.processing.chunker import FFmpegChunker
    mock_run.side_effect = FileNotFoundError("ffmpeg")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        pass

    chunker = FFmpegChunker(ffmpeg_path="nonexistent", chunk_length_minutes=20)
    source = AudioSource(filepath=tf.name, duration_seconds=600.0)

    with pytest.raises(RuntimeError, match="FFmpeg is not installed or not found in system PATH."):
        chunker.split(source)

@patch("subprocess.run")
def test_ffmpeg_chunker_subprocess_error(mock_run: MagicMock) -> None:
    import subprocess

    from meetingnoter.processing.chunker import FFmpegChunker
    mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        pass

    chunker = FFmpegChunker(chunk_length_minutes=20)
    source = AudioSource(filepath=tf.name, duration_seconds=1500.0) # multiple chunks

    with pytest.raises(RuntimeError, match="FFmpeg chunking failed"):
        chunker.split(source)

@patch("subprocess.run")
def test_ffmpeg_chunker_empty_output_multiple_chunks(mock_run: MagicMock) -> None:
    from meetingnoter.processing.chunker import FFmpegChunker

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        pass

    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = 0
        mock_stat_obj.st_mode = 33188
        mock_stat.return_value = mock_stat_obj

        chunker = FFmpegChunker(chunk_length_minutes=20)
        source = AudioSource(filepath=tf.name, duration_seconds=1500.0)

        with pytest.raises(RuntimeError, match="FFmpeg chunk 0 is empty"):
            chunker.split(source)

@patch("subprocess.run")
def test_ffmpeg_chunker_empty_output_single_chunk(mock_run: MagicMock) -> None:
    from meetingnoter.processing.chunker import FFmpegChunker

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        pass

    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = 0
        mock_stat_obj.st_mode = 33188
        mock_stat.return_value = mock_stat_obj

        chunker = FFmpegChunker(chunk_length_minutes=20)
        source = AudioSource(filepath=tf.name, duration_seconds=600.0)

        with pytest.raises(RuntimeError, match="FFmpeg chunk 0 is empty"):
            chunker.split(source)
