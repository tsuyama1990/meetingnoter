import math
import os
import subprocess
import tempfile
from pathlib import Path

from domain_models import AudioChunk, AudioSource, AudioSplitter


class FFmpegChunker(AudioSplitter):
    """Concrete implementation of AudioSplitter using FFmpeg."""

    ffmpeg_path: str
    chunk_length_seconds: int

    def __init__(self, ffmpeg_path: str = "ffmpeg", chunk_length_minutes: int = 20) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.chunk_length_seconds = chunk_length_minutes * 60

    def split(self, source: AudioSource) -> list[AudioChunk]:
        """Physically splits an audio file using FFmpeg into manageable chunks."""
        # Validate path
        source_path = Path(source.filepath)
        if not source_path.is_file():
            msg = f"Audio source file does not exist or is not a file: {source.filepath}"
            raise ValueError(msg)

        # Calculate number of chunks
        num_chunks: int = math.ceil(source.duration_seconds / self.chunk_length_seconds)

        chunks: list[AudioChunk] = []
        try:
            for i in range(num_chunks):
                start_time: float = i * self.chunk_length_seconds
                end_time: float = min((i + 1) * self.chunk_length_seconds, source.duration_seconds)
                duration: float = end_time - start_time
                chunks.append(self._process_chunk(source_path, i, start_time, end_time, duration))
        except Exception:
            # Cleanup any successfully processed chunks before failing
            for chunk in chunks:
                Path(chunk.chunk_filepath).unlink(missing_ok=True)
            raise

        return chunks

    def _process_chunk(
        self, source_path: Path, index: int, start_time: float, end_time: float, duration: float
    ) -> AudioChunk:
        """Processes a single audio chunk via FFmpeg."""
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-i",
                str(source_path),
            ]
            if start_time > 0 or duration < end_time:
                cmd.extend(["-ss", str(start_time), "-t", str(duration)])
            cmd.extend(["-c", "copy", temp_path])

            subprocess.run(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            try:
                if Path(temp_path).stat().st_size == 0:
                    msg = f"FFmpeg chunk {index} is empty."
                    raise RuntimeError(msg)
            except (FileNotFoundError, PermissionError) as e:
                msg = "Failed to stat chunk file."
                raise RuntimeError(msg) from e

            return AudioChunk(
                chunk_filepath=temp_path,
                start_time=start_time,
                end_time=end_time,
                chunk_index=index,
            )
        except FileNotFoundError as e:
            Path(temp_path).unlink(missing_ok=True)
            msg = "FFmpeg is not installed or not found in system PATH."
            raise RuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            Path(temp_path).unlink(missing_ok=True)
            msg = f"FFmpeg chunking failed: {e}"
            raise RuntimeError(msg) from e
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            raise
