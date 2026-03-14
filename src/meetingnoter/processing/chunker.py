import math
import subprocess
import tempfile

from domain_models import AudioChunk, AudioSource, AudioSplitter


class FFmpegChunker(AudioSplitter):
    """Concrete implementation of AudioSplitter using FFmpeg."""

    def __init__(self, chunk_length_minutes: int = 20) -> None:
        self.chunk_length_seconds = chunk_length_minutes * 60

    def split(self, source: AudioSource) -> list[AudioChunk]:
        """Physically splits an audio file using FFmpeg into manageable chunks."""
        # Calculate number of chunks
        num_chunks: int = math.ceil(source.duration_seconds / self.chunk_length_seconds)

        import shutil
        ffmpeg_path: str | None = shutil.which("ffmpeg")
        if not ffmpeg_path:
            msg = "FFmpeg is not installed or not found in system PATH."
            raise RuntimeError(msg)

        try:
            if num_chunks <= 1:
                # If the audio is shorter than the chunk length, return as one chunk.
                # Using ffmpeg to copy and guarantee format.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
                    pass
                subprocess.run( # noqa: S603
                    [
                        ffmpeg_path, "-y", "-i", source.filepath,
                        "-ac", "1", "-ar", "16000",
                        chunk_file.name
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                from pathlib import Path
                if Path(chunk_file.name).stat().st_size == 0:
                    msg = "FFmpeg produced an empty file"
                    raise RuntimeError(msg)

                return [AudioChunk(
                    chunk_filepath=chunk_file.name,
                    start_time=0.0,
                    end_time=source.duration_seconds,
                    chunk_index=0
                )]

            chunks: list[AudioChunk] = []
            from pathlib import Path
            for i in range(num_chunks):
                start_time: float = i * self.chunk_length_seconds
                end_time: float = min((i + 1) * self.chunk_length_seconds, source.duration_seconds)
                duration: float = end_time - start_time

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
                    pass

                # Execute real ffmpeg split
                subprocess.run( # noqa: S603
                    [
                        ffmpeg_path, "-y", "-i", source.filepath,
                        "-ss", str(start_time),
                        "-t", str(duration),
                        "-ac", "1", "-ar", "16000",
                        chunk_file.name
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )

                if Path(chunk_file.name).stat().st_size > 0:
                    chunks.append(
                        AudioChunk(
                            chunk_filepath=chunk_file.name,
                            start_time=start_time,
                            end_time=end_time,
                            chunk_index=i
                        )
                    )
                else:
                    msg = f"FFmpeg chunk {i} is empty."
                    raise RuntimeError(msg)
        except FileNotFoundError as e:
            msg = "FFmpeg is not installed or not found in system PATH."
            raise RuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            msg = f"FFmpeg chunking failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return chunks
