import gc
import typing
from pathlib import Path

from domain_models import (
    AudioChunk,
    PipelineConfig,
    SpeechSegment,
    Transcriber,
    TranscriptionSegment,
)

try:
    import torch
    from faster_whisper import WhisperModel
except ImportError as e:
    msg = f"Required library 'faster-whisper' or 'torch' is not installed: {e}"
    raise ImportError(msg) from e


class FasterWhisperTranscriber(Transcriber):
    """Concrete implementation of Transcriber using faster-whisper."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.model: WhisperModel | None = None

    def _cleanup_resources(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _validate_audio_file(self, file_path: Path) -> None:
        path = file_path.resolve()

        if path.is_symlink():
            msg = f"Audio path {path} is a symlink, which is not permitted."
            raise ValueError(msg)

        if not path.is_file():
            msg = f"Audio chunk file not found: {path}"
            raise FileNotFoundError(msg)

        allowed_dirs = [Path.cwd(), Path(__import__("tempfile").gettempdir())]
        if not any(path.is_relative_to(d) for d in allowed_dirs):
            msg = f"Audio path {path} is not within allowed directories."
            raise ValueError(msg)

    def _load_model(self) -> None:
        if self.model is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = WhisperModel(
                    self.config.transcriber_model_size,
                    device=device,
                    compute_type=self.config.transcriber_compute_type,
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    msg = "CUDA Out of Memory when trying to load Faster Whisper model."
                    raise RuntimeError(msg) from e
                msg = f"Failed to load Faster Whisper model: {e}"
                raise RuntimeError(msg) from e
            except Exception as e:
                msg = f"Failed to load Faster Whisper model: {e}"
                raise RuntimeError(msg) from e

    def transcribe(
        self, chunk: AudioChunk, speech_segments: list[SpeechSegment]
    ) -> list[TranscriptionSegment]:
        """Transcribes speech using faster-whisper logic, heavily customized for Japanese."""
        audio_path = Path(chunk.chunk_filepath)
        self._validate_audio_file(audio_path)

        self._load_model()

        if self.model:
            try:
                # Based on the ARCHITECTURE SPEC, we must override thresholds for Japanese
                segments: typing.Iterable[typing.Any]
                info: typing.Any
                segments, info = self.model.transcribe(
                    audio=str(audio_path.resolve()),
                    language=self.config.transcriber_language,
                    vad_filter=self.config.transcriber_vad_filter,
                    condition_on_previous_text=self.config.transcriber_condition_on_previous_text,
                    temperature=list(self.config.transcriber_temperature),
                    compression_ratio_threshold=None,
                    log_prob_threshold=None,
                    no_speech_threshold=None,
                )

                result: list[TranscriptionSegment] = []
                for segment in segments:
                    # Convert local chunk timestamps to global timestamps
                    start_sec: float = chunk.start_time + float(segment.start)
                    end_sec: float = chunk.start_time + float(segment.end)

                    if start_sec < end_sec:
                        result.append(
                            TranscriptionSegment(
                                start_time=start_sec,
                                end_time=end_sec,
                                text=str(segment.text.strip()),
                            )
                        )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    msg = "CUDA Out of Memory during transcription."
                    raise RuntimeError(msg) from e
                msg = f"Faster whisper transcription failed: {e}"
                raise RuntimeError(msg) from e
            except Exception as e:
                msg = f"Faster whisper transcription failed: {e}"
                raise RuntimeError(msg) from e
            finally:
                self._cleanup_resources()

            return result

        msg = "Faster Whisper model was not properly loaded."
        raise RuntimeError(msg)
