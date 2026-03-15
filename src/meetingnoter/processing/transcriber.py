import typing
from pathlib import Path

from domain_models import AudioChunk, SpeechSegment, Transcriber, TranscriptionSegment

try:
    import torch
    from faster_whisper import WhisperModel
except ImportError as e:
    msg = "Required library 'faster-whisper' or 'torch' is not installed."
    raise ImportError(msg) from e


class FasterWhisperTranscriber(Transcriber):
    """Concrete implementation of Transcriber using faster-whisper."""

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str = "int8",
        language: str = "ja",
        vad_filter: bool = True,
        condition_on_previous_text: bool = False,
        temperature: list[float] | None = None,
    ) -> None:
        if temperature is None:
            temperature = [0.0, 0.2]
        self.model_size = model_size
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text
        self.temperature = temperature
        self.model = None

    def _load_model(self) -> None:
        if self.model is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = WhisperModel(
                    self.model_size, device=device, compute_type=self.compute_type
                )
            except Exception as e:
                msg = f"Failed to load Faster Whisper model: {e}"
                raise RuntimeError(msg) from e

    def transcribe(
        self, chunk: AudioChunk, speech_segments: list[SpeechSegment]
    ) -> list[TranscriptionSegment]:
        """Transcribes speech using faster-whisper logic, heavily customized for Japanese."""
        if not Path(chunk.chunk_filepath).exists():
            msg = f"Audio chunk file not found: {chunk.chunk_filepath}"
            raise FileNotFoundError(msg)

        self._load_model()

        if self.model:
            try:
                # Based on the ARCHITECTURE SPEC, we must override thresholds for Japanese
                segments: typing.Iterable[typing.Any]
                info: typing.Any
                segments, info = self.model.transcribe(
                    audio=chunk.chunk_filepath,
                    language=self.language,
                    vad_filter=self.vad_filter,
                    condition_on_previous_text=self.condition_on_previous_text,
                    temperature=self.temperature,
                    compression_ratio_threshold=None,
                    log_prob_threshold=None,
                    no_speech_threshold=None,
                )

                result: list[TranscriptionSegment] = []
                for segment in segments:
                    # Convert local chunk timestamps to global timestamps
                    start_sec: float = chunk.start_time + segment.start
                    end_sec: float = chunk.start_time + segment.end

                    if start_sec < end_sec:
                        result.append(
                            TranscriptionSegment(
                                start_time=start_sec,
                                end_time=end_sec,
                                text=str(segment.text.strip()),
                            )
                        )
            except Exception as e:
                msg = f"Faster whisper transcription failed: {e}"
                raise RuntimeError(msg) from e
            finally:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return result

        msg = "Faster Whisper model was not properly loaded."
        raise RuntimeError(msg)
