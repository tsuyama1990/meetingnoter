
from domain_models import AudioChunk, SpeechSegment, Transcriber, TranscriptionSegment


class FasterWhisperTranscriber(Transcriber):
    """Concrete implementation of Transcriber using faster-whisper."""

    def __init__(self, model_size: str = "large-v3", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.model = None

    def _load_model(self) -> None:
        if self.model is None:
            try:
                import torch
                from faster_whisper import WhisperModel
            except ImportError as e:
                msg = "Required library 'faster-whisper' or 'torch' is not installed."
                raise ImportError(msg) from e

            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = WhisperModel(self.model_size, device=device, compute_type=self.compute_type)
            except Exception as e:
                msg = f"Failed to load Faster Whisper model: {e}"
                raise RuntimeError(msg) from e

    def transcribe(self, chunk: AudioChunk, speech_segments: list[SpeechSegment]) -> list[TranscriptionSegment]:
        """Transcribes speech using faster-whisper logic, heavily customized for Japanese."""
        self._load_model()

        from pathlib import Path
        if not Path(chunk.chunk_filepath).exists():
            msg = f"Audio chunk file not found: {chunk.chunk_filepath}"
            raise FileNotFoundError(msg)

        if self.model:
            import typing
            try:
                # Based on the ARCHITECTURE SPEC, we must override thresholds for Japanese
                segments: typing.Iterable[typing.Any]
                info: typing.Any
                segments, info = self.model.transcribe(
                    chunk.chunk_filepath,
                    language="ja",
                    vad_filter=True,
                    condition_on_previous_text=False, # Disable to prevent hallucination loops
                    compression_ratio_threshold=None, # None to prevent dropping valid Japanese speech
                    log_prob_threshold=None, # None to prevent dropping low confidence speech
                    no_speech_threshold=None,
                    temperature=[0.0, 0.2]
                )

                result = []
                for segment in segments:
                    # Convert local chunk timestamps to global timestamps
                    start_sec = chunk.start_time + segment.start
                    end_sec = chunk.start_time + segment.end

                    if start_sec < end_sec:
                        result.append(
                            TranscriptionSegment(
                                start_time=start_sec,
                                end_time=end_sec,
                                text=segment.text.strip()
                            )
                        )
            except Exception as e:
                msg = f"Faster whisper transcription failed: {e}"
                raise RuntimeError(msg) from e
            else:
                return result

        msg = "Faster Whisper model was not properly loaded."
        raise RuntimeError(msg)
