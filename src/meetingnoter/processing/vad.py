from domain_models import AudioChunk, SpeechDetector, SpeechSegment


class SileroVADDetector(SpeechDetector):
    """Concrete implementation of SpeechDetector using Silero VAD."""

    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 1000) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.model = None
        self.utils = None

    def _load_model(self) -> None:
        if self.model is None:
            try:
                import torch
            except ImportError as e:
                msg = "Required library 'torch' is not installed."
                raise ImportError(msg) from e
            try:
                # Load silero VAD securely
                model, utils = torch.hub.load( # type: ignore[no-untyped-call]
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    trust_repo=True # Security: Explicitly acknowledging trust for this specific official repo
                )
                self.model = model
                self.utils = utils
            except Exception as e:
                msg = f"Failed to load Silero VAD model: {e}"
                raise RuntimeError(msg) from e

    def detect_speech(self, chunk: AudioChunk) -> list[SpeechSegment]:
        """Detects speech segments using Silero VAD logic."""
        self._load_model()

        if self.model and self.utils:
            try:
                (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = self.utils

                # Load audio using the utils
                wav = read_audio(chunk.chunk_filepath, sampling_rate=16000)

                # Get timestamps from model
                speech_timestamps = get_speech_timestamps(
                    wav,
                    self.model,
                    sampling_rate=16000,
                    threshold=self.threshold,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    min_silence_duration_ms=self.min_silence_duration_ms
                )

                segments = []
                for ts in speech_timestamps:
                    # Convert samples to seconds and add global chunk offset
                    start_sec = chunk.start_time + (ts['start'] / 16000.0)
                    end_sec = chunk.start_time + (ts['end'] / 16000.0)
                    if start_sec < end_sec:
                        segments.append(SpeechSegment(start_time=start_sec, end_time=end_sec))
                return segments
            except Exception as e:
                msg = f"Silero VAD processing failed: {e}"
                raise RuntimeError(msg) from e

        msg = "Silero VAD model was not properly loaded."
        raise RuntimeError(msg)
