import hashlib
import typing
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain_models import AudioChunk, SpeechDetector, SpeechSegment

try:
    import torch
    import torchaudio
except ImportError as e:
    msg = "Required libraries 'torch' and/or 'torchaudio' are not installed."
    raise ImportError(msg) from e


class VADConfig(BaseModel):
    """Secure configuration model for Voice Activity Detection."""
    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(default=1000, ge=500, le=1000, description="Strict Japanese nuances.")
    model_path: str = Field(..., min_length=1)
    model_hash: str | None = Field(default=None, description="SHA256 checksum for model integrity.")
    frame_duration: float = Field(default=0.032, gt=0.0)

    @field_validator("model_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        path = Path(v).resolve()
        allowed_dirs = [Path.cwd(), Path(__import__("tempfile").gettempdir())]
        for d in allowed_dirs:
            if path.is_relative_to(d):
                return str(path)
        msg = f"Path {path} is not within allowed directories: {allowed_dirs}"
        raise ValueError(msg)


class SileroVADDetector(SpeechDetector):
    """Concrete implementation of SpeechDetector using Silero VAD."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 1000,
        model_path: str | None = None,
        model_hash: str | None = None,
        frame_duration: float = 0.032,
    ) -> None:
        if not model_path:
            msg = "A valid 'model_path' to a local Silero VAD model must be provided."
            raise ValueError(msg)

        self.config = VADConfig(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            model_path=model_path,
            model_hash=model_hash,
            frame_duration=frame_duration,
        )
        self.model: typing.Any = None

    def _verify_model_integrity(self, path: Path) -> None:
        if not self.config.model_hash:
            return

        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)

        computed = sha256.hexdigest()
        if computed != self.config.model_hash:
            msg = f"Model checksum mismatch! Expected {self.config.model_hash}, got {computed}"
            raise RuntimeError(msg)

    def _load_model(self) -> None:
        if self.model is None:
            model_file = Path(self.config.model_path)

            if not model_file.exists():
                msg = f"Silero VAD model not found at configured path {self.config.model_path}."
                raise FileNotFoundError(msg)

            self._verify_model_integrity(model_file)

            try:
                self.model = torch.jit.load(self.config.model_path)  # type: ignore[no-untyped-call]
            except Exception as e:
                msg = f"Failed to securely load Silero VAD model from local cache: {e}"
                raise RuntimeError(msg) from e

    def _merge_and_filter_chunks(self, temp_speech_chunks: list[tuple[Decimal, Decimal]]) -> list[tuple[Decimal, Decimal]]:
        merged_chunks: list[tuple[Decimal, Decimal]] = []
        for start, end in temp_speech_chunks:
            if not merged_chunks:
                merged_chunks.append((start, end))
            else:
                prev_start, prev_end = merged_chunks[-1]
                if (start - prev_end) * Decimal("1000") < Decimal(self.config.min_silence_duration_ms):
                    merged_chunks[-1] = (prev_start, end)
                else:
                    merged_chunks.append((start, end))

        return [
            (s, e) for s, e in merged_chunks if (e - s) * Decimal("1000") >= Decimal(self.config.min_speech_duration_ms)
        ]

    def _parse_probabilities(self, probs: typing.Any, chunk: AudioChunk) -> list[SpeechSegment]:
        segments: list[SpeechSegment] = []
        temp_speech_chunks: list[tuple[Decimal, Decimal]] = []
        is_speech: bool = False
        speech_start: Decimal = Decimal("0.0")

        frame_dur = Decimal(str(self.config.frame_duration))

        for idx in range(len(probs)):
            prob: float = float(probs[idx].item())
            time_sec: Decimal = Decimal(idx) * frame_dur

            if prob >= self.config.threshold and not is_speech:
                is_speech = True
                speech_start = time_sec
            elif prob < self.config.threshold and is_speech:
                is_speech = False
                temp_speech_chunks.append((speech_start, time_sec))

        if is_speech:
            temp_speech_chunks.append((speech_start, Decimal(len(probs)) * frame_dur))

        final_chunks = self._merge_and_filter_chunks(temp_speech_chunks)

        for start, end in final_chunks:
            global_start: float = chunk.start_time + float(start)
            global_end: float = chunk.start_time + float(end)
            global_end = min(global_end, chunk.end_time)

            if global_start < global_end:
                segments.append(SpeechSegment(start_time=global_start, end_time=global_end))

        if len(segments) == 0 and float(probs.mean().item()) > self.config.threshold:
            segments.append(SpeechSegment(start_time=chunk.start_time, end_time=chunk.end_time))

        return segments

    def _validate_audio_file(self, file_path: Path) -> None:
        allowed_dirs = [Path.cwd(), Path(__import__("tempfile").gettempdir())]
        if not any(file_path.is_relative_to(d) for d in allowed_dirs):
            msg = f"Audio path {file_path} is not within allowed directories."
            raise ValueError(msg)

        if file_path.suffix.lower() != ".wav":
            msg = f"Audio format must be .wav, got {file_path.suffix}"
            raise ValueError(msg)

        if file_path.stat().st_size > 1024 * 1024 * 1024: # 1GB
            msg = f"Audio file {file_path} exceeds maximum allowed size of 1GB."
            raise ValueError(msg)

    def detect_speech(self, chunk: AudioChunk) -> list[SpeechSegment]:
        """Detects speech segments using Silero VAD logic."""
        self._load_model()

        if self.model:
            audio_path = Path(chunk.chunk_filepath).resolve()
            self._validate_audio_file(audio_path)

            try:
                wav, sr = torchaudio.load(str(audio_path))
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
                wav = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav
            except Exception as e:
                msg = f"Failed to securely load or resample audio file: {e}"
                raise RuntimeError(msg) from e

            # Prevent OOM/DoS by asserting waveform size limits
            if wav.shape[1] > 16000 * 3600: # Max 1 hour of audio per chunk
                msg = "Audio chunk exceeds maximum allowed tensor length (1 hour)."
                raise RuntimeError(msg)

            try:
                with torch.no_grad():
                    out = self.model(wav)
                probs = out.squeeze()
                segments = self._parse_probabilities(probs, chunk)
            except Exception as e:
                msg = f"Silero VAD processing failed: {e}"
                raise RuntimeError(msg) from e
            else:
                return segments

        msg = "Silero VAD model was not properly loaded."
        raise RuntimeError(msg)
