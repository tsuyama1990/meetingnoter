import hashlib
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

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
    fallback_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence for fallback behavior."
    )
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(
        default=1000, ge=500, le=1000, description="Strict Japanese nuances."
    )
    model_path: str = Field(..., min_length=1)
    model_hash: str | None = Field(default=None, description="SHA256 checksum for model integrity.")
    frame_duration: float = Field(default=0.032, gt=0.0)

    # Architectural and Security Limits
    max_model_size_bytes: int = Field(default=100 * 1024 * 1024)  # 100MB
    max_audio_size_bytes: int = Field(default=1024 * 1024 * 1024)  # 1GB
    max_audio_duration_seconds: int = Field(default=3600)  # 1 Hour
    target_sample_rate: int = Field(default=16000)

    @field_validator("model_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        path = Path(v).resolve()

        # Must exist and be a file
        if not path.is_file():
            msg = f"Model path {path} must be an existing file."
            raise ValueError(msg)

        # Must have valid extension
        if path.suffix.lower() not in [".jit", ".pt"]:
            msg = f"Model path {path} has invalid extension."
            raise ValueError(msg)

        allowed_dirs = [Path.cwd(), Path(__import__("tempfile").gettempdir())]
        for d in allowed_dirs:
            if path.is_relative_to(d):
                return str(path)
        msg = f"Path {path} is not within allowed directories: {allowed_dirs}"
        raise ValueError(msg)

    @field_validator("model_hash")
    @classmethod
    def validate_hash(cls, v: str | None) -> str | None:
        if v is not None and not re.match(r"^[a-fA-F0-9]{64}$", v):
            msg = "Model hash must be a valid SHA256 hex digest."
            raise ValueError(msg)
        return v


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

        # Immediately validate the configuration via Pydantic model creation
        self.config = VADConfig(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            model_path=model_path,
            model_hash=model_hash,
            frame_duration=frame_duration,
        )
        self.model: Any = None

    def _verify_and_load_model(self) -> None:
        if self.model is not None:
            return

        model_file = Path(self.config.model_path)

        # Verify size limit to prevent DoS on hashing
        if model_file.stat().st_size > self.config.max_model_size_bytes:
            msg = f"Model file exceeds {self.config.max_model_size_bytes} bytes."
            raise ValueError(msg)

        # Open file securely and hold lock/handle while verifying and loading to prevent TOCTOU
        with model_file.open("rb") as f:
            if self.config.model_hash:
                sha256 = hashlib.sha256()
                # Streaming hash
                for block in iter(lambda: f.read(8192), b""):
                    sha256.update(block)

                computed = sha256.hexdigest()
                if computed != self.config.model_hash:
                    msg = f"Model checksum mismatch! Expected {self.config.model_hash}, got {computed}"
                    raise RuntimeError(msg)

                # Reset file pointer to beginning for loading
                f.seek(0)

            try:
                # Load model directly from the secure file handle
                self.model = torch.jit.load(f)  # type: ignore[no-untyped-call]
            except Exception as e:
                msg = f"Failed to securely load Silero VAD model: {e}"
                raise RuntimeError(msg) from e

    def _merge_and_filter_chunks(
        self, temp_speech_chunks: list[tuple[Decimal, Decimal]]
    ) -> list[tuple[Decimal, Decimal]]:
        merged_chunks: list[tuple[Decimal, Decimal]] = []
        for start, end in temp_speech_chunks:
            if not merged_chunks:
                merged_chunks.append((start, end))
            else:
                prev_start, prev_end = merged_chunks[-1]
                if (start - prev_end) * Decimal("1000") < Decimal(
                    self.config.min_silence_duration_ms
                ):
                    merged_chunks[-1] = (prev_start, end)
                else:
                    merged_chunks.append((start, end))

        return [
            (s, e)
            for s, e in merged_chunks
            if (e - s) * Decimal("1000") >= Decimal(self.config.min_speech_duration_ms)
        ]

    def _parse_probabilities(self, probs: torch.Tensor, chunk: AudioChunk) -> list[SpeechSegment]:
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

        if len(segments) == 0 and float(probs.mean().item()) > self.config.fallback_threshold:
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

        if file_path.stat().st_size > self.config.max_audio_size_bytes:
            msg = f"Audio file {file_path} exceeds maximum allowed size of {self.config.max_audio_size_bytes}."
            raise ValueError(msg)

        import wave

        try:
            with wave.open(str(file_path), "rb") as w:
                if w.getnchannels() < 1 or w.getframerate() <= 0:
                    msg = "Invalid WAV header metadata."
                    raise ValueError(msg)
        except wave.Error as e:
            msg = "Corrupted or invalid WAV file content."
            raise ValueError(msg) from e

    def _load_and_sanitize_audio(self, audio_path: Path) -> torch.Tensor:
        try:
            with audio_path.open("rb") as f:
                wav, sr = torchaudio.load(f)
                if sr != self.config.target_sample_rate:
                    wav = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.config.target_sample_rate
                    )(wav)
                wav = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav
        except Exception as e:
            msg = f"Failed to securely load or resample audio file: {e}"
            raise RuntimeError(msg) from e

        if not isinstance(wav, torch.Tensor):
            msg = "Expected torchaudio.load to return a torch.Tensor"
            raise TypeError(msg)

        if wav.dim() != 2:
            msg = f"Expected 2D audio tensor [channels, frames], got {wav.dim()}D"
            raise RuntimeError(msg)

        if wav.shape[1] > self.config.target_sample_rate * self.config.max_audio_duration_seconds:
            msg = f"Audio chunk exceeds maximum allowed tensor length ({self.config.max_audio_duration_seconds}s)."
            raise RuntimeError(msg)

        if torch.isnan(wav).any() or torch.isinf(wav).any():
            msg = "Audio tensor contains NaN or Inf values."
            raise ValueError(msg)

        return torch.clamp(wav, min=-1.0, max=1.0)

    def detect_speech(self, chunk: AudioChunk) -> list[SpeechSegment]:
        """Detects speech segments using Silero VAD logic."""
        audio_path = Path(chunk.chunk_filepath).resolve()

        self._validate_audio_file(audio_path)
        self._verify_and_load_model()

        if self.model:
            wav = self._load_and_sanitize_audio(audio_path)

            try:
                with torch.no_grad():
                    out = self.model(wav)
                probs = out.squeeze()

                if not isinstance(probs, torch.Tensor):
                    msg = "Model did not return a torch.Tensor"
                    raise TypeError(msg)

                segments = self._parse_probabilities(probs, chunk)
            except TypeError:
                raise
            except Exception as e:
                msg = f"Silero VAD processing failed: {e}"
                raise RuntimeError(msg) from e
            finally:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return segments

        msg = "Silero VAD model was not properly loaded."
        raise RuntimeError(msg)
