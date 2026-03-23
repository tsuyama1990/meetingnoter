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
    frame_duration: float = Field(default=0.032, gt=0.0)

    # Architectural and Security Limits
    max_audio_size_bytes: int = Field(default=1024 * 1024 * 1024)  # 1GB
    max_audio_duration_seconds: int = Field(default=3600)  # 1 Hour
    target_sample_rate: int = Field(default=16000)


class SileroVADDetector(SpeechDetector):
    """Concrete implementation of SpeechDetector using Silero VAD."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 1000,
        frame_duration: float = 0.032,
        model_path: str | None = None, # Left for backward compatibility in __init__ args, but ignored
        model_hash: str | None = None, # Left for backward compatibility in __init__ args, but ignored
    ) -> None:
        # Immediately validate the configuration via Pydantic model creation
        self.config = VADConfig(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            frame_duration=frame_duration,
        )
        self.model: Any = None

    def _verify_and_load_model(self) -> None:
        if self.model is not None:
            return

        try:
            # Load model dynamically via torch.hub
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:v4.0",
                model="silero_vad",
                force_reload=False,
            )
        except Exception as e:
            msg = f"Failed to load Silero VAD model via torch.hub: {e}"
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
                # 1. Calculate window size (512 samples for 16kHz)
                window_size = 512

                # 2. Call self.model.reset_states() before the loop
                self.model.reset_states()

                probs_list = []
                num_samples = wav.shape[1]

                with torch.no_grad():
                    # 3. Iterate through the wav tensor in steps of 512.
                    for i in range(0, num_samples, window_size):
                        frame = wav[:, i : i + window_size]

                        # Pad the final window with zeros if necessary
                        if frame.shape[1] < window_size:
                            padding = window_size - frame.shape[1]
                            frame = torch.nn.functional.pad(frame, (0, padding))

                        # 4. Call the model
                        frame_prob = self.model(frame, self.config.target_sample_rate)

                        # 5. Extract float probability using .item() and append to standard Python list
                        probs_list.append(frame_prob.item())

                # 6. Convert list of floats back into 1D torch.Tensor
                probs = torch.tensor(probs_list)

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
