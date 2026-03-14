
from domain_models import AudioChunk, Diarizer, SpeakerLabel


class PyannoteDiarizer(Diarizer):
    """Concrete implementation of Diarizer using pyannote.audio."""

    def __init__(self, auth_token: str) -> None:
        self.auth_token = auth_token
        self.pipeline = None

    def _load_model(self) -> None:
        if self.pipeline is None:
            try:
                import torch
                from pyannote.audio import Pipeline
            except ImportError as e:
                msg = "Required library 'pyannote.audio' or 'torch' is not installed."
                raise ImportError(msg) from e

            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.pipeline = Pipeline.from_pretrained( # type: ignore[call-arg, assignment]
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=self.auth_token
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        msg = f"Failed to load pyannote pipeline after {max_retries} attempts. Please check HF_AUTH_TOKEN: {e}"
                        raise RuntimeError(msg) from e
                    time.sleep(2)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.pipeline is not None:
                self.pipeline.to(device)

    def diarize(self, chunk: AudioChunk) -> list[SpeakerLabel]:
        """Diarizes the audio using pyannote.audio."""
        self._load_model()

        from pathlib import Path
        if not Path(chunk.chunk_filepath).exists():
            msg = f"Audio chunk file not found: {chunk.chunk_filepath}"
            raise FileNotFoundError(msg)

        if self.pipeline:
            from typing import Any
            try:
                # Apply the specific configuration for Japanese overlap speech mentioned in SPEC
                diarization: Any = self.pipeline(
                    chunk.chunk_filepath,
                    exclusive=True, # Prevent overlapping labels which cause Speaker Confusion
                    num_workers=4
                )

                labels: list[SpeakerLabel] = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    # Turn is a pyannote.core.Segment object
                    start_sec: float = chunk.start_time + turn.start
                    end_sec: float = chunk.start_time + turn.end

                    if start_sec < end_sec:
                        labels.append(
                            SpeakerLabel(
                                start_time=start_sec,
                                end_time=end_sec,
                                speaker_id=str(speaker)
                            )
                        )
            except Exception as e:
                msg = f"Pyannote diarization failed: {e}"
                raise RuntimeError(msg) from e
            else:
                return labels

        msg = "Pyannote pipeline was not properly loaded."
        raise RuntimeError(msg)
