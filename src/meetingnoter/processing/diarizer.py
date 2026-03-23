from domain_models import AudioChunk, Diarizer, SpeakerLabel


class PyannoteDiarizer(Diarizer):
    """Concrete implementation of Diarizer using pyannote.audio."""

    def __init__(self, auth_token: str) -> None:
        self.auth_token = auth_token
        self.pipeline = None

    def _load_model(self) -> None:
        if self.pipeline is None:
            if (
                not self.auth_token
                or len(self.auth_token) < 10
                or not self.auth_token.startswith("hf_")
            ):
                msg = "Invalid authentication token format for pyannote."
                raise ValueError(msg)

            try:
                import torch
                from pyannote.audio import Pipeline
            except ImportError as e:
                msg = "Failed to initialize diarization model due to missing dependencies."
                raise ImportError(msg) from e

            import logging
            import time

            logger = logging.getLogger(__name__)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.pipeline = Pipeline.from_pretrained(  # type: ignore[assignment]
                        "pyannote/speaker-diarization-3.1", token=self.auth_token
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.debug("Pyannote load failed: %s", e)
                        msg = "Failed to initialize diarization model due to an external error."
                        raise RuntimeError(msg) from e
                    time.sleep(2**attempt)  # Exponential backoff

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
                    exclusive=True,  # Prevent overlapping labels which cause Speaker Confusion
                    num_workers=4,
                )

                labels: list[SpeakerLabel] = []
                # Support pyannote.audio 4.x DiarizeOutput wrapper and older Annotation objects
                annotation = (
                    diarization.speaker_diarization
                    if hasattr(diarization, "speaker_diarization")
                    else diarization
                )
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    start_sec: float = float(turn.start)
                    end_sec: float = float(turn.end)

                    if start_sec < end_sec:
                        labels.append(
                            SpeakerLabel(
                                start_time=start_sec, end_time=end_sec, speaker_id=str(speaker)
                            )
                        )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug("Diarization failed: %s", e)
                msg = "Pyannote diarization process failed."
                raise RuntimeError(msg) from e
            else:
                return labels

        msg = "Pyannote pipeline failed to initialize."
        raise RuntimeError(msg)
