import logging
import os
import sys
from pathlib import Path
from typing import Any

# Assuming Google Drive is mounted at /content/drive
if os.path.isdir("/content/drive/MyDrive"):
    os.environ["HF_HOME"] = "/content/drive/MyDrive/MeetingNoter/models/hf_cache"
    os.environ["CTRANSLATE2_CACHE_DIR"] = "/content/drive/MyDrive/MeetingNoter/models/whisper_cache"

try:
    import torch
except ImportError:
    torch: Any = None  # type: ignore[no-redef]


from domain_models import (
    Aggregator,
    AudioChunk,
    AudioSource,
    AudioSplitter,
    DiarizedSegment,
    DiarizedTranscript,
    Diarizer,
    PipelineConfig,
    SpeechDetector,
    StorageClient,
    Transcriber,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config() -> PipelineConfig:
    # Resolves directly from ENV or google.colab.userdata via default_factory
    return PipelineConfig()


def _cleanup_memory() -> None:
    """Centralized resource manager for explicit memory scrubbing."""
    import gc

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class PipelineOrchestrator:
    def __init__(
        self,
        storage: StorageClient,
        splitter: AudioSplitter,
        detector: SpeechDetector,
        transcriber: Transcriber,
        diarizer: Diarizer,
        aggregator: Aggregator,
    ) -> None:
        self.storage = storage
        self.splitter = splitter
        self.detector = detector
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.aggregator = aggregator

    def run(self, file_id: str) -> DiarizedTranscript:
        source: AudioSource | None = None
        chunks: list[AudioChunk] = []
        try:
            source = self.storage.download(file_id)
            chunks = self.splitter.split(source)
            all_segments: list[DiarizedSegment] = []

            try:
                for chunk in chunks:
                    all_segments.extend(self._process_single_chunk(chunk))
            except Exception:
                logger.exception("Pipeline aborted during chunk processing loop.")
                raise

            _cleanup_memory()
            return DiarizedTranscript(segments=all_segments)
        finally:
            _cleanup_memory()
            if source is not None:
                Path(source.filepath).unlink(missing_ok=True)
            for chunk in chunks:
                Path(chunk.chunk_filepath).unlink(missing_ok=True)

    def _process_single_chunk(self, chunk: AudioChunk) -> list[DiarizedSegment]:
        from domain_models import SpeakerLabel, SpeechSegment, TranscriptionSegment

        try:
            speech_segments: list[SpeechSegment] = self.detector.detect_speech(chunk)
            transcriptions: list[TranscriptionSegment] = self.transcriber.transcribe(
                chunk, speech_segments
            )
            speaker_labels: list[SpeakerLabel] = self.diarizer.diarize(chunk)
            segments: list[DiarizedSegment] = self.aggregator.merge(
                chunk, transcriptions, speaker_labels
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.exception(
                    "Resource exhaustion (OOM) encountered processing chunk %d. Aborting pipeline.",
                    chunk.chunk_index,
                )
            else:
                logger.exception(
                    "Runtime error processing chunk %d. Aborting pipeline.", chunk.chunk_index
                )
            raise
        except ValueError:
            logger.exception(
                "Validation error processing chunk %d. Aborting pipeline.", chunk.chunk_index
            )
            raise
        except Exception as e:
            msg = f"Unexpected failure in pipeline processing chunk {chunk.chunk_index}: {e}"
            raise RuntimeError(msg) from e
        else:
            _cleanup_memory()
            return segments
        finally:
            _cleanup_memory()


# Backwards compatibility alias for tests
def run_pipeline(
    storage: StorageClient,
    splitter: AudioSplitter,
    detector: SpeechDetector,
    transcriber: Transcriber,
    diarizer: Diarizer,
    aggregator: Aggregator,
    file_id: str,
) -> DiarizedTranscript:
    orchestrator = PipelineOrchestrator(
        storage, splitter, detector, transcriber, diarizer, aggregator
    )
    return orchestrator.run(file_id)


def _process_single_chunk(
    chunk: AudioChunk,
    detector: SpeechDetector,
    transcriber: Transcriber,
    diarizer: Diarizer,
    aggregator: Aggregator,
) -> list[DiarizedSegment]:
    orchestrator = PipelineOrchestrator(None, None, detector, transcriber, diarizer, aggregator)  # type: ignore[arg-type]
    return orchestrator._process_single_chunk(chunk)


def main() -> None:
    """Main entry point to execute the pipeline using Dependency Injection container initialization."""
    # 1. Resolve and validate configuration
    import typing as _typing

    pydantic: _typing.Any = __import__("pydantic")

    try:
        config: PipelineConfig = get_config()
    except pydantic.ValidationError:
        logger.exception("Configuration validation failed due to invalid schema.")
        sys.exit(1)
    except ValueError:
        logger.exception("Configuration validation failed due to missing secrets.")
        sys.exit(1)

    # 2. Initialize concrete implementations using static imports to prevent injection
    from meetingnoter.ingestion.drive_client import GoogleDriveClient
    from meetingnoter.processing.aggregator import TranscriptMerger
    from meetingnoter.processing.chunker import FFmpegChunker
    from meetingnoter.processing.diarizer import PyannoteDiarizer
    from meetingnoter.processing.transcriber import FasterWhisperTranscriber
    from meetingnoter.processing.vad import SileroVADDetector

    try:
        storage: StorageClient = GoogleDriveClient(config=config)
        splitter: AudioSplitter = FFmpegChunker(
            ffmpeg_path=config.ffmpeg_path, chunk_length_minutes=config.chunk_length_minutes
        )
        detector: SpeechDetector = SileroVADDetector(
            threshold=config.vad_threshold,
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            min_silence_duration_ms=config.vad_min_silence_duration_ms,
            model_path=config.silero_vad_model_path,
        )
        transcriber: Transcriber = FasterWhisperTranscriber(config)
        aggregator: Aggregator = TranscriptMerger()

        if not config.pyannote_auth_token or not config.pyannote_auth_token.startswith("hf_"):
            msg = "Invalid Pyannote auth token. It must be a valid Hugging Face token starting with 'hf_'."
            raise ValueError(msg)

        try:
            diarizer: Diarizer = PyannoteDiarizer(auth_token=config.pyannote_auth_token)
        except Exception as e:
            msg = f"Failed to initialize PyannoteDiarizer. Ensure pyannote.audio is correctly installed: {e}"
            raise RuntimeError(msg) from e

    except Exception:
        logger.exception("Failed to statically initialize pipeline dependencies.")
        sys.exit(1)

    # 3. Execute pipeline
    try:
        orchestrator = PipelineOrchestrator(
            storage=storage,
            splitter=splitter,
            detector=detector,
            transcriber=transcriber,
            diarizer=diarizer,
            aggregator=aggregator,
        )
        transcript: DiarizedTranscript = orchestrator.run(config.file_id)
    except RuntimeError:
        logger.exception("Pipeline execution failed due to an error.")
        sys.exit(1)
    else:
        logger.info(
            "Pipeline finished successfully. Generated %d diarized segments.",
            len(transcript.segments),
        )


if __name__ == "__main__":
    main()
