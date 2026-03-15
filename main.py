import logging
import sys
from pathlib import Path
from typing import Any

try:
    import google.colab
except ImportError:
    google: Any = None  # type: ignore[no-redef]

try:
    import torch
except ImportError:
    torch: Any = None  # type: ignore[no-redef]

from domain_models import (
    AudioChunk,
    AudioSource,
    AudioSplitter,
    DiarizedSegment,
    DiarizedTranscript,
    Diarizer,
    PipelineConfig,
    SpeakerLabel,
    SpeechDetector,
    SpeechSegment,
    StorageClient,
    Transcriber,
    TranscriptionSegment,
)

# Import the concrete implementations
from meetingnoter.ingestion.drive_client import GoogleDriveClient
from meetingnoter.processing.chunker import FFmpegChunker
from meetingnoter.processing.diarizer import PyannoteDiarizer
from meetingnoter.processing.transcriber import FasterWhisperTranscriber
from meetingnoter.processing.vad import SileroVADDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config() -> PipelineConfig:
    # Resolves directly from ENV or google.colab.userdata via default_factory
    return PipelineConfig()


def run_pipeline(
    storage: StorageClient,
    splitter: AudioSplitter,
    detector: SpeechDetector,
    transcriber: Transcriber,
    diarizer: Diarizer,
    file_id: str,
) -> DiarizedTranscript:
    """Orchestrates the entire voice analysis pipeline."""
    # 1. Download audio
    source: AudioSource = storage.download(file_id)

    # 2. Split into chunks
    chunks: list[AudioChunk] = splitter.split(source)

    all_segments: list[DiarizedSegment] = []

    # 3. Process each chunk
    import requests

    for chunk in chunks:
        try:
            # VAD gating
            speech_segments: list[SpeechSegment] = detector.detect_speech(chunk)

            # Transcribe with faster-whisper
            transcriptions: list[TranscriptionSegment] = transcriber.transcribe(
                chunk, speech_segments
            )

            # Diarize with Pyannote
            speaker_labels: list[SpeakerLabel] = diarizer.diarize(chunk)

            # Aggregate results
            for t, s in zip(transcriptions, speaker_labels, strict=False):
                all_segments.append(
                    DiarizedSegment(
                        start_time=t.start_time,
                        end_time=t.end_time,
                        text=t.text,
                        speaker_id=s.speaker_id,
                    )
                )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.exception(
                    "Resource exhaustion (OOM) encountered processing chunk %d", chunk.chunk_index
                )
            else:
                logger.exception("Runtime error processing chunk %d", chunk.chunk_index)
        except requests.exceptions.RequestException:
            logger.exception(
                "Network or authentication failure processing chunk %d", chunk.chunk_index
            )
        except ValueError:
            logger.exception("Validation error processing chunk %d", chunk.chunk_index)
        except Exception as e:
            # Re-raise unexpected exceptions to fail fast
            msg = f"Unexpected failure in pipeline: {e}"
            raise RuntimeError(msg) from e
        finally:
            # Clear GPU memory after each heavy chunk to prevent OOM
            import gc

            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Cleanup temp files
    Path(source.filepath).unlink(missing_ok=True)
    for chunk in chunks:
        Path(chunk.chunk_filepath).unlink(missing_ok=True)

    return DiarizedTranscript(segments=all_segments)


def create_components(
    config: PipelineConfig,
) -> tuple[StorageClient, AudioSplitter, SpeechDetector, Transcriber, Diarizer]:
    """Factory function to build concrete implementations for dependency injection."""
    storage: StorageClient = GoogleDriveClient(config=config)
    splitter: AudioSplitter = FFmpegChunker(ffmpeg_path=config.ffmpeg_path, chunk_length_minutes=20)
    detector: SpeechDetector = SileroVADDetector(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=1000,
        model_path=config.silero_vad_model_path,
    )
    transcriber: Transcriber = FasterWhisperTranscriber(config)
    diarizer: Diarizer = PyannoteDiarizer(auth_token=config.pyannote_auth_token)
    return storage, splitter, detector, transcriber, diarizer


def main() -> None:
    """Main entry point to execute the pipeline using concrete implementations."""
    # 1. Resolve and validate configuration
    import pydantic
    import requests

    try:
        config: PipelineConfig = get_config()
    except pydantic.ValidationError:
        logger.exception("Configuration validation failed due to invalid schema.")
        sys.exit(1)
    except ValueError:
        logger.exception("Configuration validation failed due to missing secrets.")
        sys.exit(1)

    # 2. Initialize concrete implementations using Dependency Injection via Factory
    try:
        storage, splitter, detector, transcriber, diarizer = create_components(config)
    except RuntimeError:
        logger.exception("Failed to dynamically initialize pipeline dependencies.")
        sys.exit(1)

    # 3. Execute pipeline
    try:
        transcript: DiarizedTranscript = run_pipeline(
            storage=storage,
            splitter=splitter,
            detector=detector,
            transcriber=transcriber,
            diarizer=diarizer,
            file_id=config.file_id,
        )
        logger.info(
            "Pipeline finished successfully. Generated %d diarized segments.",
            len(transcript.segments),
        )
    except requests.exceptions.RequestException:
        logger.exception("Pipeline execution failed due to network connectivity.")
        sys.exit(1)
    except RuntimeError:
        logger.exception("Pipeline execution failed due to unexpected runtime error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
