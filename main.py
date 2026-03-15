import logging
import sys
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch: Any = None  # type: ignore[no-redef]

import importlib

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


def _process_single_chunk(
    chunk: AudioChunk,
    detector: SpeechDetector,
    transcriber: Transcriber,
    diarizer: Diarizer,
    aggregator: Aggregator,
) -> list[DiarizedSegment]:
    """Processes a single chunk to manage memory strictly."""
    from domain_models import SpeakerLabel, SpeechSegment, TranscriptionSegment

    try:
        # VAD gating
        speech_segments: list[SpeechSegment] = detector.detect_speech(chunk)

        # Transcribe with faster-whisper
        transcriptions: list[TranscriptionSegment] = transcriber.transcribe(chunk, speech_segments)

        # Diarize with Pyannote
        speaker_labels: list[SpeakerLabel] = diarizer.diarize(chunk)

        # Aggregate results
        segments: list[DiarizedSegment] = aggregator.merge(chunk, transcriptions, speaker_labels)
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
        # Re-raise unexpected exceptions to fail fast
        msg = f"Unexpected failure in pipeline processing chunk {chunk.chunk_index}: {e}"
        raise RuntimeError(msg) from e
    else:
        _cleanup_memory()
        return segments
    finally:
        # Clear GPU memory after each heavy chunk to prevent OOM even on error
        _cleanup_memory()


def _cleanup_memory() -> None:
    """Centralized resource manager for explicit memory scrubbing."""
    import gc

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pipeline(
    storage: StorageClient,
    splitter: AudioSplitter,
    detector: SpeechDetector,
    transcriber: Transcriber,
    diarizer: Diarizer,
    aggregator: Aggregator,
    file_id: str,
) -> DiarizedTranscript:
    """Orchestrates the entire voice analysis pipeline."""
    source: AudioSource | None = None
    chunks: list[AudioChunk] = []

    try:
        # 1. Download audio
        source = storage.download(file_id)

        # 2. Split into chunks
        chunks = splitter.split(source)

        all_segments: list[DiarizedSegment] = []

        # 3. Process each chunk

        try:
            for chunk in chunks:
                all_segments.extend(
                    _process_single_chunk(
                        chunk=chunk, detector=detector, transcriber=transcriber, diarizer=diarizer, aggregator=aggregator
                    )
                )
        except Exception:
            logger.exception("Pipeline aborted during chunk processing loop.")
            raise

        _cleanup_memory()
        return DiarizedTranscript(segments=all_segments)
    finally:
        # Guarantee final memory scrub before returning to main
        _cleanup_memory()

        # Cleanup temp files
        if source is not None:
            Path(source.filepath).unlink(missing_ok=True)
        for chunk in chunks:
            Path(chunk.chunk_filepath).unlink(missing_ok=True)


def main() -> None:
    """Main entry point to execute the pipeline using Dependency Injection container initialization."""
    # 1. Resolve and validate configuration
    import pydantic

    try:
        config: PipelineConfig = get_config()
    except pydantic.ValidationError:
        logger.exception("Configuration validation failed due to invalid schema.")
        sys.exit(1)
    except ValueError:
        logger.exception("Configuration validation failed due to missing secrets.")
        sys.exit(1)

    # 2. Initialize concrete implementations by passing dependencies dynamically rather than hardcoding imports
    try:
        drive_client_module = importlib.import_module(config.drive_client_module_path)
        chunker_module = importlib.import_module(config.chunker_module_path)
        vad_module = importlib.import_module(config.vad_module_path)
        transcriber_module = importlib.import_module(config.transcriber_module_path)
        diarizer_module = importlib.import_module(config.diarizer_module_path)
        aggregator_module = importlib.import_module(config.aggregator_module_path)

        storage: StorageClient = drive_client_module.GoogleDriveClient(config=config)
        splitter: AudioSplitter = chunker_module.FFmpegChunker(
            ffmpeg_path=config.ffmpeg_path, chunk_length_minutes=config.chunk_length_minutes
        )
        detector: SpeechDetector = vad_module.SileroVADDetector(
            threshold=config.vad_threshold,
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            min_silence_duration_ms=config.vad_min_silence_duration_ms,
            model_path=config.silero_vad_model_path,
        )
        transcriber: Transcriber = transcriber_module.FasterWhisperTranscriber(config)
        aggregator: Aggregator = aggregator_module.TranscriptMerger()

        if not config.pyannote_auth_token or not config.pyannote_auth_token.startswith("hf_"):
            msg = "Invalid Pyannote auth token. It must be a valid Hugging Face token starting with 'hf_'."
            raise ValueError(msg)

        try:
            diarizer: Diarizer = diarizer_module.PyannoteDiarizer(
                auth_token=config.pyannote_auth_token
            )
        except Exception as e:
            msg = f"Failed to initialize PyannoteDiarizer. Ensure pyannote.audio is correctly installed: {e}"
            raise RuntimeError(msg) from e

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
            aggregator=aggregator,
            file_id=config.file_id,
        )
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
