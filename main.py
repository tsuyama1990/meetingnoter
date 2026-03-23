import logging
import os
import sys
from pathlib import Path
from typing import Any

# --- Task 2: Ensure src/ is always on sys.path regardless of how the script is invoked ---
_src_path = Path(__file__).resolve().parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Map TORCH_HOME for Colab caching

os.environ["TORCH_HOME"] = os.environ.get(
    "TORCH_HOME", "/content/drive/MyDrive/MeetingNoter/models/torch_cache"
)
os.environ["HF_HOME"] = os.environ.get(
    "HF_HOME", "/content/drive/MyDrive/MeetingNoter/models/hf_cache"
)

try:
    import torch
except ImportError:
    torch: Any = None  # type: ignore[no-redef]


from domain_models import (  # noqa: E402
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
    import argparse

    parser = argparse.ArgumentParser(description="MeetingNoter Pipeline")
    parser.add_argument("file_id_or_path", nargs="?", help="File ID or Path for the audio source")
    parser.add_argument(
        "--preprocess",
        choices=["none", "loudnorm", "compressor"],
        default="none",
        help="Audio preprocessing mode",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Threshold below which transcriptions are marked uncertain",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save the output files"
    )

    args, unknown = parser.parse_known_args()

    # Override OS environment variables so PipelineConfig picks them up
    if args.preprocess:
        os.environ["PREPROCESS"] = args.preprocess
    if args.confidence_threshold is not None:
        os.environ["CONFIDENCE_THRESHOLD"] = str(args.confidence_threshold)
    if args.output_dir:
        os.environ["OUTPUT_DIR"] = args.output_dir
    if args.file_id_or_path:
        os.environ["FILE_ID"] = args.file_id_or_path

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

    def run(self, file_id: str, config: PipelineConfig | None = None) -> DiarizedTranscript:
        source: AudioSource | None = None
        preprocessed_path: str | None = None
        chunks: list[AudioChunk] = []
        try:
            source = self.storage.download(file_id)

            # Preprocess the audio
            if config is not None and config.preprocess != "none" and source is not None:
                import typing

                from meetingnoter.processing.audio_preprocessor import preprocess_audio

                logger.info("Starting audio preprocessing with mode: %s", config.preprocess)
                preprocessed_path = preprocess_audio(
                    source.filepath,
                    config.ffmpeg_path,
                    typing.cast(
                        typing.Literal["none", "loudnorm", "compressor"], config.preprocess
                    ),
                )
                source.filepath = preprocessed_path

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
            if preprocessed_path is not None and (
                source is None or preprocessed_path != source.filepath
            ):
                Path(preprocessed_path).unlink(missing_ok=True)
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
    config: PipelineConfig | None = None,
) -> DiarizedTranscript:
    orchestrator = PipelineOrchestrator(
        storage, splitter, detector, transcriber, diarizer, aggregator
    )
    return orchestrator.run(file_id, config)


def _process_single_chunk(
    chunk: AudioChunk,
    detector: SpeechDetector,
    transcriber: Transcriber,
    diarizer: Diarizer,
    aggregator: Aggregator,
) -> list[DiarizedSegment]:
    orchestrator = PipelineOrchestrator(None, None, detector, transcriber, diarizer, aggregator)  # type: ignore[arg-type]
    return orchestrator._process_single_chunk(chunk)


def main() -> None:  # noqa: C901, PLR0915
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
        transcript: DiarizedTranscript = orchestrator.run(config.file_id, config)
    except RuntimeError:
        logger.exception("Pipeline execution failed due to an error.")
        sys.exit(1)
    else:
        logger.info(
            "Pipeline finished successfully. Generated %d diarized segments.",
            len(transcript.segments),
        )

        # Output saving
        import json

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "output.json"
        md_path = output_dir / "output.md"

        # JSON Output
        segments_data = []
        for seg in transcript.segments:
            seg_dict = seg.model_dump(exclude_none=True)
            if not seg_dict.get("uncertain"):
                seg_dict.pop("uncertain", None)
            segments_data.append(seg_dict)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(segments_data, f, indent=2, ensure_ascii=False)

        # Markdown Output
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        with md_path.open("w", encoding="utf-8") as f:
            for seg in transcript.segments:
                start_str = format_timestamp(seg.start_time)
                end_str = format_timestamp(seg.end_time)
                f.write(f"[{start_str} - {end_str}] {seg.speaker_id}: {seg.text}\n")

        logger.info("Output saved to %s and %s", json_path, md_path)


if __name__ == "__main__":
    main()
