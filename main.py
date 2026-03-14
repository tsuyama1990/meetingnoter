import logging
import sys
from pathlib import Path

from domain_models import (
    AudioSplitter,
    DiarizedSegment,
    DiarizedTranscript,
    Diarizer,
    SpeechDetector,
    StorageClient,
    Transcriber,
)

# Import the concrete implementations
from meetingnoter.ingestion.drive_client import GoogleDriveClient
from meetingnoter.processing.chunker import FFmpegChunker
from meetingnoter.processing.diarizer import PyannoteDiarizer
from meetingnoter.processing.transcriber import FasterWhisperTranscriber
from meetingnoter.processing.vad import SileroVADDetector
from meetingnoter.utils.secrets import _get_secret

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline(
    storage: StorageClient,
    splitter: AudioSplitter,
    detector: SpeechDetector,
    transcriber: Transcriber,
    diarizer: Diarizer,
    file_id: str
) -> DiarizedTranscript:
    """Orchestrates the entire voice analysis pipeline."""
    # 1. Download audio
    source = storage.download(file_id)

    # 2. Split into chunks
    chunks = splitter.split(source)

    all_segments = []

    # 3. Process each chunk
    for chunk in chunks:
        try:
            # VAD gating
            speech_segments = detector.detect_speech(chunk)

            # Transcribe with faster-whisper
            transcriptions = transcriber.transcribe(chunk, speech_segments)

            # Diarize with Pyannote
            speaker_labels = diarizer.diarize(chunk)

            # Aggregate results
            for t, s in zip(transcriptions, speaker_labels, strict=False):
                all_segments.append(
                    DiarizedSegment(
                        start_time=t.start_time,
                        end_time=t.end_time,
                        text=t.text,
                        speaker_id=s.speaker_id
                    )
                )
        except Exception:
            logger.exception("Failed to process chunk %d. Skipping and continuing.", chunk.chunk_index)
        finally:
            # Clear GPU memory after each heavy chunk to prevent OOM
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # Cleanup temp files
    Path(source.filepath).unlink(missing_ok=True)
    for chunk in chunks:
        Path(chunk.chunk_filepath).unlink(missing_ok=True)

    return DiarizedTranscript(segments=all_segments)

def main() -> None:
    """Main entry point to execute the pipeline using concrete implementations."""
    file_id = _get_secret("MEETINGNOTER_FILE_ID") if len(sys.argv) <= 1 else sys.argv[1]

    if not file_id:
        logger.error("No file_id provided. Set MEETINGNOTER_FILE_ID or pass as argument.")
        sys.exit(1)

    # Initialize concrete implementations
    try:
        storage = GoogleDriveClient()
        splitter = FFmpegChunker(chunk_length_minutes=20)
        detector = SileroVADDetector(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=1000)
        transcriber = FasterWhisperTranscriber(model_size="large-v3", compute_type="int8")
        diarizer = PyannoteDiarizer()
    except ValueError:
        logger.exception("Failed to initialize pipeline components. Check required credentials.")
        sys.exit(1)

    # Execute pipeline
    try:
        transcript = run_pipeline(
            storage=storage,
            splitter=splitter,
            detector=detector,
            transcriber=transcriber,
            diarizer=diarizer,
            file_id=file_id
        )
        logger.info("Pipeline finished successfully. Generated %d diarized segments.", len(transcript.segments))
    except Exception:
        logger.exception("Pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
