from domain_models import (
    Aggregator,
    AudioSplitter,
    Diarizer,
    PipelineConfig,
    SpeechDetector,
    StorageClient,
    Transcriber,
)


class PipelineComponentFactory:
    @staticmethod
    def create_storage_client(config: PipelineConfig) -> StorageClient:
        from meetingnoter.ingestion.drive_client import GoogleDriveClient

        return GoogleDriveClient(config=config)

    @staticmethod
    def create_splitter(config: PipelineConfig) -> AudioSplitter:
        from meetingnoter.processing.chunker import FFmpegChunker

        return FFmpegChunker(
            ffmpeg_path=config.ffmpeg_path, chunk_length_minutes=config.chunk_length_minutes
        )

    @staticmethod
    def create_detector(config: PipelineConfig) -> SpeechDetector:
        from meetingnoter.processing.vad import SileroVADDetector

        return SileroVADDetector(
            threshold=config.vad_threshold,
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            min_silence_duration_ms=config.vad_min_silence_duration_ms,
            model_path=config.silero_vad_model_path,
        )

    @staticmethod
    def create_transcriber(config: PipelineConfig) -> Transcriber:
        from meetingnoter.processing.transcriber import FasterWhisperTranscriber

        return FasterWhisperTranscriber(config)

    @staticmethod
    def create_diarizer(config: PipelineConfig) -> Diarizer:
        from meetingnoter.processing.diarizer import PyannoteDiarizer

        return PyannoteDiarizer(auth_token=config.pyannote_auth_token)

    @staticmethod
    def create_aggregator() -> Aggregator:
        from meetingnoter.processing.aggregator import TranscriptMerger

        return TranscriptMerger()


def validate_pyannote_token(token: str) -> None:
    if not token or len(token) < 10 or not token.startswith("hf_"):
        msg = "Invalid Pyannote auth token format."
        raise ValueError(msg)
