from domain_models import (
    AudioChunk,
    AudioSource,
    AudioSplitter,
    DiarizedSegment,
    DiarizedTranscript,
    Diarizer,
    SpeakerLabel,
    SpeechDetector,
    SpeechSegment,
    StorageClient,
    Transcriber,
    TranscriptionSegment,
)


class DummyStorageClient:
    def download(self, file_id: str) -> AudioSource:
        return AudioSource(filepath=f"/mock_dir/{file_id}.wav", duration_seconds=60.0)


class DummyAudioSplitter:
    def split(self, source: AudioSource) -> list[AudioChunk]:
        return [
            AudioChunk(
                chunk_filepath="/mock_dir/chunk0.wav", start_time=0.0, end_time=30.0, chunk_index=0
            ),
            AudioChunk(
                chunk_filepath="/mock_dir/chunk1.wav", start_time=30.0, end_time=60.0, chunk_index=1
            ),
        ]


class DummySpeechDetector:
    def detect_speech(self, chunk: AudioChunk) -> list[SpeechSegment]:
        return [SpeechSegment(start_time=chunk.start_time, end_time=chunk.start_time + 10.0)]


class DummyTranscriber:
    def transcribe(
        self, chunk: AudioChunk, speech_segments: list[SpeechSegment]
    ) -> list[TranscriptionSegment]:
        return [
            TranscriptionSegment(
                start_time=seg.start_time, end_time=seg.end_time, text=f"Text {seg.start_time}"
            )
            for seg in speech_segments
        ]


class DummyDiarizer:
    def diarize(self, chunk: AudioChunk) -> list[SpeakerLabel]:
        return [
            SpeakerLabel(
                start_time=chunk.start_time,
                end_time=chunk.start_time + 10.0,
                speaker_id="SPEAKER_00",
            )
        ]


def test_pipeline_integration() -> None:
    # 1. Download
    storage: StorageClient = DummyStorageClient()
    source = storage.download("test_id")
    assert isinstance(source, AudioSource)

    # 2. Split
    splitter: AudioSplitter = DummyAudioSplitter()
    chunks = splitter.split(source)
    assert len(chunks) == 2

    # 3. Detect, Transcribe, Diarize
    detector: SpeechDetector = DummySpeechDetector()
    transcriber: Transcriber = DummyTranscriber()
    diarizer: Diarizer = DummyDiarizer()

    all_segments = []

    for chunk in chunks:
        speech_segments = detector.detect_speech(chunk)
        transcriptions = transcriber.transcribe(chunk, speech_segments)
        speaker_labels = diarizer.diarize(chunk)

        # Basic aggregation (simplified for cycle 1 test)
        for t, s in zip(transcriptions, speaker_labels, strict=False):
            all_segments.append(
                DiarizedSegment(
                    start_time=t.start_time,
                    end_time=t.end_time,
                    text=t.text,
                    speaker_id=s.speaker_id,
                )
            )

    transcript = DiarizedTranscript(segments=all_segments)
    assert len(transcript.segments) == 2
    assert transcript.segments[0].speaker_id == "SPEAKER_00"
    assert transcript.segments[0].text == "Text 0.0"
