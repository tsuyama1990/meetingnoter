from domain_models import (
    Aggregator,
    AudioChunk,
    DiarizedSegment,
    SpeakerLabel,
    TranscriptionSegment,
)


class TranscriptMerger(Aggregator):
    """Merges transcription segments and speaker labels, assigning the correct speaker and applying global time offsets."""

    def merge(
        self,
        chunk: AudioChunk,
        transcriptions: list[TranscriptionSegment],
        speaker_labels: list[SpeakerLabel],
    ) -> list[DiarizedSegment]:

        result: list[DiarizedSegment] = []
        for trans in transcriptions:
            max_overlap = 0.0
            best_speaker_id = "UNKNOWN"

            for label in speaker_labels:
                overlap_start = max(trans.start_time, label.start_time)
                overlap_end = min(trans.end_time, label.end_time)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker_id = label.speaker_id

            global_start = chunk.start_time + trans.start_time
            global_end = chunk.start_time + trans.end_time

            result.append(
                DiarizedSegment(
                    start_time=global_start,
                    end_time=global_end,
                    text=trans.text,
                    speaker_id=best_speaker_id,
                    confidence_score=trans.confidence_score,
                    uncertain=trans.uncertain,
                )
            )

        return result
