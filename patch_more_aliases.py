import re
with open("tutorials/UAT_AND_TUTORIAL.py", "r") as f:
    c = f.read()

c = c.replace('_AudioChunk(', 'AudioChunk(')
c = c.replace('_SpeechSegment(', 'SpeechSegment(')
c = c.replace('_PipelineConfig_err(', 'PipelineConfig(')
c = c.replace('_AudioChunk_err(', 'AudioChunk(')
c = c.replace('_c07_run_pipeline(', 'run_pipeline(')
c = c.replace('_c07_err_run_pipeline(', 'run_pipeline(')
c = c.replace('from domain_models import AudioChunk, SpeakerLabel, TranscriptionSegment', 'from src.domain_models import AudioChunk, SpeakerLabel, TranscriptionSegment')
c = c.replace('from meetingnoter import TranscriptMerger', 'from src.meetingnoter.processing.aggregator import TranscriptMerger')
c = c.replace('from domain_models import DiarizedSegment, DiarizedTranscript', 'from src.domain_models import DiarizedSegment, DiarizedTranscript')


with open("tutorials/UAT_AND_TUTORIAL.py", "w") as f:
    f.write(c)
