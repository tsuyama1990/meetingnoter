import re

with open("tutorials/UAT_AND_TUTORIAL.py", "r") as f:
    c = f.read()

# Lines 540-544
dup = """                from src.meetingnoter.ingestion.drive_client import GoogleDriveClient
                from src.meetingnoter.processing.chunker import FFmpegChunker
                from src.meetingnoter.processing.diarizer import PyannoteDiarizer
                from src.meetingnoter.processing.transcriber import FasterWhisperTranscriber
                from src.meetingnoter.processing.vad import SileroVADDetector"""

c = c.replace(dup, '', 1)
# Actually, I shouldn't just remove it because `test_c07_error_handling` relies on these imports.
# Oh wait, `GoogleDriveClient` is literally ONLY needed if we instantiate it. Let me move the imports outside the function scope to the cell scope!
# But then `PyannoteDiarizer` might cause a circular import? No.
# If we define them in the top-level cell imports, we avoid this completely.
