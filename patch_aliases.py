import re
with open("tutorials/UAT_AND_TUTORIAL.py", "r") as f:
    c = f.read()

# Replace aliases
c = re.sub(r'from src.domain_models import AudioChunk as _AudioChunk_err\n\s*from src.domain_models import PipelineConfig as _PipelineConfig_err', 'from src.domain_models import AudioChunk, PipelineConfig', c)
c = c.replace('_AudioChunk_err(', 'AudioChunk(')
c = c.replace('_PipelineConfig_err()', 'PipelineConfig()')

c = re.sub(r'from src.domain_models import PipelineConfig as _C07Config\n\s*from main import run_pipeline as _c07_run_pipeline', 'from src.domain_models import PipelineConfig\n    from main import run_pipeline', c)
c = c.replace('_C07Config()', 'PipelineConfig()')
c = c.replace('_c07_run_pipeline(', 'run_pipeline(')

c = re.sub(r'from src.domain_models import PipelineConfig as _C07ErrConfig\n\s*from main import run_pipeline as _c07_err_run_pipeline', 'from src.domain_models import PipelineConfig\n    from main import run_pipeline', c)
c = c.replace('_C07ErrConfig()', 'PipelineConfig()')
c = c.replace('_c07_err_run_pipeline(', 'run_pipeline(')

c = re.sub(r'from src.domain_models import AudioChunk as _AudioChunk\n\s*from src.domain_models import PipelineConfig as _PipelineConfig\n\s*from src.domain_models import SpeechSegment as _SpeechSegment', 'from src.domain_models import AudioChunk, PipelineConfig, SpeechSegment', c)
c = c.replace('_AudioChunk(', 'AudioChunk(')
c = c.replace('_PipelineConfig()', 'PipelineConfig()')
c = c.replace('_SpeechSegment(', 'SpeechSegment(')

with open("tutorials/UAT_AND_TUTORIAL.py", "w") as f:
    f.write(c)
