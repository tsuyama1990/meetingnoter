import subprocess
from pathlib import Path
import os
p = Path('dev_documents')
p.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['PYTHONPATH'] = 'src'
# Generate the log expected by the precommit instructions
res = subprocess.run(['uv', 'run', 'pytest', '--cov=src/domain_models', '--cov=src/meetingnoter', '--cov-report=term-missing', 'tests/unit/', 'tests/uat/'], capture_output=True, text=True, check=False, env=env)
(p / 'test_execution_log.txt').write_text(res.stdout + res.stderr)
print(f'✓ Log saved: {p / "test_execution_log.txt"}')
