import os
import subprocess

# Change to the root directory
os.chdir("/app")

# Run pytest specifically for the newly created test file
env = os.environ.copy()
env["PYTHONPATH"] = "src"
# Override addopts in pyproject.toml
res = subprocess.run(
    [
        "uv",
        "run",
        "pytest",
        "tests/unit/test_cycle_05.py",
        "--cov=meetingnoter.processing.transcriber",
        "--cov-report=term-missing",
    ],
    capture_output=True,
    text=True,
    env=env,
)

print(res.stdout)
print(res.stderr)
