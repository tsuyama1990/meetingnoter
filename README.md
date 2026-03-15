# MeetingNoter

## Overview
MeetingNoter is a highly secure, privacy-preserving voice analysis and speaker diarization pipeline. It is specifically engineered to safely transcribe and diarize highly sensitive conversational data, such as Customer Problem Fit (CPF) interviews, without transmitting any audio to commercial third-party Speech-to-Text (STT) APIs like Google Cloud Speech or OpenAI Whisper API.

By running open-source foundation models (Whisper, Silero VAD, Pyannote.audio) locally, MeetingNoter completely eliminates the risk of PII leakage while ensuring zero-marginal-cost processing.

## Features
- **Secure Processing:** Runs entirely locally, ensuring sensitive audio data never leaves your environment.
- **Robust Ingestion:** Securely downloads audio files directly from Google Drive.
- **Memory-Efficient Processing:** Automatically chunks long audio files using FFmpeg to prevent Out-Of-Memory (OOM) crashes, enabling the processing of hour-long interviews on standard hardware.
- **Aggressive Memory Management:** The orchestrated pipeline explicitly manages garbage collection and PyTorch CUDA cache clearing between heavy inference operations to guarantee stable execution.
- **Voice Activity Detection (VAD) Gating:** Uses Silero VAD to detect speech segments, mathematically eliminating Whisper's hallucination loops during silent periods.
- **High-Accuracy Transcription:** Utilizes the highly optimized `faster-whisper` engine (CTranslate2) with specific configurations for the Japanese language to prevent data loss.
- **Accurate Speaker Diarization:** Integrates `pyannote.audio` to identify speaker turns and accurately segment audio with specific overlap prevention settings to accommodate Japanese conversational styles (e.g., 'aizuchi').
- **Temporal Alignment and Data Aggregation:** Unifies output boundaries and synchronizes localized transcript timestamps back into a comprehensive, continuous global timeline suitable for final CPF qualitative analysis.

## Installation

Ensure you have Python 3.12+ and `uv` installed.

```bash
# Install dependencies
uv sync

# Ensure ffmpeg is installed on your system
# (e.g., sudo apt install ffmpeg OR brew install ffmpeg)
```

## Usage

You can run the full orchestration pipeline to transcribe and diarize audio straight from Google Drive:

```bash
# Set required environment variables
export GOOGLE_API_KEY="your_google_drive_api_key"
export PYANNOTE_AUTH_TOKEN="your_hugging_face_token"
export FILE_ID="your_google_drive_file_id"

# Run the orchestrated pipeline
uv run main.py
```

### Testing and Interactive UAT Notebook

You can verify the pipeline components using the included test suite.

```bash
# Run the test suite
uv run pytest
```

An interactive tutorial and User Acceptance Testing (UAT) notebook is available using Marimo, demonstrating the capabilities of all pipeline stages.

```bash
# Run the UAT notebook
PYTHONPATH=src uv run marimo edit tests/uat/UAT_AND_TUTORIAL.py
```

## Structure
- `src/domain_models/`: Pydantic-based data contracts and protocol interfaces.
- `src/meetingnoter/ingestion/`: Modules for securely downloading audio data.
- `src/meetingnoter/processing/`: Core processing logic (chunking, VAD, transcription, diarization, aggregation).
- `tests/`: Unit, integration, and UAT test suites.
