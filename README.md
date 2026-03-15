# MeetingNoter

## Overview
MeetingNoter is a highly secure, privacy-preserving voice analysis and speaker diarization pipeline. It is specifically engineered to safely transcribe and diarize highly sensitive conversational data, such as Customer Problem Fit (CPF) interviews, without transmitting any audio to commercial third-party Speech-to-Text (STT) APIs like Google Cloud Speech or OpenAI Whisper API.

By running open-source foundation models (Whisper, Silero VAD, Pyannote.audio) locally, MeetingNoter completely eliminates the risk of PII leakage while ensuring zero-marginal-cost processing.

## Features
- **Secure Processing:** Runs entirely locally, ensuring sensitive audio data never leaves your environment.
- **Robust Ingestion:** Securely downloads audio files directly from Google Drive.
- **Memory-Efficient Processing:** Automatically chunks long audio files using FFmpeg to prevent Out-Of-Memory (OOM) crashes, enabling the processing of hour-long interviews on standard hardware.
- **Voice Activity Detection (VAD) Gating:** Uses Silero VAD to detect speech segments, mathematically eliminating Whisper's hallucination loops during silent periods.
- **High-Accuracy Transcription:** Utilizes the highly optimized `faster-whisper` engine (CTranslate2) with specific configurations for the Japanese language to prevent data loss.

## Installation

Ensure you have Python 3.12+ and `uv` installed.

```bash
# Install dependencies
uv sync

# Ensure ffmpeg is installed on your system
# (e.g., sudo apt install ffmpeg OR brew install ffmpeg)
```

## Usage

MeetingNoter is currently in development. You can verify the pipeline components using the included test suite.

```bash
# Run the test suite
uv run pytest
```

### Interactive UAT Notebook

An interactive tutorial and User Acceptance Testing (UAT) notebook is available using Marimo.

```bash
# Run the UAT notebook
PYTHONPATH=src uv run marimo edit tests/uat/UAT_AND_TUTORIAL.py
```

## Structure
- `src/domain_models/`: Pydantic-based data contracts and protocol interfaces.
- `src/meetingnoter/ingestion/`: Modules for securely downloading audio data.
- `src/meetingnoter/processing/`: Core processing logic (chunking, VAD, transcription).
- `tests/`: Unit, integration, and UAT test suites.
