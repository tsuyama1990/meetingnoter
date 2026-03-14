# MeetingNoter

MeetingNoter is a highly secure, privacy-preserving voice analysis and speaker diarization pipeline specifically designed for the rigorous demands of qualitative research processes.

It provides robust, highly accurate Speech-to-Text transcription and speaker recognition capabilities while running fully locally (or inside an ephemeral Google Colab instance). The strict zero-commercial-API architecture ensures that highly sensitive audio data, including Personal Identifiable Information (PII) and corporate secrets, never leaves your environment.

## Features
- **Pydantic-driven Core Schema**: Strong typing and validations enforce strict boundaries for parsing audio chunks, speech segments, transcripts, and speaker labels.
- **Secure Architecture Framework**: Defines strict abstractions (`StorageClient`, `AudioSplitter`, `SpeechDetector`, `Transcriber`, `Diarizer`) to plug in models and clients without mixing concerns.

## Installation

Ensure you have `uv` installed, then run the following to install all required dependencies:

```bash
uv sync
```

## Usage

(Note: This pipeline is currently undergoing early development. Currently implemented are the core schemas and abstractions.)

To perform interactive testing of the core data schemas, run the User Acceptance Testing tutorial using Marimo:

```bash
PYTHONPATH=src uv run marimo edit tests/uat/UAT_AND_TUTORIAL.py
```

## Project Structure

- `src/domain_models/`: Core Pydantic schemas and interface protocols defining the entire project's contract boundaries.
- `tests/unit/`: Unit tests for domain models and mock protocols.
- `tests/e2e/`: End-to-end integration tests simulating a full pipeline run.
- `tests/uat/`: Interactive User Acceptance Testing scripts built with `marimo`.
