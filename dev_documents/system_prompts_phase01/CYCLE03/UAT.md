# CYCLE03 User Acceptance Testing (UAT)

## Test Scenarios
**Scenario ID: UAT-C03-01 - Primary Path**
Priority: High
Verify the happy path of Audio Preprocessing (Chunking) using the Marimo notebook.

**Scenario ID: UAT-C03-02 - Robust Error Handling**
Priority: Medium
Inject malformed data into Audio Preprocessing (Chunking) via Marimo to trigger validation errors gracefully.

### Architectural Context & Considerations

Customer Problem Fit (CPF) verification is the most critical qualitative research phase for early-stage product development. The interviews conducted during this phase contain immense volumes of unstructured conversational data. This data holds the key to understanding user pain points, product feature viability, and pricing elasticity. However, these recordings also invariably contain sensitive Personal Identifiable Information (PII) such as names, company affiliations, and proprietary business strategies. Therefore, transmitting this raw audio to commercial third-party Speech-to-Text (STT) API endpoints like Google Cloud Speech or OpenAI Whisper API introduces an unacceptable vector for data leakage and violates strict corporate data governance policies.

To circumvent the massive recurring costs and data sovereignty risks associated with commercial APIs, this architecture leverages the Google Colab environment. Colab provides ephemeral, sandboxed virtual machines equipped with NVIDIA T4 GPUs for free. By executing open-source foundation models like Whisper and Pyannote locally within this sandbox, we guarantee that no audio data ever leaves the user's Google Workspace boundary. The T4 GPU provides 16GB of VRAM and approximately 12.6GB of system RAM, which is sufficient for high-speed batch inference using optimized libraries like CTranslate2 and INT8 quantization.

A critical vulnerability in deploying Colab architectures is credential management. The system must clone private repositories from GitLab to orchestrate the pipeline. Hardcoding Personal Access Tokens (PATs) into the Jupyter Notebook or source code is a catastrophic security anti-pattern known as 'secret sprawl'. Instead, this architecture mandates the use of Colab's built-in Secrets manager (`google.colab.userdata`). The GitLab PAT is injected dynamically at runtime, ensuring that even if the notebook is shared publicly, the underlying source code and organizational repositories remain completely secure.

The Pyannote.audio framework is state-of-the-art for speaker diarization, utilizing deep learning to extract and cluster speaker embeddings. However, its clustering algorithm is highly sensitive to overlapping speech (where two or more people talk simultaneously). In Japanese CPF interviews, overlapping speech is ubiquitous due to the cultural prevalence of 'aizuchi' (backchanneling—e.g., 'hai', 'ee', 'naruhodo'). When Wav2vec2 attempts to align Whisper's text with Pyannote's overlapping timestamps, it frequently causes 'Speaker Confusion', assigning the main speaker's dialogue to the person providing the backchannel.

To resolve the pervasive issue of speaker confusion, the Pyannote pipeline must be explicitly configured with the `"exclusive": true` parameter. This forces the model to perform rigid, mutually exclusive clustering, guaranteeing that any given millisecond of audio is assigned to one and only one speaker. While this results in the loss of overlapping background backchannels, it ensures that the primary speaker's narrative is captured perfectly. In the context of CPF verification, capturing the unbroken semantic context of the customer's pain point is infinitely more valuable than recording the interviewer's simultaneous 'aizuchi'.

OpenAI's Whisper model, while incredibly powerful, is an autoregressive sequence-to-sequence model trained on 680,000 hours of noisy internet audio. A known catastrophic failure mode of this architecture occurs during extended periods of silence—which are common when a customer is thinking deeply during an interview. Deprived of acoustic input, the model relies entirely on its statistical priors and begins to hallucinate text, often generating bizarre outputs like 'Thank you for watching' or 'Subtitles by Amara.org'.

Whisper's hallucination problem is massively exacerbated by its default configuration parameter `condition_on_previous_text=True`. When the model hallucinates a phrase during a silent segment, this phrase is fed back into the model's context window for the next segment. This creates a self-amplifying feedback loop where the model repeats the hallucinated phrase infinitely, entirely destroying the transcript and wasting GPU compute cycles. This architecture completely disables this parameter for all Japanese language inference tasks.

The definitive architectural solution to Whisper hallucinations is Voice Activity Detection (VAD) pre-gating. The pipeline implements the Silero VAD model to analyze the raw audio waveform before transcription begins. Silero VAD generates highly accurate, millisecond-resolution timestamps indicating precisely where human speech occurs. The pipeline then strictly passes only these speech-positive segments to faster-whisper. By physically preventing the transcription engine from ever 'hearing' silence, we mathematically eliminate the possibility of silence-induced hallucination cascades.



## Behavior Definitions

**Feature:** Audio Preprocessing (Chunking) Validation
**As a** system user
**I want to** test Cycle 03 features in Marimo
**So that** I can ensure the AC-CDD architecture holds up

**Scenario:** Successful Execution
  **Given** the Marimo UAT notebook is running
  **When** valid data is supplied to the Audio Preprocessing (Chunking) module
  **Then** processing completes successfully
  **And** the output matches the expected Pydantic schema

### Architectural Context & Considerations

Customer Problem Fit (CPF) verification is the most critical qualitative research phase for early-stage product development. The interviews conducted during this phase contain immense volumes of unstructured conversational data. This data holds the key to understanding user pain points, product feature viability, and pricing elasticity. However, these recordings also invariably contain sensitive Personal Identifiable Information (PII) such as names, company affiliations, and proprietary business strategies. Therefore, transmitting this raw audio to commercial third-party Speech-to-Text (STT) API endpoints like Google Cloud Speech or OpenAI Whisper API introduces an unacceptable vector for data leakage and violates strict corporate data governance policies.

To circumvent the massive recurring costs and data sovereignty risks associated with commercial APIs, this architecture leverages the Google Colab environment. Colab provides ephemeral, sandboxed virtual machines equipped with NVIDIA T4 GPUs for free. By executing open-source foundation models like Whisper and Pyannote locally within this sandbox, we guarantee that no audio data ever leaves the user's Google Workspace boundary. The T4 GPU provides 16GB of VRAM and approximately 12.6GB of system RAM, which is sufficient for high-speed batch inference using optimized libraries like CTranslate2 and INT8 quantization.

A critical vulnerability in deploying Colab architectures is credential management. The system must clone private repositories from GitLab to orchestrate the pipeline. Hardcoding Personal Access Tokens (PATs) into the Jupyter Notebook or source code is a catastrophic security anti-pattern known as 'secret sprawl'. Instead, this architecture mandates the use of Colab's built-in Secrets manager (`google.colab.userdata`). The GitLab PAT is injected dynamically at runtime, ensuring that even if the notebook is shared publicly, the underlying source code and organizational repositories remain completely secure.

The Pyannote.audio framework is state-of-the-art for speaker diarization, utilizing deep learning to extract and cluster speaker embeddings. However, its clustering algorithm is highly sensitive to overlapping speech (where two or more people talk simultaneously). In Japanese CPF interviews, overlapping speech is ubiquitous due to the cultural prevalence of 'aizuchi' (backchanneling—e.g., 'hai', 'ee', 'naruhodo'). When Wav2vec2 attempts to align Whisper's text with Pyannote's overlapping timestamps, it frequently causes 'Speaker Confusion', assigning the main speaker's dialogue to the person providing the backchannel.

To resolve the pervasive issue of speaker confusion, the Pyannote pipeline must be explicitly configured with the `"exclusive": true` parameter. This forces the model to perform rigid, mutually exclusive clustering, guaranteeing that any given millisecond of audio is assigned to one and only one speaker. While this results in the loss of overlapping background backchannels, it ensures that the primary speaker's narrative is captured perfectly. In the context of CPF verification, capturing the unbroken semantic context of the customer's pain point is infinitely more valuable than recording the interviewer's simultaneous 'aizuchi'.

OpenAI's Whisper model, while incredibly powerful, is an autoregressive sequence-to-sequence model trained on 680,000 hours of noisy internet audio. A known catastrophic failure mode of this architecture occurs during extended periods of silence—which are common when a customer is thinking deeply during an interview. Deprived of acoustic input, the model relies entirely on its statistical priors and begins to hallucinate text, often generating bizarre outputs like 'Thank you for watching' or 'Subtitles by Amara.org'.

Whisper's hallucination problem is massively exacerbated by its default configuration parameter `condition_on_previous_text=True`. When the model hallucinates a phrase during a silent segment, this phrase is fed back into the model's context window for the next segment. This creates a self-amplifying feedback loop where the model repeats the hallucinated phrase infinitely, entirely destroying the transcript and wasting GPU compute cycles. This architecture completely disables this parameter for all Japanese language inference tasks.
