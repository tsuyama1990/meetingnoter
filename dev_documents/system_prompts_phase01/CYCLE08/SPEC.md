# CYCLE08 Specification: Data Aggregation and UAT Finalization

## Summary
This cycle focuses on implementing Data Aggregation and UAT Finalization. Cycle 08 finalizes the system. `merger.py` consumes the localized chunk outputs and applies the temporal offsets calculated in Cycle 03 to shift all timestamps back to the global timeline. It resolves speaker boundary IDs. Finally, we implement the `UAT_AND_TUTORIAL.py` Marimo notebook, providing a rich, reactive UI with a 'Mock Mode' for rapid testing and a 'Real Mode' for GPU execution.

### Architectural Context & Considerations

Customer Problem Fit (CPF) verification is the most critical qualitative research phase for early-stage product development. The interviews conducted during this phase contain immense volumes of unstructured conversational data. This data holds the key to understanding user pain points, product feature viability, and pricing elasticity. However, these recordings also invariably contain sensitive Personal Identifiable Information (PII) such as names, company affiliations, and proprietary business strategies. Therefore, transmitting this raw audio to commercial third-party Speech-to-Text (STT) API endpoints like Google Cloud Speech or OpenAI Whisper API introduces an unacceptable vector for data leakage and violates strict corporate data governance policies.

To circumvent the massive recurring costs and data sovereignty risks associated with commercial APIs, this architecture leverages the Google Colab environment. Colab provides ephemeral, sandboxed virtual machines equipped with NVIDIA T4 GPUs for free. By executing open-source foundation models like Whisper and Pyannote locally within this sandbox, we guarantee that no audio data ever leaves the user's Google Workspace boundary. The T4 GPU provides 16GB of VRAM and approximately 12.6GB of system RAM, which is sufficient for high-speed batch inference using optimized libraries like CTranslate2 and INT8 quantization.

A critical vulnerability in deploying Colab architectures is credential management. The system must clone private repositories from GitLab to orchestrate the pipeline. Hardcoding Personal Access Tokens (PATs) into the Jupyter Notebook or source code is a catastrophic security anti-pattern known as 'secret sprawl'. Instead, this architecture mandates the use of Colab's built-in Secrets manager (`google.colab.userdata`). The GitLab PAT is injected dynamically at runtime, ensuring that even if the notebook is shared publicly, the underlying source code and organizational repositories remain completely secure.

The Pyannote.audio framework is state-of-the-art for speaker diarization, utilizing deep learning to extract and cluster speaker embeddings. However, its clustering algorithm is highly sensitive to overlapping speech (where two or more people talk simultaneously). In Japanese CPF interviews, overlapping speech is ubiquitous due to the cultural prevalence of 'aizuchi' (backchanneling—e.g., 'hai', 'ee', 'naruhodo'). When Wav2vec2 attempts to align Whisper's text with Pyannote's overlapping timestamps, it frequently causes 'Speaker Confusion', assigning the main speaker's dialogue to the person providing the backchannel.

To resolve the pervasive issue of speaker confusion, the Pyannote pipeline must be explicitly configured with the `"exclusive": true` parameter. This forces the model to perform rigid, mutually exclusive clustering, guaranteeing that any given millisecond of audio is assigned to one and only one speaker. While this results in the loss of overlapping background backchannels, it ensures that the primary speaker's narrative is captured perfectly. In the context of CPF verification, capturing the unbroken semantic context of the customer's pain point is infinitely more valuable than recording the interviewer's simultaneous 'aizuchi'.

OpenAI's Whisper model, while incredibly powerful, is an autoregressive sequence-to-sequence model trained on 680,000 hours of noisy internet audio. A known catastrophic failure mode of this architecture occurs during extended periods of silence—which are common when a customer is thinking deeply during an interview. Deprived of acoustic input, the model relies entirely on its statistical priors and begins to hallucinate text, often generating bizarre outputs like 'Thank you for watching' or 'Subtitles by Amara.org'.



## System Architecture
The architecture demands strict isolation for this component. It must strictly adhere to the defined Pydantic boundaries.

### File Structure
```text
.
├── src/
│   ├── meetingnoter/
│   │   ├── domain/
│   │   │   ├── models.py
│   │   │   └── interfaces.py
│   │   ├── ingestion/
│   │   │   └── drive_client.py
│   │   ├── processing/
│   │   │   ├── chunker.py
│   │   │   ├── vad.py
│   │   │   ├── transcriber.py
│   │   │   └── diarizer.py
│   │   ├── aggregation/
│   │   │   └── **merger.py**
│   │   └── pipeline.py
├── tests/
│   ├── unit/
│   │   └── **test_cycle_08.py**
│   └── integration/
│       └── **test_integration_08.py**
```

### Architectural Context & Considerations

Customer Problem Fit (CPF) verification is the most critical qualitative research phase for early-stage product development. The interviews conducted during this phase contain immense volumes of unstructured conversational data. This data holds the key to understanding user pain points, product feature viability, and pricing elasticity. However, these recordings also invariably contain sensitive Personal Identifiable Information (PII) such as names, company affiliations, and proprietary business strategies. Therefore, transmitting this raw audio to commercial third-party Speech-to-Text (STT) API endpoints like Google Cloud Speech or OpenAI Whisper API introduces an unacceptable vector for data leakage and violates strict corporate data governance policies.

To circumvent the massive recurring costs and data sovereignty risks associated with commercial APIs, this architecture leverages the Google Colab environment. Colab provides ephemeral, sandboxed virtual machines equipped with NVIDIA T4 GPUs for free. By executing open-source foundation models like Whisper and Pyannote locally within this sandbox, we guarantee that no audio data ever leaves the user's Google Workspace boundary. The T4 GPU provides 16GB of VRAM and approximately 12.6GB of system RAM, which is sufficient for high-speed batch inference using optimized libraries like CTranslate2 and INT8 quantization.

A critical vulnerability in deploying Colab architectures is credential management. The system must clone private repositories from GitLab to orchestrate the pipeline. Hardcoding Personal Access Tokens (PATs) into the Jupyter Notebook or source code is a catastrophic security anti-pattern known as 'secret sprawl'. Instead, this architecture mandates the use of Colab's built-in Secrets manager (`google.colab.userdata`). The GitLab PAT is injected dynamically at runtime, ensuring that even if the notebook is shared publicly, the underlying source code and organizational repositories remain completely secure.

The Pyannote.audio framework is state-of-the-art for speaker diarization, utilizing deep learning to extract and cluster speaker embeddings. However, its clustering algorithm is highly sensitive to overlapping speech (where two or more people talk simultaneously). In Japanese CPF interviews, overlapping speech is ubiquitous due to the cultural prevalence of 'aizuchi' (backchanneling—e.g., 'hai', 'ee', 'naruhodo'). When Wav2vec2 attempts to align Whisper's text with Pyannote's overlapping timestamps, it frequently causes 'Speaker Confusion', assigning the main speaker's dialogue to the person providing the backchannel.

To resolve the pervasive issue of speaker confusion, the Pyannote pipeline must be explicitly configured with the `"exclusive": true` parameter. This forces the model to perform rigid, mutually exclusive clustering, guaranteeing that any given millisecond of audio is assigned to one and only one speaker. While this results in the loss of overlapping background backchannels, it ensures that the primary speaker's narrative is captured perfectly. In the context of CPF verification, capturing the unbroken semantic context of the customer's pain point is infinitely more valuable than recording the interviewer's simultaneous 'aizuchi'.

OpenAI's Whisper model, while incredibly powerful, is an autoregressive sequence-to-sequence model trained on 680,000 hours of noisy internet audio. A known catastrophic failure mode of this architecture occurs during extended periods of silence—which are common when a customer is thinking deeply during an interview. Deprived of acoustic input, the model relies entirely on its statistical priors and begins to hallucinate text, often generating bizarre outputs like 'Thank you for watching' or 'Subtitles by Amara.org'.

Whisper's hallucination problem is massively exacerbated by its default configuration parameter `condition_on_previous_text=True`. When the model hallucinates a phrase during a silent segment, this phrase is fed back into the model's context window for the next segment. This creates a self-amplifying feedback loop where the model repeats the hallucinated phrase infinitely, entirely destroying the transcript and wasting GPU compute cycles. This architecture completely disables this parameter for all Japanese language inference tasks.

The definitive architectural solution to Whisper hallucinations is Voice Activity Detection (VAD) pre-gating. The pipeline implements the Silero VAD model to analyze the raw audio waveform before transcription begins. Silero VAD generates highly accurate, millisecond-resolution timestamps indicating precisely where human speech occurs. The pipeline then strictly passes only these speech-positive segments to faster-whisper. By physically preventing the transcription engine from ever 'hearing' silence, we mathematically eliminate the possibility of silence-induced hallucination cascades.

The Google Colab T4 environment has a hard limit of ~12.6GB of system RAM. While faster-whisper VRAM consumption is manageable, Pyannote's agglomerative hierarchical clustering algorithm creates a pairwise distance matrix of speaker embeddings that scales quadratically with audio duration. Processing a standard 60-minute interview in a single pass will inevitably cause the matrix to exceed 25GB, resulting in a sudden, unrecoverable Out-Of-Memory (OOM) kernel crash. This represents the single largest technical barrier to utilizing the free Colab tier for enterprise-scale audio processing.

To physically guarantee that OOM crashes cannot occur, the architecture mandates an aggressive 'Audio Chunking' strategy. Before any AI inference begins, the ingestion module utilizes an FFmpeg subprocess to slice the continuous interview recording into discrete, manageable segments (e.g., 20 or 30 minutes in length). By processing these chunks sequentially rather than concurrently, the maximum size of the Pyannote distance matrix is strictly bounded, ensuring that system RAM consumption never spikes above the Colab limits.

Chunking solves the OOM problem but introduces state management complexity. After a 20-minute chunk is processed through the heavy Whisper and Pyannote models, the Python runtime will hold massive tensor objects in memory. The pipeline orchestrator must explicitly invoke `gc.collect()` to trigger Python's garbage collector, immediately followed by `torch.cuda.empty_cache()` to force PyTorch to release unoccupied VRAM back to the operating system. Only after this rigorous memory scrubbing is complete will the orchestrator proceed to load the next audio chunk.

The 'faster-whisper' implementation, utilizing the CTranslate2 engine, is significantly faster than the reference OpenAI implementation. However, it contains hardcoded threshold parameters (`compression_ratio_threshold` and `log_prob_threshold`) designed for space-separated languages like English. When applied to logographic languages like Japanese or Chinese, these default thresholds frequently misclassify valid speech as 'compression anomalies' or 'low-confidence noise', resulting in massive chunks of perfect audio being silently dropped from the final transcript.

To ensure zero data loss during Japanese transcription, the pipeline explicitly overrides the faster-whisper default heuristics. By passing `compression_ratio_threshold=None` and `log_prob_threshold=None` to the transcribe method, we force the model to return the decoded text regardless of its internal confidence scores or compression ratios. Since we have already guaranteed via Silero VAD that the audio segment contains valid human speech, these Whisper-internal heuristics are redundant and actively harmful to the pipeline's accuracy.



## Design Architecture
We design the data structures and validation rules first. Pydantic models serve as the unbreachable contract.

### Architectural Context & Considerations

Customer Problem Fit (CPF) verification is the most critical qualitative research phase for early-stage product development. The interviews conducted during this phase contain immense volumes of unstructured conversational data. This data holds the key to understanding user pain points, product feature viability, and pricing elasticity. However, these recordings also invariably contain sensitive Personal Identifiable Information (PII) such as names, company affiliations, and proprietary business strategies. Therefore, transmitting this raw audio to commercial third-party Speech-to-Text (STT) API endpoints like Google Cloud Speech or OpenAI Whisper API introduces an unacceptable vector for data leakage and violates strict corporate data governance policies.

To circumvent the massive recurring costs and data sovereignty risks associated with commercial APIs, this architecture leverages the Google Colab environment. Colab provides ephemeral, sandboxed virtual machines equipped with NVIDIA T4 GPUs for free. By executing open-source foundation models like Whisper and Pyannote locally within this sandbox, we guarantee that no audio data ever leaves the user's Google Workspace boundary. The T4 GPU provides 16GB of VRAM and approximately 12.6GB of system RAM, which is sufficient for high-speed batch inference using optimized libraries like CTranslate2 and INT8 quantization.

A critical vulnerability in deploying Colab architectures is credential management. The system must clone private repositories from GitLab to orchestrate the pipeline. Hardcoding Personal Access Tokens (PATs) into the Jupyter Notebook or source code is a catastrophic security anti-pattern known as 'secret sprawl'. Instead, this architecture mandates the use of Colab's built-in Secrets manager (`google.colab.userdata`). The GitLab PAT is injected dynamically at runtime, ensuring that even if the notebook is shared publicly, the underlying source code and organizational repositories remain completely secure.

The Pyannote.audio framework is state-of-the-art for speaker diarization, utilizing deep learning to extract and cluster speaker embeddings. However, its clustering algorithm is highly sensitive to overlapping speech (where two or more people talk simultaneously). In Japanese CPF interviews, overlapping speech is ubiquitous due to the cultural prevalence of 'aizuchi' (backchanneling—e.g., 'hai', 'ee', 'naruhodo'). When Wav2vec2 attempts to align Whisper's text with Pyannote's overlapping timestamps, it frequently causes 'Speaker Confusion', assigning the main speaker's dialogue to the person providing the backchannel.

To resolve the pervasive issue of speaker confusion, the Pyannote pipeline must be explicitly configured with the `"exclusive": true` parameter. This forces the model to perform rigid, mutually exclusive clustering, guaranteeing that any given millisecond of audio is assigned to one and only one speaker. While this results in the loss of overlapping background backchannels, it ensures that the primary speaker's narrative is captured perfectly. In the context of CPF verification, capturing the unbroken semantic context of the customer's pain point is infinitely more valuable than recording the interviewer's simultaneous 'aizuchi'.

OpenAI's Whisper model, while incredibly powerful, is an autoregressive sequence-to-sequence model trained on 680,000 hours of noisy internet audio. A known catastrophic failure mode of this architecture occurs during extended periods of silence—which are common when a customer is thinking deeply during an interview. Deprived of acoustic input, the model relies entirely on its statistical priors and begins to hallucinate text, often generating bizarre outputs like 'Thank you for watching' or 'Subtitles by Amara.org'.

Whisper's hallucination problem is massively exacerbated by its default configuration parameter `condition_on_previous_text=True`. When the model hallucinates a phrase during a silent segment, this phrase is fed back into the model's context window for the next segment. This creates a self-amplifying feedback loop where the model repeats the hallucinated phrase infinitely, entirely destroying the transcript and wasting GPU compute cycles. This architecture completely disables this parameter for all Japanese language inference tasks.



## Implementation Approach
We follow TDD. First, we write the test asserting the exact required behavior. Then we implement the core logic for Data Aggregation and UAT Finalization, ensuring cyclomatic complexity never exceeds 10.

### Architectural Context & Considerations

Customer Problem Fit (CPF) verification is the most critical qualitative research phase for early-stage product development. The interviews conducted during this phase contain immense volumes of unstructured conversational data. This data holds the key to understanding user pain points, product feature viability, and pricing elasticity. However, these recordings also invariably contain sensitive Personal Identifiable Information (PII) such as names, company affiliations, and proprietary business strategies. Therefore, transmitting this raw audio to commercial third-party Speech-to-Text (STT) API endpoints like Google Cloud Speech or OpenAI Whisper API introduces an unacceptable vector for data leakage and violates strict corporate data governance policies.

To circumvent the massive recurring costs and data sovereignty risks associated with commercial APIs, this architecture leverages the Google Colab environment. Colab provides ephemeral, sandboxed virtual machines equipped with NVIDIA T4 GPUs for free. By executing open-source foundation models like Whisper and Pyannote locally within this sandbox, we guarantee that no audio data ever leaves the user's Google Workspace boundary. The T4 GPU provides 16GB of VRAM and approximately 12.6GB of system RAM, which is sufficient for high-speed batch inference using optimized libraries like CTranslate2 and INT8 quantization.

A critical vulnerability in deploying Colab architectures is credential management. The system must clone private repositories from GitLab to orchestrate the pipeline. Hardcoding Personal Access Tokens (PATs) into the Jupyter Notebook or source code is a catastrophic security anti-pattern known as 'secret sprawl'. Instead, this architecture mandates the use of Colab's built-in Secrets manager (`google.colab.userdata`). The GitLab PAT is injected dynamically at runtime, ensuring that even if the notebook is shared publicly, the underlying source code and organizational repositories remain completely secure.

The Pyannote.audio framework is state-of-the-art for speaker diarization, utilizing deep learning to extract and cluster speaker embeddings. However, its clustering algorithm is highly sensitive to overlapping speech (where two or more people talk simultaneously). In Japanese CPF interviews, overlapping speech is ubiquitous due to the cultural prevalence of 'aizuchi' (backchanneling—e.g., 'hai', 'ee', 'naruhodo'). When Wav2vec2 attempts to align Whisper's text with Pyannote's overlapping timestamps, it frequently causes 'Speaker Confusion', assigning the main speaker's dialogue to the person providing the backchannel.

To resolve the pervasive issue of speaker confusion, the Pyannote pipeline must be explicitly configured with the `"exclusive": true` parameter. This forces the model to perform rigid, mutually exclusive clustering, guaranteeing that any given millisecond of audio is assigned to one and only one speaker. While this results in the loss of overlapping background backchannels, it ensures that the primary speaker's narrative is captured perfectly. In the context of CPF verification, capturing the unbroken semantic context of the customer's pain point is infinitely more valuable than recording the interviewer's simultaneous 'aizuchi'.

OpenAI's Whisper model, while incredibly powerful, is an autoregressive sequence-to-sequence model trained on 680,000 hours of noisy internet audio. A known catastrophic failure mode of this architecture occurs during extended periods of silence—which are common when a customer is thinking deeply during an interview. Deprived of acoustic input, the model relies entirely on its statistical priors and begins to hallucinate text, often generating bizarre outputs like 'Thank you for watching' or 'Subtitles by Amara.org'.

Whisper's hallucination problem is massively exacerbated by its default configuration parameter `condition_on_previous_text=True`. When the model hallucinates a phrase during a silent segment, this phrase is fed back into the model's context window for the next segment. This creates a self-amplifying feedback loop where the model repeats the hallucinated phrase infinitely, entirely destroying the transcript and wasting GPU compute cycles. This architecture completely disables this parameter for all Japanese language inference tasks.

The definitive architectural solution to Whisper hallucinations is Voice Activity Detection (VAD) pre-gating. The pipeline implements the Silero VAD model to analyze the raw audio waveform before transcription begins. Silero VAD generates highly accurate, millisecond-resolution timestamps indicating precisely where human speech occurs. The pipeline then strictly passes only these speech-positive segments to faster-whisper. By physically preventing the transcription engine from ever 'hearing' silence, we mathematically eliminate the possibility of silence-induced hallucination cascades.

The Google Colab T4 environment has a hard limit of ~12.6GB of system RAM. While faster-whisper VRAM consumption is manageable, Pyannote's agglomerative hierarchical clustering algorithm creates a pairwise distance matrix of speaker embeddings that scales quadratically with audio duration. Processing a standard 60-minute interview in a single pass will inevitably cause the matrix to exceed 25GB, resulting in a sudden, unrecoverable Out-Of-Memory (OOM) kernel crash. This represents the single largest technical barrier to utilizing the free Colab tier for enterprise-scale audio processing.



## Test Strategy
### Unit Testing Approach
Extensive use of `unittest.mock` is required to isolate external calls.

### Integration Testing Approach
Verify the handoff between this module and adjacent modules using synthetic datasets.

### Architectural Context & Considerations

Customer Problem Fit (CPF) verification is the most critical qualitative research phase for early-stage product development. The interviews conducted during this phase contain immense volumes of unstructured conversational data. This data holds the key to understanding user pain points, product feature viability, and pricing elasticity. However, these recordings also invariably contain sensitive Personal Identifiable Information (PII) such as names, company affiliations, and proprietary business strategies. Therefore, transmitting this raw audio to commercial third-party Speech-to-Text (STT) API endpoints like Google Cloud Speech or OpenAI Whisper API introduces an unacceptable vector for data leakage and violates strict corporate data governance policies.

To circumvent the massive recurring costs and data sovereignty risks associated with commercial APIs, this architecture leverages the Google Colab environment. Colab provides ephemeral, sandboxed virtual machines equipped with NVIDIA T4 GPUs for free. By executing open-source foundation models like Whisper and Pyannote locally within this sandbox, we guarantee that no audio data ever leaves the user's Google Workspace boundary. The T4 GPU provides 16GB of VRAM and approximately 12.6GB of system RAM, which is sufficient for high-speed batch inference using optimized libraries like CTranslate2 and INT8 quantization.

A critical vulnerability in deploying Colab architectures is credential management. The system must clone private repositories from GitLab to orchestrate the pipeline. Hardcoding Personal Access Tokens (PATs) into the Jupyter Notebook or source code is a catastrophic security anti-pattern known as 'secret sprawl'. Instead, this architecture mandates the use of Colab's built-in Secrets manager (`google.colab.userdata`). The GitLab PAT is injected dynamically at runtime, ensuring that even if the notebook is shared publicly, the underlying source code and organizational repositories remain completely secure.

The Pyannote.audio framework is state-of-the-art for speaker diarization, utilizing deep learning to extract and cluster speaker embeddings. However, its clustering algorithm is highly sensitive to overlapping speech (where two or more people talk simultaneously). In Japanese CPF interviews, overlapping speech is ubiquitous due to the cultural prevalence of 'aizuchi' (backchanneling—e.g., 'hai', 'ee', 'naruhodo'). When Wav2vec2 attempts to align Whisper's text with Pyannote's overlapping timestamps, it frequently causes 'Speaker Confusion', assigning the main speaker's dialogue to the person providing the backchannel.

To resolve the pervasive issue of speaker confusion, the Pyannote pipeline must be explicitly configured with the `"exclusive": true` parameter. This forces the model to perform rigid, mutually exclusive clustering, guaranteeing that any given millisecond of audio is assigned to one and only one speaker. While this results in the loss of overlapping background backchannels, it ensures that the primary speaker's narrative is captured perfectly. In the context of CPF verification, capturing the unbroken semantic context of the customer's pain point is infinitely more valuable than recording the interviewer's simultaneous 'aizuchi'.

OpenAI's Whisper model, while incredibly powerful, is an autoregressive sequence-to-sequence model trained on 680,000 hours of noisy internet audio. A known catastrophic failure mode of this architecture occurs during extended periods of silence—which are common when a customer is thinking deeply during an interview. Deprived of acoustic input, the model relies entirely on its statistical priors and begins to hallucinate text, often generating bizarre outputs like 'Thank you for watching' or 'Subtitles by Amara.org'.

Whisper's hallucination problem is massively exacerbated by its default configuration parameter `condition_on_previous_text=True`. When the model hallucinates a phrase during a silent segment, this phrase is fed back into the model's context window for the next segment. This creates a self-amplifying feedback loop where the model repeats the hallucinated phrase infinitely, entirely destroying the transcript and wasting GPU compute cycles. This architecture completely disables this parameter for all Japanese language inference tasks.

The definitive architectural solution to Whisper hallucinations is Voice Activity Detection (VAD) pre-gating. The pipeline implements the Silero VAD model to analyze the raw audio waveform before transcription begins. Silero VAD generates highly accurate, millisecond-resolution timestamps indicating precisely where human speech occurs. The pipeline then strictly passes only these speech-positive segments to faster-whisper. By physically preventing the transcription engine from ever 'hearing' silence, we mathematically eliminate the possibility of silence-induced hallucination cascades.

The Google Colab T4 environment has a hard limit of ~12.6GB of system RAM. While faster-whisper VRAM consumption is manageable, Pyannote's agglomerative hierarchical clustering algorithm creates a pairwise distance matrix of speaker embeddings that scales quadratically with audio duration. Processing a standard 60-minute interview in a single pass will inevitably cause the matrix to exceed 25GB, resulting in a sudden, unrecoverable Out-Of-Memory (OOM) kernel crash. This represents the single largest technical barrier to utilizing the free Colab tier for enterprise-scale audio processing.

To physically guarantee that OOM crashes cannot occur, the architecture mandates an aggressive 'Audio Chunking' strategy. Before any AI inference begins, the ingestion module utilizes an FFmpeg subprocess to slice the continuous interview recording into discrete, manageable segments (e.g., 20 or 30 minutes in length). By processing these chunks sequentially rather than concurrently, the maximum size of the Pyannote distance matrix is strictly bounded, ensuring that system RAM consumption never spikes above the Colab limits.
