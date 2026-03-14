# User Test Scenario & Tutorial Plan

## 1. Test Scenarios

### Scenario ID: TS-001 - Quick Start Pipeline Execution (Priority: High)
This fundamental scenario evaluates the user's ability to seamlessly execute the core transcription and diarization pipeline using a standardized sample audio file. The objective is to ensure that a completely new user—perhaps a product manager with limited Python experience—can comfortably configure the Google Colab environment, inject the necessary GitLab credentials via the secure Secrets manager, and run the complete processing workflow without encountering technical friction, confusing tracebacks, or memory errors.

The user will be provided with a 10-minute synthetic Japanese interview audio file. This file is specifically engineered to contain typical conversation dynamics, including rapid backchanneling (aizuchi), short pauses, and minor overlapping speech. The user is expected to mount their Google Drive, paste the single-line initialization script into their Colab notebook, and execute the pipeline cell. They will observe the system's logging output as it automatically chunks the audio, filters out non-speech segments using the Silero VAD module, transcribes the speech with faster-whisper, and identifies the distinct speakers using Pyannote.

The scenario is considered successful if the user receives a fully formatted JSON output file in their designated Google Drive folder within a highly reasonable timeframe (e.g., under 5 minutes). The final transcript must demonstrate accurate text extraction and correct speaker labels applied to the overlapping speech segments, proving that the `exclusive: true` parameter successfully assigned words to the dominant speaker. This exact workflow constitutes the system's Aha! Moment. It vividly demonstrates the immediate, immense value of the zero-cost, highly secure architecture, proving that complex Natural Language Processing tasks can be orchestrated effortlessly in a constrained, free cloud environment without relying on expensive commercial APIs.


### Scenario ID: TS-002 - Advanced Parameter Tuning and Long Audio Processing (Priority: Medium)
This advanced scenario tests the system's extreme robustness and the user's ability to manipulate the underlying configuration to handle severe edge cases. It specifically focuses on long-duration audio files that typically trigger Out of Memory (OOM) crashes in standard whisper implementations. The user will upload a massive 90-minute audio recording, simulating a complex, in-depth, multi-topic CPF interview.

The scenario requires the user to interact with the system's configuration file, intentionally modifying the default parameters to suit this extreme load. They will reduce the chunk size from the standard 30 minutes to a more conservative 15 minutes, enforcing stricter memory constraints to guarantee stability. Additionally, the user will be tasked with tuning the VAD threshold settings, specifically lowering the `min_speech_duration_ms` to capture very brief instances of agreement that might otherwise be filtered out.

Upon executing the modified pipeline, the user must monitor the system RAM and GPU VRAM usage via the Colab resource dashboard. The critical success factor is observing the memory utilization graph: it must show periodic drops, confirming that the implemented garbage collection (`gc.collect()`) and VRAM clearing utilities (`torch.cuda.empty_cache()`) are successfully preventing memory spikes between chunk processing cycles. The flawless generation of the final 90-minute transcript without a kernel crash validates the architecture's profound resilience. This scenario empowers the user to process arbitrarily long recordings securely and dependably, proving the operational stability of the system under heavy, production-level loads.


## 2. Behavior Definitions
The behavior of the system is strictly defined using precise Gherkin-style syntax. This ensures absolute clarity and establishes a shared understanding of the operational contracts among developers, stakeholders, and quality assurance teams. These definitions explicitly outline the expected technical outcomes for various user interactions and systemic edge cases, leaving no room for ambiguity during the implementation phases.

GIVEN the user has successfully authenticated and mounted their personal Google Drive to the Google Colab environment,
AND the user has securely stored their GitLab Personal Access Token in the Colab Secrets manager under the designated key 'GITLAB_TOKEN',
WHEN the user executes the pipeline initialization script in the notebook cell,
THEN the system must dynamically retrieve the token using `google.colab.userdata.get` without exposing it in the plaintext output logs,
AND successfully clone the target private repository containing the WhisperX orchestrator codebase.

GIVEN an uploaded audio file (m4a/mp3) that exceeds 60 minutes in length, representing a lengthy user interview,
WHEN the orchestrator initiates the primary processing workflow,
THEN the system must automatically invoke the FFmpeg utility to physically divide the audio into distinct temporal segments no longer than 30 minutes each,
AND sequentially process each chunk while explicitly calling Python's garbage collector and PyTorch's `empty_cache()` method between iterations,
TO absolutely guarantee that the system RAM does not exceed the Colab T4's 12.67GB limit and cause a catastrophic Out of Memory (OOM) crash.

GIVEN a Japanese audio segment containing continuous background static noise or prolonged silences indicative of a thinking pause,
WHEN the audio is processed by the Silero Voice Activity Detection (VAD) gating module,
THEN the system must strictly filter out all non-speech frames based on the configured `threshold` and `min_speech_duration_ms` parameters,
AND pass only the highly confident, isolated speech segments to the faster-whisper transcriber engine,
SO THAT the language model is entirely prevented from hallucinating irrelevant text loops caused by its statistical attempt to decode absolute silence.

GIVEN an audio segment where the interviewer and the interviewee speak simultaneously, creating overlapping speech (aizuchi),
WHEN the Pyannote.audio module performs the complex speaker diarization algorithm,
THEN the system must rigorously apply the `exclusive: true` constraint within the pipeline parameters,
AND forcefully assign the temporal segment to the dominant speaker's cluster rather than generating ambiguous multi-speaker tags,
TO ensure that the subsequent mathematical alignment with the transcribed text remains coherent and highly accurate, prioritizing main conversational context over minor background affirmations.


## 3. Tutorial Strategy
The tutorial strategy is meticulously designed to transform the User Acceptance Testing (UAT) scenarios into an interactive, educational, and highly engaging experience for new users and stakeholders. To facilitate this seamless onboarding, the system will employ **Marimo**, a reactive Python notebook environment, to create fully executable tutorials. This modern approach allows users to directly interact with the code, modify tuning parameters in real-time, and instantly visualize the impact of their acoustic changes without the immense complexities of setting up a local GPU development environment from scratch.

The tutorial strategy incorporates two distinctly tailored execution modes to cater to different user contexts and security postures:
1. **Mock Mode (CI / No-API-Key Execution):** This mode is exceptionally crucial for continuous integration environments, automated testing, and users who wish to test the system's programmatic logic without utilizing actual PII-laden audio files or requiring access to the private GitLab repository. In Mock Mode, the pipeline actively bypasses the heavy, time-consuming ML model inferences. Instead, it utilizes pre-generated dummy data arrays and mocked algorithmic responses. This allows for rapid, millisecond-level validation of the architectural flow, the data serialization processes, and the final output JSON formatting.
2. **Real Mode:** This is the full-fidelity, production-level execution mode where the user utilizes their actual Google Drive, the Colab T4 GPU resources, and their secure GitLab credentials to process genuine, sensitive audio files. The Marimo tutorial will seamlessly guide the user in transitioning from Mock Mode to Real Mode. It will ensure they thoroughly understand the profound security implications and the necessary credential prerequisites before they handle sensitive, real-world customer data.

## 4. Tutorial Plan
The tutorial plan strictly mandates the creation of a **SINGLE** Marimo Text/Python file named `tutorials/UAT_AND_TUTORIAL.py`. This consolidated, monolithic file will comprehensively contain all necessary UAT scenarios, ranging smoothly from the Quick Start pipeline execution to the Advanced Parameter Tuning workflows. Centralizing the tutorial within a single interactive file significantly reduces cognitive load for the user, providing a unified, linear progression path that is intuitive to follow and trivially easy to maintain for the development team.

The Marimo notebook will seamlessly blend richly formatted Markdown explanations alongside the executable Python cells. It will clearly delineate the GIVEN/WHEN/THEN behavioral definitions directly above the corresponding code blocks, dynamically guiding the user step-by-step toward their Aha! Moment. By avoiding scattered documentation, the user experiences a cohesive story of data sovereignty and advanced ML processing.

## 5. Tutorial Validation
To ensure the absolute reliability and perpetual correctness of the tutorial experience, the `tutorials/UAT_AND_TUTORIAL.py` file must undergo rigorous automated validation. The system's CI/CD pipeline will be explicitly configured to automatically execute the Marimo notebook in Mock Mode upon every single code commit.

This critical validation step will mathematically verify that all Python cells execute correctly without raising unhandled traceback exceptions. It ensures that the Markdown formatting renders properly across different viewers, and crucially, that the mocked outputs strictly conform to the defined Pydantic schema contracts established in Cycle 01. This guarantees that the tutorial remains permanently functional, never succumbing to bit-rot, and stays perfectly synchronized with the underlying, evolving system architecture.
