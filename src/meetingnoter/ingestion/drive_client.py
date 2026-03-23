import logging
import os
import pathlib
import tempfile
import wave

import requests

from domain_models import AudioSource, PipelineConfig, StorageClient

logger = logging.getLogger(__name__)


class GoogleDriveClient(StorageClient):
    """Concrete implementation for downloading from Google Drive securely."""

    def __init__(self, config: PipelineConfig, http_client: requests.Session | None = None) -> None:
        """
        Initializes the client with the provided PipelineConfig.
        Configuration must be injected by Dependency Injection using PipelineConfig.
        Optionally inject an http_client for testability.
        """
        self.config = config
        self.http_client = http_client or requests.Session()

    def download(self, file_id: str) -> AudioSource:  # noqa: C901, PLR0915
        """Downloads an audio file securely and returns an AudioSource."""
        temp_file_path = ""
        try:
            # Task 5: Pass API key as a query parameter (correct for API key auth).
            # Bearer token headers are for OAuth 2.0 tokens, not API keys.
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={self.config.google_api_key}"

            response = self.http_client.get(url, timeout=30, verify=True, stream=True)
            response.raise_for_status()

            raw_fd, raw_path = tempfile.mkstemp()
            try:
                with os.fdopen(raw_fd, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Use ffmpeg to safely convert the raw file into a valid WAV file
                fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)

                import shutil
                import subprocess

                ffmpeg_path = shutil.which("ffmpeg")
                if not ffmpeg_path:
                    msg = "ffmpeg is not installed or not in PATH."
                    raise RuntimeError(msg)

                try:
                    subprocess.run(  # noqa: S603
                        [ffmpeg_path, "-y", "-i", raw_path, temp_file_path],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr
                    logger.exception("FFmpeg transcoding failed: %s", error_msg)
                    msg = "FFmpeg transcoding failed. Check logs for details."
                    raise RuntimeError(msg) from e
            finally:
                raw_path_obj = pathlib.Path(raw_path)
                if raw_path_obj.exists():
                    raw_path_obj.unlink()

            with wave.open(temp_file_path, "rb") as wav:
                frames: int = wav.getnframes()
                rate: int = wav.getframerate()
                duration: float = frames / float(rate)

            return AudioSource(filepath=temp_file_path, duration_seconds=duration)
        except requests.exceptions.HTTPError as e:
            if temp_file_path and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            msg = "HTTP Error during download. Please check permissions and credentials."
            logger.exception("HTTP Error during download")
            raise RuntimeError(msg) from e
        except requests.exceptions.RequestException as e:
            if temp_file_path and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            msg = "Network or Request Error during download."
            logger.exception("Network or Request Error during download")
            raise RuntimeError(msg) from e
        except Exception as e:
            if temp_file_path and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            msg = "Unexpected error while parsing audio."
            logger.exception("Unexpected error while parsing audio")
            raise RuntimeError(msg) from e
