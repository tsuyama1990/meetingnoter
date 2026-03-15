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

    def download(self, file_id: str) -> AudioSource:
        """Downloads an audio file securely and returns an AudioSource."""
        temp_file_path = ""
        try:
            # Construct Google Drive download URL for API key usage securely
            # API key should not be in the query string to prevent log leakage.
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            headers = {"Authorization": f"Bearer {self.config.google_api_key}"}

            response = self.http_client.get(
                url, headers=headers, timeout=30, verify=True, stream=True
            )
            response.raise_for_status()

            fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

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
