import logging
import os
import pathlib
import subprocess
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

            # Convert ffmpeg_path to ffprobe safely
            ffmpeg_path = pathlib.Path(self.config.ffmpeg_path)
            ffprobe_base = ffmpeg_path.name.replace("ffmpeg", "ffprobe")
            ffprobe_path = (
                str(ffmpeg_path.parent / ffprobe_base) if ffmpeg_path.parent.name else ffprobe_base
            )

            cmd = [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                temp_file_path,
            ]
            try:
                output = subprocess.check_output(cmd)  # noqa: S603
                duration = float(output.decode("utf-8").strip())
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                # Fallback to wave if ffprobe fails (e.g., if it's missing but we have a valid wav)
                with wave.open(temp_file_path, "rb") as wav:
                    frames: int = wav.getnframes()
                    rate: int = wav.getframerate()
                    duration = frames / float(rate)

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
