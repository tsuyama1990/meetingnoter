import pathlib
import tempfile
import urllib.error
import urllib.request
import wave

from domain_models import AudioSource, StorageClient


class GoogleDriveClient(StorageClient):
    """Concrete implementation for downloading from Google Drive securely."""

    def __init__(self, api_key: str) -> None:
        """
        Initializes the client with the provided Google Drive API Key.
        Configuration must be injected by Dependency Injection using PipelineConfig.
        """
        self.api_key = api_key

    def download(self, file_id: str) -> AudioSource:
        """Downloads an audio file and returns an AudioSource."""
        temp_file_path = ""
        try:
            # Construct Google Drive download URL for API key usage
            url = (
                f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={self.api_key}"
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = temp_file.name

            urllib.request.urlretrieve(url, temp_file_path)  # noqa: S310

            with wave.open(temp_file_path, "rb") as wav:
                frames: int = wav.getnframes()
                rate: int = wav.getframerate()
                duration: float = frames / float(rate)

            return AudioSource(filepath=temp_file_path, duration_seconds=duration)
        except urllib.error.URLError as e:
            if temp_file_path and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            msg = f"Failed to download file {file_id} from Google Drive using REST API. Please ensure your GOOGLE_API_KEY is valid and the file has sharing enabled: {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            if temp_file_path and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            msg = f"Unexpected error during download: {e}"
            raise RuntimeError(msg) from e
