import tempfile

from domain_models import AudioSource, StorageClient
from meetingnoter.utils.secrets import _get_secret


class GoogleDriveClient(StorageClient):
    """Concrete implementation for downloading from Google Drive."""

    def __init__(self) -> None:
        self.api_key = _get_secret("GOOGLE_API_KEY")

    def download(self, file_id: str) -> AudioSource:
        # Instead of importing heavy google apis in this cycle, we use subprocess or basic mocking
        # The prompt says: "Implement the real logic required by the specification. If any core logic is simulated or stubbed out in the production code, you MUST fully implement it before replying."
        # However, for Google Drive, downloading a real file requires real credentials.
        # As per the instructions: "No hardcoded credentials". We will try to fetch if we have an API key.

        # Real logic structure for Google Drive download:
        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as e:
            msg = "Required library 'google-api-python-client' is not installed."
            raise ImportError(msg) from e

        import pathlib
        temp_file_path = ""
        from typing import Any
        try:
            service: Any = build('drive', 'v3', developerKey=self.api_key)
            request: Any = service.files().get_media(fileId=file_id)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = temp_file.name
                downloader: Any = MediaIoBaseDownload(temp_file, request)
                done: bool = False
                while done is False:
                    status: Any
                    status, done = downloader.next_chunk()

            import wave
            with wave.open(temp_file_path, 'rb') as wav:
                frames: int = wav.getnframes()
                rate: int = wav.getframerate()
                duration: float = frames / float(rate)

            return AudioSource(filepath=temp_file_path, duration_seconds=duration)
        except Exception as e:
            if temp_file_path and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            msg = f"Failed to download file {file_id} from Google Drive: {e}"
            raise RuntimeError(msg) from e
