import os
import tempfile

from domain_models import AudioSource, StorageClient


class GoogleDriveClient(StorageClient):
    """Concrete implementation for downloading from Google Drive."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "mock_key_for_now")
        # In a real cycle 02 scenario, we would initialize googleapiclient.discovery.build

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

        try:
            service = build('drive', 'v3', developerKey=self.api_key)
            request = service.files().get_media(fileId=file_id)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                downloader = MediaIoBaseDownload(temp_file, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

            import wave
            with wave.open(temp_file.name, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)

            return AudioSource(filepath=temp_file.name, duration_seconds=duration)
        except Exception as e:
            msg = f"Failed to download file {file_id} from Google Drive: {e}"
            raise RuntimeError(msg) from e
