class CredentialManager:
    """Secure credential manager for handling API keys without exposing them in logs or dicts."""

    def __init__(self, token: str) -> None:
        self._token = token

    def get_auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def __str__(self) -> str:
        return "<REDACTED CREDENTIAL>"

    def __repr__(self) -> str:
        return "<REDACTED CREDENTIAL>"

    def __len__(self) -> int:
        return len(self._token)
