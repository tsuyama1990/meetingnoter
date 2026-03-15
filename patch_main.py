with open("main.py", "r") as f:
    content = f.read()

content = content.replace("diarizer: Diarizer = PyannoteDiarizer(auth_token=config.pyannote_auth_token.get_auth_header()['Authorization'].split(' ')[1])", "diarizer: Diarizer = PyannoteDiarizer(auth_token=config.pyannote_auth_token._token)")

content = content.replace("google_api_key=__import__('domain_models.credentials', fromlist=['CredentialManager']).CredentialManager(secrets[\"google_api_key\"]),", "google_api_key=CredentialManager(secrets[\"google_api_key\"]),")

content = content.replace("pyannote_auth_token=__import__('domain_models.credentials', fromlist=['CredentialManager']).CredentialManager(secrets[\"pyannote_auth_token\"]),", "pyannote_auth_token=CredentialManager(secrets[\"pyannote_auth_token\"]),")

content = content.replace("from domain_models.config import resolve_secrets", "from domain_models.config import resolve_secrets\nfrom domain_models.credentials import CredentialManager")

with open("main.py", "w") as f:
    f.write(content)

with open("src/domain_models/config.py", "r") as f:
    content = f.read()

content = content.replace("    def cast_to_credential_manager(cls, v: Any) -> __import__('domain_models.credentials', fromlist=['CredentialManager']).CredentialManager:", "    def cast_to_credential_manager(cls, v: Any) -> 'CredentialManager':")

with open("src/domain_models/config.py", "w") as f:
    f.write(content)
